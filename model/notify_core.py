"""Notification engine for budget thresholds, warnings, and tips.

Deterministic and artifact-grounded (no LLM): emits events for app/email channels
and writes `artifacts/events_{client_id}.jsonl`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


# [ASSUMPTION] Standard ClearLedger budget alert ladder from the product write-up.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.70, 0.85, 0.95)

# [ASSUMPTION] Multi-channel delivery targets; v1 records payloads for all channels
# and renders the `app` channel in Streamlit (no live SMTP required for demo).
DEFAULT_CHANNELS: tuple[str, ...] = ("app", "email")

# [ASSUMPTION] Warn when runway is short even if MTD utilization is still mid-range.
SHORT_RUNWAY_DAYS = 7


@dataclass(frozen=True)
class NotificationEvent:
    event_id: str
    client_id: int
    as_of_date: str
    kind: str  # budget_threshold | warning | tip
    severity: str  # info | warning | critical
    title: str
    message: str
    channels: list[str]
    threshold_pct: float | None = None
    utilization_pct: float | None = None
    projected_utilization_pct: float | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_id(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _pct_label(x: float) -> str:
    return f"{x * 100:.0f}%"


def _fmt_money(x: float | None) -> str:
    if x is None:
        return "N/A"
    return f"${float(x):,.2f}"


def events_path(root: Path, *, client_id: int) -> Path:
    return root / "artifacts" / f"events_{int(client_id)}.jsonl"


def load_events(root: Path, *, client_id: int) -> list[dict[str, Any]]:
    path = events_path(root, client_id=client_id)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def write_events(
    root: Path,
    *,
    client_id: int,
    events: Sequence[NotificationEvent],
) -> Path:
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    path = events_path(root, client_id=client_id)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
    return path


def crossed_thresholds(
    utilization_pct: float | None,
    *,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> list[float]:
    """Return thresholds that utilization has reached or exceeded."""
    if utilization_pct is None:
        return []
    util = float(utilization_pct)
    return sorted(t for t in thresholds if util >= float(t))


def threshold_status(
    utilization_pct: float | None,
    *,
    projected_utilization_pct: float | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict[str, Any]]:
    """UI helper: status for each budget threshold (crossed by MTD and/or projected)."""
    rows: list[dict[str, Any]] = []
    mtd = float(utilization_pct) if utilization_pct is not None else None
    proj = float(projected_utilization_pct) if projected_utilization_pct is not None else None
    for thr in thresholds:
        mtd_hit = mtd is not None and mtd >= float(thr)
        proj_hit = proj is not None and proj >= float(thr)
        if mtd_hit:
            status = "crossed"
        elif proj_hit:
            status = "projected"
        else:
            status = "ok"
        rows.append(
            {
                "threshold_pct": float(thr),
                "label": _pct_label(float(thr)),
                "status": status,
                "mtd_hit": mtd_hit,
                "projected_hit": proj_hit,
            }
        )
    return rows


def generate_notifications(
    *,
    client_id: int,
    as_of_date: date | str,
    utilization_pct: float | None,
    projected_utilization_pct: float | None = None,
    mtd_spend_usd: float | None = None,
    monthly_limit_usd: float | None = None,
    remaining_budget_usd: float | None = None,
    days_to_limit_estimate: int | None = None,
    overspend_risk: bool = False,
    top_recommendation_title: str | None = None,
    top_recommendation_action: str | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    channels: Sequence[str] = DEFAULT_CHANNELS,
) -> list[NotificationEvent]:
    """Build grounded notification events for a client as-of date.

    Priority order (highest first):
    1. Critical budget thresholds (95%)
    2. Warning thresholds (85%, 70%)
    3. Overspend / short-runway warnings
    4. Personalized tip from the top recommendation
    """
    as_of = pd_to_date(as_of_date)
    as_of_s = as_of.isoformat()
    channel_list = [str(c) for c in channels]
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    events: list[NotificationEvent] = []

    mtd_hit = set(crossed_thresholds(utilization_pct, thresholds=thresholds))
    proj_hit = set(crossed_thresholds(projected_utilization_pct, thresholds=thresholds))
    # Alert if either MTD or projected utilization is past the threshold
    hit = sorted(mtd_hit | proj_hit)
    for thr in hit:
        if thr >= 0.95:
            severity = "critical"
        elif thr >= 0.85:
            severity = "warning"
        else:
            severity = "info"

        if thr in mtd_hit:
            title = f"Past {_pct_label(thr)} budget threshold"
            message = (
                f"You have passed the {_pct_label(thr)} discretionary budget threshold. "
                f"As of {as_of_s}, month-to-date spend is {_fmt_money(mtd_spend_usd)} of "
                f"{_fmt_money(monthly_limit_usd)} "
                f"({_pct_label(float(utilization_pct or 0))} utilization). "
                f"Remaining budget: {_fmt_money(remaining_budget_usd)}."
            )
        else:
            title = f"Projected to pass {_pct_label(thr)} budget threshold"
            message = (
                f"Projected month-end utilization "
                f"({_pct_label(float(projected_utilization_pct or 0))}) is at or above "
                f"the {_pct_label(thr)} threshold. "
                f"MTD spend is currently {_fmt_money(mtd_spend_usd)} of "
                f"{_fmt_money(monthly_limit_usd)} "
                f"({_pct_label(float(utilization_pct or 0))} utilization)."
            )

        events.append(
            NotificationEvent(
                event_id=_stable_id("threshold", str(client_id), as_of_s, f"{thr:.2f}"),
                client_id=int(client_id),
                as_of_date=as_of_s,
                kind="budget_threshold",
                severity=severity,
                title=title,
                message=message,
                channels=list(channel_list),
                threshold_pct=float(thr),
                utilization_pct=float(utilization_pct) if utilization_pct is not None else None,
                projected_utilization_pct=(
                    float(projected_utilization_pct)
                    if projected_utilization_pct is not None
                    else None
                ),
                created_at=now,
            )
        )

    if overspend_risk or (
        projected_utilization_pct is not None and float(projected_utilization_pct) >= 1.0
    ):
        events.append(
            NotificationEvent(
                event_id=_stable_id("warning", "overspend", str(client_id), as_of_s),
                client_id=int(client_id),
                as_of_date=as_of_s,
                kind="warning",
                severity="critical",
                title="Projected to exceed your discretionary limit",
                message=(
                    f"Projected month-end utilization is "
                    f"{_pct_label(float(projected_utilization_pct or 0))}. "
                    f"Consider cutting discretionary spend before month end."
                ),
                channels=list(channel_list),
                utilization_pct=float(utilization_pct) if utilization_pct is not None else None,
                projected_utilization_pct=(
                    float(projected_utilization_pct)
                    if projected_utilization_pct is not None
                    else None
                ),
                created_at=now,
            )
        )
    elif (
        days_to_limit_estimate is not None
        and int(days_to_limit_estimate) <= SHORT_RUNWAY_DAYS
        and int(days_to_limit_estimate) >= 0
    ):
        events.append(
            NotificationEvent(
                event_id=_stable_id("warning", "runway", str(client_id), as_of_s),
                client_id=int(client_id),
                as_of_date=as_of_s,
                kind="warning",
                severity="warning",
                title=f"Only {int(days_to_limit_estimate)} day(s) of discretionary runway left",
                message=(
                    f"At the current pace you may hit your limit in about "
                    f"{int(days_to_limit_estimate)} day(s). "
                    f"Remaining budget: {_fmt_money(remaining_budget_usd)}."
                ),
                channels=list(channel_list),
                utilization_pct=float(utilization_pct) if utilization_pct is not None else None,
                projected_utilization_pct=(
                    float(projected_utilization_pct)
                    if projected_utilization_pct is not None
                    else None
                ),
                created_at=now,
            )
        )

    if top_recommendation_title:
        tip_body = top_recommendation_title
        if top_recommendation_action:
            tip_body = f"{top_recommendation_title} — {top_recommendation_action}"
        events.append(
            NotificationEvent(
                event_id=_stable_id("tip", str(client_id), as_of_s, top_recommendation_title),
                client_id=int(client_id),
                as_of_date=as_of_s,
                kind="tip",
                severity="info",
                title="Personalized tip",
                message=tip_body,
                channels=list(channel_list),
                utilization_pct=float(utilization_pct) if utilization_pct is not None else None,
                projected_utilization_pct=(
                    float(projected_utilization_pct)
                    if projected_utilization_pct is not None
                    else None
                ),
                created_at=now,
            )
        )

    # Highest severity / threshold first for UI display
    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    events.sort(
        key=lambda e: (
            severity_rank.get(e.severity, 9),
            -(e.threshold_pct or 0.0),
            e.kind,
            e.title,
        )
    )
    return events


def pd_to_date(x: date | str) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    # local import keeps module light for notebooks that only need types
    import pandas as pd

    return pd.to_datetime(x).date()


def run_notifications_for_client(
    client_id: int,
    *,
    root: Path,
    as_of_date: date | str | None = None,
    budget: dict[str, Any] | None = None,
    prediction: dict[str, Any] | None = None,
    recommendations: Sequence[Any] | None = None,
    write_artifact: bool = True,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    channels: Sequence[str] = DEFAULT_CHANNELS,
) -> dict[str, Any]:
    """Generate notifications from budget/prediction artifacts (or provided dicts)."""
    art = root / "artifacts"

    if budget is None:
        budget_path = art / f"budget_utilization_{int(client_id)}.json"
        if budget_path.exists():
            budget = json.loads(budget_path.read_text(encoding="utf-8"))
        else:
            budget = {}

    if prediction is None:
        runway_path = art / f"runway_{int(client_id)}.json"
        if runway_path.exists():
            prediction = json.loads(runway_path.read_text(encoding="utf-8"))
        else:
            prediction = {}

    as_of = as_of_date or budget.get("as_of_date") or prediction.get("as_of_date")
    if as_of is None:
        raise ValueError("as_of_date is required when budget/prediction artifacts omit it")

    top_title = None
    top_action = None
    if recommendations:
        top = recommendations[0]
        top_title = getattr(top, "title", None) or (top.get("title") if isinstance(top, dict) else None)
        top_action = getattr(top, "action", None) or (
            top.get("action") if isinstance(top, dict) else None
        )

    events = generate_notifications(
        client_id=int(client_id),
        as_of_date=as_of,
        utilization_pct=_maybe_float(budget.get("utilization_pct")),
        projected_utilization_pct=_maybe_float(prediction.get("projected_utilization_pct")),
        mtd_spend_usd=_maybe_float(
            budget.get("mtd_discretionary_spend_usd")
            or prediction.get("mtd_discretionary_spend_usd")
        ),
        monthly_limit_usd=_maybe_float(
            budget.get("monthly_discretionary_limit_usd")
            or prediction.get("monthly_discretionary_limit_usd")
        ),
        remaining_budget_usd=_maybe_float(prediction.get("remaining_budget_usd")),
        days_to_limit_estimate=_maybe_int(prediction.get("days_to_limit_estimate")),
        overspend_risk=bool(prediction.get("overspend_risk")),
        top_recommendation_title=top_title,
        top_recommendation_action=top_action,
        thresholds=thresholds,
        channels=channels,
    )

    out_path = None
    if write_artifact:
        out_path = write_events(root, client_id=int(client_id), events=events)

    return {
        "client_id": int(client_id),
        "as_of_date": pd_to_date(as_of).isoformat(),
        "events": events,
        "events_path": str(out_path) if out_path else None,
        "app_events": [e for e in events if "app" in e.channels],
        "email_events": [e for e in events if "email" in e.channels],
    }


def _maybe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _maybe_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None
