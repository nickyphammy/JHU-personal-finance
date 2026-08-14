"""Deterministic recommendations engine + feedback loop (v1).

Goal: surface up to 3 high-impact, explainable suggestions to reduce discretionary spend.

This is intentionally rule-based (no LLM) so it is cheap, fast, and testable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


# [ASSUMPTION] Lookback window for pattern detection (keeps recommendations relevant).
DEFAULT_LOOKBACK_DAYS = 180

# [ASSUMPTION] Minimum distinct months to consider a merchant "recurring".
MIN_RECURRING_MONTHS = 3

# [ASSUMPTION] Minimum average monthly spend (USD) for a recurring merchant to be worth recommending.
# Set low enough to catch common streaming subscriptions.
MIN_RECURRING_AVG_MONTHLY_USD = 10.0


SUBSCRIPTION_KEYWORDS = (
    "subscription",
    "stream",
    "netflix",
    "spotify",
    "hulu",
    "prime",
    "membership",
    "recurring",
)


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _as_date(x: Any) -> date:
    return pd.to_datetime(x).date()


def _stable_id(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def feedback_path(root: Path, *, client_id: int) -> Path:
    return root / "artifacts" / f"recommendation_feedback_{int(client_id)}.json"


def load_feedback(root: Path, *, client_id: int) -> dict[str, dict[str, Any]]:
    path = feedback_path(root, client_id=client_id)
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def write_feedback(
    root: Path,
    *,
    client_id: int,
    rec_id: str,
    status: str,
    note: str | None = None,
) -> Path:
    fb = load_feedback(root, client_id=client_id)
    fb[str(rec_id)] = {
        "status": status,
        "note": note,
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    path = feedback_path(root, client_id=client_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fb, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _is_subscription_row(row: pd.Series) -> bool:
    cat = str(row.get("category") or "").strip().lower()
    if cat == "subscriptions":
        return True
    desc = str(row.get("mcc_description") or "").strip().lower()
    return any(k in desc for k in SUBSCRIPTION_KEYWORDS)


def detect_recurring_merchants(
    tx: pd.DataFrame, *, as_of_date: date, lookback_days: int = DEFAULT_LOOKBACK_DAYS
) -> pd.DataFrame:
    """Return recurring merchant patterns (month-level spend stability)."""
    _ensure_cols(tx, ["transaction_dt", "amount_usd", "merchant_id", "category", "mcc_description"])
    work = tx.loc[:, ["transaction_dt", "amount_usd", "merchant_id", "category", "mcc_description"]].copy()
    work["transaction_dt"] = pd.to_datetime(work["transaction_dt"], errors="coerce")
    work["amount_usd"] = pd.to_numeric(work["amount_usd"], errors="coerce")
    work = work.loc[work["transaction_dt"].notna() & work["amount_usd"].notna()].copy()
    work["spend_usd"] = work["amount_usd"].where(work["amount_usd"] > 0, 0.0)

    end = pd.to_datetime(as_of_date)
    start = end - pd.Timedelta(days=int(lookback_days))
    work = work.loc[(work["transaction_dt"] >= start) & (work["transaction_dt"] <= end)].copy()
    if work.empty:
        return pd.DataFrame()

    work["month"] = work["transaction_dt"].dt.to_period("M").astype(str)
    work["merchant_id"] = work["merchant_id"].astype(str)
    work["is_subscription"] = work.apply(_is_subscription_row, axis=1)

    by = (
        work.groupby(["merchant_id", "month"], dropna=False)["spend_usd"]
        .sum()
        .reset_index()
        .rename(columns={"spend_usd": "monthly_spend_usd"})
    )

    # aggregate across months for each merchant
    agg = (
        by.groupby("merchant_id", dropna=False)["monthly_spend_usd"]
        .agg(["count", "mean", "sum", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "active_months",
                "mean": "avg_monthly_spend_usd",
                "sum": "total_spend_usd",
                "std": "std_monthly_spend_usd",
            }
        )
    )
    agg["cv_monthly_spend"] = agg["std_monthly_spend_usd"] / agg["avg_monthly_spend_usd"].replace(0, pd.NA)

    sub = (
        work.groupby("merchant_id", dropna=False)["is_subscription"]
        .any()
        .reset_index()
        .rename(columns={"is_subscription": "is_subscription"})
    )
    agg = agg.merge(sub, on="merchant_id", how="left")

    # [ASSUMPTION] Stability threshold for recurring spend; tolerate some noise.
    stable = agg["cv_monthly_spend"].fillna(0.0) <= 0.75
    agg = agg.loc[
        (agg["active_months"] >= int(MIN_RECURRING_MONTHS))
        & (agg["avg_monthly_spend_usd"] >= float(MIN_RECURRING_AVG_MONTHLY_USD))
        & stable
    ].copy()

    return agg.sort_values(["is_subscription", "avg_monthly_spend_usd"], ascending=[False, False])


def detect_top_category_merchant(
    tx: pd.DataFrame,
    *,
    as_of_date: date,
    lookback_days: int = 90,
    discretionary_only: bool = True,
) -> dict[str, Any] | None:
    """Find the highest-spend category and, within it, the top merchant.

    Used to recommend a targeted cutback (category + merchant), not just a vague category cap.
    """
    required = ["transaction_dt", "amount_usd", "category", "merchant_id"]
    _ensure_cols(tx, required)

    cols = list(required)
    if "is_discretionary" in tx.columns:
        cols.append("is_discretionary")
    if "merchant_city" in tx.columns:
        cols.append("merchant_city")
    if "merchant_state" in tx.columns:
        cols.append("merchant_state")

    work = tx.loc[:, cols].copy()
    work["transaction_dt"] = pd.to_datetime(work["transaction_dt"], errors="coerce")
    work["amount_usd"] = pd.to_numeric(work["amount_usd"], errors="coerce")
    work = work.loc[work["transaction_dt"].notna() & work["amount_usd"].notna()].copy()
    if work.empty:
        return None

    work["spend_usd"] = work["amount_usd"].where(work["amount_usd"] > 0, 0.0)
    work["merchant_id"] = work["merchant_id"].astype(str)
    work["category"] = work["category"].astype(str)

    end = pd.to_datetime(as_of_date)
    start = end - pd.Timedelta(days=int(lookback_days))
    work = work.loc[(work["transaction_dt"] >= start) & (work["transaction_dt"] <= end)].copy()
    if work.empty:
        return None

    if discretionary_only and "is_discretionary" in work.columns:
        work = work.loc[work["is_discretionary"] == True].copy()  # noqa: E712
        if work.empty:
            return None

    by_cat = work.groupby("category", dropna=False)["spend_usd"].sum().sort_values(ascending=False)
    if by_cat.empty or float(by_cat.iloc[0]) <= 0:
        return None

    top_category = str(by_cat.index[0])
    category_spend = float(by_cat.iloc[0])
    cat_rows = work.loc[work["category"] == top_category].copy()

    by_merch = (
        cat_rows.groupby("merchant_id", dropna=False)["spend_usd"]
        .sum()
        .sort_values(ascending=False)
    )
    if by_merch.empty or float(by_merch.iloc[0]) <= 0:
        return None

    top_merchant = str(by_merch.index[0])
    merchant_spend = float(by_merch.iloc[0])
    merchant_share = merchant_spend / category_spend if category_spend > 0 else 0.0

    city = None
    state = None
    sample = cat_rows.loc[cat_rows["merchant_id"] == top_merchant]
    if not sample.empty:
        if "merchant_city" in sample.columns:
            city_val = sample["merchant_city"].dropna()
            city = None if city_val.empty else str(city_val.iloc[0])
        if "merchant_state" in sample.columns:
            state_val = sample["merchant_state"].dropna()
            state = None if state_val.empty else str(state_val.iloc[0])

    # [ASSUMPTION] 15% cut of this merchant's window spend, normalized to ~monthly.
    months = max(float(lookback_days) / 30.0, 1.0)
    estimated_monthly_savings = 0.15 * (merchant_spend / months)

    return {
        "category": top_category,
        "merchant_id": top_merchant,
        "merchant_city": city,
        "merchant_state": state,
        "category_spend_usd": category_spend,
        "merchant_spend_usd": merchant_spend,
        "merchant_share_of_category": merchant_share,
        "lookback_days": int(lookback_days),
        "estimated_monthly_savings_usd": float(max(estimated_monthly_savings, 0.0)),
    }


@dataclass(frozen=True)
class Recommendation:
    rec_id: str
    title: str
    action: str
    rationale: str
    estimated_monthly_savings_usd: float
    rule: str
    meta: dict[str, Any]


def generate_recommendations(
    tx: pd.DataFrame,
    *,
    client_id: int,
    as_of_date: date,
    monthly_limit_usd: float | None,
    root: Path,
    max_recommendations: int = 3,
) -> list[Recommendation]:
    """Return up to N actionable spend-cut recommendations, filtered by feedback.

    Only two rule families are emitted:
    - costly recurring subscriptions
    - cut back at the top merchant in the highest discretionary category

    `monthly_limit_usd` is accepted for API compatibility with the UI but is not used
    to invent generic “lower your limit” tips.
    """
    _ = monthly_limit_usd  # reserved for future impact ranking vs limit
    fb = load_feedback(root, client_id=client_id)

    recs: list[Recommendation] = []

    recurring = detect_recurring_merchants(tx, as_of_date=as_of_date)
    if not recurring.empty:
        # Costly recurring subscriptions first (already sorted by avg monthly spend desc).
        top_sub = recurring.loc[recurring["is_subscription"] == True].head(3)  # noqa: E712
        for _, r in top_sub.iterrows():
            merchant_id = str(r["merchant_id"])
            avg_monthly = float(r["avg_monthly_spend_usd"])
            rec_id = _stable_id("recurring_subscription", str(client_id), merchant_id)
            if (fb.get(rec_id) or {}).get("status") == "dismissed":
                continue
            recs.append(
                Recommendation(
                    rec_id=rec_id,
                    title="Review costly recurring subscription",
                    action="Cancel, pause, or downgrade if unused.",
                    rationale=(
                        f"Merchant `{merchant_id}` shows recurring monthly spend across "
                        f"{int(r['active_months'])} month(s) (avg ${avg_monthly:,.2f}/mo)."
                    ),
                    estimated_monthly_savings_usd=avg_monthly,
                    rule="recurring_subscription",
                    meta={"merchant_id": merchant_id, "active_months": int(r["active_months"])},
                )
            )

    # Top discretionary category + highest-spend merchant within it
    if len(recs) < max_recommendations:
        hit = detect_top_category_merchant(tx, as_of_date=as_of_date, lookback_days=90)
        if hit is not None:
            category = str(hit["category"])
            merchant_id = str(hit["merchant_id"])
            rec_id = _stable_id("cutback_top_category_merchant", str(client_id), category, merchant_id)
            if (fb.get(rec_id) or {}).get("status") != "dismissed":
                place_bits = [merchant_id]
                if hit.get("merchant_city"):
                    place_bits.append(str(hit["merchant_city"]))
                if hit.get("merchant_state"):
                    place_bits.append(str(hit["merchant_state"]))
                place = " / ".join(place_bits)
                share_pct = float(hit["merchant_share_of_category"]) * 100.0
                recs.append(
                    Recommendation(
                        rec_id=rec_id,
                        title=f"Cut back {category} at your top merchant",
                        action=(
                            f"Reduce visits or ticket size at merchant `{place}` "
                            f"(your largest {category} merchant)."
                        ),
                        rationale=(
                            f"In the last ~{int(hit['lookback_days'])} days, `{category}` is your "
                            f"highest discretionary category (${float(hit['category_spend_usd']):,.2f}). "
                            f"Merchant `{merchant_id}` alone is ${float(hit['merchant_spend_usd']):,.2f} "
                            f"({share_pct:.0f}% of that category)."
                        ),
                        estimated_monthly_savings_usd=float(hit["estimated_monthly_savings_usd"]),
                        rule="cutback_top_category_merchant",
                        meta={
                            "category": category,
                            "merchant_id": merchant_id,
                            "merchant_city": hit.get("merchant_city"),
                            "merchant_state": hit.get("merchant_state"),
                            "category_spend_usd": float(hit["category_spend_usd"]),
                            "merchant_spend_usd": float(hit["merchant_spend_usd"]),
                            "lookback_days": int(hit["lookback_days"]),
                        },
                    )
                )

    # Enforce max and keep most impactful first (estimated savings desc)
    allowed = {"recurring_subscription", "cutback_top_category_merchant"}
    recs = [r for r in recs if r.rule in allowed]
    recs = sorted(recs, key=lambda r: r.estimated_monthly_savings_usd, reverse=True)
    return recs[: int(max_recommendations)]
