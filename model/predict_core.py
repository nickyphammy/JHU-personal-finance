"""Regression-based spending prediction (budget breach / overspend risk).

Goal (v1):
- Forecast month-end discretionary spend from month-to-date behavior.
- Derive overspending risk and a days-to-limit estimate from the forecast.

Trains on all users; can score any single user (e.g., for UI selection).
"""

from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.analytics_core import is_discretionary, mcc_to_category, project_root


_CURRENCY_RE = re.compile(r"[^0-9.\\-]+")


def parse_currency_to_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    s = _CURRENCY_RE.sub("", s)
    if s in {"", "-", ".", "-."}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_user_limits(users_csv: Path) -> pd.Series:
    """Return monthly discretionary limit (USD) indexed by client_id (int)."""
    users = pd.read_csv(users_csv, dtype=str)
    if "id" not in users.columns:
        raise ValueError("users.csv missing required column 'id'")
    if "monthly_discretionary_limits" not in users.columns:
        raise ValueError("users.csv missing required column 'monthly_discretionary_limits'")

    users = users.loc[:, ["id", "monthly_discretionary_limits"]].copy()
    users["client_id"] = pd.to_numeric(users["id"], errors="coerce")
    users["limit_usd"] = users["monthly_discretionary_limits"].map(parse_currency_to_float)
    limits = users.dropna(subset=["client_id"]).set_index("client_id")["limit_usd"]
    limits.index = limits.index.astype(int)
    return limits


def iter_transactions_ndjson(
    ndjson_path: Path, *, chunksize: int = 200_000
) -> Iterable[pd.DataFrame]:
    if not ndjson_path.exists():
        raise FileNotFoundError(
            f"Missing {ndjson_path}. Run data_processing/clean.ipynb to produce artifacts."
        )
    yield from pd.read_json(ndjson_path, lines=True, dtype=False, chunksize=chunksize)  # type: ignore[arg-type]


def _ensure_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_daily_discretionary_spend(
    ndjson_path: Path, *, chunksize: int = 200_000
) -> pd.DataFrame:
    """Return daily discretionary spend for all users.

    Output columns: client_id (int), day (datetime64[ns]), discretionary_spend_usd (float)
    """
    daily_parts: list[pd.DataFrame] = []
    for chunk in iter_transactions_ndjson(ndjson_path, chunksize=chunksize):
        _ensure_required_columns(
            chunk,
            ["client_id", "transaction_dt", "amount_usd", "mcc_code", "mcc_description"],
        )

        work = chunk.loc[:, ["client_id", "transaction_dt", "amount_usd", "mcc_code", "mcc_description"]].copy()
        work["client_id"] = pd.to_numeric(work["client_id"], errors="coerce")
        work["transaction_dt"] = pd.to_datetime(work["transaction_dt"], errors="coerce")
        work["amount_usd"] = pd.to_numeric(work["amount_usd"], errors="coerce")

        work = work.loc[work["client_id"].notna() & work["transaction_dt"].notna() & work["amount_usd"].notna()].copy()
        if work.empty:
            continue

        work["category"] = work.apply(
            lambda r: mcc_to_category(r.get("mcc_code"), r.get("mcc_description")),
            axis=1,
        )
        work["is_discretionary"] = work["category"].map(is_discretionary)
        work["spend_usd"] = work["amount_usd"].where(work["amount_usd"] > 0, 0.0)
        work = work.loc[work["is_discretionary"]].copy()
        if work.empty:
            continue

        work["day"] = work["transaction_dt"].dt.normalize()
        daily = (
            work.groupby(["client_id", "day"], dropna=False)["spend_usd"]
            .sum()
            .reset_index()
            .rename(columns={"spend_usd": "discretionary_spend_usd"})
        )
        daily_parts.append(daily)

    if not daily_parts:
        return pd.DataFrame(columns=["client_id", "day", "discretionary_spend_usd"])

    daily_all = pd.concat(daily_parts, ignore_index=True)
    daily_all["client_id"] = daily_all["client_id"].astype(int)
    daily_all["day"] = pd.to_datetime(daily_all["day"], errors="coerce")
    daily_all["discretionary_spend_usd"] = pd.to_numeric(
        daily_all["discretionary_spend_usd"], errors="coerce"
    ).fillna(0.0)

    daily_all = (
        daily_all.groupby(["client_id", "day"], dropna=False)["discretionary_spend_usd"]
        .sum()
        .reset_index()
        .sort_values(["client_id", "day"])
    )
    return daily_all


def build_monthly_training_samples(
    daily: pd.DataFrame, *, user_limits: pd.Series
) -> pd.DataFrame:
    """Build supervised samples from daily discretionary spend.

    Target (regression): `month_total_discretionary_spend_usd`.
    One sample per (client_id, day) within each month.
    """
    _ensure_required_columns(daily, ["client_id", "day", "discretionary_spend_usd"])
    work = daily.copy()
    work["client_id"] = pd.to_numeric(work["client_id"], errors="coerce").astype("Int64")
    work["day"] = pd.to_datetime(work["day"], errors="coerce")
    work["discretionary_spend_usd"] = (
        pd.to_numeric(work["discretionary_spend_usd"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    work = work.loc[work["client_id"].notna() & work["day"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    work["client_id"] = work["client_id"].astype(int)
    work["year"] = work["day"].dt.year.astype(int)
    work["month"] = work["day"].dt.month.astype(int)
    work["ym"] = work["day"].dt.to_period("M").astype(str)
    work = work.sort_values(["client_id", "day"])

    # month-level totals + MTD
    work["month_total_discretionary_spend_usd"] = work.groupby(["client_id", "ym"])[
        "discretionary_spend_usd"
    ].transform("sum")
    work["mtd_discretionary_spend_usd"] = work.groupby(["client_id", "ym"])[
        "discretionary_spend_usd"
    ].cumsum()

    # date features
    work["day_of_month"] = work["day"].dt.day.astype(int)
    work["days_in_month"] = work["day"].dt.days_in_month.astype(int)
    work["days_remaining_in_month"] = (work["days_in_month"] - work["day_of_month"]).astype(int)

    # rolling spend signals within the month (short windows to capture habit changes)
    gb = work.groupby(["client_id", "ym"])["discretionary_spend_usd"]
    work["avg_daily_discretionary_spend_7d"] = (
        gb.rolling(window=7, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    )
    work["avg_daily_discretionary_spend_30d"] = (
        gb.rolling(window=30, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    )

    # join monthly limit (USD)
    limits = user_limits.copy()
    limits.index = limits.index.astype(int)
    work["monthly_discretionary_limit_usd"] = work["client_id"].map(limits)
    work["mtd_utilization_pct"] = work["mtd_discretionary_spend_usd"] / work["monthly_discretionary_limit_usd"]
    work["avg7d_utilization_pct"] = work["avg_daily_discretionary_spend_7d"] / work[
        "monthly_discretionary_limit_usd"
    ]

    # drop rows where limit is unknown (can't define overspend risk)
    work = work.loc[work["monthly_discretionary_limit_usd"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    feature_cols = [
        "year",
        "month",
        "day_of_month",
        "days_in_month",
        "days_remaining_in_month",
        "mtd_discretionary_spend_usd",
        "avg_daily_discretionary_spend_7d",
        "avg_daily_discretionary_spend_30d",
        "monthly_discretionary_limit_usd",
        "mtd_utilization_pct",
        "avg7d_utilization_pct",
    ]

    keep = ["client_id", "day", "ym", "month_total_discretionary_spend_usd", *feature_cols]
    return work.loc[:, keep].reset_index(drop=True)


@dataclass(frozen=True)
class RegressionArtifacts:
    model: Pipeline
    feature_columns: tuple[str, ...]
    metrics: dict[str, Any]


def train_month_end_spend_model(
    samples: pd.DataFrame, *, cutoff_day: date | None = None
) -> RegressionArtifacts:
    """Train Ridge regression on all-user samples with a time split."""
    required = [
        "day",
        "month_total_discretionary_spend_usd",
        "monthly_discretionary_limit_usd",
    ]
    _ensure_required_columns(samples, required)

    feature_columns = tuple(
        c
        for c in samples.columns
        if c
        not in {
            "client_id",
            "day",
            "ym",
            "month_total_discretionary_spend_usd",
        }
    )
    X = samples.loc[:, list(feature_columns)].copy()
    y = samples["month_total_discretionary_spend_usd"].astype(float)

    day_series = pd.to_datetime(samples["day"], errors="coerce")
    if cutoff_day is None:
        # [ASSUMPTION] use an 80/20 split by time (not random) for reproducibility
        cutoff = day_series.quantile(0.8)
    else:
        cutoff = pd.to_datetime(cutoff_day)

    train_mask = day_series <= cutoff
    if train_mask.sum() < 10 or (~train_mask).sum() < 10:
        raise ValueError("Not enough samples for a meaningful time split (need >=10 each side).")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=0)),
        ]
    )
    pipeline.fit(X.loc[train_mask], y.loc[train_mask])

    pred_train = pipeline.predict(X.loc[train_mask])
    pred_test = pipeline.predict(X.loc[~train_mask])

    metrics = {
        "cutoff_day": str(pd.to_datetime(cutoff).date()),
        "n_samples": int(len(samples)),
        "n_train": int(train_mask.sum()),
        "n_test": int((~train_mask).sum()),
        "mae_train_usd": float(mean_absolute_error(y.loc[train_mask], pred_train)),
        "mae_test_usd": float(mean_absolute_error(y.loc[~train_mask], pred_test)),
    }
    return RegressionArtifacts(model=pipeline, feature_columns=feature_columns, metrics=metrics)


def predict_month_end_discretionary_spend(
    artifacts: RegressionArtifacts, *, feature_row: dict[str, Any]
) -> float:
    X = pd.DataFrame([feature_row]).loc[:, list(artifacts.feature_columns)]
    return float(artifacts.model.predict(X)[0])


def derive_overspend_risk_and_days_to_limit(
    *,
    predicted_month_end_discretionary_spend_usd: float,
    mtd_discretionary_spend_usd: float,
    monthly_discretionary_limit_usd: float,
    as_of_day: date,
    days_in_month: int,
) -> dict[str, Any]:
    """Post-process the regression output into breach-risk signals.

    days_to_limit is a simple linearized estimate (v1) based on the implied remaining spend rate.
    """
    overspend_risk = bool(predicted_month_end_discretionary_spend_usd > monthly_discretionary_limit_usd)
    remaining_budget = float(monthly_discretionary_limit_usd - mtd_discretionary_spend_usd)

    day_of_month = int(as_of_day.day)
    days_remaining = max(int(days_in_month - day_of_month), 0)
    if remaining_budget <= 0:
        days_to_limit = 0
    else:
        implied_remaining_spend = max(float(predicted_month_end_discretionary_spend_usd - mtd_discretionary_spend_usd), 0.0)
        implied_daily = implied_remaining_spend / max(days_remaining, 1)
        if implied_daily <= 0:
            days_to_limit = None
        else:
            days_to_limit = int(math.ceil(remaining_budget / implied_daily))

    return {
        "overspend_risk": overspend_risk,
        "days_to_limit_estimate": days_to_limit,
        "remaining_budget_usd": remaining_budget,
    }


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def default_paths(root: Path | None = None) -> dict[str, Path]:
    root = root or project_root()
    return {
        "users_csv": root / "data" / "users.csv",
        "transactions_ndjson": root / "artifacts" / "transactions_enriched.json",
        "model_pkl": root / "artifacts" / "runway_model.pkl",
        "metrics_json": root / "artifacts" / "runway_model_metrics.json",
    }


def daily_discretionary_from_transactions(
    tx: pd.DataFrame, *, client_id: int | None = None
) -> pd.DataFrame:
    """Build daily discretionary spend from an in-memory transactions frame.

    Accepts either precomputed `category` / `is_discretionary` columns (UI path)
    or raw MCC fields (same mapping as the NDJSON builder).
    """
    if tx.empty:
        return pd.DataFrame(columns=["client_id", "day", "discretionary_spend_usd"])

    work = tx.copy()
    if "client_id" not in work.columns:
        if client_id is None:
            raise ValueError("transactions missing client_id and no client_id override provided")
        work["client_id"] = int(client_id)
    else:
        work["client_id"] = pd.to_numeric(work["client_id"], errors="coerce")
        if client_id is not None:
            work = work.loc[work["client_id"] == int(client_id)].copy()

    work["transaction_dt"] = pd.to_datetime(work["transaction_dt"], errors="coerce")
    work["amount_usd"] = pd.to_numeric(work["amount_usd"], errors="coerce")
    work = work.loc[
        work["client_id"].notna() & work["transaction_dt"].notna() & work["amount_usd"].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=["client_id", "day", "discretionary_spend_usd"])

    if "category" not in work.columns:
        _ensure_required_columns(work, ["mcc_code", "mcc_description"])
        work["category"] = work.apply(
            lambda r: mcc_to_category(r.get("mcc_code"), r.get("mcc_description")),
            axis=1,
        )
    if "is_discretionary" not in work.columns:
        work["is_discretionary"] = work["category"].map(is_discretionary)

    work = work.loc[work["is_discretionary"]].copy()
    if work.empty:
        return pd.DataFrame(columns=["client_id", "day", "discretionary_spend_usd"])

    work["spend_usd"] = work["amount_usd"].where(work["amount_usd"] > 0, 0.0)
    work["day"] = work["transaction_dt"].dt.normalize()
    daily = (
        work.groupby(["client_id", "day"], dropna=False)["spend_usd"]
        .sum()
        .reset_index()
        .rename(columns={"spend_usd": "discretionary_spend_usd"})
        .sort_values(["client_id", "day"])
    )
    daily["client_id"] = daily["client_id"].astype(int)
    return daily


def load_regression_artifacts(
    model_pkl: Path, *, metrics_json: Path | None = None
) -> RegressionArtifacts:
    if not model_pkl.exists():
        raise FileNotFoundError(
            f"Missing {model_pkl}. Run model/predict.ipynb to train and write runway_model.pkl."
        )
    payload = pickle.loads(model_pkl.read_bytes())
    if "model" not in payload or "feature_columns" not in payload:
        raise ValueError(f"Invalid model payload in {model_pkl}")
    metrics: dict[str, Any] = {}
    if metrics_json is not None and metrics_json.exists():
        metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    return RegressionArtifacts(
        model=payload["model"],
        feature_columns=tuple(payload["feature_columns"]),
        metrics=metrics,
    )


def _feature_row_as_of(samples: pd.DataFrame, *, as_of_day: date) -> pd.Series:
    if samples.empty:
        raise ValueError("No training samples available for this client/as-of date.")
    as_of_ts = pd.Timestamp(as_of_day).normalize()
    exact = samples.loc[pd.to_datetime(samples["day"]).dt.normalize() == as_of_ts]
    if not exact.empty:
        return exact.iloc[-1]

    same_month = samples.loc[
        (pd.to_datetime(samples["day"]).dt.year == as_of_day.year)
        & (pd.to_datetime(samples["day"]).dt.month == as_of_day.month)
        & (pd.to_datetime(samples["day"]).dt.normalize() <= as_of_ts)
    ]
    if not same_month.empty:
        return same_month.sort_values("day").iloc[-1]

    prior = samples.loc[pd.to_datetime(samples["day"]).dt.normalize() <= as_of_ts]
    if prior.empty:
        raise ValueError(f"No discretionary spend on or before {as_of_day} for this client.")
    return prior.sort_values("day").iloc[-1]


def score_client_as_of(
    tx: pd.DataFrame,
    *,
    client_id: int,
    as_of_day: date,
    monthly_limit_usd: float,
    artifacts: RegressionArtifacts,
) -> dict[str, Any]:
    """Score one client for a given as-of day using a trained month-end spend model."""
    if monthly_limit_usd is None or float(monthly_limit_usd) <= 0:
        raise ValueError("monthly_limit_usd must be a positive number")

    daily = daily_discretionary_from_transactions(tx, client_id=int(client_id))
    as_of_ts = pd.Timestamp(as_of_day).normalize()
    daily = daily.loc[pd.to_datetime(daily["day"]).dt.normalize() <= as_of_ts].copy()
    if daily.empty:
        raise ValueError(
            f"No discretionary spend on or before {as_of_day} for client_id={client_id}."
        )

    limits = pd.Series({int(client_id): float(monthly_limit_usd)})
    samples = build_monthly_training_samples(daily, user_limits=limits)
    row = _feature_row_as_of(samples, as_of_day=as_of_day)

    feature_row = {c: row[c] for c in artifacts.feature_columns}
    predicted = predict_month_end_discretionary_spend(artifacts, feature_row=feature_row)
    days_in_month = int(row["days_in_month"])
    mtd = float(row["mtd_discretionary_spend_usd"])
    signals = derive_overspend_risk_and_days_to_limit(
        predicted_month_end_discretionary_spend_usd=predicted,
        mtd_discretionary_spend_usd=mtd,
        monthly_discretionary_limit_usd=float(monthly_limit_usd),
        as_of_day=as_of_day,
        days_in_month=days_in_month,
    )

    projected_util = (
        float(predicted) / float(monthly_limit_usd) if float(monthly_limit_usd) else None
    )
    return {
        "client_id": int(client_id),
        "as_of_date": str(as_of_day),
        "monthly_discretionary_limit_usd": float(monthly_limit_usd),
        "mtd_discretionary_spend_usd": mtd,
        "predicted_month_end_discretionary_spend_usd": float(predicted),
        "projected_utilization_pct": projected_util,
        **signals,
        "feature_row": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in feature_row.items()},
        "model_metrics": artifacts.metrics,
    }


def run_prediction_for_client(
    client_id: int,
    *,
    as_of_date: date | pd.Timestamp,
    transactions: pd.DataFrame,
    monthly_limit_usd: float,
    write_artifacts: bool = True,
    root: Path | None = None,
) -> dict[str, Any]:
    """UI entrypoint: load trained model, score client, optionally write runway JSON."""
    root = root or project_root()
    paths = default_paths(root)
    artifacts = load_regression_artifacts(
        paths["model_pkl"], metrics_json=paths["metrics_json"]
    )
    as_of_day = pd.to_datetime(as_of_date).date()
    prediction = score_client_as_of(
        transactions,
        client_id=int(client_id),
        as_of_day=as_of_day,
        monthly_limit_usd=float(monthly_limit_usd),
        artifacts=artifacts,
    )

    out_path = root / "artifacts" / f"runway_{int(client_id)}.json"
    if write_artifacts:
        # Keep artifact compact (omit nested feature dump duplication in metrics)
        artifact = {k: v for k, v in prediction.items() if k not in {"feature_row", "model_metrics"}}
        artifact["model_cutoff_day"] = artifacts.metrics.get("cutoff_day")
        artifact["mae_test_usd"] = artifacts.metrics.get("mae_test_usd")
        write_json(out_path, artifact)

    return {"prediction": prediction, "artifact_path": out_path if write_artifacts else None}
