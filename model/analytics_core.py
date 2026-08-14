"""Backend analytics: category mapping, aggregations, and artifact I/O.

The Streamlit UI should only call into this module (or read the artifacts it writes).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Dining", ["restaurant", "fast food", "drinking places", "eating places"]),
    ("Groceries", ["grocery", "supermarkets", "miscellaneous food stores"]),
    ("Utilities", ["utilities", "electric", "gas", "water", "sanitary"]),
    ("Transportation", ["service stations", "tolls", "taxicabs", "limousines", "parking"]),
    ("Entertainment", ["amusement", "motion picture", "theaters", "video"]),
    (
        "Shopping",
        ["department stores", "discount stores", "book stores", "home furnishing", "lumber"],
    ),
    ("Transfers", ["money transfer", "transfer"]),
    ("Subscriptions", ["subscription"]),
    ("Fees & Interest", ["fee", "interest"]),
]

DISCRETIONARY_CATEGORIES = {
    "Dining",
    "Entertainment",
    "Shopping",
    "Travel",
    "Subscriptions",
}

# Kept identical when overlaying LLM labels — only these count toward MTD discretionary /
# utilization / runway. LLM-only labels like Healthcare/Housing/Transportation stay non-discretionary.


def project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").is_dir() and (candidate / "artifacts").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate project root containing 'data/' and 'artifacts/'")


def artifacts_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "artifacts"


def mcc_to_category(mcc_code: Any, mcc_description: Any) -> str:
    """Deterministic MCC-based category mapping (rule-based, no LLM)."""
    desc = str(mcc_description).strip().lower() if mcc_description is not None else ""
    code = str(mcc_code).strip() if mcc_code is not None else ""

    if not desc and not code:
        return "Other/Uncategorized"

    for category, keywords in CATEGORY_RULES:
        if any(k in desc for k in keywords):
            return category

    if code.startswith("54"):
        return "Groceries"
    if code.startswith("58"):
        return "Dining"
    if code.startswith("49"):
        return "Utilities"
    if code.startswith(("41", "47")):
        return "Transportation"
    if code.startswith(("53", "59", "52", "57")):
        return "Shopping"

    return "Other/Uncategorized"


def is_discretionary(category: str) -> bool:
    return category in DISCRETIONARY_CATEGORIES


def transaction_categories_path(client_id: int, *, root: Path | None = None) -> Path:
    """Preferred per-client LLM categorize artifact path."""
    return artifacts_dir(root) / f"transaction_categories_{int(client_id)}.jsonl"


@lru_cache(maxsize=64)
def _load_category_overrides_cached(
    client_id: int, path_str: str, mtime_ns: int
) -> tuple[tuple[str, str, str | None], ...]:
    """Return immutable rows: (transaction_id, category, source)."""
    path = Path(path_str)
    rows: list[tuple[str, str, str | None]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "client_id" in obj and str(obj.get("client_id")) != str(client_id):
                continue
            tx_id = obj.get("id")
            category = obj.get("category")
            if tx_id is None or category is None:
                continue
            source = obj.get("source")
            rows.append((str(tx_id), str(category), None if source is None else str(source)))
    return tuple(rows)


def load_transaction_category_overrides(
    client_id: int, *, root: Path | None = None
) -> dict[str, dict[str, str | None]]:
    """Load id → {category, source} from categorize artifacts (empty if missing)."""
    root = root or project_root()
    path = transaction_categories_path(client_id, root=root)
    if not path.exists():
        # All-users file (optional)
        path = artifacts_dir(root) / "transaction_categories.jsonl"
        if not path.exists():
            return {}

    rows = _load_category_overrides_cached(int(client_id), str(path), path.stat().st_mtime_ns)
    out: dict[str, dict[str, str | None]] = {}
    for tx_id, category, source in rows:
        # For the all-users file, rows for other clients may be present; filter above.
        out[tx_id] = {"category": category, "source": source}
    return out


def apply_category_overrides(
    df: pd.DataFrame,
    overrides: dict[str, dict[str, str | None]],
) -> pd.DataFrame:
    """Prefer LLM/artifact categories by transaction id; keep MCC where missing."""
    work = df.copy()
    if "id" not in work.columns:
        work["category_source"] = "mcc"
        work["is_discretionary"] = work["category"].map(is_discretionary)
        return work

    ids = work["id"].astype(str)
    mapped_cat = ids.map(lambda i: (overrides.get(i) or {}).get("category"))
    mapped_src = ids.map(lambda i: (overrides.get(i) or {}).get("source"))

    use_override = mapped_cat.notna()
    work.loc[use_override, "category"] = mapped_cat.loc[use_override].astype(str)
    work["category_source"] = "mcc"
    work.loc[use_override, "category_source"] = mapped_src.loc[use_override].fillna("llm_artifact")
    work["is_discretionary"] = work["category"].map(is_discretionary)
    return work


@lru_cache(maxsize=256)
def _scan_all_users_ndjson_for_client(client_id: int, ndjson_path: str) -> tuple[dict[str, Any], ...]:
    path = Path(ndjson_path)
    if not path.exists():
        raise FileNotFoundError(
            "Missing artifacts/transactions_enriched.json. Run data_processing/clean.ipynb first."
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if str(obj.get("client_id")) == str(client_id):
                records.append(obj)
    return tuple(records)


def load_transaction_records(
    client_id: int, *, root: Path | None = None
) -> tuple[list[dict[str, Any]], Path]:
    """Load one client's transactions. Prefer focus JSON; fall back to all-users NDJSON."""
    root = root or project_root()
    art = artifacts_dir(root)
    focus_path = art / f"transactions_enriched_{client_id}.json"
    all_path = art / "transactions_enriched.json"

    if focus_path.exists():
        records = json.loads(focus_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list in {focus_path}")
        return records, focus_path

    records = list(_scan_all_users_ndjson_for_client(int(client_id), str(all_path)))
    return records, all_path


def load_transactions_for_client(client_id: int, *, root: Path | None = None) -> pd.DataFrame:
    records, _source = load_transaction_records(client_id, root=root)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    if "client_id" in df.columns:
        df = df.loc[df["client_id"].astype(str) == str(client_id)].copy()

    if "transaction_dt" in df.columns:
        df["transaction_dt"] = pd.to_datetime(df["transaction_dt"], errors="coerce")
    if "amount_usd" in df.columns:
        df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")

    # Baseline: deterministic MCC mapping (always available).
    df["category"] = df.apply(
        lambda r: mcc_to_category(r.get("mcc_code"), r.get("mcc_description")), axis=1
    )
    # Overlay: prefer LLM categorize artifact when present for this client.
    overrides = load_transaction_category_overrides(int(client_id), root=root)
    df = apply_category_overrides(df, overrides)
    return df


def compute_analytics(df: pd.DataFrame, *, as_of_date: pd.Timestamp) -> dict[str, Any]:
    """Compute spend/budget/pattern analytics for an already-loaded client dataframe."""
    work = df.copy()
    work = work.loc[work["amount_usd"].notna()].copy()
    work["spend_usd"] = work["amount_usd"].where(work["amount_usd"] > 0, 0.0)

    by_category = (
        work.groupby("category", dropna=False)["spend_usd"].sum().sort_values(ascending=False)
    )
    spend_by_category = {str(k): float(v) for k, v in by_category.items()}

    by_mcc = (
        work.groupby(["mcc_code", "mcc_description"], dropna=False)["spend_usd"]
        .sum()
        .sort_values(ascending=False)
    )
    spend_by_mcc = [
        {
            "mcc_code": (None if pd.isna(k[0]) else str(k[0])),
            "mcc_description": (None if pd.isna(k[1]) else str(k[1])),
            "spend_usd": float(v),
        }
        for k, v in by_mcc.items()
    ]

    work = work.loc[work["transaction_dt"].notna()].copy()
    work["month"] = work["transaction_dt"].dt.to_period("M").astype(str)
    work["txn_date"] = work["transaction_dt"].dt.normalize()
    work["dow"] = work["transaction_dt"].dt.day_name()

    by_month = work.groupby("month")["spend_usd"].sum().sort_index()

    # Average daily spend by weekday (not lifetime totals for every Monday, etc.)
    daily = work.groupby(["txn_date", "dow"], dropna=False)["spend_usd"].sum().reset_index()
    avg_by_dow = daily.groupby("dow", dropna=False)["spend_usd"].mean()
    dow_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    by_dow_avg = {d: float(avg_by_dow[d]) for d in dow_order if d in avg_by_dow.index}

    top_merchants = (
        work.groupby(["merchant_id", "merchant_city", "merchant_state"], dropna=False)["spend_usd"]
        .sum()
        .sort_values(ascending=False)
        .head(25)
    )

    patterns = {
        "total_spend_usd": float(work["spend_usd"].sum()),
        "by_month_usd": {str(k): float(v) for k, v in by_month.items()},
        "avg_daily_spend_by_day_of_week_usd": by_dow_avg,
        # Backward-compatible alias used by older UI code
        "by_day_of_week_usd": by_dow_avg,
        "top_merchants_usd": [
            {
                "merchant_id": (None if pd.isna(k[0]) else str(k[0])),
                "merchant_city": (None if pd.isna(k[1]) else str(k[1])),
                "merchant_state": (None if pd.isna(k[2]) else str(k[2])),
                "spend_usd": float(v),
            }
            for k, v in top_merchants.items()
        ],
    }

    if "is_discretionary" not in work.columns:
        work["is_discretionary"] = work["category"].map(is_discretionary)

    monthly_limit_series = pd.to_numeric(
        df.get("monthly_discretionary_limit_usd"), errors="coerce"
    ).dropna()
    monthly_limit = float(monthly_limit_series.iloc[0]) if not monthly_limit_series.empty else None

    as_of = as_of_date.normalize()
    month_start = as_of.replace(day=1)
    mtd = work.loc[(work["transaction_dt"] >= month_start) & (work["transaction_dt"] <= as_of)].copy()
    mtd_discretionary = mtd.loc[mtd["is_discretionary"]]
    mtd_discretionary_spend = float(mtd_discretionary["spend_usd"].sum())
    utilization_pct = (mtd_discretionary_spend / monthly_limit) if monthly_limit else None

    category_source_mix: dict[str, int] = {}
    if "category_source" in work.columns:
        category_source_mix = {
            str(k): int(v) for k, v in work["category_source"].value_counts(dropna=False).items()
        }

    budget = {
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "month_start": month_start.strftime("%Y-%m-%d"),
        "monthly_discretionary_limit_usd": monthly_limit,
        "mtd_discretionary_spend_usd": mtd_discretionary_spend,
        "utilization_pct": utilization_pct,
        "discretionary_categories": sorted(DISCRETIONARY_CATEGORIES),
        "category_source_mix": category_source_mix,
    }

    return {
        "spend_by_category_usd": spend_by_category,
        "spend_by_mcc_usd": spend_by_mcc,
        "spending_patterns": patterns,
        "budget_utilization": budget,
    }


def artifact_paths(client_id: int, *, root: Path | None = None) -> dict[str, Path]:
    art = artifacts_dir(root)
    return {
        "spend_by_category": art / f"spend_by_category_{client_id}.json",
        "spend_by_mcc": art / f"spend_by_mcc_{client_id}.json",
        "budget_utilization": art / f"budget_utilization_{client_id}.json",
        "spending_patterns": art / f"spending_patterns_{client_id}.json",
    }


def write_analytics_artifacts(
    client_id: int, computed: dict[str, Any], *, root: Path | None = None
) -> dict[str, Path]:
    paths = artifact_paths(client_id, root=root)
    paths["spend_by_category"].write_text(
        json.dumps(
            {"client_id": int(client_id), "spend_by_category_usd": computed["spend_by_category_usd"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["spend_by_mcc"].write_text(
        json.dumps(
            {"client_id": int(client_id), "spend_by_mcc_usd": computed["spend_by_mcc_usd"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["budget_utilization"].write_text(
        json.dumps({"client_id": int(client_id), **computed["budget_utilization"]}, indent=2),
        encoding="utf-8",
    )
    paths["spending_patterns"].write_text(
        json.dumps({"client_id": int(client_id), **computed["spending_patterns"]}, indent=2),
        encoding="utf-8",
    )
    return paths


def run_analytics_for_client(
    client_id: int,
    *,
    as_of_date: pd.Timestamp | None = None,
    write_artifacts: bool = True,
    root: Path | None = None,
    transactions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Backend entrypoint: load cleaned data, compute analytics, optionally persist artifacts."""
    root = root or project_root()
    _records, source_path = load_transaction_records(client_id, root=root)
    tx = transactions if transactions is not None else load_transactions_for_client(client_id, root=root)
    if tx.empty:
        raise ValueError(f"No transactions found for client_id={client_id}")

    max_dt = tx["transaction_dt"].max()
    min_dt = tx["transaction_dt"].min()
    if pd.isna(max_dt) or pd.isna(min_dt):
        raise ValueError(f"Could not determine transaction date range for client_id={client_id}")

    as_of = pd.to_datetime(as_of_date) if as_of_date is not None else max_dt
    computed = compute_analytics(tx, as_of_date=as_of)

    paths: dict[str, Path] | None = None
    if write_artifacts:
        paths = write_analytics_artifacts(client_id, computed, root=root)

    return {
        "client_id": int(client_id),
        "transactions": tx,
        "source_path": source_path,
        "min_dt": min_dt,
        "max_dt": max_dt,
        "as_of_date": as_of,
        "computed": computed,
        "artifact_paths": paths,
    }


def clear_caches() -> None:
    _scan_all_users_ndjson_for_client.cache_clear()
    _load_category_overrides_cached.cache_clear()
