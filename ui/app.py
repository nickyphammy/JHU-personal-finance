from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_CLIENT_ID = 1696


def project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").is_dir() and (candidate / "artifacts").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate project root containing 'data/' and 'artifacts/'")


ROOT = project_root()
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

USERS_CSV = DATA_DIR / "users.csv"
ALL_USERS_NDJSON = ARTIFACTS_DIR / "transactions_enriched.json"


def parse_client_ids(users_df: pd.DataFrame) -> list[int]:
    if "id" not in users_df.columns:
        raise ValueError("users_df missing required column 'id'")
    client_ids = (
        pd.to_numeric(users_df["id"], errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return [int(x) for x in client_ids]


def mcc_to_category(mcc_code: Any, mcc_description: Any) -> str:
    """Deterministic MCC-based category mapping (rule-based, no LLM)."""
    desc = (str(mcc_description).strip().lower() if mcc_description is not None else "")
    code = str(mcc_code).strip() if mcc_code is not None else ""

    if not desc and not code:
        return "Other/Uncategorized"

    rules: list[tuple[str, list[str]]] = [
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

    for category, keywords in rules:
        if any(k in desc for k in keywords):
            return category

    # Coarse prefix fallback rules
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


@lru_cache(maxsize=256)
def scan_all_users_ndjson_for_client(client_id: int) -> list[dict[str, Any]]:
    if not ALL_USERS_NDJSON.exists():
        raise FileNotFoundError(
            "Missing artifacts/transactions_enriched.json. Run data_processing/clean.ipynb first."
        )

    records: list[dict[str, Any]] = []
    with ALL_USERS_NDJSON.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if str(obj.get("client_id")) == str(client_id):
                records.append(obj)
    return records


@lru_cache(maxsize=64)
def load_transaction_records(client_id: int) -> tuple[list[dict[str, Any]], str]:
    """Load one client's transactions. Prefer focus JSON; fall back to all-users NDJSON."""
    focus_path = ARTIFACTS_DIR / f"transactions_enriched_{client_id}.json"
    if focus_path.exists():
        records = json.loads(focus_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list in {focus_path}")
        return records, str(focus_path)

    records = scan_all_users_ndjson_for_client(client_id)
    return records, str(ALL_USERS_NDJSON)


@lru_cache(maxsize=64)
def load_transactions_for_client(client_id: int) -> pd.DataFrame:
    records, _source = load_transaction_records(client_id)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    # Keep only this client (belt-and-suspenders if focus file is wrong)
    if "client_id" in df.columns:
        df = df.loc[df["client_id"].astype(str) == str(client_id)].copy()

    if "transaction_dt" in df.columns:
        df["transaction_dt"] = pd.to_datetime(df["transaction_dt"], errors="coerce")
    if "amount_usd" in df.columns:
        df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")

    df["category"] = df.apply(
        lambda r: mcc_to_category(r.get("mcc_code"), r.get("mcc_description")), axis=1
    )
    return df


def compute_analytics(df: pd.DataFrame, *, as_of_date: pd.Timestamp) -> dict[str, Any]:
    work = df.copy()
    work = work.loc[work["amount_usd"].notna()].copy()
    work["spend_usd"] = work["amount_usd"].where(work["amount_usd"] > 0, 0.0)

    # Spend by category
    by_category = (
        work.groupby("category", dropna=False)["spend_usd"].sum().sort_values(ascending=False)
    )
    spend_by_category = {k: float(v) for k, v in by_category.items()}

    # Spend by MCC
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

    # Patterns
    work = work.loc[work["transaction_dt"].notna()].copy()
    work["month"] = work["transaction_dt"].dt.to_period("M").astype(str)
    work["dow"] = work["transaction_dt"].dt.day_name()

    by_month = work.groupby("month")["spend_usd"].sum().sort_index()
    by_dow = work.groupby("dow")["spend_usd"].sum()

    top_merchants = (
        work.groupby(["merchant_id", "merchant_city", "merchant_state"], dropna=False)["spend_usd"]
        .sum()
        .sort_values(ascending=False)
        .head(25)
    )

    patterns = {
        "total_spend_usd": float(work["spend_usd"].sum()),
        "by_month_usd": {k: float(v) for k, v in by_month.items()},
        "by_day_of_week_usd": {k: float(v) for k, v in by_dow.items()},
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

    # Budget utilization
    discretionary_categories = {"Dining", "Entertainment", "Shopping", "Travel", "Subscriptions"}
    work["is_discretionary"] = work["category"].isin(discretionary_categories)

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

    budget = {
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "month_start": month_start.strftime("%Y-%m-%d"),
        "monthly_discretionary_limit_usd": monthly_limit,
        "mtd_discretionary_spend_usd": mtd_discretionary_spend,
        "utilization_pct": utilization_pct,
        "discretionary_categories": sorted(list(discretionary_categories)),
    }

    return {
        "spend_by_category": spend_by_category,
        "spend_by_mcc": spend_by_mcc,
        "patterns": patterns,
        "budget": budget,
    }


def main() -> None:
    st.set_page_config(page_title="ClearLedger Coach v1", layout="wide")
    st.title("ClearLedger — Personal Finance Coach v1")
    st.caption("Select a client_id in the sidebar to load that user's spending.")

    if not USERS_CSV.exists():
        st.error("Missing dataset file `data/users.csv`.")
        st.write(f"Not found: `{USERS_CSV}`")
        st.stop()

    if not ALL_USERS_NDJSON.exists():
        st.error("Missing cleaned artifact `artifacts/transactions_enriched.json`.")
        st.write("Run `data_processing/clean.ipynb` first.")
        st.stop()

    users_df = pd.read_csv(USERS_CSV, dtype=str)
    if "id" not in users_df.columns:
        st.error("`data/users.csv` missing required column `id`.")
        st.stop()

    client_ids = parse_client_ids(users_df)
    if not client_ids:
        st.error("No client ids found in `data/users.csv`.")
        st.stop()

    default_index = (
        client_ids.index(DEFAULT_CLIENT_ID) if DEFAULT_CLIENT_ID in client_ids else 0
    )

    with st.sidebar:
        st.subheader("Client")
        selected_client_id = st.selectbox(
            "client_id",
            options=client_ids,
            index=default_index,
            key="client_id_select",
        )
        selected_client_id = int(selected_client_id)

        # Reset date/category widgets whenever the selected client changes
        prev_client = st.session_state.get("active_client_id")
        if prev_client != selected_client_id:
            st.session_state["active_client_id"] = selected_client_id
            for key in list(st.session_state.keys()):
                if key.startswith("as_of_") or key.startswith("tx_start_") or key.startswith(
                    "tx_end_"
                ) or key.startswith("tx_category_"):
                    del st.session_state[key]

        if st.button("Clear data cache"):
            scan_all_users_ndjson_for_client.cache_clear()
            load_transaction_records.cache_clear()
            load_transactions_for_client.cache_clear()
            st.rerun()

    with st.spinner(f"Loading transactions for client {selected_client_id}..."):
        try:
            _records, source_path = load_transaction_records(selected_client_id)
            tx = load_transactions_for_client(selected_client_id)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()

    if tx.empty:
        st.error(f"No transactions found for client_id={selected_client_id}.")
        st.info("That client exists in users.csv but has no rows in the cleaned artifact.")
        st.stop()

    max_dt = tx["transaction_dt"].max() if "transaction_dt" in tx.columns else None
    min_dt = tx["transaction_dt"].min() if "transaction_dt" in tx.columns else None
    if pd.isna(max_dt) or pd.isna(min_dt):
        st.error("Could not determine transaction date range for this client.")
        st.stop()

    min_date = min_dt.date()
    max_date = max_dt.date()

    st.subheader(f"Client {selected_client_id}")
    st.caption(
        f"{len(tx):,} transactions · {min_date} → {max_date} · source: `{Path(source_path).name}`"
    )

    as_of = st.date_input(
        "AS_OF_DATE",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key=f"as_of_{selected_client_id}",
    )

    as_of_ts = pd.to_datetime(as_of)
    with st.spinner("Computing analytics..."):
        computed = compute_analytics(tx, as_of_date=as_of_ts)

    budget = computed["budget"]
    spend_by_category = computed["spend_by_category"]
    spend_by_mcc = computed["spend_by_mcc"]
    patterns = computed["patterns"]

    tab_overview, tab_transactions, tab_patterns, tab_mcc = st.tabs(
        ["Overview", "Transactions", "Patterns", "MCC Breakdown"]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("As of", budget.get("as_of_date"))
        lim = budget.get("monthly_discretionary_limit_usd")
        c2.metric("Monthly Discretionary Limit", f"${lim:,.2f}" if lim is not None else "N/A")
        c3.metric("MTD Discretionary Spend", f"${budget.get('mtd_discretionary_spend_usd'):,.2f}")
        util = budget.get("utilization_pct")
        util_pct = (util * 100.0) if util is not None else None
        c4.metric("Utilization", f"{util_pct:.1f}%" if util_pct is not None else "N/A")

        if util is not None:
            st.progress(min(max(float(util), 0.0), 1.0))

        spend_cat_df = (
            pd.DataFrame([{"category": k, "spend_usd": v} for k, v in spend_by_category.items()])
            .sort_values("spend_usd", ascending=False)
            .head(12)
        )
        st.plotly_chart(
            px.bar(
                spend_cat_df,
                x="category",
                y="spend_usd",
                title=f"Top Categories (Client {selected_client_id})",
            ),
            use_container_width=True,
        )

    with tab_transactions:
        col_a, col_b, col_c = st.columns([2, 2, 2])
        start = col_a.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key=f"tx_start_{selected_client_id}",
        )
        end = col_b.date_input(
            "End date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key=f"tx_end_{selected_client_id}",
        )
        category = col_c.selectbox(
            "Category",
            options=["(All)"] + sorted(tx["category"].dropna().unique().tolist()),
            key=f"tx_category_{selected_client_id}",
        )

        tx_view = tx.copy()
        if start and end:
            tx_view = tx_view.loc[
                (tx_view["transaction_dt"] >= pd.to_datetime(start))
                & (tx_view["transaction_dt"] <= pd.to_datetime(end))
            ]
        if category != "(All)":
            tx_view = tx_view.loc[tx_view["category"] == category]

        cols = [
            c
            for c in [
                "transaction_dt",
                "amount_usd",
                "category",
                "merchant_city",
                "merchant_state",
                "mcc_code",
                "mcc_description",
                "use_chip",
            ]
            if c in tx_view.columns
        ]
        st.dataframe(
            tx_view[cols].sort_values("transaction_dt", ascending=False),
            use_container_width=True,
        )

    with tab_patterns:
        by_month = patterns.get("by_month_usd", {})
        if by_month:
            df_month = (
                pd.DataFrame([{"month": k, "spend_usd": v} for k, v in by_month.items()])
                .sort_values("month")
            )
            st.plotly_chart(
                px.line(
                    df_month,
                    x="month",
                    y="spend_usd",
                    title=f"Spend by Month (Client {selected_client_id})",
                ),
                use_container_width=True,
            )

        by_dow = patterns.get("by_day_of_week_usd", {})
        if by_dow:
            df_dow = pd.DataFrame([{"day": k, "spend_usd": v} for k, v in by_dow.items()])
            st.plotly_chart(
                px.bar(
                    df_dow,
                    x="day",
                    y="spend_usd",
                    title=f"Spend by Day of Week (Client {selected_client_id})",
                ),
                use_container_width=True,
            )

        tm = patterns.get("top_merchants_usd", [])
        if tm:
            st.subheader("Top merchants")
            st.dataframe(pd.DataFrame(tm), use_container_width=True)

    with tab_mcc:
        st.subheader("Spend by MCC")
        mcc_df = pd.DataFrame(spend_by_mcc).sort_values("spend_usd", ascending=False).head(50)
        st.dataframe(mcc_df, use_container_width=True)


if __name__ == "__main__":
    main()
