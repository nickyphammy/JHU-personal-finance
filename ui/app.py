from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


CLIENT_ID = 1696


def _project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").is_dir() and (candidate / "artifacts").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate project root containing 'data/' and 'artifacts/'")


ROOT = _project_root()
ARTIFACTS_DIR = ROOT / "artifacts"

PATHS = {
    "transactions": ARTIFACTS_DIR / f"transactions_enriched_{CLIENT_ID}.json",
    "spend_by_category": ARTIFACTS_DIR / f"spend_by_category_{CLIENT_ID}.json",
    "spend_by_mcc": ARTIFACTS_DIR / f"spend_by_mcc_{CLIENT_ID}.json",
    "budget_utilization": ARTIFACTS_DIR / f"budget_utilization_{CLIENT_ID}.json",
    "spending_patterns": ARTIFACTS_DIR / f"spending_patterns_{CLIENT_ID}.json",
}


def mcc_to_category(mcc_code: Any, mcc_description: Any) -> str:
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
        ("Shopping", ["department stores", "discount stores", "book stores", "home furnishing", "lumber"]),
        ("Transfers", ["money transfer", "transfer"]),
        ("Subscriptions", ["subscription"]),
        ("Fees & Interest", ["fee", "interest"]),
    ]

    for category, keywords in rules:
        if any(k in desc for k in keywords):
            return category

    if code.startswith("54"):
        return "Groceries"
    if code.startswith("58"):
        return "Dining"
    if code.startswith("49"):
        return "Utilities"
    if code.startswith("41") or code.startswith("47"):
        return "Transportation"
    if code.startswith(("53", "59", "52", "57")):
        return "Shopping"

    return "Other/Uncategorized"


@lru_cache(maxsize=64)
def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def load_transactions(path: Path) -> pd.DataFrame:
    records = load_json(path)
    df = pd.DataFrame.from_records(records)
    if "transaction_dt" in df.columns:
        df["transaction_dt"] = pd.to_datetime(df["transaction_dt"], errors="coerce")
    if "amount_usd" in df.columns:
        df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")
    if "mcc_code" in df.columns or "mcc_description" in df.columns:
        df["category"] = df.apply(
            lambda r: mcc_to_category(r.get("mcc_code"), r.get("mcc_description")), axis=1
        )
    return df


def missing_artifacts() -> list[Path]:
    return [p for p in PATHS.values() if not p.exists()]

def main() -> None:
    st.set_page_config(page_title="ClearLedger Coach v1 (1696)", layout="wide")

    st.title("ClearLedger — Personal Finance Coach v1")
    st.caption(f"Local demo UI (read-only artifacts) · client_id = {CLIENT_ID}")

    missing = missing_artifacts()
    if missing:
        st.error("Required artifacts are missing.")
        st.write("Run the notebooks in order:")
        st.code("\n".join(["1) data_processing/clean.ipynb", "2) model/analytics.ipynb"]))
        st.write("Missing files:")
        for p in missing:
            st.write(f"- `{p}`")
        st.stop()

    with st.sidebar:
        st.subheader("Artifacts")
        for k, p in PATHS.items():
            st.write(f"- `{k}`: `{p}`")
        if st.button("Refresh cache"):
            load_json.cache_clear()
            load_transactions.cache_clear()
            st.rerun()

    budget = load_json(PATHS["budget_utilization"])
    spend_by_category = load_json(PATHS["spend_by_category"])["spend_by_category_usd"]
    spend_by_mcc = load_json(PATHS["spend_by_mcc"])["spend_by_mcc_usd"]
    patterns = load_json(PATHS["spending_patterns"])
    tx = load_transactions(PATHS["transactions"])

    tab_overview, tab_transactions, tab_patterns, tab_mcc = st.tabs(
        ["Overview", "Transactions", "Patterns", "MCC Breakdown"]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("As of", budget.get("as_of_date"))
        c2.metric(
            "Monthly Discretionary Limit",
            f"${budget.get('monthly_discretionary_limit_usd'):,.2f}",
        )
        c3.metric("MTD Discretionary Spend", f"${budget.get('mtd_discretionary_spend_usd'):,.2f}")
        util = budget.get("utilization_pct")
        util_pct = (util * 100.0) if util is not None else None
        c4.metric("Utilization", f"{util_pct:.1f}%" if util_pct is not None else "N/A")

        if util is not None:
            st.progress(min(max(float(util), 0.0), 1.0))
            if util >= 0.95:
                st.warning("Utilization ≥ 95% (critical threshold).")
            elif util >= 0.85:
                st.warning("Utilization ≥ 85% (warning threshold).")
            elif util >= 0.70:
                st.info("Utilization ≥ 70% (heads-up threshold).")

        spend_cat_df = (
            pd.DataFrame([{"category": k, "spend_usd": v} for k, v in spend_by_category.items()])
            .sort_values("spend_usd", ascending=False)
            .head(12)
        )
        fig = px.bar(spend_cat_df, x="category", y="spend_usd", title="Top Categories (Spend USD)")
        st.plotly_chart(fig, use_container_width=True)

    with tab_transactions:
        st.subheader("Transactions (cleaned + enriched)")
        col_a, col_b, col_c = st.columns([2, 2, 2])
        min_dt = tx["transaction_dt"].min() if "transaction_dt" in tx.columns else None
        max_dt = tx["transaction_dt"].max() if "transaction_dt" in tx.columns else None

        start = col_a.date_input("Start date", value=min_dt.date() if pd.notna(min_dt) else None)
        end = col_b.date_input("End date", value=max_dt.date() if pd.notna(max_dt) else None)
        category = col_c.selectbox(
            "Category",
            options=["(All)"] + sorted(tx["category"].dropna().unique().tolist())
            if "category" in tx.columns
            else ["(All)"],
        )

        tx_view = tx.copy()
        if "transaction_dt" in tx_view.columns and start and end:
            tx_view = tx_view.loc[
                (tx_view["transaction_dt"] >= pd.to_datetime(start))
                & (tx_view["transaction_dt"] <= pd.to_datetime(end))
            ]
        if category != "(All)" and "category" in tx_view.columns:
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
        st.dataframe(tx_view[cols].sort_values("transaction_dt", ascending=False), use_container_width=True)

    with tab_patterns:
        st.subheader("Spending patterns")
        by_month = patterns.get("by_month_usd", {})
        if by_month:
            df_month = (
                pd.DataFrame([{"month": k, "spend_usd": v} for k, v in by_month.items()])
                .sort_values("month")
            )
            st.plotly_chart(
                px.line(df_month, x="month", y="spend_usd", title="Spend by Month"),
                use_container_width=True,
            )

        by_dow = patterns.get("by_day_of_week_usd", {})
        if by_dow:
            df_dow = pd.DataFrame([{"day": k, "spend_usd": v} for k, v in by_dow.items()])
            st.plotly_chart(
                px.bar(df_dow, x="day", y="spend_usd", title="Spend by Day of Week"),
                use_container_width=True,
            )

        st.subheader("Top merchants")
        tm = patterns.get("top_merchants_usd", [])
        if tm:
            st.dataframe(pd.DataFrame(tm), use_container_width=True)

    with tab_mcc:
        st.subheader("Spend by MCC")
        mcc_df = pd.DataFrame(spend_by_mcc).sort_values("spend_usd", ascending=False).head(50)
        st.dataframe(mcc_df, use_container_width=True)


if __name__ == "__main__":
    main()
