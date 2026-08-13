"""Streamlit frontend only.

All analytics computation lives in `model/analytics_core.py`.
This file loads clients, calls the backend, and renders charts/tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow `streamlit run ui/app.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.analytics_core import (  # noqa: E402
    clear_caches,
    load_transaction_records,
    load_transactions_for_client,
    project_root,
    run_analytics_for_client,
)

DEFAULT_CLIENT_ID = 1696


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


def main() -> None:
    st.set_page_config(page_title="ClearLedger Coach v1", layout="wide")
    st.title("ClearLedger — Personal Finance Coach v1")
    st.caption("Frontend UI · analytics computed by `model/analytics_core.py`")

    root = project_root()
    users_csv = root / "data" / "users.csv"
    cleaned = root / "artifacts" / "transactions_enriched.json"

    if not users_csv.exists():
        st.error("Missing dataset file `data/users.csv`.")
        st.stop()
    if not cleaned.exists():
        st.error("Missing cleaned artifact `artifacts/transactions_enriched.json`.")
        st.write("Run `data_processing/clean.ipynb` first.")
        st.stop()

    users_df = pd.read_csv(users_csv, dtype=str)
    client_ids = parse_client_ids(users_df)
    if not client_ids:
        st.error("No client ids found in `data/users.csv`.")
        st.stop()

    default_index = client_ids.index(DEFAULT_CLIENT_ID) if DEFAULT_CLIENT_ID in client_ids else 0

    with st.sidebar:
        st.subheader("Client")
        selected_client_id = int(
            st.selectbox(
                "client_id",
                options=client_ids,
                index=default_index,
                key="client_id_select",
            )
        )

        prev_client = st.session_state.get("active_client_id")
        if prev_client != selected_client_id:
            st.session_state["active_client_id"] = selected_client_id
            for key in list(st.session_state.keys()):
                if key.startswith(("as_of_", "tx_start_", "tx_end_", "tx_category_")):
                    del st.session_state[key]

        if st.button("Clear analytics cache"):
            clear_caches()
            st.rerun()

    with st.spinner(f"Loading client {selected_client_id} via analytics backend..."):
        try:
            _records, source_path = load_transaction_records(selected_client_id, root=root)
            tx = load_transactions_for_client(selected_client_id, root=root)
        except Exception as exc:  # noqa: BLE001 - surface backend errors in UI
            st.error(str(exc))
            st.stop()

    if tx.empty:
        st.error(f"No transactions found for client_id={selected_client_id}.")
        st.stop()

    min_dt = tx["transaction_dt"].min()
    max_dt = tx["transaction_dt"].max()
    if pd.isna(min_dt) or pd.isna(max_dt):
        st.error("Could not determine transaction date range for this client.")
        st.stop()

    min_date = min_dt.date()
    max_date = max_dt.date()

    st.subheader(f"Client {selected_client_id}")
    st.caption(
        f"{len(tx):,} transactions · {min_date} → {max_date} · "
        f"source: `{Path(source_path).name}`"
    )

    as_of = st.date_input(
        "AS_OF_DATE",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key=f"as_of_{selected_client_id}",
    )

    with st.spinner("Running analytics backend (writes artifacts)..."):
        result = run_analytics_for_client(
            selected_client_id,
            as_of_date=pd.to_datetime(as_of),
            write_artifacts=True,
            root=root,
            transactions=tx,
        )

    computed = result["computed"]
    budget = computed["budget_utilization"]
    spend_by_category = computed["spend_by_category_usd"]
    spend_by_mcc = computed["spend_by_mcc_usd"]
    patterns = computed["spending_patterns"]

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
