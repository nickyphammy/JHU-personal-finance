"""Streamlit frontend only.

Analytics live in `model/analytics_core.py`; prediction in `model/predict_core.py`.
This file loads clients, calls backends, and renders charts/tables.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow `streamlit run ui/app.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model.analytics_core as analytics_core  # noqa: E402
import model.predict_core as predict_core  # noqa: E402
from importlib import reload  # noqa: E402

reload(analytics_core)  # avoid stale Streamlit imports after backend edits
reload(predict_core)

load_transaction_records = analytics_core.load_transaction_records
load_transactions_for_client = analytics_core.load_transactions_for_client
project_root = analytics_core.project_root
run_analytics_for_client = analytics_core.run_analytics_for_client
run_prediction_for_client = predict_core.run_prediction_for_client
default_predict_paths = predict_core.default_paths


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


def format_usd_yaxis(fig: Any, *, title: str = "Spend (USD)") -> Any:
    fig.update_yaxes(title=title, tickprefix="$", tickformat=",.0f")
    return fig


# Matches the “Spending by Category” card palette (donut + legend).
CATEGORY_DONUT_COLORS = [
    "#6C5CE7",  # purple
    "#0984E3",  # blue
    "#00B894",  # teal
    "#E17055",  # coral/orange
    "#FDCB6E",  # yellow
    "#E84393",  # pink
    "#00CEC9",  # cyan
    "#B2BEC3",  # grey (Other)
]


def compute_category_share(spend_by_category: dict[str, float], *, top_n: int = 5) -> pd.DataFrame:
    if not spend_by_category:
        return pd.DataFrame(columns=["category", "spend_usd", "share_pct"])

    df = pd.DataFrame(
        [{"category": str(k), "spend_usd": float(v)} for k, v in spend_by_category.items()]
    ).sort_values("spend_usd", ascending=False)

    total = float(df["spend_usd"].sum())
    if total <= 0:
        df["share_pct"] = 0.0
        return df.loc[:, ["category", "spend_usd", "share_pct"]]

    head = df.head(int(top_n)).copy()
    tail = df.iloc[int(top_n) :].copy()
    if not tail.empty:
        other = pd.DataFrame(
            [{"category": "Other", "spend_usd": float(tail["spend_usd"].sum())}]
        )
        head = pd.concat([head, other], ignore_index=True)

    head["share_pct"] = (head["spend_usd"] / total) * 100.0
    return head.loc[:, ["category", "spend_usd", "share_pct"]].reset_index(drop=True)


def compute_mtd_category_spend(
    tx: pd.DataFrame, *, as_of_date: Any
) -> dict[str, float]:
    """Positive MTD spend by category through as_of_date (inclusive)."""
    if not {"transaction_dt", "category", "amount_usd"}.issubset(set(tx.columns)):
        return {}

    as_of = pd.to_datetime(as_of_date).normalize()
    month_start = as_of.replace(day=1)
    work = tx.loc[:, ["transaction_dt", "category", "amount_usd"]].copy()
    work["transaction_dt"] = pd.to_datetime(work["transaction_dt"], errors="coerce")
    work["amount_usd"] = pd.to_numeric(work["amount_usd"], errors="coerce")
    work = work.loc[
        work["transaction_dt"].notna()
        & work["amount_usd"].notna()
        & (work["transaction_dt"] >= month_start)
        & (work["transaction_dt"] <= as_of)
    ].copy()
    if work.empty:
        return {}

    work["spend_usd"] = work["amount_usd"].where(work["amount_usd"] > 0, 0.0)
    by_cat = work.groupby("category", dropna=False)["spend_usd"].sum().sort_values(ascending=False)
    return {str(k): float(v) for k, v in by_cat.items() if float(v) > 0}


def render_spending_by_category_donut(
    spend_by_category: dict[str, float],
    *,
    client_id: int,
    top_n: int = 5,
    key: str,
) -> None:
    """Donut + right-side legend card (category, amount, %)."""
    share = compute_category_share(spend_by_category, top_n=top_n)
    if share.empty:
        st.info("No category spend available.")
        return

    total = float(share["spend_usd"].sum())
    colors = [
        CATEGORY_DONUT_COLORS[i % len(CATEGORY_DONUT_COLORS)] for i in range(len(share))
    ]
    # Keep “Other” grey when present
    for i, cat in enumerate(share["category"].tolist()):
        if str(cat).lower() in {"other", "other/uncategorized"}:
            colors[i] = "#B2BEC3"

    with st.container(border=True):
        st.markdown(
            "<h5 style='color:#FFFFFF;margin:0 0 0.5rem 0;'>Spending by Category</h5>",
            unsafe_allow_html=True,
        )
        chart_col, legend_col = st.columns([1.15, 1.0], gap="large")

        with chart_col:
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=share["category"].tolist(),
                        values=share["spend_usd"].tolist(),
                        hole=0.64,
                        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=3)),
                        textinfo="none",
                        sort=False,
                        direction="clockwise",
                        hovertemplate=(
                            "<b>%{label}</b><br>$%{value:,.2f}"
                            "<br>%{percent}<extra></extra>"
                        ),
                    )
                ]
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(t=8, b=8, l=8, r=8),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                annotations=[
                    dict(
                        text=(
                            f"<b>${total:,.2f}</b>"
                            "<br><span style='color:#FFFFFF;font-size:12px;"
                            "letter-spacing:0.04em;opacity:0.85'>TOTAL</span>"
                        ),
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=22, color="#FFFFFF", family="Inter, sans-serif"),
                        align="center",
                    )
                ],
            )
            st.plotly_chart(fig, use_container_width=True, key=key)

        with legend_col:
            st.write("")  # vertical align with chart title spacing
            for color, row in zip(colors, share.itertuples(index=False), strict=True):
                cat, spend_usd, share_pct = row.category, row.spend_usd, row.share_pct
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;justify-content:space-between;
                                padding:0.35rem 0;border-bottom:1px solid rgba(255,255,255,0.15);">
                      <div style="display:flex;align-items:center;gap:0.55rem;min-width:0;">
                        <span style="width:10px;height:10px;border-radius:50%;
                                     background:{color};display:inline-block;flex-shrink:0;"></span>
                        <span style="color:#FFFFFF;font-size:0.95rem;font-weight:500;
                                     white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                          {cat}
                        </span>
                      </div>
                      <div style="display:flex;gap:1.1rem;align-items:baseline;flex-shrink:0;
                                  font-variant-numeric:tabular-nums;">
                        <span style="color:#FFFFFF;font-weight:600;">${spend_usd:,.2f}</span>
                        <span style="color:#FFFFFF;min-width:3.2rem;text-align:right;opacity:0.9;">
                          {share_pct:.1f}%
                        </span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.expander("View full breakdown →", expanded=False):
            full = (
                pd.DataFrame(
                    [
                        {"category": str(k), "spend_usd": float(v)}
                        for k, v in spend_by_category.items()
                    ]
                )
                .sort_values("spend_usd", ascending=False)
                .reset_index(drop=True)
            )
            full_total = float(full["spend_usd"].sum()) if not full.empty else 0.0
            if full_total > 0:
                full["share_pct"] = (full["spend_usd"] / full_total) * 100.0
            else:
                full["share_pct"] = 0.0
            full["spend_usd"] = full["spend_usd"].map(lambda x: f"${x:,.2f}")
            full["share_pct"] = full["share_pct"].map(lambda x: f"{x:.1f}%")
            st.dataframe(
                full.rename(
                    columns={
                        "category": "Category",
                        "spend_usd": "Spend",
                        "share_pct": "Share",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(
                f"<p style='color:#FFFFFF;opacity:0.85;font-size:0.85rem;'>"
                f"Client {client_id} · total ${total:,.2f} shown in chart</p>",
                unsafe_allow_html=True,
            )


def compute_monthly_category_trend(tx: pd.DataFrame, *, top_n: int = 8) -> pd.DataFrame:
    """Monthly spend trend by category (positive spend only)."""
    if not {"transaction_dt", "category"}.issubset(set(tx.columns)):
        return pd.DataFrame(columns=["month", "category", "spend_usd"])

    work = tx.loc[:, ["transaction_dt", "category", "amount_usd"]].copy()
    work["transaction_dt"] = pd.to_datetime(work["transaction_dt"], errors="coerce")
    work["amount_usd"] = pd.to_numeric(work["amount_usd"], errors="coerce")
    work = work.loc[work["transaction_dt"].notna() & work["amount_usd"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["month", "category", "spend_usd"])

    work["spend_usd"] = work["amount_usd"].where(work["amount_usd"] > 0, 0.0)
    top_categories = (
        work.groupby("category", dropna=False)["spend_usd"]
        .sum()
        .sort_values(ascending=False)
        .head(int(top_n))
        .index.tolist()
    )

    work = work.loc[work["category"].isin(top_categories)].copy()
    work["month"] = work["transaction_dt"].dt.to_period("M").astype(str)
    trend = (
        work.groupby(["month", "category"], dropna=False)["spend_usd"]
        .sum()
        .reset_index()
        .sort_values(["month", "category"])
    )
    return trend


def render_monthly_category_trend(
    tx: pd.DataFrame, *, client_id: int, top_n: int = 8, key: str
) -> None:
    trend = compute_monthly_category_trend(tx, top_n=top_n)
    if trend.empty:
        st.info("Not enough data to plot monthly category spend trends.")
        return

    fig = px.line(
        trend,
        x="month",
        y="spend_usd",
        color="category",
        title=f"Monthly Spend Trend by Category (Client {client_id})",
        labels={"month": "Month", "spend_usd": "Spend (USD)", "category": "Category"},
        markers=True,
    )
    format_usd_yaxis(fig)
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True, key=key)


DOW_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def compute_avg_daily_spend_by_dow(tx: pd.DataFrame) -> pd.DataFrame:
    """Average spend on calendar days for each weekday (not lifetime weekday totals)."""
    if not {"transaction_dt", "amount_usd"}.issubset(set(tx.columns)):
        return pd.DataFrame(columns=["day", "avg_daily_spend_usd"])

    work = tx.loc[:, ["transaction_dt", "amount_usd"]].copy()
    work["transaction_dt"] = pd.to_datetime(work["transaction_dt"], errors="coerce")
    work["amount_usd"] = pd.to_numeric(work["amount_usd"], errors="coerce")
    work = work.loc[work["transaction_dt"].notna() & work["amount_usd"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["day", "avg_daily_spend_usd"])

    work["spend_usd"] = work["amount_usd"].where(work["amount_usd"] > 0, 0.0)
    work["txn_date"] = work["transaction_dt"].dt.normalize()
    work["day"] = work["transaction_dt"].dt.day_name()

    daily = work.groupby(["txn_date", "day"], dropna=False)["spend_usd"].sum().reset_index()
    avg = daily.groupby("day", dropna=False)["spend_usd"].mean()
    return pd.DataFrame(
        [{"day": d, "avg_daily_spend_usd": float(avg[d])} for d in DOW_ORDER if d in avg.index]
    )


def render_avg_daily_spend_by_dow(tx: pd.DataFrame, *, client_id: int, key: str) -> None:
    df_dow = compute_avg_daily_spend_by_dow(tx)
    if df_dow.empty:
        st.info("Not enough data to plot average daily spend by weekday.")
        return

    fig = px.bar(
        df_dow,
        x="day",
        y="avg_daily_spend_usd",
        title=f"Average Daily Spend by Day of Week (Client {client_id})",
        labels={"day": "Day of week", "avg_daily_spend_usd": "Avg daily spend (USD)"},
        category_orders={"day": DOW_ORDER},
    )
    format_usd_yaxis(fig, title="Avg daily spend (USD)")
    # Keep y-axis near typical daily amounts (hundreds), not lifetime totals
    ymax = float(df_dow["avg_daily_spend_usd"].max())
    fig.update_yaxes(range=[0, max(ymax * 1.25, 1.0)])
    st.plotly_chart(fig, use_container_width=True, key=key)
    example = df_dow.iloc[0]
    st.caption(
        "Each bar is the mean spend across calendar days for that weekday "
        f"(example: typical {example['day']} ≈ ${example['avg_daily_spend_usd']:,.0f}). "
        "This is NOT the sum of all historical Mondays/Tuesdays across years."
    )


def main() -> None:
    st.set_page_config(page_title="ClearLedger Coach v1", layout="wide")
    st.title("ClearLedger — Personal Finance Coach v1")
    st.caption(
        "Frontend UI · analytics via `model/analytics_core.py` · "
        "forecast via `model/predict_core.py`"
    )

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

    with st.sidebar:
        selected_client_id = int(st.selectbox("client_id", options=client_ids, key="client_id_select"))

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

    prediction: dict[str, Any] | None = None
    predict_error: str | None = None
    model_pkl = default_predict_paths(root)["model_pkl"]
    lim = budget.get("monthly_discretionary_limit_usd")
    if not model_pkl.exists():
        predict_error = (
            "Missing trained model `artifacts/runway_model.pkl`. "
            "Run `model/predict.ipynb` once to train on all users."
        )
    elif lim is None:
        predict_error = "No monthly discretionary limit available for this client."
    else:
        with st.spinner("Running prediction backend..."):
            try:
                pred_result = run_prediction_for_client(
                    selected_client_id,
                    as_of_date=as_of,
                    transactions=tx,
                    monthly_limit_usd=float(lim),
                    write_artifacts=True,
                    root=root,
                )
                prediction = pred_result["prediction"]
            except Exception as exc:  # noqa: BLE001 - surface backend errors in UI
                predict_error = str(exc)

    tab_overview, tab_forecast, tab_transactions, tab_patterns, tab_mcc = st.tabs(
        ["Overview", "Forecast", "Transactions", "Patterns", "MCC Breakdown"]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("As of", budget.get("as_of_date"))
        c2.metric("Monthly Discretionary Limit", f"${lim:,.2f}" if lim is not None else "N/A")
        c3.metric("MTD Discretionary Spend", f"${budget.get('mtd_discretionary_spend_usd'):,.2f}")
        util = budget.get("utilization_pct")
        util_pct = (util * 100.0) if util is not None else None
        c4.metric("Utilization", f"{util_pct:.1f}%" if util_pct is not None else "N/A")

        if util is not None:
            st.progress(min(max(float(util), 0.0), 1.0))

        if prediction is not None:
            st.subheader("Month-end forecast")
            p1, p2, p3, p4 = st.columns(4)
            projected = prediction.get("predicted_month_end_discretionary_spend_usd")
            p1.metric(
                "Projected month-end",
                f"${projected:,.2f}" if projected is not None else "N/A",
            )
            days_left = prediction.get("days_to_limit_estimate")
            p2.metric(
                "Days to limit",
                "N/A" if days_left is None else str(days_left),
            )
            risk = prediction.get("overspend_risk")
            p3.metric("Overspend risk", "Yes" if risk else "No")
            proj_util = prediction.get("projected_utilization_pct")
            p4.metric(
                "Projected utilization",
                f"{proj_util * 100:.1f}%" if proj_util is not None else "N/A",
            )
            if lim is not None and projected is not None:
                meter = min(max(float(projected) / float(lim), 0.0), 1.5)
                st.caption("Projected vs limit (capped display at 150%)")
                st.progress(min(meter, 1.0))
        elif predict_error:
            st.warning(predict_error)

        left, right = st.columns([1.15, 1.35], gap="large")
        with left:
            mtd_category = compute_mtd_category_spend(tx, as_of_date=as_of)
            render_spending_by_category_donut(
                mtd_category if mtd_category else spend_by_category,
                client_id=selected_client_id,
                top_n=5,
                key=f"category_donut_{selected_client_id}",
            )
            if mtd_category:
                st.caption("MTD spend by category through AS_OF_DATE.")
        with right:
            render_monthly_category_trend(
                tx,
                client_id=selected_client_id,
                top_n=8,
                key=f"monthly_category_trend_overview_{selected_client_id}",
            )

    with tab_forecast:
        st.subheader("Discretionary runway forecast")
        if predict_error:
            st.warning(predict_error)
        elif prediction is None:
            st.info("No prediction available.")
        else:
            projected = prediction.get("predicted_month_end_discretionary_spend_usd")
            mtd = prediction.get("mtd_discretionary_spend_usd")
            remaining = prediction.get("remaining_budget_usd")
            days_left = prediction.get("days_to_limit_estimate")
            risk = bool(prediction.get("overspend_risk"))

            f1, f2, f3 = st.columns(3)
            f1.metric("MTD discretionary", f"${mtd:,.2f}" if mtd is not None else "N/A")
            f2.metric(
                "Projected month-end",
                f"${projected:,.2f}" if projected is not None else "N/A",
            )
            f3.metric(
                "Remaining budget",
                f"${remaining:,.2f}" if remaining is not None else "N/A",
            )

            if risk:
                st.error(
                    "Overspend risk: projected month-end discretionary spend exceeds the "
                    f"monthly limit"
                    + (
                        f" (${lim:,.2f})."
                        if lim is not None
                        else "."
                    )
                )
            else:
                st.success("On track: projected month-end spend is within the monthly limit.")

            if days_left is None:
                st.info(
                    "Days-to-limit is undefined (implied remaining daily spend is ~0, "
                    "or no breach path from the current forecast)."
                )
            elif days_left == 0:
                st.warning("Limit already reached or breached on a linearized runway basis.")
            else:
                st.write(
                    f"Estimated **{days_left}** day(s) until the discretionary limit is hit "
                    "at the implied remaining spend rate."
                )

            metrics = prediction.get("model_metrics") or {}
            if metrics:
                st.caption(
                    "Model holdout MAE: "
                    f"${metrics.get('mae_test_usd'):,.2f}"
                    if metrics.get("mae_test_usd") is not None
                    else "Model metrics available after training."
                )
                st.caption(f"Train cutoff day: {metrics.get('cutoff_day', 'n/a')}")

            compare = pd.DataFrame(
                [
                    {"label": "MTD actual", "spend_usd": float(mtd or 0.0)},
                    {
                        "label": "Projected month-end",
                        "spend_usd": float(projected or 0.0),
                    },
                    {
                        "label": "Monthly limit",
                        "spend_usd": float(lim or 0.0),
                    },
                ]
            )
            fig = px.bar(
                compare,
                x="label",
                y="spend_usd",
                title=f"Actual vs projected vs limit (Client {selected_client_id})",
                labels={"label": "", "spend_usd": "USD"},
            )
            format_usd_yaxis(fig)
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"forecast_compare_{selected_client_id}",
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
        render_monthly_category_trend(
            tx,
            client_id=selected_client_id,
            top_n=8,
            key=f"monthly_category_trend_patterns_{selected_client_id}",
        )

        render_avg_daily_spend_by_dow(
            tx,
            client_id=selected_client_id,
            key=f"avg_daily_spend_dow_{selected_client_id}",
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
