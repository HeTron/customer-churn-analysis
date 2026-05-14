"""Cohort Analysis page — interactive EDA on the Telco churn dataset."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import load_clean
from src.features import add_engineered_features

st.set_page_config(
    page_title="Cohort Analysis · Churn Edge",
    page_icon="📊",
    layout="wide",
)
st.title("📊 Cohort Analysis")

# ── Data ──────────────────────────────────────────────────────────────────────
df_raw = load_clean()
df = add_engineered_features(df_raw)

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    contract_opts = sorted(df["Contract"].unique().tolist())
    selected_contracts = st.multiselect(
        "Contract type",
        options=contract_opts,
        default=contract_opts,
    )

    internet_opts = sorted(df["InternetService"].unique().tolist())
    selected_internet = st.multiselect(
        "Internet service",
        options=internet_opts,
        default=internet_opts,
    )

    tenure_min = int(df["tenure"].min())
    tenure_max = int(df["tenure"].max())
    tenure_range = st.slider(
        "Tenure (months)",
        min_value=tenure_min,
        max_value=tenure_max,
        value=(tenure_min, tenure_max),
    )

mask = (
    df["Contract"].isin(selected_contracts)
    & df["InternetService"].isin(selected_internet)
    & df["tenure"].between(tenure_range[0], tenure_range[1])
)
filtered = df[mask].copy()

if filtered.empty:
    st.warning("No customers match the current filters.")
    st.stop()

# ── Headline metrics ───────────────────────────────────────────────────────────
total_customers = len(filtered)
churn_rate = filtered["Churn"].mean()
avg_mrr = filtered["MonthlyCharges"].mean()
total_mrr = filtered["MonthlyCharges"].sum()
mrr_at_risk = filtered.loc[filtered["Churn"] == 1, "MonthlyCharges"].sum()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Customers", f"{total_customers:,}")
m2.metric("Churn Rate", f"{churn_rate:.1%}")
m3.metric("Avg MRR", f"${avg_mrr:.2f}")
m4.metric("Total MRR", f"${total_mrr:,.0f}")
m5.metric("MRR at Risk", f"${mrr_at_risk:,.0f}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Churn by Contract",
    "Churn by Tenure",
    "Churn by Charges",
    "Services & Churn",
    "Payment Method",
])

# Tab 1 — Churn by Contract
with tab1:
    ct = (
        filtered.groupby("Contract")["Churn"]
        .mean()
        .reset_index()
        .rename(columns={"Churn": "churn_rate"})
        .sort_values("churn_rate", ascending=False)
    )
    fig = px.bar(
        ct,
        x="Contract",
        y="churn_rate",
        title="Churn Rate by Contract Type",
        labels={"churn_rate": "Churn Rate"},
        template="plotly_dark",
        color_discrete_sequence=["#00D4AA"],
    )
    fig.update_layout(yaxis_tickformat=".0%", height=420)
    st.plotly_chart(fig, use_container_width=True)

# Tab 2 — Churn by Tenure Bucket
with tab2:
    bucket_order = ["0-12", "13-24", "25-48", "49+"]
    tb = (
        filtered.groupby("tenure_bucket")["Churn"]
        .mean()
        .reindex(bucket_order)
        .reset_index()
        .rename(columns={"Churn": "churn_rate"})
    )
    fig2 = px.line(
        tb,
        x="tenure_bucket",
        y="churn_rate",
        title="Churn Rate by Tenure Bucket",
        labels={"tenure_bucket": "Tenure (months)", "churn_rate": "Churn Rate"},
        template="plotly_dark",
        markers=True,
        color_discrete_sequence=["#00D4AA"],
    )
    fig2.update_layout(yaxis_tickformat=".0%", height=420)
    st.plotly_chart(fig2, use_container_width=True)

# Tab 3 — Churn by Monthly Charges
with tab3:
    churners = filtered[filtered["Churn"] == 1]["MonthlyCharges"]
    non_churners = filtered[filtered["Churn"] == 0]["MonthlyCharges"]

    fig3 = go.Figure()
    fig3.add_trace(
        go.Histogram(
            x=non_churners,
            name="No Churn",
            opacity=0.7,
            marker_color="#888888",
            nbinsx=30,
        )
    )
    fig3.add_trace(
        go.Histogram(
            x=churners,
            name="Churned",
            opacity=0.7,
            marker_color="#00D4AA",
            nbinsx=30,
        )
    )
    fig3.update_layout(
        barmode="overlay",
        title="Monthly Charges Distribution by Churn Status",
        xaxis_title="Monthly Charges ($)",
        yaxis_title="Customer Count",
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig3, use_container_width=True)

# Tab 4 — Services Count & Churn
with tab4:
    sc = (
        filtered.groupby("services_count")["Churn"]
        .mean()
        .reset_index()
        .rename(columns={"Churn": "churn_rate"})
    )
    fig4 = px.bar(
        sc,
        x="services_count",
        y="churn_rate",
        title="Churn Rate vs Number of Active Services",
        labels={"services_count": "Services Count", "churn_rate": "Churn Rate"},
        template="plotly_dark",
        color_discrete_sequence=["#00D4AA"],
    )
    fig4.update_layout(yaxis_tickformat=".0%", height=420)
    st.plotly_chart(fig4, use_container_width=True)

# Tab 5 — Payment Method
with tab5:
    pm = (
        filtered.groupby("PaymentMethod")["Churn"]
        .mean()
        .reset_index()
        .rename(columns={"Churn": "churn_rate"})
        .sort_values("churn_rate", ascending=False)
    )
    fig5 = px.bar(
        pm,
        x="PaymentMethod",
        y="churn_rate",
        title="Churn Rate by Payment Method",
        labels={"PaymentMethod": "Payment Method", "churn_rate": "Churn Rate"},
        template="plotly_dark",
        color_discrete_sequence=["#00D4AA"],
    )
    fig5.update_layout(yaxis_tickformat=".0%", height=420)
    st.plotly_chart(fig5, use_container_width=True)
