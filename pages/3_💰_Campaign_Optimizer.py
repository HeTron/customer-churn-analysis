"""Campaign Optimizer — batch scoring, revenue-at-risk, and ROI-optimal campaign sizing."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.business import campaign_roi, optimize_top_n, rank_customers, revenue_at_risk
from src.data import load_clean
from src.features import build_feature_matrix, get_feature_cols
from src.model import predict_proba, train_model

st.set_page_config(
    page_title="Campaign Optimizer · Churn Edge",
    page_icon="💰",
    layout="wide",
)
st.title("💰 Campaign Optimizer")

st.markdown(
    "Score every customer in the dataset, rank by revenue-at-risk, "
    "and find the optimal retention campaign size that maximises ROI."
)

# ── Sidebar — campaign assumptions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Campaign Assumptions")

    cost_per_customer = st.number_input(
        "Cost per retention touch ($)",
        min_value=1.0,
        max_value=500.0,
        value=50.0,
        step=1.0,
        help="Total cost to reach one customer (call, offer, gift card, etc.)",
    )

    success_rate = st.slider(
        "Intervention success rate",
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.01,
        format="%.2f",
        help="Fraction of targeted churners you expect to retain.",
    )

    ltv_months = st.slider(
        "Customer LTV (months)",
        min_value=6,
        max_value=60,
        value=24,
        step=1,
        help="How many months of MRR you recover per successfully retained customer.",
    )

run = st.button("Run Campaign Analysis", type="primary")

if not run:
    st.info("Configure assumptions in the sidebar, then click **Run Campaign Analysis**.")
    st.stop()

# ── Score full dataset ─────────────────────────────────────────────────────────
df = load_clean()
X_full, y_full = build_feature_matrix(df)
feature_cols = get_feature_cols(X_full)

with st.spinner("Training LightGBM on full dataset..."):
    model = train_model(X_full, y_full)

with st.spinner("Scoring all customers..."):
    probas = predict_proba(model, X_full)

# Reattach MonthlyCharges to the scored frame
scored_df = df.copy()
scored_df["churn_proba"] = probas
# Align index after clean() reset_index
scored_df = scored_df.reset_index(drop=True)

with st.spinner("Ranking customers by revenue at risk..."):
    ranked = rank_customers(scored_df, mrr_col="MonthlyCharges", proba_col="churn_proba")

# ── Headline metrics ───────────────────────────────────────────────────────────
total_customers = len(ranked)
total_mrr = ranked["MonthlyCharges"].sum()
rar = revenue_at_risk(ranked, mrr_col="MonthlyCharges", proba_col="churn_proba")
high_risk_count = int((ranked["churn_proba"] >= 0.7).sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Customers", f"{total_customers:,}")
m2.metric("Total Monthly Revenue", f"${total_mrr:,.0f}")
m3.metric("Monthly Revenue at Risk", f"${rar:,.0f}")
m4.metric("High-Risk Customers (≥70%)", f"{high_risk_count:,}")

st.caption(
    "⚠️ Scores above are in-sample (model trained and scored on same data). "
    "For honest ranking, use the out-of-fold probabilities from the CV page."
)

st.divider()

# ── ROI vs Campaign Size ───────────────────────────────────────────────────────
st.subheader("ROI vs Campaign Size")

with st.spinner("Computing ROI across campaign sizes..."):
    roi_df = optimize_top_n(
        ranked,
        cost_per_customer=cost_per_customer,
        intervention_success_rate=success_rate,
        ltv_months=ltv_months,
        mrr_col="MonthlyCharges",
        proba_col="churn_proba",
    )

if not roi_df.empty:
    best_row = roi_df.loc[roi_df["roi"].idxmax()]
    best_n = int(best_row["top_n"])

    fig_roi = go.Figure()
    fig_roi.add_trace(
        go.Scatter(
            x=roi_df["top_n"],
            y=roi_df["roi"],
            mode="lines+markers",
            name="ROI",
            line=dict(color="#00D4AA", width=2),
            marker=dict(size=7),
        )
    )
    fig_roi.add_vline(
        x=best_n,
        line_dash="dash",
        line_color="#FFA500",
        annotation_text=f"Peak ROI at N={best_n}",
        annotation_position="top right",
    )
    fig_roi.update_layout(
        xaxis_title="Customers Targeted (top N by revenue at risk)",
        yaxis_title="Return on Investment",
        template="plotly_dark",
        height=400,
        yaxis_tickformat=".1f",
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    # ── Recommended campaign ───────────────────────────────────────────────────
    st.subheader("Recommended Campaign")

    best_campaign = campaign_roi(
        ranked,
        top_n=best_n,
        retention_cost_per_customer=cost_per_customer,
        intervention_success_rate=success_rate,
        customer_ltv_months=ltv_months,
    )

    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    rc1.metric("Optimal Target Size", f"{best_campaign['top_n']:,}")
    rc2.metric("Campaign Spend", f"${best_campaign['spend']:,.0f}")
    rc3.metric("Expected Revenue Saved", f"${best_campaign['expected_saves']:,.0f}")
    rc4.metric("Net Value", f"${best_campaign['net']:,.0f}")
    rc5.metric("ROI", f"{best_campaign['roi']:.1f}x")

    st.caption(
        f"Avg churn probability in target group: "
        f"{best_campaign['avg_proba_in_target']:.1%}"
    )

st.divider()

# ── Top 50 customers to target ─────────────────────────────────────────────────
st.subheader("Top 50 Customers to Target")

top50 = ranked.head(50).copy()
top50.insert(0, "Customer", [f"CUST-{i+1:04d}" for i in range(len(top50))])

display_cols = ["Customer", "MonthlyCharges", "churn_proba", "expected_loss", "risk_tier"]
display_df = top50[display_cols].copy()
display_df = display_df.rename(columns={
    "MonthlyCharges": "Monthly Charges ($)",
    "churn_proba": "Churn Probability",
    "expected_loss": "Expected Annual Loss ($)",
    "risk_tier": "Risk Tier",
})
display_df["Monthly Charges ($)"] = display_df["Monthly Charges ($)"].map("${:.2f}".format)
display_df["Churn Probability"] = display_df["Churn Probability"].map("{:.1%}".format)
display_df["Expected Annual Loss ($)"] = display_df["Expected Annual Loss ($)"].map("${:,.0f}".format)

st.dataframe(display_df, use_container_width=True, hide_index=True)
