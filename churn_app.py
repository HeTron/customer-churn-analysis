"""Churn Edge — home / landing page."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Churn Edge",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔁 Churn Edge")
st.subheader(
    "Customer churn prediction with business-value scoring "
    "and Claude-powered retention recommendations."
)

st.caption(
    "Portfolio project — IBM Telco churn dataset · LightGBM · Stratified CV · "
    "Revenue-at-risk scoring · Claude Sonnet explanations"
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Cohort Analysis")
    st.markdown(
        "Explore churn patterns across contract types, tenure buckets, "
        "payment methods, and service add-ons. Filter the dataset interactively "
        "and watch headline metrics update in real time."
    )

with col2:
    st.markdown("### 🤖 Predict")
    st.markdown(
        "Score a single customer: configure their plan details, run the full "
        "pipeline — features → LightGBM → probability — and get a Claude-generated "
        "retention recommendation in plain English."
    )

with col3:
    st.markdown("### 💰 Campaign Optimizer")
    st.markdown(
        "Score every customer in the dataset, rank by annualised revenue at risk, "
        "and find the optimal retention campaign size that maximises ROI given your "
        "intervention cost and success-rate assumptions."
    )

st.divider()

st.markdown("## How it works")
st.markdown(
    """
```
IBM Telco CSV (7,043 customers)
  └─► Clean
        Coerce TotalCharges · drop 11 blank rows · map Churn Yes/No → 1/0
  └─► Engineered Features (9 new signals)
        tenure_bucket · services_count · has_security_addon · has_streaming
        avg_charge_per_month · charge_vs_monthly_ratio
        is_month_to_month · is_auto_pay · is_fiber
  └─► One-hot encoding → feature matrix
  └─► LightGBM Classifier
        class_weight=balanced · 300 estimators · depth 6
  └─► Stratified 5-fold Cross-Validation
        AUC · Accuracy · Precision · Recall · F1 · Log-loss per fold
        Out-of-fold probabilities for honest downstream ranking
  └─► Claude Explanation (Sonnet)
        Customer profile + top risk factors → plain-English retention rec
  └─► Revenue-at-risk scoring
        Expected loss = MonthlyCharges × churn_proba × 12 per customer
  └─► Campaign Optimizer
        Sweep campaign size → ROI curve → optimal top-N recommendation
```
"""
)

st.divider()
st.caption(
    "Built by Jason Eid · "
    "[github.com/HeTron/customer-churn-analysis](https://github.com/HeTron/customer-churn-analysis)"
)
