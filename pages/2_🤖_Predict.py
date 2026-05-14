"""Predict page — single-customer churn scorer + Claude retention recommendation."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.data import load_clean
from src.explain import explain_churn
from src.features import build_feature_matrix, get_feature_cols, prepare_single_customer
from src.model import cross_validated_metrics, feature_importance, predict_proba, top_risk_factors, train_model

st.set_page_config(
    page_title="Predict · Churn Edge",
    page_icon="🤖",
    layout="wide",
)
st.title("🤖 Single-Customer Churn Predictor")

# ── Load data + train (cached) ─────────────────────────────────────────────────
df = load_clean()
X_full, y_full = build_feature_matrix(df)
feature_cols = get_feature_cols(X_full)

# ── Sidebar — customer input form ──────────────────────────────────────────────
with st.sidebar:
    st.header("Customer Profile")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5)
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(monthly_charges * max(tenure, 1)),
        step=1.0,
        help="Defaults to MonthlyCharges × tenure. Override if needed.",
    )

    st.subheader("Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.subheader("Plan & Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

run = st.button("Predict Churn Risk", type="primary")

if not run:
    st.info("Configure the customer profile in the sidebar, then click **Predict Churn Risk**.")
    st.stop()

# ── Pipeline ──────────────────────────────────────────────────────────────────
customer_dict = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

with st.spinner("Training LightGBM on full dataset..."):
    model = train_model(X_full, y_full)

with st.spinner("Running stratified 5-fold cross-validation..."):
    cv_metrics = cross_validated_metrics(X_full, y_full, n_splits=5)

with st.spinner("Scoring customer..."):
    customer_row = prepare_single_customer(customer_dict, feature_cols)
    churn_prob = float(predict_proba(model, customer_row)[0])

risk_tier = "High" if churn_prob >= 0.7 else "Medium" if churn_prob >= 0.4 else "Low"
tier_color = "#FF4B4B" if risk_tier == "High" else "#FFA500" if risk_tier == "Medium" else "#00D4AA"

# ── Churn probability display ──────────────────────────────────────────────────
col_gauge, col_metrics = st.columns([1, 1])

with col_gauge:
    st.subheader("Churn Probability")
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Risk Tier: {risk_tier}", "font": {"color": tier_color, "size": 18}},
            number={"suffix": "%", "font": {"size": 48}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": tier_color},
                "steps": [
                    {"range": [0, 40], "color": "#1A2A2A"},
                    {"range": [40, 70], "color": "#2A2A1A"},
                    {"range": [70, 100], "color": "#2A1A1A"},
                ],
                "threshold": {
                    "line": {"color": tier_color, "width": 4},
                    "thickness": 0.75,
                    "value": churn_prob * 100,
                },
            },
        )
    )
    fig_gauge.update_layout(template="plotly_dark", height=320, margin=dict(t=60, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_metrics:
    st.subheader("CV Validation Metrics")
    st.caption(
        "Computed on stratified 5-fold cross-validation — true out-of-sample performance."
    )
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Mean AUC", f"{cv_metrics['mean_auc']:.3f}")
    mc2.metric("Mean Accuracy", f"{cv_metrics['mean_accuracy']:.3f}")
    mc3.metric("Mean F1", f"{cv_metrics['mean_f1']:.3f}")

    st.markdown("&nbsp;")

    mc4, mc5 = st.columns(2)
    mc4.metric("Mean Precision", f"{cv_metrics['mean_precision']:.3f}")
    mc5.metric("Mean Recall", f"{cv_metrics['mean_recall']:.3f}")

st.divider()

# ── Top risk factors ───────────────────────────────────────────────────────────
st.subheader("Top Risk Factors")

risk_factors = top_risk_factors(model, customer_row, feature_cols, n=5)

if risk_factors:
    fig_rf = go.Figure(
        go.Bar(
            x=[f["weight"] for f in risk_factors],
            y=[f["feature"] for f in risk_factors],
            orientation="h",
            marker_color="#00D4AA",
        )
    )
    fig_rf.update_layout(
        template="plotly_dark",
        height=280,
        xaxis_title="Risk Contribution (feature value × normalised importance)",
        margin=dict(l=200),
    )
    st.plotly_chart(fig_rf, use_container_width=True)
    st.caption(
        "Approximate: feature value × normalised global importance. "
        "Not SHAP — directional signal only."
    )
else:
    st.info("No positive risk factors identified for this customer profile.")

st.divider()

# ── Feature importance (global) ────────────────────────────────────────────────
with st.expander("Global Feature Importance (top 15)"):
    fi_df = feature_importance(model, feature_cols)
    top15 = fi_df.head(15).sort_values("importance")
    fig_fi = go.Figure(
        go.Bar(
            x=top15["importance"],
            y=top15["feature"],
            orientation="h",
            marker_color="#00D4AA",
        )
    )
    fig_fi.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=200),
        xaxis_title="Importance (splits)",
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ── Claude's retention recommendation ─────────────────────────────────────────
st.subheader("Claude's Retention Recommendation")

with st.spinner("Asking Claude to analyse this customer..."):
    explanation = explain_churn(
        customer_profile=customer_dict,
        churn_proba=churn_prob,
        top_factors=risk_factors,
        model_metrics=cv_metrics,
    )

st.markdown(
    f"""
<div style="background:#1A1F2E;padding:1.2rem 1.5rem;border-left:4px solid #00D4AA;border-radius:4px;margin-top:0.5rem">
{explanation}
</div>
""",
    unsafe_allow_html=True,
)
