"""Claude-powered plain-English churn explanation and retention recommendation."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a customer retention analyst. Given a telecom customer's profile, "
    "their predicted churn probability, and the top features driving that risk, "
    "explain in 3-4 plain-English sentences WHY this customer is likely to churn "
    "and WHAT retention action is recommended. "
    "Be specific — point to the exact features (e.g. month-to-month contract, "
    "fiber optic without security add-ons, high monthly charges). "
    "Avoid hedging. End with one concrete action the retention team should take "
    "(e.g. 'Offer a 12-month contract discount', 'Add TechSupport to their plan at no cost'). "
    "Write for a non-technical account manager, not a data scientist."
)


@st.cache_data(ttl=300)
def explain_churn(
    customer_profile: dict[str, Any],
    churn_proba: float,
    top_factors: list[dict],
    model_metrics: dict[str, Any],
) -> str:
    """
    Call Claude Sonnet to explain a customer's churn risk.
    Cached 5 minutes to avoid redundant API calls for the same customer.
    Gracefully degrades when ANTHROPIC_API_KEY is absent.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return (
            "_Claude explanation unavailable — set ANTHROPIC_API_KEY in .env "
            "or Streamlit secrets._"
        )

    client = Anthropic(api_key=api_key)

    profile_lines = "\n".join(
        f"  - {k}: {v}" for k, v in customer_profile.items()
    )

    factor_lines = "\n".join(
        f"  - {f['feature']} (value={f['value']:.2f}, risk weight={f['weight']:.4f})"
        for f in top_factors
    )

    user_msg = (
        f"Customer profile:\n{profile_lines}\n\n"
        f"Predicted churn probability: {churn_proba:.1%}\n\n"
        f"Top contributing risk factors:\n{factor_lines}\n\n"
        f"Model validation (stratified 5-fold CV):\n"
        f"  - Mean AUC: {model_metrics.get('mean_auc', 'N/A')}\n"
        f"  - Mean F1: {model_metrics.get('mean_f1', 'N/A')}\n\n"
        "Please explain this customer's churn risk and recommend one retention action."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text
