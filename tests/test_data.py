"""Tests for src/data.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import clean, load_raw


def _make_raw_df() -> pd.DataFrame:
    """Minimal synthetic Telco-like dataframe for testing."""
    return pd.DataFrame({
        "customerID": ["AAA-001", "AAA-002", "AAA-003"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 0, 1],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [1, 34, 72],
        "PhoneService": ["No", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes"],
        "InternetService": ["DSL", "DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes", "No"],
        "OnlineBackup": ["Yes", "No", "No"],
        "DeviceProtection": ["No", "Yes", "No"],
        "TechSupport": ["No", "No", "No"],
        "StreamingTV": ["No", "No", "Yes"],
        "StreamingMovies": ["No", "No", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 56.95, 100.35],
        "TotalCharges": ["29.85", "", "7362.90"],  # blank to trigger coercion
        "Churn": ["No", "No", "Yes"],
    })


class TestClean:
    def test_drops_customer_id(self):
        raw = _make_raw_df()
        cleaned = clean(raw)
        assert "customerID" not in cleaned.columns

    def test_handles_total_charges_blank(self):
        raw = _make_raw_df()
        cleaned = clean(raw)
        # Row with blank TotalCharges should be dropped; 2 rows remain
        assert len(cleaned) == 2

    def test_churn_mapped_to_int(self):
        raw = _make_raw_df()
        cleaned = clean(raw)
        assert set(cleaned["Churn"].unique()).issubset({0, 1})

    def test_churn_yes_maps_to_1(self):
        raw = _make_raw_df()
        cleaned = clean(raw)
        # Only row with Churn=Yes that survives is row index 2 (tenure 72)
        assert cleaned.loc[cleaned["tenure"] == 72, "Churn"].iloc[0] == 1

    def test_total_charges_is_numeric(self):
        raw = _make_raw_df()
        cleaned = clean(raw)
        assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])

    def test_no_nan_in_churn(self):
        raw = _make_raw_df()
        cleaned = clean(raw)
        assert cleaned["Churn"].isna().sum() == 0
