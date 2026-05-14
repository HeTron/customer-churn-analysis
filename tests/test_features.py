"""Tests for src/features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_engineered_features,
    build_feature_matrix,
    prepare_single_customer,
)


def _make_clean_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Synthetic cleaned Telco dataframe (customerID already dropped, Churn as 0/1)."""
    rng = np.random.default_rng(seed)
    contracts = ["Month-to-month", "One year", "Two year"]
    internet = ["DSL", "Fiber optic", "No"]
    yes_no = ["Yes", "No"]
    payment = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    def pick(opts, size):
        return rng.choice(opts, size=size)

    tenure = rng.integers(0, 73, size=n)
    monthly = rng.uniform(18, 120, size=n).round(2)
    total = (monthly * tenure.clip(min=1)).round(2)

    return pd.DataFrame({
        "gender": pick(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, size=n),
        "Partner": pick(yes_no, n),
        "Dependents": pick(yes_no, n),
        "tenure": tenure,
        "PhoneService": pick(yes_no, n),
        "MultipleLines": pick(["Yes", "No", "No phone service"], n),
        "InternetService": pick(internet, n),
        "OnlineSecurity": pick(["Yes", "No", "No internet service"], n),
        "OnlineBackup": pick(["Yes", "No", "No internet service"], n),
        "DeviceProtection": pick(["Yes", "No", "No internet service"], n),
        "TechSupport": pick(["Yes", "No", "No internet service"], n),
        "StreamingTV": pick(["Yes", "No", "No internet service"], n),
        "StreamingMovies": pick(["Yes", "No", "No internet service"], n),
        "Contract": pick(contracts, n),
        "PaperlessBilling": pick(yes_no, n),
        "PaymentMethod": pick(payment, n),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": rng.integers(0, 2, size=n),
    })


class TestServicesCount:
    def test_all_yes_gives_max(self):
        row = pd.Series({
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
        })
        df = pd.DataFrame([{**row.to_dict(), "tenure": 12, "MonthlyCharges": 80,
                            "TotalCharges": 960, "InternetService": "Fiber optic",
                            "Contract": "Month-to-month", "SeniorCitizen": 0,
                            "gender": "Male", "Partner": "No", "Dependents": "No",
                            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
                            "Churn": 0}])
        engineered = add_engineered_features(df)
        assert engineered["services_count"].iloc[0] == 8

    def test_all_no_gives_zero(self):
        row_dict = {
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "tenure": 5,
            "MonthlyCharges": 20,
            "TotalCharges": 100,
            "InternetService": "No",
            "Contract": "Month-to-month",
            "SeniorCitizen": 0,
            "gender": "Male",
            "Partner": "No",
            "Dependents": "No",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "Churn": 0,
        }
        df = pd.DataFrame([row_dict])
        engineered = add_engineered_features(df)
        assert engineered["services_count"].iloc[0] == 0


class TestHasSecurityAddon:
    def test_online_security_yes(self):
        df = _make_clean_df(10)
        df["OnlineSecurity"] = "Yes"
        df["TechSupport"] = "No"
        engineered = add_engineered_features(df)
        assert (engineered["has_security_addon"] == 1).all()

    def test_tech_support_yes(self):
        df = _make_clean_df(10)
        df["OnlineSecurity"] = "No"
        df["TechSupport"] = "Yes"
        engineered = add_engineered_features(df)
        assert (engineered["has_security_addon"] == 1).all()

    def test_neither_gives_zero(self):
        df = _make_clean_df(10)
        df["OnlineSecurity"] = "No"
        df["TechSupport"] = "No"
        engineered = add_engineered_features(df)
        assert (engineered["has_security_addon"] == 0).all()


class TestIsMonthToMonth:
    def test_month_to_month_is_1(self):
        df = _make_clean_df(10)
        df["Contract"] = "Month-to-month"
        engineered = add_engineered_features(df)
        assert (engineered["is_month_to_month"] == 1).all()

    def test_one_year_is_0(self):
        df = _make_clean_df(10)
        df["Contract"] = "One year"
        engineered = add_engineered_features(df)
        assert (engineered["is_month_to_month"] == 0).all()


class TestBuildFeatureMatrix:
    def test_no_nan_in_X(self):
        df = _make_clean_df(100)
        X, y = build_feature_matrix(df)
        assert not X.isna().any().any(), "X must not contain NaN"

    def test_same_length_X_y(self):
        df = _make_clean_df(100)
        X, y = build_feature_matrix(df)
        assert len(X) == len(y)

    def test_y_is_binary(self):
        df = _make_clean_df(100)
        _, y = build_feature_matrix(df)
        assert set(y.unique()).issubset({0, 1})

    def test_churn_not_in_X(self):
        df = _make_clean_df(100)
        X, _ = build_feature_matrix(df)
        assert "Churn" not in X.columns


class TestPrepareSingleCustomer:
    def test_columns_match_training(self):
        df = _make_clean_df(100)
        X, _ = build_feature_matrix(df)
        feature_cols = list(X.columns)

        customer = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 24,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.0,
            "TotalCharges": 2040.0,
        }

        row = prepare_single_customer(customer, feature_cols)
        assert list(row.columns) == feature_cols
        assert len(row) == 1

    def test_no_nan_in_prepared_row(self):
        df = _make_clean_df(100)
        X, _ = build_feature_matrix(df)
        feature_cols = list(X.columns)

        customer = {
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 30.0,
            "TotalCharges": 30.0,
        }

        row = prepare_single_customer(customer, feature_cols)
        assert not row.isna().any().any()
