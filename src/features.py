"""Feature engineering — pure functions, no Streamlit dependencies."""

from __future__ import annotations

import numpy as np
import pandas as pd

_YES_NO_SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

_CATEGORICAL_COLS = [
    "gender",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "tenure_bucket",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PhoneService",
    "PaperlessBilling",
    "Partner",
    "Dependents",
]


def _services_count(row: pd.Series) -> int:
    """Count how many services a customer has active (Yes)."""
    count = 0
    for col in _YES_NO_SERVICE_COLS:
        if str(row.get(col, "")).strip() == "Yes":
            count += 1
    return count


def _tenure_bucket(tenure: float) -> str:
    """Bin tenure (months) into descriptive ranges."""
    if tenure <= 12:
        return "0-12"
    elif tenure <= 24:
        return "13-24"
    elif tenure <= 48:
        return "25-48"
    else:
        return "49+"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-driven engineered features to the cleaned Telco dataframe.
    All inputs must already be clean (no blank TotalCharges, Churn as 0/1).
    """
    df = df.copy()

    df["tenure_bucket"] = df["tenure"].apply(_tenure_bucket)

    df["services_count"] = df.apply(_services_count, axis=1)

    df["has_security_addon"] = (
        (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)

    df["has_streaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)

    # Avoid division by zero for new customers (tenure == 0)
    safe_tenure = df["tenure"].clip(lower=1)
    df["avg_charge_per_month"] = df["TotalCharges"] / safe_tenure

    # Ratio close to 1.0 means billing has been consistent; <1 suggests a recent price hike
    df["charge_vs_monthly_ratio"] = df["avg_charge_per_month"] / df["MonthlyCharges"].replace(0, np.nan)
    df["charge_vs_monthly_ratio"] = df["charge_vs_monthly_ratio"].fillna(1.0)

    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    df["is_auto_pay"] = df["PaymentMethod"].isin(
        {"Bank transfer (automatic)", "Credit card (automatic)"}
    ).astype(int)

    # Fiber customers churn at materially higher rates in this dataset
    df["is_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Add engineered features, one-hot encode categoricals, and return (X, y).

    X contains no target column and no NaNs.
    y is the Churn series (0/1).
    """
    df = add_engineered_features(df)

    y = df["Churn"].copy()
    df = df.drop(columns=["Churn"])

    present_cats = [c for c in _CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=present_cats, drop_first=True)

    # Ensure no NaN survives (charge_vs_monthly_ratio already filled; belt-and-suspenders)
    df = df.fillna(0)

    # Convert all bool columns that get_dummies may produce to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.copy()
    return X, y


# Populated after the first call in tests; also built here for module-level access.
# Real column order is data-dependent — use get_feature_cols() for authoritative list.
FEATURE_COLS: list[str] = []


def get_feature_cols(X: pd.DataFrame) -> list[str]:
    """Return the column list from a built feature matrix and cache in FEATURE_COLS."""
    global FEATURE_COLS
    FEATURE_COLS = list(X.columns)
    return FEATURE_COLS


def prepare_single_customer(
    customer_dict: dict,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Encode a single customer dict (from the Predict UI) into the same column
    space as the training feature matrix.

    Missing dummy columns are zero-filled; extra columns are dropped.
    """
    # Build a one-row DataFrame
    row_df = pd.DataFrame([customer_dict])

    # Apply engineered features
    row_df = add_engineered_features(row_df)

    # Drop Churn if present (shouldn't be, but guard anyway)
    if "Churn" in row_df.columns:
        row_df = row_df.drop(columns=["Churn"])

    present_cats = [c for c in _CATEGORICAL_COLS if c in row_df.columns]
    row_df = pd.get_dummies(row_df, columns=present_cats, drop_first=True)

    bool_cols = row_df.select_dtypes(include="bool").columns
    row_df[bool_cols] = row_df[bool_cols].astype(int)

    # Align to training columns: add missing as 0, drop extras
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feature_cols]

    return row_df.fillna(0)
