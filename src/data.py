"""Telco churn dataset loading and cleaning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_CSV = _PROJECT_ROOT / "churn_data.csv"

_STRING_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "Churn",
]


def load_raw(csv_path: str | Path = _DEFAULT_CSV) -> pd.DataFrame:
    """Read the Telco churn CSV as-is."""
    return pd.read_csv(csv_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the IBM Telco churn dataframe:
    - Strip whitespace from string columns (TotalCharges blank → NaN via to_numeric).
    - Coerce TotalCharges to float; drop rows where coercion fails (11 blanks).
    - Drop customerID (not a feature).
    - Map Churn Yes/No → 1/0.
    """
    df = df.copy()

    for col in _STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df.reset_index(drop=True)


@st.cache_data
def load_clean(csv_path: str | Path = _DEFAULT_CSV) -> pd.DataFrame:
    """Load and clean the Telco churn dataset. Cached by Streamlit."""
    return clean(load_raw(csv_path))
