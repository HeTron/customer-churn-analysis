"""LightGBM classifier, cross-validation, and per-customer risk factor analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

DEFAULT_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
}


@st.cache_resource
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
) -> LGBMClassifier:
    """Fit LGBMClassifier on full (X, y). Cached by Streamlit per unique call."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    model = LGBMClassifier(**p)
    model.fit(X, y)
    return model


def predict_proba(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return P(churn=1) for each row in X."""
    return model.predict_proba(X)[:, 1]


def cross_validated_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    params: dict | None = None,
) -> dict:
    """
    Stratified k-fold cross-validation.

    Returns per-fold metric lists + means + out_of_fold_probas
    (same length as y, useful for calibration plots and honest ranking).
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aucs, accs, precs, recs, f1s, lls = [], [], [], [], [], []
    oof_probas = np.zeros(len(y))

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = LGBMClassifier(**p)
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)

        oof_probas[val_idx] = proba

        aucs.append(float(roc_auc_score(y_val, proba)))
        accs.append(float(accuracy_score(y_val, preds)))
        precs.append(float(precision_score(y_val, preds, zero_division=0)))
        recs.append(float(recall_score(y_val, preds, zero_division=0)))
        f1s.append(float(f1_score(y_val, preds, zero_division=0)))
        lls.append(float(log_loss(y_val, proba)))

    return {
        "auc": aucs,
        "accuracy": accs,
        "precision": precs,
        "recall": recs,
        "f1": f1s,
        "log_loss": lls,
        "mean_auc": float(np.mean(aucs)),
        "mean_accuracy": float(np.mean(accs)),
        "mean_precision": float(np.mean(precs)),
        "mean_recall": float(np.mean(recs)),
        "mean_f1": float(np.mean(f1s)),
        "mean_log_loss": float(np.mean(lls)),
        "out_of_fold_probas": oof_probas,
        "n_splits": n_splits,
    }


def feature_importance(
    model: LGBMClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return a DataFrame sorted by feature importance descending."""
    return (
        pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def top_risk_factors(
    model: LGBMClassifier,
    customer_row: pd.DataFrame,
    feature_names: list[str],
    n: int = 3,
) -> list[dict]:
    """
    Approximate per-feature contributions for a single customer.

    Uses a simple approach: scale each feature value by the global feature
    importance. This is not SHAP but is fast and good enough for a "why this
    customer might churn" UI explanation.

    Returns top-n positive contributors as a list of dicts with keys:
    feature, value, weight.
    """
    importances = model.feature_importances_.astype(float)

    row_values = customer_row.iloc[0].values.astype(float)

    # Normalise importance to [0, 1]
    imp_norm = importances / (importances.max() + 1e-9)

    # Simple weighted contribution: value × normalised importance
    contributions = row_values * imp_norm

    # Top-n positive contributors
    top_idx = np.argsort(contributions)[::-1][:n]

    return [
        {
            "feature": feature_names[i],
            "value": float(row_values[i]),
            "weight": float(contributions[i]),
        }
        for i in top_idx
        if contributions[i] > 0
    ]
