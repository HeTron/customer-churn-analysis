"""Business-value functions — pure, no Streamlit dependencies."""

from __future__ import annotations

import pandas as pd

_RISK_GRID_DEFAULT = [50, 100, 200, 500, 1000, 2000]


def revenue_at_risk(
    df_with_proba: pd.DataFrame,
    mrr_col: str = "MonthlyCharges",
    proba_col: str = "churn_proba",
) -> float:
    """
    Expected monthly revenue lost to churn.

    Sum of MonthlyCharges × churn_proba across all customers.
    """
    return float((df_with_proba[mrr_col] * df_with_proba[proba_col]).sum())


def rank_customers(
    df_with_proba: pd.DataFrame,
    mrr_col: str = "MonthlyCharges",
    proba_col: str = "churn_proba",
) -> pd.DataFrame:
    """
    Add expected_loss (annualised) and risk_tier, then sort by expected_loss desc.

    expected_loss = MonthlyCharges × churn_proba × 12  (annualised MRR at risk)
    risk_tier     = High (≥0.7), Medium (0.4–0.7), Low (<0.4)
    """
    df = df_with_proba.copy()
    df["expected_loss"] = df[mrr_col] * df[proba_col] * 12

    df["risk_tier"] = pd.cut(
        df[proba_col],
        bins=[-0.001, 0.4, 0.7, 1.001],
        labels=["Low", "Medium", "High"],
    )

    return df.sort_values("expected_loss", ascending=False).reset_index(drop=True)


def campaign_roi(
    ranked_df: pd.DataFrame,
    top_n: int,
    retention_cost_per_customer: float,
    intervention_success_rate: float,
    customer_ltv_months: int = 24,
    mrr_col: str = "MonthlyCharges",
    proba_col: str = "churn_proba",
) -> dict:
    """
    ROI of targeting the top-N highest-risk customers.

    spend          = top_n × retention_cost_per_customer
    expected_saves = sum(MRR × proba × success_rate × LTV_months) for top-N rows
    net            = expected_saves − spend
    roi            = net / spend  (e.g. 2.5 = 250% ROI)
    """
    target = ranked_df.head(top_n)

    spend = top_n * retention_cost_per_customer

    expected_saves = float(
        (target[mrr_col] * target[proba_col] * intervention_success_rate * customer_ltv_months).sum()
    )

    net = expected_saves - spend
    roi = net / spend if spend > 0 else 0.0

    return {
        "top_n": top_n,
        "spend": spend,
        "expected_saves": expected_saves,
        "net": net,
        "roi": roi,
        "customers_targeted": len(target),
        "avg_proba_in_target": float(target[proba_col].mean()) if len(target) > 0 else 0.0,
    }


def optimize_top_n(
    ranked_df: pd.DataFrame,
    cost_per_customer: float,
    intervention_success_rate: float,
    ltv_months: int = 24,
    n_grid: list[int] | None = None,
    mrr_col: str = "MonthlyCharges",
    proba_col: str = "churn_proba",
) -> pd.DataFrame:
    """
    Sweep top_n across a grid, computing ROI at each point.

    Returns a DataFrame with columns [top_n, spend, expected_saves, net, roi]
    useful for plotting the optimal campaign size.
    """
    if n_grid is None:
        n_grid = _RISK_GRID_DEFAULT

    n_max = len(ranked_df)
    results = []
    for n in n_grid:
        n = min(n, n_max)
        if n == 0:
            continue
        row = campaign_roi(
            ranked_df,
            top_n=n,
            retention_cost_per_customer=cost_per_customer,
            intervention_success_rate=intervention_success_rate,
            customer_ltv_months=ltv_months,
            mrr_col=mrr_col,
            proba_col=proba_col,
        )
        results.append(row)
        if n == n_max:
            break

    return pd.DataFrame(results).drop_duplicates(subset=["top_n"])
