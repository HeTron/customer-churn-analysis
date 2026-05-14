"""Tests for src/business.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.business import campaign_roi, optimize_top_n, rank_customers, revenue_at_risk


def _make_scored_df(n: int = 20) -> pd.DataFrame:
    """Simple dataframe with MonthlyCharges and churn_proba for business logic tests."""
    import numpy as np
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "MonthlyCharges": rng.uniform(20, 100, n).round(2),
        "churn_proba": rng.uniform(0, 1, n).round(3),
    })


class TestRevenueAtRisk:
    def test_exact_sum(self):
        df = pd.DataFrame({
            "MonthlyCharges": [100.0, 50.0, 80.0],
            "churn_proba": [0.5, 0.2, 0.8],
        })
        expected = 100.0 * 0.5 + 50.0 * 0.2 + 80.0 * 0.8
        result = revenue_at_risk(df)
        assert abs(result - expected) < 1e-6

    def test_zero_proba_gives_zero(self):
        df = pd.DataFrame({
            "MonthlyCharges": [100.0, 200.0],
            "churn_proba": [0.0, 0.0],
        })
        assert revenue_at_risk(df) == pytest.approx(0.0)

    def test_full_proba_equals_sum_mrr(self):
        df = pd.DataFrame({
            "MonthlyCharges": [100.0, 50.0],
            "churn_proba": [1.0, 1.0],
        })
        assert revenue_at_risk(df) == pytest.approx(150.0)


class TestRankCustomers:
    def test_sorted_by_expected_loss_desc(self):
        df = _make_scored_df(30)
        ranked = rank_customers(df)
        losses = ranked["expected_loss"].tolist()
        assert losses == sorted(losses, reverse=True)

    def test_expected_loss_formula(self):
        df = pd.DataFrame({
            "MonthlyCharges": [100.0],
            "churn_proba": [0.5],
        })
        ranked = rank_customers(df)
        # expected_loss = 100 × 0.5 × 12 = 600
        assert ranked["expected_loss"].iloc[0] == pytest.approx(600.0)

    def test_risk_tier_high(self):
        df = pd.DataFrame({
            "MonthlyCharges": [80.0],
            "churn_proba": [0.75],
        })
        ranked = rank_customers(df)
        assert ranked["risk_tier"].iloc[0] == "High"

    def test_risk_tier_low(self):
        df = pd.DataFrame({
            "MonthlyCharges": [50.0],
            "churn_proba": [0.2],
        })
        ranked = rank_customers(df)
        assert ranked["risk_tier"].iloc[0] == "Low"


class TestCampaignRoi:
    def test_zero_success_rate_gives_negative_roi(self):
        df = pd.DataFrame({
            "MonthlyCharges": [100.0] * 10,
            "churn_proba": [0.8] * 10,
        })
        ranked = rank_customers(df)
        result = campaign_roi(ranked, top_n=5, retention_cost_per_customer=50, intervention_success_rate=0.0)
        # With 0% success, expected_saves=0, spend=250, net=-250
        assert result["expected_saves"] == pytest.approx(0.0)
        assert result["net"] < 0
        assert result["roi"] < 0

    def test_high_success_rate_gives_positive_roi(self):
        df = pd.DataFrame({
            "MonthlyCharges": [200.0] * 10,
            "churn_proba": [0.9] * 10,
        })
        ranked = rank_customers(df)
        # Cost $10 per customer, 100% success, 24-month LTV → saves far exceed cost
        result = campaign_roi(
            ranked,
            top_n=5,
            retention_cost_per_customer=10.0,
            intervention_success_rate=1.0,
            customer_ltv_months=24,
        )
        assert result["expected_saves"] > result["spend"]
        assert result["roi"] > 0

    def test_spend_calculation(self):
        df = _make_scored_df(20)
        ranked = rank_customers(df)
        result = campaign_roi(ranked, top_n=10, retention_cost_per_customer=25.0, intervention_success_rate=0.3)
        assert result["spend"] == pytest.approx(250.0)

    def test_customers_targeted_capped_at_top_n(self):
        df = _make_scored_df(5)
        ranked = rank_customers(df)
        result = campaign_roi(ranked, top_n=100, retention_cost_per_customer=50, intervention_success_rate=0.2)
        # Can't target more customers than exist
        assert result["customers_targeted"] <= 5


class TestOptimizeTopN:
    def test_returns_row_for_each_grid_point(self):
        df = _make_scored_df(500)
        ranked = rank_customers(df)
        grid = [10, 50, 100, 200]
        result = optimize_top_n(ranked, cost_per_customer=50, intervention_success_rate=0.2, n_grid=grid)
        # All grid points ≤ len(ranked) should appear
        assert len(result) >= 1
        for n in grid:
            if n <= len(ranked):
                assert n in result["top_n"].values

    def test_output_columns(self):
        df = _make_scored_df(100)
        ranked = rank_customers(df)
        result = optimize_top_n(ranked, cost_per_customer=50, intervention_success_rate=0.2)
        for col in ["top_n", "spend", "expected_saves", "net", "roi"]:
            assert col in result.columns
