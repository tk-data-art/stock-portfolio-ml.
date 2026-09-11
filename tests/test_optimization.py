"""Unit tests for Markowitz quadratic portfolio optimization."""

import numpy as np
import pandas as pd
import pytest
from stock_portfolio.optimization.markowitz import MarkowitzOptimizer


@pytest.fixture
def sample_market_inputs() -> tuple[pd.Series, pd.DataFrame]:
    """Provide deterministic test returns and covariance matrix."""
    tickers = ["AAPL", "MSFT", "GOOGL"]
    # AAPL and GOOGL positive, MSFT negative
    expected_returns = pd.Series([0.20, -0.10, 0.15], index=tickers)

    # 3x3 positive semi-definite covariance matrix
    cov_data = np.array(
        [
            [0.08, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.07],
        ]
    )
    cov_matrix = pd.DataFrame(cov_data, index=tickers, columns=tickers)
    return expected_returns, cov_matrix


def test_weights_sum_to_one(
    sample_market_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    """Validate full capital deployment constraint: sum(w_i) == 1.0."""
    mu, sigma = sample_market_inputs
    optimizer = MarkowitzOptimizer(risk_free_rate=0.04)
    result = optimizer.optimize_max_sharpe(mu, sigma)

    total_weight = sum(result.weights.values())
    assert pytest.approx(total_weight, abs=1e-3) == 1.0


def test_long_only_constraint(
    sample_market_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    """Validate that all weights are non-negative (no short positions)."""
    mu, sigma = sample_market_inputs
    optimizer = MarkowitzOptimizer(risk_free_rate=0.04)
    result = optimizer.optimize_max_sharpe(mu, sigma)

    for ticker, weight in result.weights.items():
        assert weight >= 0.0, f"Weight for {ticker} is negative: {weight}"


def test_negative_return_asset_receives_zero_weight(
    sample_market_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    """Ensure an asset with negative expected return gets 0% weight."""
    mu, sigma = sample_market_inputs
    optimizer = MarkowitzOptimizer(risk_free_rate=0.04)
    result = optimizer.optimize_max_sharpe(mu, sigma)

    assert result.weights["MSFT"] == 0.0
