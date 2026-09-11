"""Unit tests for data validation and Prophet dataframe transformations."""

import pandas as pd
import pytest
from stock_portfolio.data.collector import MarketDataCollector


def test_prepare_prophet_series_structure() -> None:
    """Validate Prophet dataframe schema requirement: exactly 'ds' and 'y' columns."""
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"AAPL": [150.0, 152.0, 151.0, 153.0, 155.0]}, index=dates)

    df_prophet = MarketDataCollector.prepare_prophet_series(prices, "AAPL")

    assert list(df_prophet.columns) == ["ds", "y"]
    assert len(df_prophet) == 5
    assert df_prophet["y"].iloc[-1] == 155.0


def test_prepare_prophet_series_missing_ticker() -> None:
    """Ensure KeyError is raised when requesting a non-existent ticker."""
    prices = pd.DataFrame({"AAPL": [150.0, 152.0]})
    with pytest.raises(KeyError):
        MarketDataCollector.prepare_prophet_series(prices, "INVALID_TICKER")
