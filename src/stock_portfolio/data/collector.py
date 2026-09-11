"""Market data acquisition, cleaning, and returns calculation."""

from datetime import datetime, timedelta
from typing import cast

import pandas as pd
import yfinance as yf


class MarketDataCollector:
    """Fetches historical market data and computes empirical statistics."""

    def __init__(self, lookback_years: int = 5) -> None:
        self.lookback_years = lookback_years

    def fetch_historical_prices(self, tickers: list[str]) -> pd.DataFrame:
        if not tickers:
            raise ValueError("Ticker list cannot be empty.")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_years * 365)

        raw_data = yf.download(
            tickers=tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )

        if raw_data.empty:
            raise ValueError(f"No market data retrieved for tickers: {tickers}")

        if "Adj Close" in raw_data.columns:
            prices = raw_data["Adj Close"]
        elif "Close" in raw_data.columns:
            prices = raw_data["Close"]
        else:
            raise KeyError("Adjusted Close / Close column not found in market feed.")

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices.ffill().bfill().dropna(axis=1, how="all")

        missing_tickers = set(tickers) - set(prices.columns)
        if missing_tickers:
            raise ValueError(f"Failed to retrieve data for: {missing_tickers}")

        return cast(pd.DataFrame, prices)

    @staticmethod
    def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
        return cast(pd.DataFrame, prices.pct_change().dropna())

    @staticmethod
    def compute_annualized_covariance(daily_returns: pd.DataFrame) -> pd.DataFrame:
        return cast(pd.DataFrame, daily_returns.cov() * 252)

    @staticmethod
    def prepare_prophet_series(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if ticker not in prices.columns:
            raise KeyError(f"Ticker {ticker} not found in prices DataFrame.")

        df_prophet = prices[[ticker]].reset_index()
        df_prophet.columns = ["ds", "y"]
        return cast(pd.DataFrame, df_prophet)
