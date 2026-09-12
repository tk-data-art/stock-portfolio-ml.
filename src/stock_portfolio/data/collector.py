from typing import cast
import pandas as pd
import yfinance as yf


class MarketDataCollector:
    """Fetches, cleans, and structures historical market price data."""

    def fetch_historical_prices(
        self,
        tickers: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        period: str = "2y",
    ) -> pd.DataFrame:
        """Fetch daily close prices for the provided tickers."""
        if start_date and end_date:
            data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        else:
            data = yf.download(tickers, period=period, auto_adjust=True, progress=False)

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.levels[0]:
                df = data["Close"]
            else:
                df = data
        elif "Close" in data.columns:
            df = data[["Close"]]
        else:
            df = data

        if isinstance(df, pd.Series):
            df = df.to_frame(name=tickers[0])

        cleaned_df = df.dropna(how="all").ffill().bfill()
        return cast(pd.DataFrame, cleaned_df)

    @staticmethod
    def prepare_prophet_series(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Format price series into Prophet schema with 'ds' and 'y' columns."""
        if ticker not in prices.columns:
            raise KeyError(f"Ticker '{ticker}' not found in prices DataFrame.")

        series = prices[ticker].dropna()
        df = series.reset_index()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        if hasattr(df["ds"].dt, "tz") and df["ds"].dt.tz is not None:
            df["ds"] = df["ds"].dt.tz_localize(None)

        return cast(pd.DataFrame, df)
