from typing import cast

import pandas as pd
import yfinance as yf

from stock_portfolio.data.storage import MarketDataStorage


class MarketDataCollector:
    """Fetches, cleans, and caches historical market price data."""

    def __init__(self, db_path: str = "data/market_data.duckdb") -> None:
        self.storage = MarketDataStorage(db_path=db_path)

    def fetch_historical_prices(
        self,
        tickers: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        period: str = "2y",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch close prices, querying DuckDB cache before falling back to Yahoo Finance."""
        if not force_refresh:
            cached_df = self.storage.load_prices(
                tickers, start_date=start_date, end_date=end_date
            )
            # Use cache if all requested tickers are present and contain sufficient history
            if not cached_df.empty and set(tickers).issubset(set(cached_df.columns)):
                if len(cached_df) > 30:
                    return cached_df[tickers]

        # Fetch from Yahoo Finance if missing or refresh requested
        if start_date and end_date:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )
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

        cleaned_df = cast(pd.DataFrame, df.dropna(how="all").ffill().bfill())

        # Persist cleaned historical bars into DuckDB
        if not cleaned_df.empty:
            self.storage.save_prices(cleaned_df)

        return cleaned_df

    @staticmethod
    def prepare_prophet_series(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Format price series into Prophet schema with 'ds' and 'y' columns."""
        if ticker not in prices.columns:
            raise KeyError(f"Ticker '{ticker}' not found in prices DataFrame.")

        series = prices[ticker].dropna()
        df = series.reset_index().rename(
            columns={"index": "ds", "Date": "ds", "date": "ds"}
        )
        df = df[["ds", ticker]].rename(columns={ticker: "y"})
        df["ds"] = pd.to_datetime(df["ds"])
        if hasattr(df["ds"].dt, "tz") and df["ds"].dt.tz is not None:
            df["ds"] = df["ds"].dt.tz_localize(None)

        return cast(pd.DataFrame, df)
