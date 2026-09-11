"""Database persistence module with local SQLite fallback."""

import sqlite3
from datetime import date

import pandas as pd


class DatabaseManager:
    """Manages CRUD operations against a local SQLite database or Supabase."""

    def __init__(self, db_path: str = "portfolio.db") -> None:
        self.db_path = db_path
        self._init_sqlite_tables()

    def _init_sqlite_tables(self) -> None:
        """Create tables locally if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 1. Forecasts table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    forecast_date TEXT NOT NULL,
                    yhat REAL NOT NULL,
                    yhat_lower REAL NOT NULL,
                    yhat_upper REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # 2. Allocations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_allocations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT NOT NULL,
                    tickers TEXT NOT NULL,
                    weights TEXT NOT NULL,
                    expected_return REAL NOT NULL,
                    volatility REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def save_forecasts(self, ticker: str, forecast_df: pd.DataFrame) -> None:
        """Persist out-of-sample Prophet predictions to SQLite."""
        records: list[tuple[str, str, float, float, float]] = []
        for _, row in forecast_df.iterrows():
            f_date = (
                row["ds"].strftime("%Y-%m-%d")
                if hasattr(row["ds"], "strftime")
                else str(row["ds"])[:10]
            )
            records.append(
                (
                    ticker,
                    f_date,
                    float(row["yhat"]),
                    float(row["yhat_lower"]),
                    float(row["yhat_upper"]),
                )
            )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO stock_forecasts (ticker, forecast_date, yhat, yhat_lower, yhat_upper)
                VALUES (?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

    def save_allocation(
        self,
        tickers: list[str],
        weights: list[float],
        expected_return: float,
        volatility: float,
        sharpe_ratio: float,
    ) -> None:
        """Persist calculated Markowitz optimal weights to SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO portfolio_allocations 
                (run_date, tickers, weights, expected_return, volatility, sharpe_ratio)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    date.today().isoformat(),
                    ",".join(tickers),
                    ",".join(str(round(w, 4)) for w in weights),
                    round(expected_return, 4),
                    round(volatility, 4),
                    round(sharpe_ratio, 4),
                ),
            )
            conn.commit()
