"""Prophet Bayesian time-series model implementation for stock trajectory forecasting."""

import logging
import os
import sys

import pandas as pd
from prophet import Prophet

# Suppress cmdstanpy and Prophet noisy optimization outputs
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


class ForecastEngine:
    """Trains Prophet models and produces forward-looking returns."""

    def __init__(self, horizon_days: int = 365) -> None:
        self.horizon_days = horizon_days

    def fit_and_forecast(self, df_prophet: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """
        Train Prophet on historical (ds, y) series and generate future projections.

        Returns:
            tuple[pd.DataFrame, float]:
                - Forecast DataFrame containing columns: ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
                - Annualized expected rate of return (mu_hat)
        """
        if len(df_prophet) < 60:
            raise ValueError(
                "Insufficient price history. Minimum 60 trading days required."
            )

        # Configure Prophet:
        # - daily_seasonality=False: stocks trade only on weekdays
        # - weekly_seasonality=True: captures day-of-week market biases
        # - yearly_seasonality=True: captures calendar-year cycles
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95,  # 95% Bayesian credible interval
        )

        # Redirect C-level stdout/stderr to silence Stan optimization lines
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                model.fit(df_prophet)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        # Generate future evaluation dataframe
        future = model.make_future_dataframe(periods=self.horizon_days)
        forecast = model.predict(future)

        # Extract latest known price and terminal forecast price
        current_price = float(df_prophet["y"].iloc[-1])
        terminal_forecast_price = float(forecast["yhat"].iloc[-1])

        # Compute annualized expected return: mu_hat = (P_end - P_start) / P_start * (365 / horizon)
        raw_return = (terminal_forecast_price - current_price) / current_price
        annualized_return = raw_return * (365.0 / self.horizon_days)

        output_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
        return forecast[output_cols], annualized_return
