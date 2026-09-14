import logging

import pandas as pd
from prophet import Prophet

# Suppress Prophet / cmdstanpy verbose logs
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ProphetEngine:
    """Bayesian structural time-series forecasting engine using Meta Prophet."""

    def __init__(self, changepoint_prior_scale: float = 0.05, interval_width: float = 0.80) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        self.interval_width = interval_width

    def fit_predict(self, series: pd.Series, horizon_days: int = 90) -> pd.DataFrame:
        """Fit a Prophet model on a price series and forecast horizon_days ahead.

        Args:
            series: Historical daily close price series indexed by datetime.
            horizon_days: Number of days to forecast into the future.

        Returns:
            DataFrame with columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper'].
        """
        # Format DataFrame according to Prophet specifications
        df_prophet = series.reset_index()
        df_prophet.columns = ["ds", "y"]
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"]).dt.tz_localize(None)

        model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            interval_width=self.interval_width,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=horizon_days)
        forecast = model.predict(future)

        # Return only relevant projection horizon
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_days)
