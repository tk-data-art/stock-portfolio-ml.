"""Production Streamlit Dashboard for Stock Forecasting & Portfolio Optimization."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from stock_portfolio.config import get_settings
from stock_portfolio.data.collector import MarketDataCollector
from stock_portfolio.data.db import DatabaseManager
from stock_portfolio.forecasting.prophet_engine import ForecastEngine
from stock_portfolio.optimization.markowitz import MarkowitzOptimizer

# Page Configuration
st.set_page_config(
    page_title="QuantML | Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

settings = get_settings()
db = DatabaseManager()


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(tickers: tuple[str, ...], lookback_years: int) -> pd.DataFrame:
    """Fetch and cache historical adjusted close prices."""
    collector = MarketDataCollector(lookback_years=lookback_years)
    return collector.fetch_historical_prices(list(tickers))


# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Engine Parameters")

default_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
selected_tickers = st.sidebar.multiselect(
    "Asset Universe",
    options=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "SPY"],
    default=default_tickers[:3],
)

lookback = st.sidebar.slider(
    "Historical Lookback (Years)",
    min_value=1,
    max_value=10,
    value=settings.default_lookback_years,
)

horizon = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=30,
    max_value=730,
    value=settings.forecast_horizon_days,
    step=30,
)

rf_rate = st.sidebar.number_input(
    "Risk-Free Rate (Rf)",
    min_value=0.0,
    max_value=0.15,
    value=settings.risk_free_rate,
    step=0.005,
    format="%.3f",
)

run_button = st.sidebar.button("⚡ Run Forecast & Optimize", use_container_width=True)

# --- MAIN VIEW ---
st.title("📈 Machine Learning Stock Forecasting & MPT Allocation")
st.caption(
    "Bayesian Time-Series Modeling (Prophet) coupled with Convex Modern Portfolio Theory (CVXPY)"
)

if not selected_tickers or len(selected_tickers) < 2:
    st.warning("Please select at least 2 assets to evaluate portfolio diversification.")
    st.stop()

if run_button:
    with st.spinner("1/3 Ingesting historical adjusted close series..."):
        try:
            prices = load_market_data(tuple(selected_tickers), lookback)
            returns = MarketDataCollector.compute_daily_returns(prices)
            cov_matrix = MarketDataCollector.compute_annualized_covariance(returns)
        except Exception as e:
            st.error(f"Data ingestion error: {e}")
            st.stop()

    forecasts: dict[str, pd.DataFrame] = {}
    expected_returns: dict[str, float] = {}

    with st.spinner("2/3 Fitting Prophet models across selected assets..."):
        engine = ForecastEngine(horizon_days=horizon)
        for ticker in selected_tickers:
            df_prophet = MarketDataCollector.prepare_prophet_series(prices, ticker)
            forecast_df, mu = engine.fit_and_forecast(df_prophet)
            forecasts[ticker] = forecast_df
            expected_returns[ticker] = mu
            # Persist forecast to database
            db.save_forecasts(ticker, forecast_df.tail(horizon))

    mu_series = pd.Series(expected_returns)

    with st.spinner("3/3 Solving convex portfolio optimization..."):
        optimizer = MarkowitzOptimizer(risk_free_rate=rf_rate)
        result = optimizer.optimize_max_sharpe(mu_series, cov_matrix)
        frontier_df = optimizer.compute_efficient_frontier(mu_series, cov_matrix)

        # Persist optimal allocation to database
        db.save_allocation(
            tickers=list(result.weights.keys()),
            weights=list(result.weights.values()),
            expected_return=result.expected_return,
            volatility=result.volatility,
            sharpe_ratio=result.sharpe_ratio,
        )

    # --- TOP METRICS CARDS ---
    st.markdown("### Optimal Portfolio Performance (Max Sharpe)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Annual Return", f"{result.expected_return * 100:.2f}%")
    col2.metric("Portfolio Volatility (Risk)", f"{result.volatility * 100:.2f}%")
    col3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")

    st.markdown("---")

    # --- ROW 1: WEIGHT ALLOCATION & ASSET RETURN EXPECTATIONS ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Optimal Asset Weights")
        weights_df = pd.DataFrame(
            list(result.weights.items()), columns=["Ticker", "Weight"]
        )
        fig_donut = px.pie(
            weights_df,
            values="Weight",
            names="Ticker",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig_donut.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.subheader("Forecasted Annual Returns (μ)")
        returns_df = pd.DataFrame(
            list(expected_returns.items()), columns=["Ticker", "Expected Return"]
        )
        returns_df["Color"] = returns_df["Expected Return"].apply(
            lambda x: "Positive" if x >= 0 else "Negative"
        )
        fig_bar = px.bar(
            returns_df,
            x="Ticker",
            y="Expected Return",
            color="Color",
            color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
            text_auto=".2%",
        )
        fig_bar.update_layout(showlegend=False, yaxis_tickformat=".1%")
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- TABS FOR DEEP DIVE ANALYTICS ---
    tab_frontier, tab_forecasts, tab_covariance = st.tabs(
        [
            "📊 Efficient Frontier",
            "📈 Individual Forecast Trajectories",
            "🧮 Covariance Matrix",
        ]
    )

    with tab_frontier:
        st.subheader("Markowitz Efficient Frontier")
        fig_frontier = go.Figure()
        fig_frontier.add_trace(
            go.Scatter(
                x=frontier_df["volatility"],
                y=frontier_df["expected_return"],
                mode="lines",
                name="Efficient Frontier",
                line=dict(color="#3498db", width=3),
            )
        )
        # Highlight optimal portfolio
        fig_frontier.add_trace(
            go.Scatter(
                x=[result.volatility],
                y=[result.expected_return],
                mode="markers+text",
                name="Max Sharpe Portfolio",
                text=["★ Optimal Allocation"],
                textposition="top center",
                marker=dict(color="#e74c3c", size=14, symbol="star"),
            )
        )
        fig_frontier.update_layout(
            xaxis_title="Annualized Volatility (Risk)",
            yaxis_title="Expected Return",
            xaxis_tickformat=".1%",
            yaxis_tickformat=".1%",
            hovermode="x unified",
        )
        st.plotly_chart(fig_frontier, use_container_width=True)

    with tab_forecasts:
        st.subheader("Prophet Forecast Trajectories (Historical vs 1-Yr Projections)")
        target_ticker = st.selectbox("Inspect Asset Curve", options=selected_tickers)
        f_df = forecasts[target_ticker]

        fig_forecast = go.Figure()
        # Historical & Point Forecast
        fig_forecast.add_trace(
            go.Scatter(
                x=f_df["ds"],
                y=f_df["yhat"],
                mode="lines",
                name="Forecast (yhat)",
                line=dict(color="#2980b9", width=2),
            )
        )
        # Upper Bound
        fig_forecast.add_trace(
            go.Scatter(
                x=f_df["ds"],
                y=f_df["yhat_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                name="Upper Bound",
            )
        )
        # Lower Bound with shaded fill
        fig_forecast.add_trace(
            go.Scatter(
                x=f_df["ds"],
                y=f_df["yhat_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(41, 128, 185, 0.2)",
                name="95% Credible Interval",
            )
        )
        fig_forecast.update_layout(
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

    with tab_covariance:
        st.subheader("Annualized Covariance Matrix (Σ)")
        st.dataframe(
            cov_matrix.style.format("{:.4f}").background_gradient(cmap="Blues")
        )

else:
    st.info(
        "👈 Configure your asset universe and risk parameters in the sidebar, then click **Run Forecast & Optimize**."
    )
