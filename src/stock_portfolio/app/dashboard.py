from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_portfolio.data.collector import MarketDataCollector
from stock_portfolio.forecasting.prophet_engine import ProphetEngine
from stock_portfolio.optimization.markowitz import MarkowitzOptimizer

st.set_page_config(page_title="Stock Forecast & Portfolio Optimizer", layout="wide")

st.title("Stock Price Forecasting & Portfolio Optimization")
st.caption("Bayesian Time-Series (Prophet) & Convex Markowitz Portfolio Engine")

BENCHMARK_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "JPM",
    "V",
    "SPY",
]

# --- Sidebar Configuration ---
st.sidebar.header("Asset Selection & Parameters")

selected_tickers = st.sidebar.multiselect(
    "Select Assets",
    options=BENCHMARK_UNIVERSE,
    default=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    help="Select between 2 and 10 assets.",
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)", min_value=30, max_value=180, value=90, step=30
)
rf_rate_input = st.sidebar.number_input(
    "Annual Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.5, step=0.25
)
risk_free_rate = float(rf_rate_input) / 100.0

run_button = st.sidebar.button("Run Forecast & Optimization", type="primary")

# --- Execution Flow ---
if run_button:
    if len(selected_tickers) < 2:
        st.error("Please select at least 2 assets to optimize portfolio weights.")
        st.stop()

    collector = MarketDataCollector()
    forecaster = ProphetEngine()
    optimizer = MarkowitzOptimizer(risk_free_rate=risk_free_rate)

    start_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")

    # 1. Fetch Market Data
    with st.spinner("Retrieving historical price series..."):
        price_df = collector.fetch_historical_prices(
            selected_tickers, start_date=start_date, end_date=end_date
        )

    if price_df.empty or len(price_df.columns) < len(selected_tickers):
        st.error("Failed to retrieve price data for all selected tickers. Try again.")
        st.stop()

    # 2. Compute Bayesian Forecasts
    forecast_results: dict[str, pd.DataFrame] = {}
    forecast_rows: list[dict[str, Any]] = []
    annualized_returns: dict[str, float] = {}

    progress_bar = st.progress(0)
    for idx, ticker in enumerate(selected_tickers):
        with st.spinner(f"Fitting Prophet model for {ticker}..."):
            history = price_df[ticker]
            f_df = forecaster.fit_predict(history, horizon_days=forecast_days)
            forecast_results[ticker] = f_df

            current_price = float(history.iloc[-1])
            target_price = float(f_df["yhat"].iloc[-1])
            lower_bound = float(f_df["yhat_lower"].iloc[-1])
            upper_bound = float(f_df["yhat_upper"].iloc[-1])
            annualized_return = ((target_price - current_price) / current_price) * (
                365.0 / forecast_days
            )

            annualized_returns[ticker] = annualized_return
            forecast_rows.append(
                {
                    "Ticker": ticker,
                    "Current Price": f"${current_price:.2f}",
                    f"Target ({forecast_days}d)": f"${target_price:.2f}",
                    "Lower Bound (80%)": f"${lower_bound:.2f}",
                    "Upper Bound (80%)": f"${upper_bound:.2f}",
                    "Expected Return (Ann.)": f"{annualized_return * 100:.2f}%",
                }
            )
        progress_bar.progress((idx + 1) / len(selected_tickers))

    # --- UI Layout: Tab Navigation ---
    tab_forecasts, tab_portfolio = st.tabs(
        ["Stock Price Forecasts", "Portfolio Optimization"]
    )

    with tab_forecasts:
        st.subheader("Asset Price Projections")
        summary_table = pd.DataFrame(forecast_rows)
        st.dataframe(summary_table, use_container_width=True)

        st.subheader("Price Forecast Trajectory")
        active_ticker = st.selectbox("Select Asset to Plot", options=selected_tickers)

        if active_ticker:
            hist_series = price_df[active_ticker]
            f_df = forecast_results[active_ticker]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist_series.index,
                    y=hist_series.values,
                    mode="lines",
                    name="Historical Price",
                    line=dict(color="#1f77b4", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=f_df["ds"],
                    y=f_df["yhat"],
                    mode="lines",
                    name=f"Prophet Forecast ({forecast_days}d)",
                    line=dict(color="#2ca02c", width=2, dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(f_df["ds"]) + list(f_df["ds"])[::-1],
                    y=list(f_df["yhat_upper"]) + list(f_df["yhat_lower"])[::-1],
                    fill="toself",
                    fillcolor="rgba(44, 160, 44, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="80% Confidence Interval",
                )
            )
            fig.update_layout(
                title=f"{active_ticker} Historical Close & {forecast_days}-Day Forecast Cone",
                xaxis_title="Date",
                yaxis_title="Stock Price ($)",
                template="plotly_white",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_portfolio:
        st.subheader("Markowitz Tangency Allocation")

        daily_returns = price_df.pct_change().dropna()
        cov_matrix = daily_returns.cov() * 252.0
        mu_series = pd.Series(annualized_returns)

        # Asset-level annual standard deviations
        asset_vols = daily_returns.std() * np.sqrt(252.0)

        # Solve for Maximum Sharpe
        result = optimizer.optimize_max_sharpe(mu_series, cov_matrix)

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Portfolio Return", f"{result.expected_return * 100:.2f}%")
        col2.metric("Annualized Volatility (Risk)", f"{result.volatility * 100:.2f}%")
        col3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")

        # --- Section A: Efficient Frontier & Asset Risk-Return Map ---
        st.subheader("Risk-Return Spectrum & Efficient Frontier")

        fig_ef = go.Figure()

        # 1. Individual Assets Scatter
        fig_ef.add_trace(
            go.Scatter(
                x=[float(v) * 100.0 for v in asset_vols],
                y=[float(m) * 100.0 for m in mu_series],
                mode="markers+text",
                text=selected_tickers,
                textposition="top center",
                marker=dict(size=12, color="#4a7c59", symbol="circle"),
                name="Individual Assets",
            )
        )

        # 2. Tangency Portfolio Point
        fig_ef.add_trace(
            go.Scatter(
                x=[result.volatility * 100.0],
                y=[result.expected_return * 100.0],
                mode="markers+text",
                text=["Optimal Tangency Portfolio"],
                textposition="bottom right",
                marker=dict(size=18, color="#d9534f", symbol="star"),
                name="Max Sharpe Tangency",
            )
        )

        # 3. Capital Allocation Line (from Risk-Free Rate to Tangency)
        cal_x = [0.0, result.volatility * 150.0]
        cal_y = [
            risk_free_rate * 100.0,
            risk_free_rate * 100.0 + result.sharpe_ratio * (result.volatility * 150.0),
        ]
        fig_ef.add_trace(
            go.Scatter(
                x=cal_x,
                y=cal_y,
                mode="lines",
                line=dict(color="#d9534f", dash="dot", width=2),
                name="Capital Allocation Line (CAL)",
            )
        )

        fig_ef.update_layout(
            title="Asset Risk vs. Expected Return & Capital Allocation Line",
            xaxis_title="Annualized Volatility / Risk (%)",
            yaxis_title="Expected Annual Return (%)",
            template="plotly_white",
            hovermode="closest",
        )
        st.plotly_chart(fig_ef, use_container_width=True)

        # --- Section B: Allocation Breakdown Donut Chart ---
        st.subheader("Asset Allocation Breakdown")

        # Filter for holdings > 0.5%
        active_weights = {k: v * 100.0 for k, v in result.weights.items() if v > 0.005}

        col_donut, col_table = st.columns([1, 1])

        with col_donut:
            fig_donut = go.Figure(
                go.Pie(
                    labels=list(active_weights.keys()),
                    values=list(active_weights.values()),
                    hole=0.45,
                    textinfo="label+percent",
                    marker=dict(
                        colors=["#2b5c8f", "#4682b4", "#5c9ebb", "#82b7d4", "#a8cfe8"]
                    ),
                )
            )
            fig_donut.update_layout(
                title="Capital Deployment", template="plotly_white", showlegend=True
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_table:
            weight_df = pd.DataFrame(
                {
                    "Ticker": list(result.weights.keys()),
                    "Target Weight": [
                        f"{w * 100.0:.2f}%" for w in result.weights.values()
                    ],
                    "Annual Volatility": [
                        f"{asset_vols[t] * 100.0:.2f}%" for t in result.weights.keys()
                    ],
                    "Expected Return": [
                        f"{mu_series[t] * 100.0:.2f}%" for t in result.weights.keys()
                    ],
                }
            )
            st.dataframe(weight_df, use_container_width=True, hide_index=True)
