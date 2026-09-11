"""Markowitz Modern Portfolio Theory (MPT) optimization using CVXPY."""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    """Encapsulates optimized portfolio metrics."""

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


class MarkowitzOptimizer:
    """Convex quadratic programming optimizer for asset portfolios."""

    def __init__(self, risk_free_rate: float = 0.045) -> None:
        self.rf = risk_free_rate

    def optimize_max_sharpe(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
    ) -> PortfolioResult:
        """
        Solve for the maximum Sharpe ratio portfolio using Charnes-Cooper transformation.

        Enforces:
        - Long-only constraint (w_i >= 0)
        - Full capital deployment (sum(w_i) == 1)
        """
        tickers = list(expected_returns.index)
        n = len(tickers)
        mu = expected_returns.to_numpy()
        sigma = cov_matrix.to_numpy()

        # Slight regularization to guarantee positive semi-definiteness
        sigma = sigma + np.eye(n) * 1e-7

        excess_returns = mu - self.rf

        # Fallback to minimum variance if all returns are below risk-free rate
        if np.all(excess_returns <= 0):
            return self.optimize_min_volatility(expected_returns, cov_matrix)

        # Charnes-Cooper quadratic transformation
        y = cp.Variable(n, nonneg=True)
        objective = cp.Minimize(0.5 * cp.quad_form(y, sigma))
        constraints = [excess_returns @ y == 1]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6)

        if problem.status not in ["optimal", "optimal_inaccurate"] or y.value is None:
            return self.optimize_min_volatility(expected_returns, cov_matrix)

        y_val = np.maximum(y.value, 0.0)
        total_y = np.sum(y_val)
        if total_y == 0:
            return self.optimize_min_volatility(expected_returns, cov_matrix)

        w = y_val / total_y

        ret = float(w @ mu)
        vol = float(np.sqrt(w @ sigma @ w))
        sharpe = (ret - self.rf) / vol if vol > 0 else 0.0

        weights_dict = {
            ticker: round(float(weight), 4)
            for ticker, weight in zip(tickers, w, strict=False)
        }
        return PortfolioResult(
            weights=weights_dict,
            expected_return=round(ret, 4),
            volatility=round(vol, 4),
            sharpe_ratio=round(sharpe, 4),
        )

    def optimize_min_volatility(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
    ) -> PortfolioResult:
        """Solve for the Global Minimum Variance (GMV) portfolio."""
        tickers = list(expected_returns.index)
        n = len(tickers)
        mu = expected_returns.to_numpy()
        sigma = cov_matrix.to_numpy() + np.eye(n) * 1e-7

        w = cp.Variable(n, nonneg=True)
        objective = cp.Minimize(0.5 * cp.quad_form(w, sigma))
        constraints = [cp.sum(w) == 1]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        w_val = np.maximum(w.value, 0.0) if w.value is not None else np.ones(n) / n
        w_val = w_val / np.sum(w_val)

        ret = float(w_val @ mu)
        vol = float(np.sqrt(w_val @ sigma @ w_val))
        sharpe = (ret - self.rf) / vol if vol > 0 else 0.0

        weights_dict = {
            ticker: round(float(weight), 4)
            for ticker, weight in zip(tickers, w_val, strict=False)
        }
        return PortfolioResult(
            weights=weights_dict,
            expected_return=round(ret, 4),
            volatility=round(vol, 4),
            sharpe_ratio=round(sharpe, 4),
        )

    def compute_efficient_frontier(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        num_points: int = 25,
    ) -> pd.DataFrame:
        """Calculate the Efficient Frontier risk-return curve."""
        tickers = list(expected_returns.index)
        n = len(tickers)
        mu = expected_returns.to_numpy()
        sigma = cov_matrix.to_numpy() + np.eye(n) * 1e-7

        frontier_data: list[dict[str, float]] = []
        gammas = np.logspace(-2, 2, num_points)

        for gamma in gammas:
            w = cp.Variable(n, nonneg=True)
            objective = cp.Maximize(mu @ w - (gamma / 2.0) * cp.quad_form(w, sigma))
            constraints = [cp.sum(w) == 1]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.OSQP)

            if w.value is not None:
                w_val = np.maximum(w.value, 0.0)
                w_val = w_val / np.sum(w_val)
                r = float(w_val @ mu)
                v = float(np.sqrt(w_val @ sigma @ w_val))
                s = (r - self.rf) / v if v > 0 else 0.0
                frontier_data.append(
                    {"volatility": v, "expected_return": r, "sharpe": s}
                )

        return pd.DataFrame(frontier_data).drop_duplicates().sort_values("volatility")
