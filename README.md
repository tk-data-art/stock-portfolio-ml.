# Real-Time Stock Forecasting & Convex Portfolio Optimization

[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/GDyfJvLbK1Mho7C82AmdaY/UyqqHUoXUqfbc7NnbXrVnk/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/GDyfJvLbK1Mho7C82AmdaY/UyqqHUoXUqfbc7NnbXrVnk/tree/main)
![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)
![Linter](https://img.shields.io/badge/linter-ruff-black.svg)
![Type Checking](https://img.shields.io/badge/types-mypy-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An enterprise-grade quantitative finance platform combining **Bayesian structural time-series forecasting** (Prophet) with **Convex Modern Portfolio Theory** (CVXPY). Built under strict production gates: deterministic dependency management, static type safety, automated linting, containerization, and continuous integration.

---

## System Architecture

```mermaid
flowchart TD
    A[Yahoo Finance Feed] --> B[Data Collector & Cleaning]
    B --> C[Prophet Time-Series Engine]
    B --> D[Covariance Matrix Engine]
    C -->|Expected Returns μ| E[CVXPY Convex Optimizer]
    D -->|Covariance Matrix Σ| E
    E -->|Charnes-Cooper QP| F[Optimal Portfolio Weights]
    F --> G[Streamlit Dashboard]
    F --> H[(SQLite Database)]