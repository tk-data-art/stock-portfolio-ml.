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
```

---

## Quality Engineering & CI Gates

* **Dependency Management**: Fully locked, reproducible environments using `Poetry`.
* **Linting & Code Quality**: Enforced via `Ruff`.
* **Static Type Safety**: 100% typed interfaces verified via `Mypy`.
* **Automated Testing**: Complete constraint validation via `pytest`.
* **CI/CD Automation**: Automated multi-stage test pipeline running on `CircleCI`.

---

## Quickstart

### 1. Local Development
```bash
# Install dependencies
poetry install

# Run static quality analysis
poetry run ruff check src/ tests/
poetry run mypy src/

# Run unit tests
poetry run pytest --cov=src/stock_portfolio -v

# Launch the dashboard
poetry run streamlit run src/stock_portfolio/app/dashboard.py
```

### 2. Docker Execution
```bash
docker build -t stock-portfolio-app:latest .
docker run -d -p 8501:8501 --name stock-app stock-portfolio-app:latest
```
Access the application at `http://localhost:8501`.
