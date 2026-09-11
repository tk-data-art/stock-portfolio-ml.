FROM python:3.12-slim

# Install system dependencies required for Stan and CVXPY C++ compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install isolated Poetry runtime and configure network resilience
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_INSTALLER_MAX_WORKERS=4 \
    POETRY_REQUESTS_TIMEOUT=120

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

# Cache dependency specifications
COPY pyproject.toml poetry.lock README.md ./

# 1. Install dependencies only (skips project root so layer can be cached)
RUN poetry install --without dev --no-root --no-interaction --no-ansi

# 2. Copy application source code
COPY src/ ./src/

# 3. Install the project package itself
RUN poetry install --without dev --no-interaction --no-ansi

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/stock_portfolio/app/dashboard.py"]
