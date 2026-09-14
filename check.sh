#!/usr/bin/env bash
set -e

echo "==> 1. Running Ruff (lint & formatting)..."
poetry run ruff check --fix src/ tests/
poetry run ruff format src/ tests/

echo "==> 2. Running Mypy (strict typing)..."
poetry run mypy src/

echo "==> 3. Running Pytest (unit tests & constraints)..."
poetry run pytest tests/

echo " All checks passed cleanly! Safe to push."
