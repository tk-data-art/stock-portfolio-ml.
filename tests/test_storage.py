import pandas as pd
from stock_portfolio.data.storage import MarketDataStorage


def test_storage_save_and_load_roundtrip() -> None:
    """Verify storing and querying prices from DuckDB reproduces the DataFrame."""
    storage = MarketDataStorage(db_path=":memory:")
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    df_in = pd.DataFrame(
        {"AAPL": [150.0, 152.0, 155.0], "MSFT": [300.0, 305.0, 310.0]}, index=dates
    )

    storage.save_prices(df_in)
    df_out = storage.load_prices(["AAPL", "MSFT"])

    assert not df_out.empty
    assert list(df_out.columns) == ["AAPL", "MSFT"]
    assert len(df_out) == 3
    assert df_out.loc["2026-01-01", "AAPL"] == 150.0
    storage.close()


def test_storage_idempotent_upsert() -> None:
    """Ensure duplicate dates update values without duplicating rows."""
    storage = MarketDataStorage(db_path=":memory:")
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    df_initial = pd.DataFrame({"AAPL": [150.0, 152.0]}, index=dates)
    storage.save_prices(df_initial)

    # Overwrite second day with an updated price
    df_update = pd.DataFrame({"AAPL": [150.0, 159.0]}, index=dates)
    storage.save_prices(df_update)

    df_out = storage.load_prices(["AAPL"])
    assert len(df_out) == 2
    assert df_out.loc["2026-01-02", "AAPL"] == 159.0
    storage.close()
