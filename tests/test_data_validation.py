import numpy as np
import pandas as pd
import pytest

from pairs_research.analyzer import prepare_price_frame, safe_log_price_frame
from pairs_research.screeners.crypto import normalize_crypto_ticker
from pairs_research.screeners.ihsg_banks import normalize_ticker
from pairs_research.screeners.sp500 import yahoo_normalize_us_ticker


def test_prepare_price_frame_masks_invalid_prices() -> None:
    index = pd.date_range("2025-01-01", periods=3)
    prices = pd.DataFrame({"AAA": [10.0, 0.0, -1.0]}, index=index)
    clean = prepare_price_frame(prices)
    assert clean["AAA"].notna().tolist() == [True, False, False]


def test_safe_log_price_frame_returns_finite_values() -> None:
    index = pd.date_range("2025-01-01", periods=2)
    prices = pd.DataFrame({"AAA": [1.0, np.e], "BBB": [2.0, 4.0]}, index=index)
    logged = safe_log_price_frame(prices)
    assert np.isfinite(logged.to_numpy()).all()
    assert logged.loc[index[1], "AAA"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("normalizer", "raw", "expected"),
    [
        (normalize_crypto_ticker, "btc", "BTC-USD"),
        (normalize_ticker, "bbca", "BBCA.JK"),
        (yahoo_normalize_us_ticker, "brk.b", "BRK-B"),
    ],
)
def test_ticker_normalization(normalizer, raw: str, expected: str) -> None:
    assert normalizer(raw) == expected
