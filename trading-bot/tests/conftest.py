import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def make_ohlcv(n: int, start_price: float = 100.0, drift: float = 0.0, vol: float = 1.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, n) / 100.0
    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = rng.uniform(10, 100, n)
    index = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)
