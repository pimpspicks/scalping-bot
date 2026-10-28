import numpy as np
import pandas as pd

from scalper.rules3 import hurst


def test_hurst_near_half_for_random_walk_and_high_for_trend():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=3000, freq="15min", tz="UTC")
    walk = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, 3000))), index=idx)
    assert abs(hurst(walk).iloc[-500:].mean() - 0.5) < 0.08
    # persistent returns (AR(1) with positive coefficient) look trending
    r = np.zeros(3000)
    for i in range(1, 3000):
        r[i] = 0.6 * r[i - 1] + rng.normal(0, 0.002)
    trend = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    assert hurst(trend).iloc[-500:].mean() > 0.6
