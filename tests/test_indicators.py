import numpy as np
import pandas as pd

from scalper.indicators import base_tf, higher_tf


def _minutes(n=60 * 24 * 6, seed=0):
    idx = pd.date_range("2026-01-05", periods=n, freq="1min", tz="UTC")
    c = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 1e-3, n)))
    return pd.DataFrame({"open": c, "high": c * 1.0005, "low": c * 0.9995, "close": c, "volume": 1.0}, index=idx)


def _bars(m, tf="15min"):
    return m.resample(tf).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})


def test_higher_timeframe_only_uses_finished_candles():
    m = _minutes()
    b = _bars(m)
    f = higher_tf(m, b.index, pd.Timedelta("15min"), "1h")
    hourly = _bars(m, "1h")
    # The 15-min bar opening 10:00 closes 10:15; the newest finished hourly candle is 09:00-10:00.
    t = pd.Timestamp("2026-01-08 10:00", tz="UTC")
    assert np.isclose(f.loc[t, "1h__roc_1"], hourly.close.loc[t - pd.Timedelta("1h")] / hourly.close.loc[t - pd.Timedelta("2h")] - 1)
    # The 15-min bar opening 10:45 closes 11:00, when the 10:00 hourly candle has just finished.
    t2 = pd.Timestamp("2026-01-08 10:45", tz="UTC")
    assert np.isclose(f.loc[t2, "1h__roc_1"], hourly.close.loc[t] / hourly.close.loc[t - pd.Timedelta("1h")] - 1)


def test_base_features_do_not_change_when_future_bars_change():
    m = _minutes()
    b = _bars(m)
    full = base_tf(b, np.zeros(len(b)))
    cut = 400
    altered = b.copy()
    altered.iloc[cut:, :4] *= 1.5  # a huge future move
    f2 = base_tf(altered, np.zeros(len(b)))
    pd.testing.assert_frame_equal(full.iloc[:cut], f2.iloc[:cut])
