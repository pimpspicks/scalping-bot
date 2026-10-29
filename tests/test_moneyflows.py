import numpy as np
import pandas as pd

from scalper.moneyflows import LAG, _at, flow_signals


def _flows(n=500):
    t = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    up = np.linspace(1, 2, n)
    return pd.DataFrame({"usdt_d": 1 / up, "stables": up, "btc_ex": 1 / up, "eth_ex": 1 / up}, index=t)


def test_rising_money_is_bullish_and_lagged():
    s = flow_signals(_flows())
    assert (s[("USDT.D", 7)] == 1).all() and (s[("BTC exchanges", 14)] == 1).all()
    assert s[("USDT.D", 7)].index[0] == pd.Timestamp("2020-01-08", tz="UTC") + LAG


def test_at_uses_only_known_values():
    s = pd.Series([1, -1], index=pd.to_datetime(["2020-01-02 04:00", "2020-01-03 04:00"], utc=True))
    t = pd.to_datetime(["2020-01-02 03:59", "2020-01-02 04:00", "2020-01-03 12:00"], utc=True)
    assert list(_at(s, t)) == [0, 1, -1]
