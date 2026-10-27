import numpy as np
import pandas as pd

from scalper.swing import Bars, simulate


def _bars(rows, atr=1.0, funding=None):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="15min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)
    fc = np.zeros(len(rows)) if funding is None else np.asarray(funding, float)
    return Bars("BTC", df, np.full(len(rows), atr), fc)


def test_target_stop_and_time_exits():
    flat = [100, 100.2, 99.8, 100]
    up = [100, 102.5, 99.9, 102]
    b = _bars([flat, flat, up, flat, flat])
    sig = np.array([1, 0, 0, 0, 0])
    t = simulate(b, sig, stop_atr=1, target_atr=2, hold_bars=3, fee_bps=12)
    # Enter at bar 1 open (100), target 102 hit in bar 2 (the stop at 99 is never touched).
    assert t.exit.tolist() == ["target"]
    assert np.isclose(t.gross_bps[0], 200) and np.isclose(t.net_bps[0], 188)

    down = [100, 100.1, 98.5, 99]
    t = simulate(_bars([flat, flat, down, flat]), sig[:4], 1, 2, 3, 0)
    assert t.exit.tolist() == ["stop"] and np.isclose(t.gross_bps[0], -100)

    t = simulate(_bars([flat] * 5), sig, 1, 2, 2, 0)
    assert t.exit.tolist() == ["time"]


def test_both_touched_counts_as_stop_and_gaps_fill_at_open():
    flat = [100, 100.2, 99.8, 100]
    wild = [100, 103, 98, 100]
    t = simulate(_bars([flat, flat, wild]), np.array([1, 0, 0]), 1, 2, 3, 0)
    assert t.exit.tolist() == ["stop"]
    gap = [97, 97.5, 96.5, 97]
    t = simulate(_bars([flat, flat, gap]), np.array([1, 0, 0]), 1, 2, 3, 0)
    assert np.isclose(t.gross_bps[0], -300)


def test_short_pays_negative_funding_and_one_position_at_a_time():
    flat = [100, 100.2, 99.8, 100]
    # Long-pays funding of +0.0001 accrues between entry and exit: a short receives it.
    b = _bars([flat] * 6, funding=[0, 0, 0.0001, 0.0001, 0.0001, 0.0001])
    t = simulate(b, np.array([-1, -1, -1, 0, 0, 0]), 5, 5, 3, 0)
    assert len(t) == 1 and np.isclose(t.funding_bps[0], -1.0)
