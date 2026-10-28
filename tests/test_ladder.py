import numpy as np
import pandas as pd

from scalper.ladder import simulate_ladder
from scalper.swing import Bars


def _bars(rows):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="5min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)
    return Bars("BTC", df, np.ones(len(rows)), np.zeros(len(rows)))


FLAT = [100, 100.1, 99.9, 100]
SIG = np.array([1] + [0] * 5)


def test_full_ladder_hits_all_three_targets():
    # step 1%: targets 101, 102, 103; stop 99.
    t = simulate_ladder(_bars([FLAT, FLAT, [100, 101.2, 99.8, 101], [101, 102.1, 100.5, 102],
                               [102, 103.5, 101.5, 103], FLAT]), SIG, 0.01, False, 0, 10)
    assert t.exit[0] == "t3" and t.targets_hit[0] == 3
    assert np.isclose(t.gross_bps[0], 0.5 * 100 + 0.25 * 200 + 0.25 * 300)


def test_plain_stop_after_first_target_loses_second_half():
    rows = [FLAT, FLAT, [100, 101.2, 99.8, 101], [101, 101.1, 98.5, 98.8], FLAT, FLAT]
    plain = simulate_ladder(_bars(rows), SIG, 0.01, False, 0, 10)
    assert plain.exit[0] == "stop_after_t1" and np.isclose(plain.gross_bps[0], 0.5 * 100 - 0.5 * 100)
    be = simulate_ladder(_bars(rows), SIG, 0.01, True, 0, 10)
    assert be.exit[0] == "stop_after_t1" and np.isclose(be.gross_bps[0], 0.5 * 100)


def test_stop_and_target_in_same_bar_counts_as_stop():
    t = simulate_ladder(_bars([FLAT, FLAT, [100, 101.5, 98.5, 100], FLAT]), SIG[:4], 0.01, True, 12, 10)
    assert t.exit[0] == "stop" and np.isclose(t.net_bps[0], -100 - 12)


def test_short_and_time_exit():
    sig = np.array([-1, 0, 0, 0])
    t = simulate_ladder(_bars([FLAT, FLAT, [100, 100.2, 98.8, 99], [99, 99.5, 98.9, 99.5]]), sig, 0.01, False, 0, 2)
    # Short from 100 for at most 2 bars: first target 99 hit, the rest closed at bar 2's close (99).
    assert t.exit[0] == "time" and np.isclose(t.gross_bps[0], 0.5 * 100 + 0.5 * 100)
