import numpy as np
import pandas as pd

from scalper.ict import fvg_retrace, market_structure_shift, sweep_reversal, window_mask


def _df(rows, start="2026-01-05 13:00", freq="5min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_bullish_fvg_signals_on_retrace_only_once():
    df = _df([
        [100, 101, 99, 100.5],   # bar 0: high 101
        [100.5, 104, 100.5, 103.5],
        [103.5, 105, 102, 104.5],  # bar 2: low 102 > high[0] 101 -> gap 101..102
        [104.5, 106, 103, 105],
        [105, 105.5, 101.5, 102.5],  # trades into the gap, closes above 101 -> long
        [102.5, 103, 101.2, 102],    # gap already used
    ])
    sig = fvg_retrace(df, np.ones(len(df)), max_age=5, min_size_atr=0.0)
    assert sig.tolist() == [0, 0, 0, 0, 1, 0]


def test_fvg_invalidated_by_close_through_gap():
    df = _df([
        [100, 101, 99, 100.5], [100.5, 104, 100.5, 103.5], [103.5, 105, 102, 104.5],
        [104.5, 104.6, 100, 100.2],  # closes below the gap bottom (101): no trade
    ])
    assert fvg_retrace(df, np.ones(len(df)), 5, 0.0).tolist() == [0, 0, 0, 0]


def test_sweep_of_recent_high_closing_back_inside_is_short():
    rows = [[100, 101, 99, 100]] * 4 + [[100, 101.5, 99.5, 100.5]]
    assert sweep_reversal(_df(rows), "n4").tolist()[-1] == -1
    rows[-1] = [100, 101.5, 99.5, 101.2]  # closed above the old high: a breakout, not a sweep
    assert sweep_reversal(_df(rows), "n4").tolist()[-1] == 0


def test_killzone_uses_new_york_time():
    # 2026-01-05 is winter: New York = UTC-5. A 5-min bar opening 13:25 UTC closes 08:30 NY.
    df = _df([[1, 1, 1, 1]] * 3, start="2026-01-05 13:25")
    assert window_mask(df, ((8, 30, 11, 0),)).tolist() == [False, True, True]


def test_mss_long_after_low_sweep_and_break_of_swing_high():
    rows = [[100, 101, 99, 100]] * 10
    rows += [[100, 102, 99.5, 101], [101, 101.5, 100, 100.5], [100.5, 100.8, 99.8, 100],  # swing high 102
             [100, 100.2, 98.5, 99.5],                                                       # sweeps lows, closes back in
             [99.5, 101, 99.4, 100.8], [100.8, 102.6, 100.6, 102.4]]                         # closes above 102
    sig = market_structure_shift(_df(rows), sweep_window=3)
    assert sig[-1] == 1 and (sig[:-1] >= 0).all()
