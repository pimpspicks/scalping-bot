"""ICT ("Inner Circle Trader") setups written as strict, testable rules.

ICT is taught as a discretionary style, so each concept below is one mechanical reading of it.
Every signal is decided at a bar's close using only that bar and earlier ones (swing points are
only used once they are confirmed), and the trade enters at the next bar's open.

* Fair value gap (FVG): a 3-bar imbalance. Bullish when bar k's low is above bar k-2's high.
  Entry when price later trades back into the gap and closes without filling it.
* Liquidity sweep: price trades beyond an obvious high/low (recent extreme, previous day's
  high/low, or the Asian session range) and closes back inside it; trade the reversal.
* Market structure shift (MSS): after a sweep of a low, a close above the last confirmed swing
  high (and the mirror image for shorts).
* Killzones, New York time: London 02:00-05:00, NY AM 08:30-11:00, NY PM 13:30-16:00.
  Silver bullet windows: 03:00-04:00, 10:00-11:00, 14:00-15:00.
"""
import numpy as np
import pandas as pd

KILLZONES = ((2, 0, 5, 0), (8, 30, 11, 0), (13, 30, 16, 0))
SILVER_BULLET = ((3, 0, 4, 0), (10, 0, 11, 0), (14, 0, 15, 0))


def _close_times_ny(df: pd.DataFrame) -> pd.DatetimeIndex:
    step = df.index[1] - df.index[0]
    return (df.index + step).tz_convert("America/New_York")


def window_mask(df: pd.DataFrame, windows) -> np.ndarray:
    """True for bars whose close falls inside any (h0, m0, h1, m1) New York time window."""
    t = _close_times_ny(df)
    minutes = t.hour * 60 + t.minute
    mask = np.zeros(len(df), bool)
    for h0, m0, h1, m1 in windows:
        mask |= (minutes > h0 * 60 + m0) & (minutes <= h1 * 60 + m1)
    return mask


def fvg_retrace(df: pd.DataFrame, atr: np.ndarray, max_age: int, min_size_atr: float,
                allowed: np.ndarray | None = None, form_window: np.ndarray | None = None) -> np.ndarray:
    """+1/-1 when price first trades back into a still-open FVG within `max_age` bars of it forming.
    `form_window` limits which bars may create the gap (used for the silver bullet)."""
    h, l, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    n = len(df)
    sig = np.zeros(n, int)
    active = []  # [direction, top, bottom, expires]
    for i in range(2, n):
        still = []
        for d, top, bot, exp in active:
            if i > exp:
                continue
            if d > 0:
                if c[i] < bot:            # closed through the gap: invalidated
                    continue
                if l[i] <= top:           # traded into the gap and held
                    sig[i] = sig[i] or 1
                    continue
            else:
                if c[i] > top:
                    continue
                if h[i] >= bot:
                    sig[i] = sig[i] or -1
                    continue
            still.append([d, top, bot, exp])
        active = still
        size_ok = atr[i] > 0 and np.isfinite(atr[i])
        if form_window is not None and not form_window[i]:
            size_ok = False
        if size_ok and l[i] - h[i - 2] > min_size_atr * atr[i]:
            active.append([1, l[i], h[i - 2], i + max_age])        # bullish gap: top = low[i], bottom = high[i-2]
        if size_ok and l[i - 2] - h[i] > min_size_atr * atr[i]:
            active.append([-1, l[i - 2], h[i], i + max_age])       # bearish gap: top = low[i-2], bottom = high[i]
    if allowed is not None:
        sig = np.where(allowed, sig, 0)
    return sig


def _levels(df: pd.DataFrame, level: str) -> tuple[np.ndarray, np.ndarray]:
    """Liquidity levels (high, low) known before each bar."""
    if level.startswith("n"):
        n = int(level[1:])
        return df.high.rolling(n).max().shift().to_numpy(), df.low.rolling(n).min().shift().to_numpy()
    t = _close_times_ny(df)
    if level == "prev_day":
        day = pd.Series(t.date, index=df.index)
        daily = df.groupby(day.values).agg(high=("high", "max"), low=("low", "min"))
        prev = daily.shift()
        return (day.map(prev.high).to_numpy(dtype=float), day.map(prev.low).to_numpy(dtype=float))
    if level == "asia":
        # Asian range: 20:00-24:00 New York time, used from midnight until the next evening.
        minutes = t.hour * 60 + t.minute
        in_asia = minutes > 20 * 60
        session = pd.Series((t + pd.Timedelta(hours=4)).date, index=df.index)  # 20:00 starts the next session
        rng = df[in_asia].groupby(session[in_asia].values).agg(high=("high", "max"), low=("low", "min"))
        hi = session.map(rng.high).to_numpy(dtype=float)
        lo = session.map(rng.low).to_numpy(dtype=float)
        hi[in_asia], lo[in_asia] = np.nan, np.nan  # range not finished yet
        return hi, lo
    raise ValueError(level)


def sweep_reversal(df: pd.DataFrame, level: str, allowed: np.ndarray | None = None) -> np.ndarray:
    hi, lo = _levels(df, level)
    h, l, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    with np.errstate(invalid="ignore"):
        short = (h > hi) & (c < hi)
        long = (l < lo) & (c > lo)
    sig = np.where(long & ~short, 1, np.where(short & ~long, -1, 0))
    if allowed is not None:
        sig = np.where(allowed, sig, 0)
    return sig


def market_structure_shift(df: pd.DataFrame, sweep_window: int, allowed: np.ndarray | None = None,
                           k: int = 2) -> np.ndarray:
    """Sweep of the last `4 * sweep_window`-bar extreme, then within `sweep_window` bars a close
    beyond the most recent confirmed swing point (fractal with `k` bars each side) the other way."""
    h, l, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    n = len(df)
    sweep = sweep_reversal(df, f"n{4 * sweep_window}")
    sig = np.zeros(n, int)
    last_sh = last_sl = np.nan
    since_low_sweep = since_high_sweep = 10**9
    for i in range(n):
        j = i - k  # swing point at j is confirmed at bar i
        if j >= k:
            if h[j] == h[j - k:i + 1].max():
                last_sh = h[j]
            if l[j] == l[j - k:i + 1].min():
                last_sl = l[j]
        since_low_sweep = 0 if sweep[i] > 0 else since_low_sweep + 1
        since_high_sweep = 0 if sweep[i] < 0 else since_high_sweep + 1
        if since_low_sweep <= sweep_window and c[i] > last_sh and (i == 0 or c[i - 1] <= last_sh):
            sig[i] = 1
        elif since_high_sweep <= sweep_window and c[i] < last_sl and (i == 0 or c[i - 1] >= last_sl):
            sig[i] = -1
    if allowed is not None:
        sig = np.where(allowed, sig, 0)
    return sig


def ict_grid(df: pd.DataFrame, atr: np.ndarray, bar_min: int) -> dict[str, dict[tuple, np.ndarray]]:
    kz = window_mask(df, KILLZONES)
    sb = window_mask(df, SILVER_BULLET)
    per_h = max(1, 60 // bar_min)
    masks = {"any_time": None, "killzone": kz}
    return {
        "ict_fvg": {("age", age, "min_atr", m, w): fvg_retrace(df, atr, age, m, mk)
                    for age in (per_h, 4 * per_h) for m in (0.0, 0.5) for w, mk in masks.items()},
        "ict_sweep": {(lvl, w): sweep_reversal(df, lvl, mk)
                      for lvl in (f"n{per_h}", f"n{4 * per_h}", "prev_day", "asia") for w, mk in masks.items()},
        "ict_mss": {("window", win, w): market_structure_shift(df, win, mk)
                    for win in (max(2, per_h // 2), 2 * per_h) for w, mk in masks.items()},
        "ict_silver_bullet": {("age", age): fvg_retrace(df, atr, age, 0.0, sb, form_window=sb)
                              for age in (max(2, per_h // 2), per_h)},
    }
