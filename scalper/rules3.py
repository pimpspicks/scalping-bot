"""Third set: niche ideas collected from trader forums, Quantpedia and papers (see the research note).

Each rule uses the setting stated in its source, or one fixed in advance, with no tuning:
CME weekend gap fill, the 21-23 UTC hour, funding-settlement reversal, options-expiry Friday
breakout, round-number levels, a Hurst-exponent regime switch on standard trend / reversal rules,
realized skewness, and liquidation wicks.

Same conventions as rules.py: decided at a bar's close with that bar and earlier ones only.
Timing rules fire on the bar that closes at the stated time, so they need bars of 1 h or less.
"""
import numpy as np
import pandas as pd

from .indicators import _atr, _rsi
from .rules2 import _down, _first, _per_hour, _sig, _up

ROUND_STEP = {"BTC": 1000.0, "ETH": 100.0, "SOL": 10.0}


def _closes_at(df, tz, weekday, hour) -> np.ndarray:
    """True on bars whose close falls exactly on `hour`:00 local time (weekday None = any day)."""
    close = (df.index + (df.index[1] - df.index[0])).tz_convert(tz)
    ok = (close.hour == hour) & (close.minute == 0)
    if weekday is not None:
        ok &= close.dayofweek == weekday
    return np.asarray(ok)


def hurst(c: pd.Series, window: int = 200, lag: int = 8) -> pd.Series:
    """Variance-ratio Hurst estimate: 0.5 = random walk, above = trending, below = mean reverting."""
    lc = np.log(c)
    v1 = lc.diff().rolling(window).var()
    vk = lc.diff(lag).rolling(window).var()
    return 0.5 * np.log(vk / v1) / np.log(lag)


def niche_rule_signals(symbol: str, all_bars: dict) -> dict[tuple[str, str], np.ndarray]:
    b = all_bars[symbol]
    df = b.df
    o, h, l, c, v = df.open, df.high, df.low, df.close, df.volume
    ph = _per_hour(df)
    atr = _atr(df)
    n = len(df)
    R = {}

    # 1. CME weekend gap: CME bitcoin futures close Fri 16:00 and reopen Sun 17:00 Chicago time. At the reopen,
    # trade back toward the Friday close if price has moved at least 0.5% away from it over the weekend.
    if ph <= 4:
        fri = _closes_at(df, "America/Chicago", 4, 16)
        sun = _closes_at(df, "America/Chicago", 6, 17)
        fri_px = pd.Series(np.where(fri, c, np.nan), index=df.index).ffill()
        gap = c / fri_px - 1
        R[("cme gap", "CME weekend gap fill (0.5%+)")] = _sig(sun & (gap < -0.005), sun & (gap > 0.005))

        # 2. Late-evening hour effect: long at 21:00 UTC every day.
        at21 = _closes_at(df, "UTC", None, 21)
        R[("hour effect", "Long at 21:00 UTC")] = _sig(at21, np.zeros(n, bool))

        # 3. Funding settlement (00/08/16 UTC): fade the move of the hour into the snapshot, just after it.
        settle = np.zeros(n, bool)
        for hr in (0, 8, 16):
            settle |= _closes_at(df, "UTC", None, hr)
        pre = c.pct_change(int(ph))
        R[("funding window", "Fade the hour into funding settlement")] = _sig(settle & (pre < 0), settle & (pre > 0))

        # 4. Options expiry Friday (Deribit 08:00 UTC): after expiry, trade a break of the Friday 00-08 UTC range
        # (first close outside it before 16:00 UTC). Also the last Friday of the month alone (the big expiry).
        ct = (df.index + (df.index[1] - df.index[0])).tz_convert("UTC")  # bar close times
        day = ct.floor("D")
        mins = ct.hour * 60 + ct.minute
        friday = np.asarray(ct.dayofweek == 4)
        in_range = friday & np.asarray((mins > 0) & (mins <= 8 * 60))
        after = friday & np.asarray((mins > 8 * 60) & (mins <= 16 * 60))
        rh = pd.Series(np.where(in_range, h, np.nan), index=df.index).groupby(day).cummax()
        rl = pd.Series(np.where(in_range, l, np.nan), index=df.index).groupby(day).cummin()
        rh, rl = rh.groupby(day).ffill(), rl.groupby(day).ffill()
        brk = pd.Series(np.where(after & (c > rh), 1, np.where(after & (c < rl), -1, 0)), index=df.index)
        first = brk.ne(0) & (brk.ne(0).astype(int).groupby(day).cumsum() == 1)
        expiry = np.where(first, brk, 0)
        R[("options expiry", "Friday post-expiry range break")] = expiry
        last_fri = np.asarray((ct + pd.Timedelta(days=7)).month != ct.month)
        R[("options expiry", "Last-Friday (monthly expiry) range break")] = np.where(last_fri, expiry, 0)

    # 5. Round numbers (BTC 1000s, ETH 100s, SOL 10s): a close across a level.
    step = ROUND_STEP.get(symbol, 10 ** np.floor(np.log10(c.median())))
    lvl_now, lvl_prev = np.floor(c / step), np.floor(c.shift() / step)
    R[("round numbers", "Close across a round number")] = _sig(lvl_now > lvl_prev, lvl_now < lvl_prev)
    # A wick through the level that closes back on the original side = rejection.
    above_lvl = np.ceil(c / step) * step  # nearest level above the close
    below_lvl = np.floor(c / step) * step
    R[("round numbers", "Round-number rejection")] = _sig((l < below_lvl) & (c > below_lvl) & (o > below_lvl),
                                                          (h > above_lvl) & (c < above_lvl) & (o < above_lvl))

    # 6. Hurst regime switch: trend rules only when H > 0.55, reversal rules only when H < 0.45.
    H = hurst(c)
    trending, choppy = H > 0.55, H < 0.45
    e21, e50 = c.ewm(span=21, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    don_hi, don_lo = h.rolling(20).max().shift(), l.rolling(20).min().shift()
    rsi = _rsi(c)
    m, sd = c.rolling(20).mean(), c.rolling(20).std()
    pctb = (c - (m - 2 * sd)) / (4 * sd)
    R[("hurst switch", "EMA 21/50 cross when trending")] = _sig(_up(e21 - e50) & trending, _down(e21 - e50) & trending)
    R[("hurst switch", "Donchian 20 breakout when trending")] = _first(_sig((c > don_hi) & trending, (c < don_lo) & trending))
    R[("hurst switch", "RSI 30/70 reversal when choppy")] = _sig(_up(rsi, 30) & choppy, _down(rsi, 70) & choppy)
    R[("hurst switch", "Bollinger reversal when choppy")] = _sig(_up(pctb, 0) & choppy, _down(pctb, 1) & choppy)

    # 7. Realized skewness of the last day / week of bar returns: fade the top / bottom 20% (of the last 90 days),
    # checked once a day at 00:00 UTC.
    if ph <= 4:
        r = np.log(c).diff()
        midnight = _closes_at(df, "UTC", None, 0)
        for label, win in (("1-day", int(24 * ph)), ("7-day", int(7 * 24 * ph))):
            sk = r.rolling(win).skew()
            daily = sk[midnight]
            hi_q = daily.rolling(90, min_periods=30).quantile(0.8).reindex(df.index)
            lo_q = daily.rolling(90, min_periods=30).quantile(0.2).reindex(df.index)
            R[("realized skew", f"Fade extreme {label} skew")] = _sig(midnight & (sk < lo_q), midnight & (sk > hi_q))

    # 8. Liquidation wicks: a wick of at least 2 ATR making up 60%+ of the bar, on volume 3 std above normal: fade it.
    rng = (h - l).replace(0, np.nan)
    low_wick = np.minimum(o, c) - l
    up_wick = h - np.maximum(o, c)
    lv = np.log1p(v)
    vz = (lv - lv.rolling(96).mean()) / lv.rolling(96).std()
    prev_atr = atr.shift()
    R[("liquidation wick", "Fade liquidation wick")] = _sig((low_wick >= 2 * prev_atr) & (low_wick / rng >= 0.6) & (vz > 3),
                                                            (up_wick >= 2 * prev_atr) & (up_wick / rng >= 0.6) & (vz > 3))
    return {k: np.nan_to_num(np.asarray(s, float)).astype(int) for k, s in R.items()}
