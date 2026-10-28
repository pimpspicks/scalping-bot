"""Second set of theories, none of which were in the first round.

Volatility compression, session opening-range breakouts, candlestick patterns, other trend tools
(Parabolic SAR, Aroon, Vortex, TRIX, Chandelier exit, Turtle, time-series momentum), volume
profile, Wyckoff springs/upthrusts, timing effects (funding settlement, weekend, turn of month)
and coin relationships (spread stretches against BTC, BTC leading the alts).

Same conventions as rules.py: decided at a bar's close with that bar and earlier ones only,
firing on the bar a condition starts, standard textbook settings.
"""
import numpy as np
import pandas as pd

from .indicators import _atr, _rma


def _up(x, level=0):
    return (x > level) & (x.shift() <= level)


def _down(x, level=0):
    return (x < level) & (x.shift() >= level)


def _sig(long, short) -> np.ndarray:
    long, short = np.asarray(long, bool), np.asarray(short, bool)
    return np.where(long & ~short, 1, np.where(short & ~long, -1, 0))


def _first(sig: np.ndarray) -> np.ndarray:
    prev = np.r_[0, sig[:-1]]
    return np.where(sig != prev, sig, 0)


def _per_hour(df) -> float:
    return 3600 / (df.index[1] - df.index[0]).total_seconds()


def _psar(df, step=0.02, max_step=0.2) -> np.ndarray:
    h, l = df.high.to_numpy(), df.low.to_numpy()
    n = len(h)
    trend = np.zeros(n, int)
    up, sar, ep, af = True, l[0], h[0], step
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, max_step)
        trend[i] = 1 if up else -1
    return trend


def _value_area(df, bins=40, share=0.7):
    """Previous New York day's volume profile: point of control and value area (70% of volume),
    with volume spread evenly over each bar's high-low range."""
    t = df.index.tz_convert("America/New_York")
    day = pd.Series(t.date, index=df.index)
    out = {}
    for d, g in df.groupby(day.values):
        lo, hi = g.low.min(), g.high.max()
        if not np.isfinite(lo) or hi <= lo:
            continue
        edges = np.linspace(lo, hi, bins + 1)
        mids = (edges[:-1] + edges[1:]) / 2
        vol = np.zeros(bins)
        for bl, bh, v in zip(g.low.to_numpy(), g.high.to_numpy(), g.volume.to_numpy()):
            inside = (mids >= bl) & (mids <= bh)
            k = inside.sum()
            if k:
                vol[inside] += v / k
        poc = int(vol.argmax())
        a, b, tot = poc, poc, vol[poc]
        while tot < share * vol.sum() and (a > 0 or b < bins - 1):
            left = vol[a - 1] if a > 0 else -1
            right = vol[b + 1] if b < bins - 1 else -1
            if right >= left:
                b += 1
                tot += vol[b]
            else:
                a -= 1
                tot += vol[a]
        out[d] = (mids[poc], edges[a], edges[b + 1])
    prof = pd.DataFrame.from_dict(out, orient="index", columns=["poc", "val", "vah"]).shift()  # previous day
    return (day.map(prof.poc).to_numpy(float), day.map(prof.val).to_numpy(float), day.map(prof.vah).to_numpy(float))


def _opening_range(df, start_hm, minutes, until_hm):
    """High/low of the bars opening in [start, start+minutes) New York time; breakout allowed until `until`."""
    t = df.index.tz_convert("America/New_York")
    mins = t.hour * 60 + t.minute
    s0 = start_hm[0] * 60 + start_hm[1]
    in_rng = (mins >= s0) & (mins < s0 + minutes)
    after = (mins >= s0 + minutes) & (mins < until_hm[0] * 60 + until_hm[1])
    day = pd.Series(t.date, index=df.index)
    rng = df[in_rng].groupby(day[in_rng].values).agg(hi=("high", "max"), lo=("low", "min"))
    hi = day.map(rng.hi).to_numpy(float)
    lo = day.map(rng.lo).to_numpy(float)
    c = df.close.to_numpy()
    long = after & (c > hi)
    short = after & (c < lo)
    sig = _sig(long, short)
    # Only the first breakout of each day.
    seen = pd.Series(sig != 0, index=df.index).groupby(day.values).cumsum().to_numpy()
    return np.where((seen == 1) & (sig != 0), sig, 0)


def new_rule_signals(symbol: str, all_bars: dict) -> dict[tuple[str, str], np.ndarray]:
    b = all_bars[symbol]
    df = b.df
    o, h, l, c, v = df.open, df.high, df.low, df.close, df.volume
    ph = _per_hour(df)
    atr = _atr(df)
    rng = h - l
    body = (c - o).abs()
    R = {}

    # Volatility compression
    m, sd = c.rolling(20).mean(), c.rolling(20).std()
    kc_mid = c.ewm(span=20, adjust=False).mean()
    squeeze = ((m + 2 * sd) < (kc_mid + 1.5 * atr)) & ((m - 2 * sd) > (kc_mid - 1.5 * atr))
    sq_run = squeeze.astype(int).groupby((~squeeze).cumsum()).cumsum()
    release = (~squeeze) & (sq_run.shift(fill_value=0) >= 6)
    mom = c - (h.rolling(20).max() + l.rolling(20).min() + 2 * m) / 4
    R[("compression", "TTM squeeze release")] = _sig(release & (mom > 0), release & (mom < 0))
    nr7 = rng == rng.rolling(7).min()
    R[("compression", "NR7 breakout")] = _sig(nr7.shift(1, fill_value=False) & (c > h.shift(1)), nr7.shift(1, fill_value=False) & (c < l.shift(1)))
    inside = (h < h.shift()) & (l > l.shift())
    R[("compression", "Inside bar breakout")] = _sig(inside.shift(1, fill_value=False) & (c > h.shift(2)), inside.shift(1, fill_value=False) & (c < l.shift(2)))

    # Session opening-range breakouts (New York time)
    if ph >= 1:
        R[("session", "New York open range breakout")] = _opening_range(df, (9, 30) if ph > 1 else (9, 0), 30 if ph > 1 else 60, (16, 0))
        R[("session", "London open range breakout")] = _opening_range(df, (3, 0), 60, (11, 0))

    # Candlesticks
    bull_engulf = (c > o) & (c.shift() < o.shift()) & (c >= o.shift()) & (o <= c.shift())
    bear_engulf = (c < o) & (c.shift() > o.shift()) & (c <= o.shift()) & (o >= c.shift())
    R[("candles", "Engulfing")] = _sig(bull_engulf, bear_engulf)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    hammer = (lower_wick >= 2 * body) & (lower_wick >= 0.6 * rng) & (c >= l + 0.66 * rng) & (rng > 0.5 * atr)
    star = (upper_wick >= 2 * body) & (upper_wick >= 0.6 * rng) & (c <= l + 0.34 * rng) & (rng > 0.5 * atr)
    R[("candles", "Pin bar (hammer / shooting star)")] = _sig(hammer, star)
    ha_c = (o + h + l + c) / 4
    ha_o = ha_c.copy()
    ha_o.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    hc, hoa = ha_c.to_numpy(), ha_o.to_numpy(copy=True)
    for i in range(1, len(hoa)):
        hoa[i] = (hoa[i - 1] + hc[i - 1]) / 2
    ha_green = pd.Series(hc > hoa, index=df.index)
    g1, g2, g3 = (ha_green.shift(k, fill_value=False) for k in (1, 2, 3))
    r1, r2, r3 = (~ha_green.shift(k, fill_value=True) for k in (1, 2, 3))
    prev3_red, prev3_green = r1 & r2 & r3, g1 & g2 & g3
    R[("candles", "Heikin Ashi colour flip")] = _sig(ha_green & prev3_red, ~ha_green & prev3_green)
    green, red = c > o, c < o
    soldiers = green & green.shift(1, fill_value=False) & green.shift(2, fill_value=False) & (c > c.shift()) & (c.shift() > c.shift(2))
    crows = red & red.shift(1, fill_value=False) & red.shift(2, fill_value=False) & (c < c.shift()) & (c.shift() < c.shift(2))
    R[("candles", "Three soldiers / crows")] = _first(_sig(soldiers, crows))

    # Other trend tools
    sar = pd.Series(_psar(df), index=df.index)
    R[("trend tools", "Parabolic SAR flip")] = _sig(_up(sar), _down(sar))
    n = 25
    aroon_up = 100 * h.rolling(n + 1).apply(np.argmax, raw=True) / n
    aroon_dn = 100 * l.rolling(n + 1).apply(np.argmin, raw=True) / n
    R[("trend tools", "Aroon cross")] = _sig(_up(aroon_up - aroon_dn), _down(aroon_up - aroon_dn))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    vip = (h - l.shift()).abs().rolling(14).sum() / tr.rolling(14).sum()
    vim = (l - h.shift()).abs().rolling(14).sum() / tr.rolling(14).sum()
    R[("trend tools", "Vortex cross")] = _sig(_up(vip - vim), _down(vip - vim))
    trix = c.ewm(span=15).mean().ewm(span=15).mean().ewm(span=15).mean().pct_change()
    R[("trend tools", "TRIX signal cross")] = _sig(_up(trix - trix.ewm(span=9).mean()), _down(trix - trix.ewm(span=9).mean()))
    a22 = _rma(tr, 22)
    long_stop = h.rolling(22).max() - 3 * a22
    short_stop = l.rolling(22).min() + 3 * a22
    ch = pd.Series(np.nan, index=df.index)
    ch[c > short_stop.shift()] = 1
    ch[c < long_stop.shift()] = -1
    ch = ch.ffill().fillna(0)
    R[("trend tools", "Chandelier exit flip")] = _sig(_up(ch), _down(ch))
    R[("trend tools", "Turtle 55-bar breakout")] = _first(_sig(c > h.rolling(55).max().shift(), c < l.rolling(55).min().shift()))
    week = int(round(24 * 7 * ph))
    R[("trend tools", "Weekly time-series momentum")] = _sig(_up(c.pct_change(week)), _down(c.pct_change(week)))

    # Volume profile (previous New York day)
    poc, val, vah = _value_area(df)
    vals, vahs = pd.Series(val, index=df.index), pd.Series(vah, index=df.index)
    below, above = c < vals, c > vahs
    R[("volume profile", "Value area re-entry (80% rule)")] = _sig(below.shift(1, fill_value=False) & (c > vals),
                                                                   above.shift(1, fill_value=False) & (c < vahs))
    R[("volume profile", "Value area breakout")] = _sig(_up(c - vahs), _down(c - vals))

    # Wyckoff: a quiet range, then a false break with high volume
    look = int(round(48 * ph)) if ph <= 1 else 48
    r_hi, r_lo = h.rolling(look).max().shift(), l.rolling(look).min().shift()
    quiet = (r_hi - r_lo) < 6 * atr.shift()
    vz = (v - v.rolling(96).mean()) / v.rolling(96).std()
    R[("wyckoff", "Spring / upthrust")] = _sig(quiet & (l < r_lo) & (c > r_lo) & (vz > 1),
                                             quiet & (h > r_hi) & (c < r_hi) & (vz > 1))

    # Timing effects (UTC)
    t = df.index
    close_t = t + (t[1] - t[0])
    fund8 = pd.Series(b.funding_cum, index=t).diff(int(round(8 * ph)) or 1)
    pre_funding = np.isin(close_t.hour, (23, 7, 15)) & (close_t.minute == 0)
    R[("timing", "Before funding: trade against funding")] = _sig(pre_funding & (fund8 < 0), pre_funding & (fund8 > 0))
    monday_open = (close_t.dayofweek == 0) & (close_t.hour == 0) & (close_t.minute == 0)
    wk_ret = c.pct_change(int(round(48 * ph)) or 1)
    R[("timing", "Monday open: fade the weekend")] = _sig(monday_open & (wk_ret < 0), monday_open & (wk_ret > 0))
    last_day = (close_t + pd.Timedelta(days=1)).day == 1
    tom = last_day & (close_t.hour == 22) & (close_t.minute == 0)
    R[("timing", "Turn of month long")] = _sig(tom, np.zeros(len(df), bool))

    # Coin relationships (alts against BTC)
    if symbol != "BTC" and "BTC" in all_bars:
        bc = all_bars["BTC"].df.close.reindex(df.index).ffill()
        ratio = np.log(c / bc)
        w = int(round(7 * 24 * ph))
        z = (ratio - ratio.rolling(w).mean()) / ratio.rolling(w).std()
        R[("coin pairs", "Spread vs BTC stretched (fade)")] = _sig(_down(z, -2), _up(z, 2))
        br = np.log(bc).diff()
        ar = np.log(c).diff()
        big = br.abs() > 2 * br.rolling(w).std()
        lag = ar.abs() < 0.5 * br.abs()
        R[("coin pairs", "BTC moves first, alt follows")] = _sig(big & lag & (br > 0), big & lag & (br < 0))
    return {k: np.nan_to_num(np.asarray(v, float)).astype(int) for k, v in R.items()}
