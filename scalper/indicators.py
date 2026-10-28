"""Technical-analysis feature library, grouped by theory, computed on OHLCV bars.

Every value at a bar uses only that bar and earlier ones. Prices are expressed relative to the
close (percent distances, ratios) so the model sees the same scale on BTC, ETH and SOL.
Higher timeframes (1 h, 4 h) are computed on their own bars and joined at the moment each bar
closes, so a 15-minute row only sees 1 h / 4 h candles that had already finished.
"""
import numpy as np
import pandas as pd

from .ict import KILLZONES, _levels, fvg_retrace, sweep_reversal, window_mask

# Which theory each feature belongs to (used to report importance by family).
FAMILIES = {
    "trend": ("ema_", "macd", "adx", "di_", "ichi_", "supertrend"),
    "momentum": ("rsi", "stoch", "roc_", "willr", "cci"),
    "volatility": ("bb_", "kc_", "atr_pct", "rvol_", "vol_ratio"),
    "volume": ("vwap", "obv", "mfi", "cmf", "vlm_z"),
    "structure": ("pivot_", "fib_", "pdh", "pdl", "pwh", "pwl", "range_pos"),
    "ict": ("fvg_", "sweep_", "asia_", "killzone"),
    "funding": ("funding",),
    "cross_coin": ("x_",),
    "time": ("hour_", "dow_"),
}


def family_of(col: str) -> str:
    name = col.split("__")[-1]
    for fam, prefixes in FAMILIES.items():
        if any(name.startswith(p) for p in prefixes):
            return fam
    return "other"


def _rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False).mean()


def _atr(df, n=14):
    prev = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()], axis=1).max(axis=1)
    return _rma(tr, n)


def _rsi(c, n=14):
    d = c.diff()
    up, dn = _rma(d.clip(lower=0), n), _rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def core(df: pd.DataFrame) -> pd.DataFrame:
    """Indicators that make sense on any timeframe."""
    o, h, l, c, v = df.open, df.high, df.low, df.close, df.volume
    f = pd.DataFrame(index=df.index)
    atr = _atr(df)
    # Trend
    for n in (9, 21, 50, 200):
        e = c.ewm(span=n, adjust=False).mean()
        f[f"ema_{n}_dist"] = c / e - 1
    f["ema_21_slope"] = c.ewm(span=21, adjust=False).mean().pct_change(4)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    pdi, mdi = 100 * _rma(plus, 14) / atr, 100 * _rma(minus, 14) / atr
    f["di_diff"] = pdi - mdi
    f["adx"] = _rma(100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan), 14)
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    f["ichi_tk"] = (tenkan - kijun) / c
    top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    f["ichi_cloud"] = np.where(c > top, c / top - 1, np.where(c < bot, c / bot - 1, 0.0))
    f.loc[top.isna(), "ichi_cloud"] = np.nan
    f["supertrend"] = _supertrend(df, atr)
    # Momentum
    f["rsi_14"] = _rsi(c)
    f["rsi_7"] = _rsi(c, 7)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    f["stoch_k"], f["stoch_d"] = k, k.rolling(3).mean()
    f["willr"] = k - 100
    for n in (1, 4, 16, 64):
        f[f"roc_{n}"] = c.pct_change(n)
    tp = (h + l + c) / 3
    f["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * (tp - tp.rolling(20).mean()).abs().rolling(20).mean())
    # Volatility
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (m - 2 * s)) / (4 * s).replace(0, np.nan)
    f["bb_width"] = 4 * s / m
    kc_mid = c.ewm(span=20, adjust=False).mean()
    f["kc_pos"] = (c - kc_mid) / (2 * atr)
    f["atr_pct"] = atr / c
    r = np.log(c).diff()
    f["rvol_16"] = r.rolling(16).std()
    f["vol_ratio"] = f.rvol_16 / r.rolling(128).std()
    # Volume
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_slope"] = (obv.diff(16) / v.rolling(64).sum().replace(0, np.nan))
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(14).sum()
    f["mfi"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    f["cmf"] = (mfm * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    lv = np.log1p(v)
    f["vlm_z"] = (lv - lv.rolling(96).mean()) / lv.rolling(96).std()
    f["range_pos"] = (c - l.rolling(96).min()) / (h.rolling(96).max() - l.rolling(96).min()).replace(0, np.nan)
    return f


def _supertrend(df, atr, mult=3.0) -> pd.Series:
    hl2 = ((df.high + df.low) / 2).to_numpy()
    a, c = atr.to_numpy(), df.close.to_numpy()
    upper, lower = hl2 + mult * a, hl2 - mult * a
    out = np.zeros(len(c))
    d, fu, fl = 1, upper[0], lower[0]
    for i in range(1, len(c)):
        fu = upper[i] if (upper[i] < fu or c[i - 1] > fu) else fu
        fl = lower[i] if (lower[i] > fl or c[i - 1] < fl) else fl
        if d == 1 and c[i] < fl:
            d = -1
        elif d == -1 and c[i] > fu:
            d = 1
        out[i] = d
    return pd.Series(out, index=df.index)


def base_tf(df: pd.DataFrame, funding_cum: np.ndarray) -> pd.DataFrame:
    """Everything for the trading timeframe: core indicators plus structure, ICT, funding, time."""
    f = core(df)
    c = df.close
    t = df.index.tz_convert("America/New_York")
    # VWAP anchored at the New York day.
    day = pd.Series(t.date, index=df.index)
    tp = (df.high + df.low + c) / 3
    vwap = (tp * df.volume).groupby(day.values).cumsum() / df.volume.groupby(day.values).cumsum().replace(0, np.nan)
    f["vwap_dist"] = c / vwap.to_numpy() - 1
    # Structure: previous day / week levels, classic pivots, Fibonacci position in the prior day's range.
    pdh, pdl = _levels(df, "prev_day")
    f["pdh_dist"], f["pdl_dist"] = c / pdh - 1, c / pdl - 1
    daily = df.groupby(day.values).agg(high=("high", "max"), low=("low", "min"), close=("close", "last")).shift()
    pp = (daily.high + daily.low + daily.close) / 3
    piv = day.map(pp).to_numpy(dtype=float)
    r1 = day.map(2 * pp - daily.low).to_numpy(dtype=float)
    s1 = day.map(2 * pp - daily.high).to_numpy(dtype=float)
    f["pivot_p"], f["pivot_r1"], f["pivot_s1"] = c / piv - 1, c / r1 - 1, c / s1 - 1
    f["fib_pos"] = (c - pdl) / (pdh - pdl)
    week = pd.Series(t.tz_localize(None).to_period("W").astype(str), index=df.index)
    wk = df.groupby(week.values).agg(high=("high", "max"), low=("low", "min")).shift()
    f["pwh_dist"] = c / week.map(wk.high).to_numpy(dtype=float) - 1
    f["pwl_dist"] = c / week.map(wk.low).to_numpy(dtype=float) - 1
    # ICT
    atr = _atr(df).to_numpy()
    fvg = fvg_retrace(df, atr, 16, 0.0)
    f["fvg_bull_recent"] = pd.Series(fvg > 0, index=df.index).rolling(16).sum()
    f["fvg_bear_recent"] = pd.Series(fvg < 0, index=df.index).rolling(16).sum()
    for lvl in ("prev_day", "asia"):
        sw = sweep_reversal(df, lvl)
        f[f"sweep_{lvl}_long"] = pd.Series(sw > 0, index=df.index).rolling(16).sum()
        f[f"sweep_{lvl}_short"] = pd.Series(sw < 0, index=df.index).rolling(16).sum()
    ah, al = _levels(df, "asia")
    f["asia_high_dist"], f["asia_low_dist"] = c / ah - 1, c / al - 1
    f["killzone"] = window_mask(df, KILLZONES).astype(float)
    # Funding: rate paid by longs over the last 8 h and 24 h.
    fc = pd.Series(funding_cum, index=df.index)
    per_h = 3600 / (df.index[1] - df.index[0]).total_seconds()
    f["funding_8h"] = fc.diff(int(8 * per_h))
    f["funding_24h"] = fc.diff(int(24 * per_h))
    # Time
    hour = t.hour + t.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * t.dayofweek / 7), np.cos(2 * np.pi * t.dayofweek / 7)
    return f


def higher_tf(minute_df: pd.DataFrame, base_index: pd.DatetimeIndex, base_step: pd.Timedelta, tf: str) -> pd.DataFrame:
    """Core indicators on a higher timeframe, aligned to the base bars by close time."""
    hdf = minute_df.resample(tf).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                      "volume": "sum"}).dropna()
    f = core(hdf)[["ema_21_dist", "ema_50_dist", "macd_hist", "adx", "di_diff", "supertrend", "rsi_14",
                   "bb_pctb", "roc_1", "roc_4", "range_pos", "ichi_cloud"]]
    f.index = f.index + pd.Timedelta(tf)  # known when the higher-timeframe bar closes
    closes = base_index + base_step
    aligned = f.reindex(f.index.union(closes)).ffill().reindex(closes)
    aligned.index = base_index
    return aligned.add_prefix(f"{tf}__")
