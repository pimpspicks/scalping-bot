"""Walk-forward tuning of the signals that came out break-even or better, judged only out of sample.

Each signal family gets a grid of settings. Every setting is simulated on every coin with the 20x
limit-order ladder. Then, at the start of each quarter, the setting with the best average ROI per trade
on all earlier trades (all coins pooled, at least `min_trades`) is chosen, and only its trades in that
quarter count. The textbook setting is scored on the same quarters for comparison.
"""
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .indicators import _atr
from .ladder import simulate_ladder_limit
from .maker_test import to_15m
from .rules import _first
from .rules3 import hurst
from .swing import make_bars

log = logging.getLogger(__name__)


def _cross_up(x):
    return (x > 0) & (x.shift() <= 0)


def _cross_dn(x):
    return (x < 0) & (x.shift() >= 0)


def _sig(long, short):
    long, short = np.asarray(long, bool), np.asarray(short, bool)
    return np.where(long & ~short, 1, np.where(short & ~long, -1, 0))


def ema_cross(df, fast, slow):
    d = df.close.ewm(span=fast, adjust=False).mean() - df.close.ewm(span=slow, adjust=False).mean()
    return _sig(_cross_up(d), _cross_dn(d))


def donchian_hurst(df, n, h_min):
    c = df.close
    hi, lo = df.high.rolling(n).max().shift(), df.low.rolling(n).min().shift()
    ok = hurst(c) > h_min if h_min else pd.Series(True, index=df.index)
    return _first(_sig((c > hi) & ok, (c < lo) & ok))


def keltner(df, mult):
    c = df.close
    pos = (c - c.ewm(span=20, adjust=False).mean()) / (mult * _atr(df))
    return _sig(_cross_up(pos - 1), _cross_dn(pos + 1))


def friday_break(df, until_hour):
    """First close outside the Friday 00:00-08:00 UTC range, between 08:00 and `until_hour` UTC."""
    ct = (df.index + (df.index[1] - df.index[0])).tz_convert("UTC")
    day = ct.floor("D")
    mins = ct.hour * 60 + ct.minute
    fri = np.asarray(ct.dayofweek == 4)
    in_range = fri & np.asarray((mins > 0) & (mins <= 8 * 60))
    after = fri & np.asarray((mins > 8 * 60) & (mins <= until_hour * 60))
    rh = pd.Series(np.where(in_range, df.high, np.nan), index=df.index).groupby(day).cummax().groupby(day).ffill()
    rl = pd.Series(np.where(in_range, df.low, np.nan), index=df.index).groupby(day).cummin().groupby(day).ffill()
    brk = pd.Series(np.where(after & (df.close > rh), 1, np.where(after & (df.close < rl), -1, 0)), index=df.index)
    first = brk.ne(0) & (brk.ne(0).astype(int).groupby(day).cumsum() == 1)
    return np.where(first, brk, 0)


# family -> {setting label: (chart minutes, function, kwargs)}; TEXTBOOK = the setting used so far
GRID = {
    "EMA cross": {f"{tf}m EMA {f}/{s}": (tf, ema_cross, {"fast": f, "slow": s})
                  for tf in (60, 240) for f, s in ((9, 21), (12, 26), (21, 50), (20, 100), (50, 200))},
    "Donchian + Hurst": {f"{tf}m Donchian {n}, Hurst>{h or 'off'}": (tf, donchian_hurst, {"n": n, "h_min": h})
                         for tf in (15, 60) for n in (10, 20, 34, 55) for h in (0, 0.5, 0.55, 0.6)},
    "Keltner breakout": {f"{tf}m Keltner {m} ATR": (tf, keltner, {"mult": m})
                         for tf in (60, 240) for m in (1.5, 2.0, 2.5, 3.0)},
    "Friday expiry break": {f"{tf}m Friday break until {u}:00": (tf, friday_break, {"until_hour": u})
                            for tf in (15, 60) for u in (12, 16, 20)},
}
TEXTBOOK = {"EMA cross": "240m EMA 21/50", "Donchian + Hurst": "15m Donchian 20, Hurst>0.55",
            "Keltner breakout": "240m Keltner 2.0 ATR", "Friday expiry break": "60m Friday break until 16:00"}


def all_trades(symbols_roots: dict[str, Path], lev=20, stop_margin=0.25, maker_bps=1, taker_bps=6, max_hold_h=72,
               wait_bars=4) -> pd.DataFrame:
    out = []
    for s, root in symbols_roots.items():
        b15 = make_bars(s, 15, root)
        bars = {15: b15, 60: make_bars(s, 60, root), 240: make_bars(s, 240, root)}
        for fam, settings in GRID.items():
            for label, (tf, fn, kw) in settings.items():
                b = bars[tf]
                sg = to_15m(fn(b.df, **kw), b.df.index, tf, b15.df.index)
                t = simulate_ladder_limit(b15, sg, stop_margin / lev, maker_bps, taker_bps, int(max_hold_h * 4), wait_bars)
                out.append(t.assign(symbol=s, family=fam, setting=label, roi_pct=t.net_bps / 100 * lev))
        log.info("%s done", s)
    t = pd.concat(out, ignore_index=True)
    return t[t.exit != "open"]


def walk_forward(t: pd.DataFrame, first_quarter: str, min_trades: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(out-of-sample trades of the tuned and textbook versions, the setting chosen each quarter)."""
    q = t.entry_time.dt.tz_localize(None).dt.to_period("Q")
    t = t.assign(quarter=q)
    picks, oos = [], []
    for fam, ft in t.groupby("family"):
        for quarter in sorted(ft.quarter.unique()):
            if quarter < pd.Period(first_quarter, "Q"):
                continue
            past = ft[ft.quarter < quarter]
            score = past.groupby("setting").roi_pct.agg(["mean", "count"])
            score = score[score["count"] >= min_trades]
            if score.empty:
                continue
            best = score["mean"].idxmax()
            picks.append({"family": fam, "quarter": str(quarter), "chosen": best,
                          "past_avg_roi": score.loc[best, "mean"]})
            now = ft[ft.quarter == quarter]
            oos.append(now[now.setting == best].assign(version="tuned"))
            oos.append(now[now.setting == TEXTBOOK[fam]].assign(version="textbook"))
    return pd.concat(oos, ignore_index=True), pd.DataFrame(picks)


def summarize(oos: pd.DataFrame, n_symbols: int) -> pd.DataFrame:
    rows = []
    for (fam, version), x in oos.groupby(["family", "version"]):
        rows.append({"family": fam, "version": version, "trades": len(x), "avg_roi_pct": x.roi_pct.mean(),
                     "t_stat": x.roi_pct.mean() / (x.roi_pct.std() / np.sqrt(len(x))),
                     "coins_positive": f"{int((x.groupby('symbol').roi_pct.mean() > 0).sum())} of {n_symbols}",
                     "quarters_positive": f"{int((x.groupby('quarter').roi_pct.mean() > 0).sum())} of {x.quarter.nunique()}"})
    return pd.DataFrame(rows)
