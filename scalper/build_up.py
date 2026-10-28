"""Build a strategy up one piece at a time from a base signal, keeping only what helps.

Base: Ichimoku cloud break on 1-hour candles. Candidates are confirmation filters (the 1 h signal
only counts if the filter agrees with its direction) and 15-minute entry triggers (after the 1 h
signal, wait for a pullback-and-resume on 15 m before entering). Greedy forward selection: at each
step add the candidate that most improves average ROI per trade on the development period, and
stop when nothing improves it by at least `min_gain` percentage points or trades would drop below
`min_trades`. The chosen recipe is then run once on the locked test period.

All trades are simulated on 15-minute candles with the leveraged ladder, so stops and targets are
checked at 15-minute resolution whatever the signal timeframe.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .candles import load_candles
from .indicators import base_tf, core, higher_tf
from .ladder import simulate_ladder
from .swing import make_bars

log = logging.getLogger(__name__)


def _up(x, level=0):
    return (x > level) & (x.shift() <= level)


def _down(x, level=0):
    return (x < level) & (x.shift() >= level)


def prepare(symbols, root: Path):
    data = {}
    for s in symbols:
        m = load_candles(s, root)
        b15, b60 = make_bars(s, 15, root), make_bars(s, 60, root)
        f60 = base_tf(b60.df, b60.funding_cum).join(higher_tf(m, b60.df.index, pd.Timedelta("1h"), "4h"))
        data[s] = {"b15": b15, "f60": f60, "f15": core(b15.df)}
    return data


def candidates(data, s) -> dict[str, tuple[pd.Series, pd.Series]]:
    """name -> (allows long, allows short), on the 1 h index."""
    f = data[s]["f60"]
    fund = f.funding_24h
    f_hi = fund.rolling(24 * 30, min_periods=24 * 7).quantile(0.8)
    f_lo = fund.rolling(24 * 30, min_periods=24 * 7).quantile(0.2)
    other = "ETH" if s == "BTC" else "BTC"
    o = data[other]["f60"].reindex(f.index) if other in data else f
    t = f.index.tz_convert("America/New_York")
    weekday = pd.Series(t.dayofweek < 5, index=f.index)
    c = {
        "RSI > 55 / < 45 (momentum)": (f.rsi_14 > 55, f.rsi_14 < 45),
        "RSI > 60 / < 40 (momentum)": (f.rsi_14 > 60, f.rsi_14 < 40),
        "MFI > 60 / < 40 (money flow)": (f.mfi > 60, f.mfi < 40),
        "4h Supertrend agrees": (f["4h__supertrend"] > 0, f["4h__supertrend"] < 0),
        "4h above/below EMA 50": (f["4h__ema_50_dist"] > 0, f["4h__ema_50_dist"] < 0),
        "4h Ichimoku cloud agrees": (f["4h__ichi_cloud"] > 0, f["4h__ichi_cloud"] < 0),
        "ADX > 20 (trending)": (f.adx > 20, f.adx > 20),
        "ADX > 25 (strong trend)": (f.adx > 25, f.adx > 25),
        "Funding not crowded": (fund < f_hi, fund > f_lo),
        "Volume above average": (f.vlm_z > 0, f.vlm_z > 0),
        "Price above/below VWAP": (f.vwap_dist > 0, f.vwap_dist < 0),
        "MACD histogram agrees": (f.macd_hist > 0, f.macd_hist < 0),
        "Tenkan above/below Kijun": (f.ichi_tk > 0, f.ichi_tk < 0),
        "Above/below daily pivot": (f.pivot_p > 0, f.pivot_p < 0),
        "Other coin Supertrend agrees": (o.supertrend > 0, o.supertrend < 0),  # ETH for BTC, BTC otherwise
        "Weekdays only": (weekday, weekday),
        "Keltner position agrees": (f.kc_pos > 0, f.kc_pos < 0),
    }
    return {k: (v[0].fillna(False), v[1].fillna(False)) for k, v in c.items()}


TRIGGERS = ("15m RSI pullback", "15m EMA 21 pullback")


def to_15m(data, s, sig60: pd.Series, trigger: str | None, wait_bars: int = 16) -> np.ndarray:
    """Place 1 h signals on the 15 m grid. Without a trigger, the signal sits on the 15 m bar that
    closes with the 1 h candle. With a trigger, it moves to the first 15 m bar within `wait_bars`
    where price pulled back and resumed in the signal's direction (or is dropped)."""
    b15 = data[s]["b15"].df
    idx = b15.index
    out = np.zeros(len(idx), int)
    pos = idx.get_indexer(sig60.index[sig60 != 0] + pd.Timedelta("45min"))
    dirs = sig60[sig60 != 0].to_numpy()
    if trigger is None:
        ok = pos >= 0
        out[pos[ok]] = dirs[ok]
        return out
    f15 = data[s]["f15"]
    if trigger == "15m RSI pullback":  # 15 m RSI dips through 50 against the signal, then crosses back
        rsi = f15.rsi_14.to_numpy()
        prev = np.r_[np.nan, rsi[:-1]]
        long_go, short_go = (prev <= 50) & (rsi > 50), (prev >= 50) & (rsi < 50)
    else:  # price dips to the 15 m EMA 21 and closes back above it (mirror for shorts)
        ema = b15.close.ewm(span=21, adjust=False).mean().to_numpy()
        lo, hi, cl = b15.low.to_numpy(), b15.high.to_numpy(), b15.close.to_numpy()
        long_go, short_go = (lo <= ema) & (cl > ema), (hi >= ema) & (cl < ema)
    for p, d in zip(pos, dirs):
        if p < 0:
            continue
        go = (long_go if d > 0 else short_go)[p + 1:p + 1 + wait_bars]
        hit = np.flatnonzero(go)
        if hit.size:
            out[p + 1 + hit[0]] = d
    return out


def evaluate(data, symbols, filters: list[str], trigger, lo, hi, lev, stop_margin, fee_bps, hold_15m):
    trades = []
    for s in symbols:
        f = data[s]["f60"]
        long = _up(f.ichi_cloud)
        short = _down(f.ichi_cloud)
        cands = candidates(data, s)
        for name in filters:
            al, as_ = cands[name]
            long, short = long & al, short & as_
        sig60 = pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=f.index)
        sig60[(sig60.index < lo) | (sig60.index >= hi)] = 0
        sig = to_15m(data, s, sig60, trigger)
        b15 = data[s]["b15"]
        t = simulate_ladder(b15, sig, stop_margin / lev, False, fee_bps, hold_15m)
        trades.append(t.assign(symbol=s, roi_pct=t.net_bps / 100 * lev))
    return pd.concat(trades, ignore_index=True)


def stats(t: pd.DataFrame) -> dict:
    if len(t) < 2:
        return {"trades": len(t), "avg_roi_pct": np.nan}
    months = t.entry_time.dt.tz_localize(None).dt.to_period("M")
    acct = (1 + 0.1 * t.sort_values("entry_time").roi_pct / 100).cumprod()
    return {"trades": len(t), "hit_first_target": (t.targets_hit >= 1).mean(), "avg_roi_pct": t.roi_pct.mean(),
            "t_stat": t.roi_pct.mean() / (t.roi_pct.std() / np.sqrt(len(t))),
            "coins_positive": int((t.groupby("symbol").roi_pct.mean() > 0).sum()),
            "positive_months": (t.roi_pct.groupby(months).sum() > 0).mean(),
            "account_return": acct.iloc[-1] - 1, "account_max_dd": (acct / acct.cummax() - 1).min()}


def run(symbols, lev, stop_margin, fee_bps, max_hold_h, start, test_start, min_trades, min_gain,
        root: Path, report_dir: Path) -> Path:
    data = prepare(symbols, root)
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(test_start, tz="UTC")
    end = max(d["b15"].df.index.max() for d in data.values()) + pd.Timedelta("1min")
    hold = int(max_hold_h * 4)
    args = (lev, stop_margin, fee_bps, hold)
    names = list(candidates(data, symbols[0]).keys())

    def dev(filters, trig):
        return stats(evaluate(data, symbols, filters, trig, lo, hi, *args))

    chosen, trigger = [], None
    base = dev([], None)
    steps = [{"step": 0, "added": "Ichimoku cloud break (1h) alone", **base}]
    tried = []
    current = base["avg_roi_pct"]
    while True:
        best = None
        options = [("filter", n) for n in names if n not in chosen] + \
                  ([("trigger", tname) for tname in TRIGGERS] if trigger is None else [])
        for kind, n in options:
            f_list = chosen + ([n] if kind == "filter" else [])
            trig = n if kind == "trigger" else trigger
            r = dev(f_list, trig)
            tried.append({"step": len(steps), "candidate": n, **r})
            if r["trades"] >= min_trades and r["avg_roi_pct"] > current + min_gain and \
                    (best is None or r["avg_roi_pct"] > best[2]["avg_roi_pct"]):
                best = (kind, n, r)
            log.info("step %d try %-32s trades %5d roi %.2f", len(steps), n, r["trades"], r["avg_roi_pct"])
        if best is None:
            break
        kind, n, r = best
        if kind == "filter":
            chosen.append(n)
        else:
            trigger = n
        current = r["avg_roi_pct"]
        steps.append({"step": len(steps), "added": n, **r})
    recipe = ["Ichimoku cloud break (1h)"] + chosen + ([trigger] if trigger else [])
    final = []
    for label, f_list, trig in (("base alone", [], None), ("final recipe", chosen, trigger)):
        for period, (a, b) in (("development", (lo, hi)), ("LOCKED TEST", (hi, end))):
            t = evaluate(data, symbols, f_list, trig, a, b, *args)
            final.append({"strategy": label, "period": period, **stats(t)})
            if label == "final recipe" and period == "LOCKED TEST":
                test_trades = t
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "build-up.md"
    test_trades.to_csv(path.with_suffix(".test-trades.csv"), index=False)
    pd.DataFrame(tried).to_csv(path.with_suffix(".all-tries.csv"), index=False)
    path.write_text(_render(pd.DataFrame(steps), pd.DataFrame(final), pd.DataFrame(tried), recipe, lev,
                            stop_margin, fee_bps, max_hold_h, start, test_start, min_trades, min_gain))
    return path


def _fmt(df):
    d = df.copy()
    for c in d.columns:
        if c in ("hit_first_target", "positive_months", "account_return", "account_max_dd"):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        elif d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return d.to_markdown(index=False)


def _render(steps, final, tried, recipe, lev, stop_margin, fee_bps, max_hold_h, start, test_start, min_trades,
            min_gain) -> str:
    first = tried[tried.step == 1].sort_values("avg_roi_pct", ascending=False)
    return f"""# Building up from the 1h Ichimoku cloud break ({lev:g}x ladder)

Exits: stop -{stop_margin:.0%} of margin, half off at +{stop_margin:.0%}, a quarter at +{2 * stop_margin:.0%}, rest at
+{3 * stop_margin:.0%}, max hold {max_hold_h:g} h, checked on 15-minute candles. Fees {fee_bps} bps of position
({fee_bps / 100 * lev:.1f}% of margin) + funding. BTC, ETH, SOL pooled.
Selection uses development data only ({start} to {test_start}); a piece is kept if it raises average ROI per
trade by at least {min_gain} points with at least {min_trades} trades left. The locked test ({test_start} on)
was run once, after the recipe was fixed. `account_*` = 10% of the account as margin per trade.

**Final recipe:** {" + ".join(recipe)}

## Final check

{_fmt(final)}

## Steps kept (development data)

{_fmt(steps)}

## Every piece tried on its own at step 1 (development data)

{_fmt(first[["candidate", "trades", "hit_first_target", "avg_roi_pct", "t_stat", "coins_positive", "positive_months"]])}
"""
