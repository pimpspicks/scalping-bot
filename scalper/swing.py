"""Swing-trade research on Phemex 1-minute candle history (15 min to several hour holds).

Every strategy produces entry signals at bar closes. A trade enters at the next bar's open and
exits at a stop (k x ATR), a take profit (m x ATR) or after a maximum hold, whichever comes
first. If a bar touches both the stop and the target we assume the stop hit first. One position
per symbol at a time. Fees are charged on both sides, plus the funding paid or received at
every 8-hour funding time the position is held through.

Parameters are chosen walk-forward: for each test month, the combination with the best net
result over the previous `train_months` is used, and only the test month's trades are kept. So
every reported number is out of sample.
"""
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .candles import load_candles, load_funding

log = logging.getLogger(__name__)

STOPS = (1.0, 2.0, 3.0)        # x ATR
TARGETS = (1.0, 2.0, 4.0)      # x ATR
HOLDS_H = (1, 4, 12)           # max hold, hours


@dataclass
class Bars:
    symbol: str
    df: pd.DataFrame        # open/high/low/close/volume, UTC index at bar open
    atr: np.ndarray         # ATR in price units known at each bar's close
    funding_cum: np.ndarray  # cumulative funding rate paid by a long, up to each bar's open


def make_bars(symbol: str, bar_min: int, root: Path) -> Bars:
    m = load_candles(symbol, root)
    df = m.resample(f"{bar_min}min").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                          "volume": "sum"}).dropna()
    prev = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f = load_funding(symbol, root)
    cum = f.cumsum().reindex(df.index.union(f.index)).ffill().reindex(df.index).fillna(0.0)
    return Bars(symbol, df, atr.to_numpy(), cum.to_numpy())


# ---------- entry signals: arrays of +1 / -1 / 0 at each bar close ----------

def donchian(b: Bars, n: int) -> np.ndarray:
    hi = b.df.high.rolling(n).max().shift()
    lo = b.df.low.rolling(n).min().shift()
    return np.where(b.df.close > hi, 1, np.where(b.df.close < lo, -1, 0))


def ema_cross(b: Bars, fast: int, slow: int) -> np.ndarray:
    d = np.sign(b.df.close.ewm(span=fast).mean() - b.df.close.ewm(span=slow).mean())
    return np.where(d != d.shift(), d, 0).astype(int)


def mean_revert(b: Bars, n: int, k: float) -> np.ndarray:
    z = (b.df.close - b.df.close.rolling(n).mean()) / b.df.close.rolling(n).std()
    return np.where(z < -k, 1, np.where(z > k, -1, 0))


def model_features(b: Bars, others: list[Bars]) -> pd.DataFrame:
    c = np.log(b.df.close)
    f = pd.DataFrame(index=b.df.index)
    for n in (1, 4, 16, 64):
        f[f"ret_{n}"] = c.diff(n)
    f["vol_16"] = c.diff().rolling(16).std()
    f["vol_ratio"] = f.vol_16 / c.diff().rolling(256).std()
    f["vlm_z"] = np.log1p(b.df.volume).pipe(lambda v: (v - v.rolling(256).mean()) / v.rolling(256).std())
    f["ema_dist"] = c - c.ewm(span=96).mean()
    f["range_pos"] = (b.df.close - b.df.low.rolling(64).min()) / (b.df.high.rolling(64).max() - b.df.low.rolling(64).min())
    f["funding"] = pd.Series(b.funding_cum, index=b.df.index).diff(int(8 * 60 / _bar_minutes(b)) or 1)
    hour = b.df.index.hour + b.df.index.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    for o in others:
        oc = np.log(o.df.close).reindex(b.df.index).ffill()
        f[f"x_{o.symbol}_ret_4"] = oc.diff(4)
        f[f"x_{o.symbol}_ret_16"] = oc.diff(16)
    return f


def _bar_minutes(b: Bars) -> int:
    return int((b.df.index[1] - b.df.index[0]).total_seconds() // 60)


def model_signals(b: Bars, others: list[Bars], horizon: int, thresholds) -> dict[float, np.ndarray]:
    """Logistic regression on past-only data, retrained at the start of every month (expanding
    window, purged by `horizon` bars). Label: sign of the close-to-close return `horizon` bars ahead."""
    f = model_features(b, others)
    c = np.log(b.df.close)
    y = np.sign(c.shift(-horizon) - c)
    p = pd.Series(np.nan, index=f.index)
    months = f.index.tz_localize(None).to_period("M").unique()
    for m in months[3:]:  # need some history before the first fit
        start = m.start_time.tz_localize("UTC")
        end = (m + 1).start_time.tz_localize("UTC")
        train_end = f.index.searchsorted(start) - horizon
        tr = f.iloc[:train_end].join(y.rename("y")).dropna()
        tr = tr[tr.y != 0]
        if len(tr) < 1000:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
        model.fit(tr.drop(columns="y"), (tr.y > 0).astype(int))
        te = f[(f.index >= start) & (f.index < end)].dropna()
        if len(te):
            p.loc[te.index] = model.predict_proba(te)[:, 1]
    edge = (p - 0.5).fillna(0).to_numpy()
    return {t: np.where(edge > t, 1, np.where(edge < -t, -1, 0)) for t in thresholds}


# ---------- trade simulation ----------

def simulate(b: Bars, signal: np.ndarray, stop_atr: float, target_atr: float, hold_bars: int,
             fee_bps: float) -> pd.DataFrame:
    o, h, l, c = (b.df[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(o)
    trades = []
    idx = np.flatnonzero(signal)
    for i in idx:
        if trades and i < trades[-1][1]:  # still in the previous trade
            continue
        e = i + 1
        if e >= n or not np.isfinite(b.atr[i]) or b.atr[i] <= 0:
            continue
        d = signal[i]
        entry = o[e]
        stop = entry - d * stop_atr * b.atr[i]
        target = entry + d * target_atr * b.atr[i]
        last = min(e + hold_bars, n) - 1
        hi, lo = h[e:last + 1], l[e:last + 1]
        hit_stop = (lo <= stop) if d > 0 else (hi >= stop)
        hit_tgt = (hi >= target) if d > 0 else (lo <= target)
        js = np.argmax(hit_stop) if hit_stop.any() else None
        jt = np.argmax(hit_tgt) if hit_tgt.any() else None
        if js is not None and (jt is None or js <= jt):
            j = e + js
            # A gap through the stop fills at the open, not the stop price.
            exit_px = min(o[j], stop) if d > 0 else max(o[j], stop)
            why = "stop"
        elif jt is not None:
            j = e + jt
            exit_px = target
            why = "target"
        else:
            j = last
            exit_px = c[j]
            why = "time"
        gross = d * (exit_px / entry - 1) * 1e4
        exit_bar = min(j + 1, n - 1)  # funding counted through the end of the exit bar
        funding = d * (b.funding_cum[exit_bar] - b.funding_cum[e]) * 1e4
        trades.append((i, j + 1, b.df.index[e], d, gross, funding, gross - funding - fee_bps, why))
    t = pd.DataFrame(trades, columns=["sig_bar", "next_free", "entry_time", "dir", "gross_bps",
                                      "funding_bps", "net_bps", "exit"])
    t["entry_time"] = pd.to_datetime(t.entry_time, utc=True)
    return t


def strategy_grid(b: Bars, others: list[Bars], bar_min: int) -> dict[str, dict[tuple, np.ndarray]]:
    per_h = 60 // bar_min if bar_min <= 60 else 1
    grids = {
        "breakout": {("n", n): donchian(b, n) for n in (4 * per_h, 12 * per_h, 24 * per_h)},
        "trend": {("ema", f, s): ema_cross(b, f, s) for f, s in ((per_h, 4 * per_h), (4 * per_h, 16 * per_h),
                                                                   (12 * per_h, 48 * per_h))},
        "mean_reversion": {("z", n, k): mean_revert(b, n, k) for n in (4 * per_h, 24 * per_h) for k in (2.0, 3.0)},
    }
    sigs = model_signals(b, others, horizon=4 * per_h, thresholds=(0.02, 0.05, 0.08))
    grids["model"] = {("thr", t): s for t, s in sigs.items()}
    return grids


def walk_forward(trades_by_combo: dict[tuple, pd.DataFrame], train_months: int, min_train_trades: int = 10):
    """For each month, pick the combo with the best total net bps over the previous months and
    keep its trades in that month. Returns (out-of-sample trades, chosen combo per month)."""
    all_months = sorted({m for t in trades_by_combo.values() for m in t.entry_time.dt.tz_localize(None).dt.to_period("M")})
    picked, kept = [], []
    for m in all_months[train_months:]:
        lo = (m - train_months).start_time
        hi = m.start_time - pd.Timedelta(hours=max(HOLDS_H))  # training trades must have closed
        best, best_val = None, -np.inf
        for combo, t in trades_by_combo.items():
            et = t.entry_time.dt.tz_localize(None)
            tr = t[(et >= lo) & (et < hi)]
            if len(tr) >= min_train_trades and tr.net_bps.sum() > best_val:
                best, best_val = combo, tr.net_bps.sum()
        if best is None:
            continue
        t = trades_by_combo[best]
        et = t.entry_time.dt.tz_localize(None)
        kept.append(t[(et >= m.start_time) & (et < (m + 1).start_time)].assign(params=str(best)))
        picked.append({"month": str(m), "params": str(best), "train_net_bps": round(best_val, 1)})
    oos = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    return oos, pd.DataFrame(picked)


def summarize(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    months = max(t.entry_time.dt.tz_localize(None).dt.to_period("M").nunique(), 1)
    eq = (1 + t.net_bps / 1e4).cumprod()
    wins, losses = t.net_bps[t.net_bps > 0].sum(), -t.net_bps[t.net_bps < 0].sum()
    return {
        "trades": len(t),
        "trades_per_month": len(t) / months,
        "win_rate": (t.net_bps > 0).mean(),
        "avg_gross_bps": t.gross_bps.mean(),
        "avg_funding_bps": t.funding_bps.mean(),
        "avg_net_bps": t.net_bps.mean(),
        "profit_factor": wins / losses if losses else np.inf,
        "total_return_1x": eq.iloc[-1] - 1,
        "max_drawdown_1x": (eq / eq.cummax() - 1).min(),
        "positive_months": (t.groupby(t.entry_time.dt.tz_localize(None).dt.to_period("M")).net_bps.sum() > 0).mean(),
    }


def run(symbols, bar_min: int, fee_bps: float, train_months: int, root: Path, report_dir: Path) -> Path:
    bars = {s: make_bars(s, bar_min, root) for s in symbols}
    rows, monthly, picks, all_oos = [], [], [], []
    per_h = 60 / bar_min
    exits = list(itertools.product(STOPS, TARGETS, [int(h * per_h) for h in HOLDS_H]))
    for s, b in bars.items():
        others = [o for k, o in bars.items() if k != s]
        for strat, entries in strategy_grid(b, others, bar_min).items():
            combos = {}
            for (ekey, sig), (st, tg, hb) in itertools.product(entries.items(), exits):
                combos[(*ekey, "stop", st, "tp", tg, "hold_h", hb / per_h)] = simulate(b, sig, st, tg, hb, fee_bps)
            oos, pk = walk_forward(combos, train_months)
            log.info("%s %s: %d out-of-sample trades", s, strat, len(oos))
            rows.append({"symbol": s, "strategy": strat, **summarize(oos)})
            picks.append(pk.assign(symbol=s, strategy=strat))
            if not oos.empty:
                all_oos.append(oos.assign(symbol=s, strategy=strat))
                monthly.append(oos.groupby(oos.entry_time.dt.tz_localize(None).dt.to_period("M").astype(str))
                               .net_bps.sum().rename(f"{s} {strat}"))
    first_test = min(p.month.min() for p in picks if not p.empty)
    for s, b in bars.items():
        c = b.df.close[b.df.index.tz_localize(None) >= pd.Period(first_test).start_time]
        rows.append({"symbol": s, "strategy": "buy & hold (no fees)", "total_return_1x": c.iloc[-1] / c.iloc[0] - 1,
                     "max_drawdown_1x": (c / c.cummax() - 1).min()})
    summary = pd.DataFrame(rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"swing-phemex-{bar_min}m.md"
    span = {s: f"{b.df.index.min():%Y-%m-%d} to {b.df.index.max():%Y-%m-%d}" for s, b in bars.items()}
    monthly_df = pd.concat(monthly, axis=1).fillna(0).round(0) if monthly else pd.DataFrame()
    path.write_text(_render(summary, monthly_df, pd.concat(picks), bar_min, fee_bps, train_months, span, first_test))
    if all_oos:
        pd.concat(all_oos).to_csv(path.with_suffix(".trades.csv"), index=False)
    return path


def _render(summary, monthly, picks, bar_min, fee_bps, train_months, span, first_test) -> str:
    s = summary.copy()
    pct = ("win_rate", "total_return_1x", "max_drawdown_1x", "positive_months")
    for c in s.columns:
        if c in pct:
            s[c] = s[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        elif s[c].dtype.kind == "f":
            s[c] = s[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    data = "; ".join(f"{k} {v}" for k, v in span.items())
    return f"""# Swing backtest: Phemex USDT perps, {bar_min}-minute bars

Data: {data}. Fees: {fee_bps} bps round trip, plus funding. Out-of-sample test period starts {first_test}.

Parameters (signal settings, stop {STOPS} x ATR, target {TARGETS} x ATR, max hold {HOLDS_H} h) are re-picked
every month from the previous {train_months} months only; the table shows the following months' trades.
`total_return_1x` compounds every trade at 1x with the whole account; at N x leverage multiply per-trade
numbers by N (and the drawdown risk with them).

## Out-of-sample results

{s.to_markdown(index=False)}

## Net bps per month (sum of trade results)

{monthly.to_markdown() if not monthly.empty else "none"}

## Parameters picked each month

{picks.to_markdown(index=False)}
"""
