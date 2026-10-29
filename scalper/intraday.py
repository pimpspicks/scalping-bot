"""Intraday setups held at most 5 hours, with volatility-based stop and target and 0.5%-risk sizing.

Plan and pass rule: reports-2026-10-29-intraday/plan.md (written before running).

Entries are decided at a 15-minute close and filled at the next open. Stop = K x ATR(14) of 15-minute bars,
target = R x the stop distance, forced exit at the close of the 20th bar (5 hours). A bar touching both stop and
target counts as the stop. A stop gapped through fills at the open. Costs `COST` per side plus funding.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .fresh import DESIGN, EXAM, Coin, load_coin, score
from .indicators import _rsi

log = logging.getLogger(__name__)

RISK, COST, MAX_SIZE, HOLD = 0.005, 0.0008, 10.0, 20
ENTRIES = ("RSI pullback", "Donchian 20", "1h EMA 21 cross", "Asia open range", "US open range")


def _cross(a, b):
    b_prev = b.shift() if isinstance(b, pd.Series) else b
    return (a > b) & (a.shift() <= b_prev)


def _first_per_day(ev: np.ndarray, t: pd.DatetimeIndex) -> np.ndarray:
    day = t.floor("D")
    nz = pd.Series(ev != 0, index=t)
    first = nz & (nz.astype(int).groupby(day).cumsum() == 1)
    return np.where(first.to_numpy(), ev, 0)


def _orb(cn: Coin, r0: int, r1: int, until: int) -> np.ndarray:
    """Opening range = bars whose close (minutes after midnight) is in (r0, r1]; first break after r1, up to `until`."""
    t = cn.t_close
    mins = np.asarray(t.hour * 60 + t.minute)
    mins = np.where(mins == 0, 1440, mins)  # a bar closing at 00:00 belongs to the previous day
    day = (t - pd.Timedelta("1min")).floor("D")
    in_r = (mins > r0) & (mins <= r1)
    after = (mins > r1) & (mins <= until)
    hi = pd.Series(np.where(in_r, cn.h, np.nan)).groupby(day).transform("max").to_numpy()
    lo = pd.Series(np.where(in_r, cn.l, np.nan)).groupby(day).transform("min").to_numpy()
    ev = np.where(after & (cn.c > hi), 1, np.where(after & (cn.c < lo), -1, 0))
    return _first_per_day(ev, day + pd.Timedelta("1min"))


def entries(cn: Coin) -> dict[str, np.ndarray]:
    c, h, l = pd.Series(cn.c), pd.Series(cn.h), pd.Series(cn.l)
    rsi = _rsi(c)
    E = {"RSI pullback": np.where(_cross(rsi, 40), 1, np.where(_cross(-rsi, -60), -1, 0))}
    hi, lo = h.rolling(20).max().shift(), l.rolling(20).min().shift()
    up, dn = c > hi, c < lo
    E["Donchian 20"] = np.where(up & ~up.shift(fill_value=False), 1, np.where(dn & ~dn.shift(fill_value=False), -1, 0))
    hourly = pd.Series(cn.c, index=cn.t_close)
    hourly = hourly[(cn.t_close.minute == 0)]
    ema = hourly.ewm(span=21, adjust=False).mean().reindex(cn.t_close, method="ffill")
    ema = pd.Series(ema.to_numpy())
    E["1h EMA 21 cross"] = np.where(_cross(c, ema), 1, np.where(_cross(ema, c), -1, 0))
    E["Asia open range"] = _orb(cn, 0, 60, 8 * 60)
    E["US open range"] = _orb(cn, 13 * 60 + 30, 14 * 60 + 30, 20 * 60)
    return {k: np.asarray(v, int) for k, v in E.items()}


def simulate(cn: Coin, sig: np.ndarray, k_stop, r_target, period, stop_level: np.ndarray | None = None) -> pd.DataFrame:
    """k_stop: ATR multiple, or "swing" (beyond the last 8 bars' extreme + 0.1 ATR). r_target: multiple of the
    stop distance, or "level" (the last 24 hours' high / low, taken only if at least 1x the stop away).
    stop_level: with "swing", a setup's own stop price per signal bar (NaN: use the 8-bar swing)."""
    if k_stop == "swing":
        lo8, hi8 = pd.Series(cn.l).rolling(8).min().to_numpy(), pd.Series(cn.h).rolling(8).max().to_numpy()
    if r_target == "level":
        lo96, hi96 = pd.Series(cn.l).rolling(96).min().to_numpy(), pd.Series(cn.h).rolling(96).max().to_numpy()
    inside = np.flatnonzero((cn.t_close >= period[0]) & (cn.t_close < period[1]))
    if not len(inside):
        return pd.DataFrame()
    i0, end = inside[0], inside[-1] + 1
    rows, free = [], 0
    for i in np.flatnonzero(sig[i0:end - 1]) + i0:
        if i < free or not np.isfinite(cn.atr[i]) or cn.atr[i] <= 0:
            continue
        d, e = int(sig[i]), i + 1
        entry = cn.o[e]
        if k_stop == "swing":
            lvl = stop_level[i] if stop_level is not None and np.isfinite(stop_level[i]) else (lo8[i] if d > 0 else hi8[i])
            dist = (entry - lvl if d > 0 else lvl - entry) + 0.1 * cn.atr[i]
        else:
            dist = k_stop * cn.atr[i]
        if not np.isfinite(dist) or dist <= 0:
            continue
        if r_target == "level":
            move = hi96[i] - entry if d > 0 else entry - lo96[i]
            if not np.isfinite(move) or move < dist:
                continue
        else:
            move = r_target * dist
        last = min(e + HOLD, end) - 1
        x_lo = d * cn.l[e:last + 1] if d > 0 else -cn.h[e:last + 1]  # adverse extreme, in direction space
        x_hi = d * cn.h[e:last + 1] if d > 0 else -cn.l[e:last + 1]
        x_stop, x_tgt = d * entry - dist, d * entry + move
        s = np.flatnonzero(x_lo <= x_stop)
        t = np.flatnonzero(x_hi >= x_tgt)
        s0 = s[0] if len(s) else HOLD + 1
        t0 = t[0] if len(t) else HOLD + 1
        if s0 <= t0 and len(s):
            k = int(s0)
            x_exit = min(d * cn.o[e + k], x_stop) if k > 0 else x_stop
            kind = "stop"
        elif len(t):
            k, x_exit, kind = int(t0), x_tgt, "target"
        else:
            k = last - e
            x_exit, kind = d * cn.c[last], "time"
        j = e + k
        gross = (x_exit - d * entry) / entry
        funding = d * (cn.fcum[min(j + 1, len(cn.fcum) - 1)] - cn.fcum[e])
        net = gross - 2 * COST - funding
        size = min(RISK / (dist / entry), MAX_SIZE)
        rows.append((cn.t_close[j], d, kind, gross, net, size, move / entry, cn.t_close[i]))
        free = j
    return pd.DataFrame(rows, columns=["exit_time", "dir", "exit", "gross", "net", "size", "target_move", "signal_time"])


def ranking_filter(coins: dict[str, Coin]) -> dict[str, np.ndarray]:
    """+1 for the 4 coins up most over 3 weeks as of the latest Monday 00:00, -1 for the 4 down most, else 0."""
    closes = pd.DataFrame({s: pd.Series(cn.c, index=cn.t_close) for s, cn in coins.items()})
    daily = closes[closes.index.hour.isin([0]) & (closes.index.minute == 0)]
    mon = daily[daily.index.dayofweek == 0]
    ret = (mon / mon.shift(3) - 1)
    side = pd.DataFrame(0, index=mon.index, columns=mon.columns)
    for t, r in ret.iterrows():
        r = r.dropna()
        if len(r) >= 10:
            o = r.sort_values()
            side.loc[t, o.index[-4:]] = 1
            side.loc[t, o.index[:4]] = -1
    return {s: side[s].reindex(side.index.union(cn.t_close)).ffill().reindex(cn.t_close).fillna(0).to_numpy()
            for s, cn in coins.items()}


GRID = [(e, f, k, r) for e in ENTRIES for f in (False, True) for k in (1.5, 2.0, 3.0) for r in (1.0, 2.0, 3.0)] \
    + [(e, f, "swing", r) for e in ENTRIES for f in (False, True) for r in ("level", 2.0)]


def _label(e, f, k, r):
    stop = "swing stop" if k == "swing" else f"stop {k} ATR"
    tgt = "target 24h high/low" if r == "level" else f"target {r:.0f}R"
    return f"{e}{' + ranking' if f else ''}, {stop}, {tgt}"


def all_runs(coins, filt, period, allow=None):
    """allow: optional {coin: bool array}; signals on bars marked False are dropped."""
    ents = {s: entries(cn) for s, cn in coins.items()}
    out = {}
    for e, f, k, r in GRID:
        trades = []
        for s, cn in coins.items():
            sig = ents[s][e]
            if f:
                sig = np.where(sig == filt[s], sig, 0)
            if allow is not None:
                sig = np.where(allow[s], sig, 0)
            t = simulate(cn, sig, k, r, period)
            if len(t):
                trades.append(t.assign(coin=s))
        out[(e, f, k, r)] = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return out


def _daily(t: pd.DataFrame) -> pd.Series:
    if not len(t):
        return pd.Series(dtype=float)
    return (t.net * t["size"]).groupby(t.exit_time.dt.floor("D")).sum()


def _row(t, period):
    s = score(_daily(t), period)
    if not len(t):
        return {**s, "Trades": 0}
    return {**s, "Trades": len(t), "Win %": (t.net > 0).mean() * 100, "Avg net per trade (bps of price)": t.net.mean() * 1e4,
            "Avg gross (bps)": t.gross.mean() * 1e4, "Target hit %": (t.exit == "target").mean() * 100,
            "Avg target move %": t.target_move.mean() * 100}


def run(symbols, root: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    coins = {}
    for s in symbols:
        try:
            coins[s] = load_coin(s, 15, root)
        except SystemExit:
            pass
    filt = ranking_filter(coins)
    design = all_runs(coins, filt, DESIGN)
    dsc = {g: _row(t, DESIGN) for g, t in design.items()}
    best = None
    for e, f, k, r in GRID:
        s = dsc[(e, f, k, r)]
        if k == "swing":
            nbs = [(e, f, k, "level" if r == 2.0 else 2.0)]
        else:
            nbs = [(e, f, kk, r) for kk in (1.5, 2.0, 3.0) if kk != k] + [(e, f, k, rr) for rr in (1.0, 2.0, 3.0) if rr != r]
        ok = s["Years positive"] >= 3 and all(dsc[n]["Sharpe"] > 0 for n in nbs)
        if ok and (best is None or s["Sharpe"] > dsc[best]["Sharpe"]):
            best = (e, f, k, r)
    log.info("chosen: %s", best)
    exam = all_runs(coins, filt, EXAM)
    esc = {g: _row(t, EXAM) for g, t in exam.items()}

    def fmt(d):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}

    table = pd.DataFrame([{"Version": _label(*g), **{f"Design {k}": v for k, v in fmt(dsc[g]).items()},
                           **{f"Exam {k}": v for k, v in fmt(esc[g]).items()}} for g in GRID])
    table = table.sort_values("Design Sharpe", ascending=False)
    table.to_csv(report_dir / "all-versions.csv", index=False)
    cols = ["Version", "Design Trades", "Design Win %", "Design Avg net per trade (bps of price)", "Design Sharpe",
            "Design Return/yr %", "Design Max DD %", "Design By year %", "Exam Sharpe", "Exam Return/yr %",
            "Exam Max DD %", "Exam By year %"]
    lines = ["# Intraday setups (max 5h hold): results\n", "Plan: plan.md (written before running).\n",
             f"Chosen on the design period: {_label(*best) if best else 'none eligible'}\n"]
    if best:
        lines += ["## Chosen version\n", pd.DataFrame([{"Period": "Design 2023-26", **fmt(dsc[best])},
                                                     {"Period": "Exam 2020-22", **fmt(esc[best])}]).to_markdown(index=False)]
    ex = table["Exam Sharpe"]
    lines += [f"\n## Context: all {len(table)} versions\n",
              f"Design: {(table['Design Sharpe'] > 0).mean():.0%} with Sharpe > 0. Exam: {(ex > 0).mean():.0%} with Sharpe > 0, "
              f"median {ex.median():.2f}. Rank correlation design vs exam Sharpe: {table['Design Sharpe'].corr(ex, method='spearman'):.2f}\n",
              "Average net per trade by entry and filter (design, bps of price):\n",
              table.assign(Entry=[v.split(",")[0] for v in table.Version]).groupby("Entry")[
                  ["Design Avg net per trade (bps of price)", "Exam Avg net per trade (bps of price)"]].mean().round(1).to_markdown(),
              "\nTop 15 by design Sharpe:\n", table.head(15)[cols].to_markdown(index=False)]
    path = report_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path
