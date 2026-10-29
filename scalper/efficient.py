"""Getting more out of the 2h move-1 pullback without changing its rules: cheaper fills, risk per trade, and
running it next to the move-3 stochastic fade (USDT.D filter).

Maker fills: a limit order only fills if price trades through it by at least `TICK` of price; maker fee `MAKER`
per side. Stops are always market orders at `COST` (8 bps, fee plus slippage).
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .fresh import load_coin
from .intraday import COST, _row
from .moneyflows import _at, flow_signals, load_flows
from .refine import FULL, TF, a_signals
from .understanding import signals, simulate as u_simulate

log = logging.getLogger(__name__)

MAKER, TICK = 0.0002, 0.0001
EXEC = ("all market orders", "limit target", "limit entry + limit target")
RISKS = (0.005, 0.01, 0.015, 0.02)
PERIODS = {"2020-22": ("2020-01-01", "2023-01-01"), "2023-26": ("2023-01-01", "2026-10-01"), "2020-26": ("2020-01-01", "2026-10-01")}


def move1_trades(cn, bias, how: str, risk=0.005, period=FULL) -> pd.DataFrame:
    """Setup A, 2h, 2 ATR swings, stop 0.1 ATR beyond the pullback low, 2R target, stablecoin filter."""
    sig, stop = a_signals(cn, 2.0, "any", 0.1)
    return trades_2r(cn, np.where(sig * bias > 0, sig, 0), stop, how, risk, period)


def trades_2r(cn, sig, stop, how: str, risk=0.005, period=FULL) -> pd.DataFrame:
    """Trades for entry signals with a stop price each and a 2R target. A trade still running at the period's end
    is closed at the last close with exit "open at end"."""
    inside = np.flatnonzero((cn.t_close >= period[0]) & (cn.t_close < period[1]))
    i0, end = inside[0], inside[-1] + 1
    rows, free = [], 0
    for i in np.flatnonzero(sig[i0:end - 2]) + i0:
        if i < free:
            continue
        d = int(sig[i])
        if how == "limit entry + limit target":
            # limit at the signal bar's close, good for 2 bars; skipped if not traded through
            px = cn.c[i]
            fill = [k for k in (i + 1, i + 2) if (cn.l[k] < px * (1 - TICK) if d > 0 else cn.h[k] > px * (1 + TICK))]
            if not fill:
                continue
            e, entry, fee_in = fill[0], px, MAKER
            if d * (entry - stop[i]) <= 0 or (d > 0 and cn.l[e] <= stop[i]) or (d < 0 and cn.h[e] >= stop[i]):
                continue  # filled bar already through the stop: skip rather than guess the order of events
            start = e + 1
        else:
            e, entry, fee_in = i + 1, cn.o[i + 1], COST
            start = e
        dist = d * (entry - stop[i])
        if not np.isfinite(dist) or dist <= 0:
            continue
        x_stop, x_tgt = d * stop[i], d * entry + 2 * dist
        through = TICK * entry if how != "all market orders" else 0.0
        w, k_exit = 256, None
        while True:
            j1 = min(start + w, end)
            lo = cn.l[start:j1] if d > 0 else -cn.h[start:j1]
            hi = cn.h[start:j1] if d > 0 else -cn.l[start:j1]
            s = np.flatnonzero(lo <= x_stop)
            t = np.flatnonzero(hi >= x_tgt + through)
            if len(s) and (not len(t) or s[0] <= t[0]):
                k = int(s[0])
                x_exit = min(d * cn.o[start + k], x_stop) if (k or start > e) else x_stop
                k_exit, kind, fee_out = start + k, "stop", COST
                break
            if len(t):
                k_exit, x_exit, kind = start + int(t[0]), x_tgt, "target"
                fee_out = MAKER if how != "all market orders" else COST
                break
            if j1 == end:
                k_exit, x_exit, kind, fee_out = end - 1, d * cn.c[end - 1], "open at end", COST
                break
            w *= 4
        gross = (x_exit - d * entry) / entry
        funding = d * (cn.fcum[min(k_exit + 1, len(cn.fcum) - 1)] - cn.fcum[e])
        net = gross - fee_in - fee_out - funding
        size = min(risk / (dist / entry), 10.0)
        rows.append((cn.t_close[i], cn.t_close[k_exit], d, kind, gross, net, size, 2 * dist / entry,
                     cn.t_close[e] if start > e else cn.t_close[i], entry, stop[i], entry + 2 * d * dist))
        free = k_exit
    return pd.DataFrame(rows, columns=["entry_time", "exit_time", "dir", "exit", "gross", "net", "size", "target_move",
                                       "fill_time", "entry_price", "stop_price", "target_price"])


def fade_trades(cn, bias, risk=0.005, period=FULL) -> pd.DataFrame:
    """Setup C2 (move-3 stochastic divergence fade), 2h, 3 ATR swings, 2R target, USDT.D 30d filter."""
    sig, st, tg = signals(cn, 3.0, "C2 move-3 stochastic divergence")
    sig = np.where(sig * bias > 0, sig, 0)
    t = u_simulate(cn, sig, st, tg, "2R", None, period)
    if len(t):
        t["size"] *= risk / 0.005
        t["entry_time"] = t.exit_time - pd.to_timedelta(t.hours, unit="h")
    return t


def exposure(t: pd.DataFrame) -> pd.Series:
    """Total open notional (multiple of the account) over time, sampled daily."""
    if not len(t):
        return pd.Series(dtype=float)
    ev = pd.concat([pd.Series(t["size"].values, index=t.entry_time), pd.Series(-t["size"].values, index=t.exit_time)])
    return ev.groupby(level=0).sum().sort_index().cumsum().resample("1D").max().ffill()


def run(symbols, root: Path, flow_root: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    fs = flow_signals(load_flows(flow_root))
    move1 = {h: [] for h in EXEC}
    fade = []
    for sym in symbols:
        try:
            cn = load_coin(sym, TF, root)
        except SystemExit:
            continue
        b_st, b_ud = _at(fs[("Stables", 30)], cn.t_close), _at(fs[("USDT.D", 30)], cn.t_close)
        for h in EXEC:
            move1[h].append(move1_trades(cn, b_st, h).assign(coin=sym))
        fade.append(fade_trades(cn, b_ud).assign(coin=sym))
        log.info("done %s", sym)
    move1 = {h: pd.concat(v, ignore_index=True) for h, v in move1.items()}
    fade = pd.concat(fade, ignore_index=True)

    def per(t, p):
        a, b = (pd.Timestamp(x, tz="UTC") for x in PERIODS[p])
        return _row(t[(t.exit_time >= a) & (t.exit_time < b)], (a, b))

    def fmt(d):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}

    cols = ["Sharpe", "Return/yr %", "Max DD %", "By year %", "Trades", "Win %", "Avg net per trade (bps of price)"]
    ex = pd.DataFrame([{"Execution": h, "Period": p, **{k: fmt(per(move1[h], p))[k] for k in cols}}
                       for h in EXEC for p in PERIODS])
    best = "limit target"
    base = move1[best]
    risk_rows = []
    for r in RISKS:
        t = base.assign(size=base["size"] * r / 0.005)
        e = exposure(t)
        for p in ("2020-22", "2023-26"):
            risk_rows.append({"Risk per trade %": r * 100, "Period": p, **{k: fmt(per(t, p))[k] for k in cols[:4]},
                              "Max open notional (x account)": round(e.max(), 1),
                              "Median open notional": round(e.median(), 1)})
    combo_rows = []
    for name, t in (("Move-1 pullback", base), ("Move-3 fade", fade),
                    ("Both together", pd.concat([base, fade], ignore_index=True))):
        for p in PERIODS:
            combo_rows.append({"Strategy": name, "Period": p, **{k: fmt(per(t, p))[k] for k in cols[:5]}})
    d1 = base.assign(day=base.exit_time.dt.floor("D")).groupby("day").apply(lambda x: (x.net * x["size"]).sum())
    d2 = fade.assign(day=fade.exit_time.dt.floor("D")).groupby("day").apply(lambda x: (x.net * x["size"]).sum())
    corr = pd.concat([d1, d2], axis=1).fillna(0).resample("W").sum().corr().iloc[0, 1]
    lines = ["# Making the 2h move-1 pullback more efficient: results\n",
             "## 1. Cheaper fills (0.5% risk per trade)\n", ex.to_markdown(index=False),
             f"\n## 2. Risk per trade (execution: {best})\n", pd.DataFrame(risk_rows).to_markdown(index=False),
             "\n## 3. Running it next to the move-3 stochastic fade (USDT.D 30d), 0.5% risk each\n",
             pd.DataFrame(combo_rows).to_markdown(index=False),
             f"\nWeekly P&L correlation between the two: {corr:.2f}\n"]
    path = report_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path
