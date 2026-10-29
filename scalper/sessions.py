"""Intraday setups limited to one market open at a time, plus order-block retests (max 5-hour hold).

Plan and pass rule: reports-2026-10-29-sessions/plan.md (written before running). Trade simulation, sizing and
costs are those of `intraday.simulate`.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .fresh import DESIGN, EXAM, Coin, load_coin
from .intraday import _row, entries as base_entries, ranking_filter, simulate

log = logging.getLogger(__name__)

SESSIONS = {"China/Asia open": ("Asia/Shanghai", 9, 30), "London open": ("Europe/London", 8, 0),
            "New York open": ("America/New_York", 9, 30)}
ENTRIES = ("RSI pullback", "Donchian 20", "1h EMA 21 cross", "Session opening range", "Order block")
EXITS = {"stop 2 ATR, target 2R": (2.0, 2.0), "stop 3 ATR, target 2R": (3.0, 2.0),
         "chart stop, target 24h high/low": ("swing", "level"), "chart stop, target 2R": ("swing", 2.0)}
SIBLING = {"stop 2 ATR, target 2R": "stop 3 ATR, target 2R", "stop 3 ATR, target 2R": "stop 2 ATR, target 2R",
           "chart stop, target 24h high/low": "chart stop, target 2R", "chart stop, target 2R": "chart stop, target 24h high/low"}


def minutes_after_open(cn: Coin, tz: str, h: int, m: int) -> np.ndarray:
    """Minutes from today's open (local time) to each bar's close; negative before the open."""
    lt = cn.t_close.tz_convert(tz)
    return np.asarray((lt.hour * 60 + lt.minute) - (h * 60 + m))


def session_orb(cn: Coin, mins: np.ndarray, day: np.ndarray) -> np.ndarray:
    in_r = (mins > 0) & (mins <= 30)
    after = (mins > 30) & (mins <= 180)
    hi = pd.Series(np.where(in_r, cn.h, np.nan)).groupby(day).transform("max").to_numpy()
    lo = pd.Series(np.where(in_r, cn.l, np.nan)).groupby(day).transform("min").to_numpy()
    ev = np.where(after & (cn.c > hi), 1, np.where(after & (cn.c < lo), -1, 0))
    nz = pd.Series(ev != 0)
    first = nz & (nz.astype(int).groupby(day).cumsum() == 1)
    return np.where(first.to_numpy(), ev, 0)


def order_blocks(cn: Coin, lookback=20, ob_window=10, valid=24) -> tuple[np.ndarray, np.ndarray]:
    """Order-block retest signals and each signal's stop level (the block's far edge)."""
    o, h, l, c = cn.o, cn.h, cn.l, cn.c
    n = len(c)
    hi20 = pd.Series(h).rolling(lookback).max().shift().to_numpy()
    lo20 = pd.Series(l).rolling(lookback).min().shift().to_numpy()
    sig, lvl = np.zeros(n, int), np.full(n, np.nan)
    for d, bos in ((1, np.flatnonzero(c > hi20)), (-1, np.flatnonzero(c < lo20))):
        last_end = -1
        for b in bos:
            if b <= last_end:  # one block per structure break run
                continue
            k0 = max(b - ob_window, 0)
            opp = np.flatnonzero((c[k0:b] < o[k0:b]) if d > 0 else (c[k0:b] > o[k0:b]))
            if not len(opp):
                continue
            k = k0 + opp[-1]
            top, bot = h[k], l[k]
            for i in range(b + 1, min(b + 1 + valid, n)):
                if d > 0 and l[i] <= top:
                    if c[i] >= bot:
                        sig[i], lvl[i] = 1, bot
                    last_end = i
                    break
                if d < 0 and h[i] >= bot:
                    if c[i] <= top:
                        sig[i], lvl[i] = -1, top
                    last_end = i
                    break
            else:
                last_end = b + valid
    return sig, lvl


GRID = [(e, s, f, x) for e in ENTRIES for s in SESSIONS for f in (False, True) for x in EXITS]


def _label(e, s, f, x):
    return f"{s}: {e}{' + ranking' if f else ''}, {x}"


def all_runs(coins, filt, period):
    prep = {}
    for sym, cn in coins.items():
        E = base_entries(cn)
        ob, ob_lvl = order_blocks(cn)
        per_s = {}
        for sname, (tz, h, m) in SESSIONS.items():
            mins = minutes_after_open(cn, tz, h, m)
            day = np.asarray(cn.t_close.tz_convert(tz).floor("D").asi8)
            win = (mins > 0) & (mins <= 180)
            per_s[sname] = {"RSI pullback": np.where(win, E["RSI pullback"], 0),
                            "Donchian 20": np.where(win, E["Donchian 20"], 0),
                            "1h EMA 21 cross": np.where(win, E["1h EMA 21 cross"], 0),
                            "Session opening range": session_orb(cn, mins, day),
                            "Order block": np.where(win, ob, 0)}
        prep[sym] = (per_s, ob_lvl)
    out = {}
    for e, s, f, x in GRID:
        k, r = EXITS[x]
        trades = []
        for sym, cn in coins.items():
            sig = prep[sym][0][s][e]
            if f:
                sig = np.where(sig == filt[sym], sig, 0)
            t = simulate(cn, sig, k, r, period, prep[sym][1] if e == "Order block" else None)
            if len(t):
                trades.append(t.assign(coin=sym))
        out[(e, s, f, x)] = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return out


def run(symbols, root: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    coins = {}
    for sym in symbols:
        try:
            coins[sym] = load_coin(sym, 15, root)
        except SystemExit:
            pass
    filt = ranking_filter(coins)
    design = all_runs(coins, filt, DESIGN)
    dsc = {g: _row(t, DESIGN) for g, t in design.items()}
    best = None
    for g in GRID:
        e, s, f, x = g
        ok = dsc[g]["Years positive"] >= 3 and dsc[(e, s, f, SIBLING[x])]["Sharpe"] > 0
        if ok and (best is None or dsc[g]["Sharpe"] > dsc[best]["Sharpe"]):
            best = g
    log.info("chosen: %s", best)
    exam = all_runs(coins, filt, EXAM)
    esc = {g: _row(t, EXAM) for g, t in exam.items()}

    def fmt(d):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}

    table = pd.DataFrame([{"Version": _label(*g), "Entry": g[0], "Session": g[1], "Ranking": g[2], "Exit": g[3],
                           **{f"Design {k}": v for k, v in fmt(dsc[g]).items()},
                           **{f"Exam {k}": v for k, v in fmt(esc[g]).items()}} for g in GRID])
    table = table.sort_values("Design Sharpe", ascending=False)
    table.to_csv(report_dir / "all-versions.csv", index=False)
    g1 = ["Design Avg gross (bps)", "Design Avg net per trade (bps of price)", "Exam Avg gross (bps)",
          "Exam Avg net per trade (bps of price)"]
    cols = ["Version", "Design Trades", "Design Win %", "Design Avg gross (bps)", "Design Avg net per trade (bps of price)",
            "Design Sharpe", "Design By year %", "Exam Sharpe", "Exam By year %"]
    lines = ["# Session opens + order blocks: results\n", "Plan: plan.md (written before running).\n",
             f"Chosen on the design period: {_label(*best) if best else 'none eligible'}\n"]
    if best:
        lines += [pd.DataFrame([{"Period": "Design 2023-26", **fmt(dsc[best])},
                                {"Period": "Exam 2020-22", **fmt(esc[best])}]).to_markdown(index=False)]
    lines += [f"\nDesign: {(table['Design Sharpe'] > 0).sum()} of {len(table)} versions with Sharpe > 0. "
              f"Exam: {(table['Exam Sharpe'] > 0).sum()}.\n",
              "## Average per trade by session (bps of price)\n", table.groupby("Session")[g1].mean().round(1).to_markdown(),
              "\n## Average per trade by entry (bps of price)\n", table.groupby("Entry")[g1].mean().round(1).to_markdown(),
              "\n## Top 15 by design Sharpe\n", table.head(15)[cols].to_markdown(index=False)]
    path = report_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path
