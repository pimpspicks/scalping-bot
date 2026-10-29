"""A bot from scratch: slow trend and coin-ranking strategies with volatility-based stops and risk sizing.

Plan and pass rule: reports-2026-10-29-fresh-bot/plan.md (written before running).

Every position risks `RISK` of the account at its stop: size (notional, as a fraction of equity) =
RISK / stop distance. The stop starts K x ATR(14) from the entry and trails K x ATR behind the best close
since entry (a chandelier stop). Optionally half the position is taken off at 3R. Entries fill at the next
bar's open; a stop fills at the stop price, or at the open if the bar gaps through it. Costs `COST` per side,
plus funding. Positions are marked to market at every bar close, and the account's daily return is the sum of
all positions' P&L that day (sizes are fractions of the account, so returns add).
"""
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .rules3 import hurst
from .swing import make_bars

log = logging.getLogger(__name__)

RISK, COST, MAX_SIZE = 0.005, 0.0008, 1.0
DESIGN = (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))
EXAM = (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC"))


@dataclass
class Coin:
    symbol: str
    tf: int
    t_close: pd.DatetimeIndex
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    atr: np.ndarray
    fcum: np.ndarray


def load_coin(s: str, tf: int, root: Path) -> Coin:
    b = make_bars(s, tf, root)
    df = b.df
    return Coin(s, tf, df.index + pd.Timedelta(minutes=tf), *(df[k].to_numpy() for k in ("open", "high", "low", "close")),
                b.atr, b.funding_cum)


def trade(cn: Coin, e: int, d: int, dist: float, tp: bool, end: int):
    """One trade entered at bar e's open in direction d with stop distance `dist` (price units).
    Forced exit at bar end-1's close if still open. Returns (exit bar, per-bar P&L per unit of entry price for
    bars e..exit, fraction of the position still open at the end (for funding))."""
    entry = cn.o[e]
    x_stop0 = d * entry - dist
    x_tp = d * entry + 3 * dist
    n = end - e
    length = 256
    while True:
        j1 = e + min(length, n)
        xc = d * cn.c[e:j1]
        best_prev = np.r_[-np.inf, np.maximum.accumulate(xc)[:-1]]
        stop = np.maximum(x_stop0, best_prev - dist)
        adv = d * cn.l[e:j1] if d > 0 else -cn.h[e:j1]
        fav = d * cn.h[e:j1] if d > 0 else -cn.l[e:j1]
        hit = np.flatnonzero(adv <= stop)
        if len(hit) or j1 == end:
            break
        length *= 4
    xo = d * cn.o[e:j1]
    if len(hit):
        k = int(hit[0])
        x_exit = min(xo[k], stop[k]) if k > 0 else stop[k]
    else:
        k = j1 - e - 1
        x_exit = xc[k]
    path = xc[:k + 1].copy()
    path[k] = x_exit
    prev = np.r_[d * entry, path[:-1]]
    frac = np.ones(k + 1)
    if tp:
        th = np.flatnonzero(fav[:k + 1] >= x_tp)
        if len(th) and th[0] < k:
            t = int(th[0])
            frac[t + 1:] = 0.5
            pnl = (path - prev) * frac
            pnl[t] = 0.5 * (x_tp - prev[t]) + 0.5 * (path[t] - prev[t])
            return e + k, pnl / entry, 0.5
    return e + k, (path - prev) * frac / entry, 1.0


def _book(daily: dict, cn: Coin, e: int, j: int, pnl: np.ndarray, size: float, d: int, frac_end: float):
    days = cn.t_close[e:j + 1].floor("D")
    v = pnl * size
    v[0] -= COST * size   # entry
    v[-1] -= COST * size  # exits (the take-profit half is charged here too)
    fund = d * (cn.fcum[min(j + 1, len(cn.fcum) - 1)] - cn.fcum[e]) * size * (1 + frac_end) / 2
    v[-1] -= fund
    for day, x in zip(days, v):
        daily[day] = daily.get(day, 0.0) + x


# ---------- family 1: trend per coin ----------

def trend_signals(cn: Coin, n: int, hurst_min: float | None) -> np.ndarray:
    h, l, c = pd.Series(cn.h), pd.Series(cn.l), pd.Series(cn.c)
    up = c > h.rolling(n).max().shift()
    dn = c < l.rolling(n).min().shift()
    if hurst_min:
        ok = hurst(c) > hurst_min
        up, dn = up & ok, dn & ok
    return np.where(up, 1, np.where(dn, -1, 0))


def run_trend(coins: dict[str, Coin], ranks: pd.DataFrame, n, k, hurst_min, long_only, mom, tp, period):
    daily, trades = {}, 0
    for s, cn in coins.items():
        sig = trend_signals(cn, n, hurst_min)
        if long_only:
            sig = np.where(sig > 0, sig, 0)
        if mom:
            rk = ranks[s].reindex(ranks.index.union(cn.t_close)).ffill().reindex(cn.t_close).to_numpy() \
                if s in ranks else np.full(len(cn.c), np.nan)
            sig = np.where(((sig > 0) & (rk > 0.5)) | ((sig < 0) & (rk < 0.5)), sig, 0)
        inside = np.flatnonzero((cn.t_close >= period[0]) & (cn.t_close < period[1]))
        if not len(inside):
            continue
        end = inside[-1] + 1
        i = inside[0]
        cand = np.flatnonzero(sig[:end - 1])
        cand = cand[cand >= i]
        free = 0
        for i in cand:
            if i < free or not np.isfinite(cn.atr[i]) or cn.atr[i] <= 0:
                continue
            e, d, dist = i + 1, int(sig[i]), k * cn.atr[i]
            size = min(RISK / (dist / cn.o[e]), MAX_SIZE)
            j, pnl, fe = trade(cn, e, d, dist, tp, end)
            _book(daily, cn, e, j, pnl, size, d, fe)
            trades += 1
            free = j  # a new signal at the exit bar's close may enter at the next open
    return pd.Series(daily).sort_index(), trades


# ---------- family 2: weekly coin ranking ----------

def run_ranking(coins: dict[str, Coin], daily_close: pd.DataFrame, daily_atr: pd.DataFrame, weeks_back, k, period):
    """Every Monday 00:00: long the 4 best, short the 4 worst past `weeks_back`-week returns. Sized on k x daily
    ATR (k=None: sized on 3 x ATR, no stop, held to the next Monday). Stopped coins wait for the next Monday."""
    mondays = daily_close.index[(daily_close.index.dayofweek == 0) & (daily_close.index >= period[0])
                                & (daily_close.index < period[1])]
    ret = daily_close / daily_close.shift(7 * weeks_back) - 1
    daily, trades = {}, 0
    for t in mondays:
        r = ret.loc[t].dropna()
        r = r[[s for s in r.index if s in coins]]
        if len(r) < 10:
            continue
        o = r.sort_values()
        picks = {**{s: -1 for s in o.index[:4]}, **{s: 1 for s in o.index[-4:]}}
        t_end = min(t + pd.Timedelta("7D"), period[1])
        for s, d in picks.items():
            cn = coins[s]
            e = cn.t_close.searchsorted(t, side="right")  # first bar that opens at/after t
            end = cn.t_close.searchsorted(t_end, side="right")
            if e >= end:
                continue
            atr_d = daily_atr.loc[t, s]
            if not np.isfinite(atr_d):
                continue
            dist = (k or 3) * atr_d
            size = min(RISK / (dist / cn.o[e]), MAX_SIZE)
            j, pnl, fe = trade(cn, e, d, dist if k else 1e12, False, end)
            _book(daily, cn, e, j, pnl, size, d, fe)
            trades += 1
    return pd.Series(daily).sort_index(), trades


# ---------- scoring ----------

def score(r: pd.Series, period) -> dict:
    days = pd.date_range(period[0], period[1] - pd.Timedelta("1D"), freq="D")
    r = r.reindex(days, fill_value=0.0)
    eq = (1 + r).cumprod()
    yrs = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    return {"Sharpe": r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0.0,
            "Return/yr %": ((eq.iloc[-1]) ** (365 / len(r)) - 1) * 100,
            "Max DD %": (eq / eq.cummax() - 1).min() * 100,
            "Years positive": int((yrs > 0).sum()), "Years": len(yrs),
            "By year %": ", ".join(f"{y} {v * 100:+.0f}" for y, v in yrs.items())}


TREND_GRID = [dict(tf=tf, n=n, k=k, hurst_min=hm, long_only=lo, mom=mo, tp=tp)
              for tf in (240, 1440) for n in (20, 55, 100) for k in (2, 3, 4, 6) for hm in (None, 0.55)
              for lo in (False, True) for mo in (False, True) for tp in (False, True)]
RANK_GRID = [dict(weeks=w, k=k) for w in (1, 2, 3, 4) for k in (3, 4, None)]


def _label(cfg):
    if "weeks" in cfg:
        return f"Ranking {cfg['weeks']}w, stop {cfg['k'] or 'none'}"
    return (f"Trend {'4h' if cfg['tf'] == 240 else '1d'} N{cfg['n']} K{cfg['k']}"
            f"{' Hurst' if cfg['hurst_min'] else ''}{' long-only' if cfg['long_only'] else ''}"
            f"{' mom' if cfg['mom'] else ''}{' TP3R' if cfg['tp'] else ''}")


def all_runs(symbols, root: Path, period) -> tuple[dict, dict]:
    coins = {tf: {} for tf in (240, 1440)}
    for s in symbols:
        for tf in (240, 1440):
            try:
                coins[tf][s] = load_coin(s, tf, root)
            except SystemExit:
                pass
    d = coins[1440]
    daily_close = pd.DataFrame({s: pd.Series(cn.c, index=cn.t_close) for s, cn in d.items()})
    daily_atr = pd.DataFrame({s: pd.Series(cn.atr, index=cn.t_close) for s, cn in d.items()})
    ranks = (daily_close / daily_close.shift(21) - 1).rank(axis=1, pct=True)
    series, trades = {}, {}
    for cfg in TREND_GRID:
        cs = {s: cn for s, cn in coins[cfg["tf"]].items()}
        r, nt = run_trend(cs, ranks, cfg["n"], cfg["k"], cfg["hurst_min"], cfg["long_only"], cfg["mom"], cfg["tp"], period)
        series[_label(cfg)], trades[_label(cfg)] = r, nt
    for cfg in RANK_GRID:
        r, nt = run_ranking(coins[240], daily_close, daily_atr, cfg["weeks"], cfg["k"], period)
        series[_label(cfg)], trades[_label(cfg)] = r, nt
    return series, trades


def _neighbours(cfg):
    out = []
    if "weeks" in cfg:
        for w in (cfg["weeks"] - 1, cfg["weeks"] + 1):
            if 1 <= w <= 4:
                out.append({**cfg, "weeks": w})
        ks = [3, 4, None]
        i = ks.index(cfg["k"])
        out += [{**cfg, "k": ks[j]} for j in (i - 1, i + 1) if 0 <= j < 3]
        return out
    for key, vals in (("n", [20, 55, 100]), ("k", [2, 3, 4, 6])):
        i = vals.index(cfg[key])
        out += [{**cfg, key: vals[j]} for j in (i - 1, i + 1) if 0 <= j < len(vals)]
    return out


def run(symbols, root: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    dser, dtr = all_runs(symbols, root, DESIGN)
    dsc = {k: score(v, DESIGN) for k, v in dser.items()}
    chosen = {}
    for fam, grid in (("Trend", TREND_GRID), ("Ranking", RANK_GRID)):
        best = None
        for cfg in grid:
            lab = _label(cfg)
            s = dsc[lab]
            ok = s["Years positive"] >= 3 and all(dsc[_label(nb)]["Sharpe"] > 0 for nb in _neighbours(cfg))
            if ok and (best is None or s["Sharpe"] > dsc[best]["Sharpe"]):
                best = lab
        chosen[fam] = best
    log.info("chosen: %s", chosen)
    eser, etr = all_runs(symbols, root, EXAM)
    rows = []
    for name, src, per, tr in (("Design 2023-26", dser, DESIGN, dtr), ("Exam 2020-22", eser, EXAM, etr)):
        picks = {f"{fam}: {lab}": src[lab] for fam, lab in chosen.items() if lab}
        if len(picks) == 2:
            a, b = list(picks.values())
            idx = a.index.union(b.index)
            picks["50/50 mix"] = (a.reindex(idx, fill_value=0) + b.reindex(idx, fill_value=0)) / 2
        for lab, r in picks.items():
            s = score(r, per)
            rows.append({"Period": name, "Strategy": lab, "Sharpe": round(s["Sharpe"], 2),
                         "Return/yr %": round(s["Return/yr %"], 1), "Max DD %": round(s["Max DD %"], 1),
                         "By year %": s["By year %"]})
    table = pd.DataFrame([{"Version": k, "Trades (design)": dtr[k], **{f"Design {a}": (round(b, 2) if isinstance(b, float) else b)
                                                                    for a, b in dsc[k].items()},
                           **{f"Exam {a}": (round(b, 2) if isinstance(b, float) else b)
                              for a, b in score(eser[k], EXAM).items()}} for k in dser])
    table.sort_values("Design Sharpe", ascending=False).to_csv(report_dir / "all-versions.csv", index=False)
    pd.DataFrame({k: v for k, v in dser.items()}).to_csv(report_dir / "design-daily.csv.gz", compression="gzip")
    pd.DataFrame({k: v for k, v in eser.items()}).to_csv(report_dir / "exam-daily.csv.gz", compression="gzip")
    ex = table["Exam Sharpe"]
    lines = ["# A bot from scratch: results\n", "Plan: plan.md (written before running).\n",
             f"Chosen on the design period: {chosen}\n", "## Chosen designs\n", pd.DataFrame(rows).to_markdown(index=False),
             f"\n## Context: all {len(table)} versions on the exam period\n",
             f"Exam Sharpe: median {ex.median():.2f}, share above 0: {(ex > 0).mean():.0%}, share >= 0.5: {(ex >= 0.5).mean():.0%}. "
             f"Rank correlation design Sharpe vs exam Sharpe: {table['Design Sharpe'].corr(ex, method='spearman'):.2f}\n",
             "Top 15 by design Sharpe:\n",
             table.sort_values("Design Sharpe", ascending=False).head(15)[
                 ["Version", "Trades (design)", "Design Sharpe", "Design Return/yr %", "Design Max DD %",
                  "Design By year %", "Exam Sharpe", "Exam Return/yr %", "Exam Max DD %", "Exam By year %"]
             ].to_markdown(index=False)]
    path = report_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path
