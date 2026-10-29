"""Refining the 2h move-1 pullback (Understanding Crypto setup A) with the stablecoin filter.

Plan and pass rule: reports-2026-10-29-refine-move1/plan.md (written before running). Both 2020-22 and 2023-26 had
already been looked at, so tweaks are judged by a yearly walk-forward (pick on earlier years only) and by a coin
split (pick on 10 coins, check on the other 10).
"""
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .fresh import Coin, load_coin, score
from .intraday import COST, RISK, _row
from .moneyflows import _at, flow_signals, load_flows
from .understanding import zigzag

log = logging.getLogger(__name__)

TF = 120
FULL = (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))
EXITS = ("2R", "1.5R", "3R", "half at 1R, rest 3R", "trail swings")
DEPTHS = ("any", "38-79%")
BUFFERS = (0.1, 0.5)
XS = (1.5, 2.0, 2.5)
FLOWS = ("Stables 30d", "Stables 14d", "Stables + USDT.D 30d")
BASE = ("2R", "any", 0.1, 2.0, "Stables 30d")
GRID = list(itertools.product(EXITS, DEPTHS, BUFFERS, XS, FLOWS))
WF_YEARS = range(2021, 2027)


def a_signals(cn: Coin, x: float, depth: str, buf: float):
    """Move-1 pullback entries (see understanding.signals, setup A) with a depth rule and stop buffer."""
    n, c, a = len(cn.c), cn.c, cn.atr
    sig, stop = np.zeros(n, int), np.full(n, np.nan)
    piv = zigzag(c, a, x)
    for k in range(4, len(piv)):
        pm2, pm1, p0, p1, p2 = (q[0] for q in piv[k - 4:k + 1])
        d, i = -piv[k][1], piv[k][2]
        if not (d * c[p0] < d * c[pm2] and d * c[p1] > d * c[pm1] and d * c[p2] > d * c[p0]) or i + 1 >= n:
            continue
        if depth != "any":
            r = (c[p1] - c[p2]) / (c[p1] - c[p0])
            if not 0.382 <= r <= 0.786:
                continue
        ext = cn.l[p2:i + 1].min() if d > 0 else cn.h[p2:i + 1].max()
        sig[i], stop[i] = d, ext - d * buf * a[i]
    return sig, stop


def trail_levels(cn: Coin, x: float):
    """Latest confirmed small-swing low (bar low) and high (bar high) as known at each bar's close."""
    n = len(cn.c)
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    for p, kind, ci in zigzag(cn.c, cn.atr, x / 2):
        if kind < 0:
            lo[ci] = cn.l[p:ci + 1].min()
        else:
            hi[ci] = cn.h[p:ci + 1].max()
    return pd.Series(lo).ffill().to_numpy(), pd.Series(hi).ffill().to_numpy()


def _first(mask):
    f = np.flatnonzero(mask)
    return int(f[0]) if len(f) else None


def run_exit(cn: Coin, e: int, d: int, stop0: float, kind: str, buf: float, trail, end: int):
    """Returns (exit bar, gross return per unit of entry price, label)."""
    entry = cn.o[e]
    dist = d * (entry - stop0)
    xs0, xe = d * stop0, d * entry
    w = 256
    while True:
        j1 = min(e + w, end)
        lo = cn.l[e:j1] if d > 0 else -cn.h[e:j1]
        hi = cn.h[e:j1] if d > 0 else -cn.l[e:j1]
        xo = d * cn.o[e:j1]
        if kind == "trail swings":
            lvl = trail[0][e - 1:j1 - 1] if d > 0 else -trail[1][e - 1:j1 - 1]
            lvl = np.where(np.isfinite(lvl), lvl - buf * cn.atr[e - 1:j1 - 1], -np.inf)
            stp = np.maximum(xs0, np.maximum.accumulate(lvl))
            k = _first(lo <= stp)
            if k is not None:
                return e + k, ((min(xo[k], stp[k]) if k else stp[k]) - xe) / entry, "trail"
        elif kind == "half at 1R, rest 3R":
            ks = _first(lo <= xs0)
            k1 = _first(hi >= xe + dist)
            if ks is not None and (k1 is None or ks <= k1):
                x = min(xo[ks], xs0) if ks else xs0
                return e + ks, (x - xe) / entry, "stop"
            if k1 is not None:
                lo2, hi2, xo2 = lo[k1 + 1:], hi[k1 + 1:], xo[k1 + 1:]
                ks2, kt2 = _first(lo2 <= xe), _first(hi2 >= xe + 3 * dist)
                if ks2 is not None and (kt2 is None or ks2 <= kt2):
                    x2 = min(xo2[ks2], xe)
                    return e + k1 + 1 + ks2, (0.5 * dist + 0.5 * (x2 - xe)) / entry, "1R + entry"
                if kt2 is not None:
                    return e + k1 + 1 + kt2, 2 * dist / entry, "1R + 3R"
        else:
            m = float(kind[:-1])
            ks, kt = _first(lo <= xs0), _first(hi >= xe + m * dist)
            if ks is not None and (kt is None or ks <= kt):
                return e + ks, ((min(xo[ks], xs0) if ks else xs0) - xe) / entry, "stop"
            if kt is not None:
                return e + kt, m * dist / entry, "target"
        if j1 == end:
            return end - 1, (d * cn.c[end - 1] - xe) / entry, "open at end"
        w *= 4


def simulate(cn: Coin, sig, stop, bias, kind, buf, trail, period=FULL) -> pd.DataFrame:
    inside = np.flatnonzero((cn.t_close >= period[0]) & (cn.t_close < period[1]))
    i0, end = inside[0], inside[-1] + 1
    sig = np.where(sig * bias > 0, sig, 0)
    rows, free = [], 0
    for i in np.flatnonzero(sig[i0:end - 1]) + i0:
        if i < free:
            continue
        d, e = int(sig[i]), i + 1
        dist = d * (cn.o[e] - stop[i])
        if not np.isfinite(dist) or dist <= 0:
            continue
        j, gross, how = run_exit(cn, e, d, stop[i], kind, buf, trail, end)
        funding = d * (cn.fcum[min(j + 1, len(cn.fcum) - 1)] - cn.fcum[e])
        net = gross - 2 * COST - funding
        size = min(RISK / (dist / cn.o[e]), 10.0)
        rows.append((cn.t_close[j], d, how, gross, net, size, dist / cn.o[e], (j - e + 1) * cn.tf / 60))
        free = j
    return pd.DataFrame(rows, columns=["exit_time", "dir", "exit", "gross", "net", "size", "target_move", "hours"])


def bias_series(flows: pd.DataFrame) -> dict[str, pd.Series]:
    fs = flow_signals(flows)
    both = (fs[("Stables", 30)] + fs[("USDT.D", 30)].reindex(fs[("Stables", 30)].index).fillna(0)) / 2
    return {"Stables 30d": fs[("Stables", 30)], "Stables 14d": fs[("Stables", 14)],
            "Stables + USDT.D 30d": both.where(both.abs() == 1, 0)}


def all_trades(coins, flows) -> dict:
    """{variant: {symbol: trades over 2020-01..2026-09}}."""
    biases = bias_series(flows)
    out = {g: {} for g in GRID}
    for sym, cn in coins.items():
        bias = {k: _at(v, cn.t_close) for k, v in biases.items()}
        trails = {x: trail_levels(cn, x) for x in XS}
        for depth, buf, x in itertools.product(DEPTHS, BUFFERS, XS):
            sig, stop = a_signals(cn, x, depth, buf)
            for kind, fl in itertools.product(EXITS, FLOWS):
                out[(kind, depth, buf, x, fl)][sym] = simulate(cn, sig, stop, bias[fl], kind, buf, trails[x]).assign(coin=sym)
        log.info("done %s", sym)
    return out


def _cat(per_coin: dict, syms=None) -> pd.DataFrame:
    parts = [t for s, t in per_coin.items() if (syms is None or s in syms) and len(t)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _label(g):
    kind, depth, buf, x, fl = g
    return f"{kind}, depth {depth}, buffer {buf} ATR, swings {x} ATR, {fl}"


def run(symbols, root: Path, flow_root: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    coins = {}
    for s in symbols:
        try:
            coins[s] = load_coin(s, TF, root)
        except SystemExit:
            pass
    trades = all_trades(coins, load_flows(flow_root))
    full = {g: _cat(trades[g]) for g in GRID}

    def per(t, a, b):
        p = (pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"))
        return _row(t[(t.exit_time >= p[0]) & (t.exit_time < p[1])] if len(t) else t, p)

    def fmt(d):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}

    # 1. each tweak alone
    one = []
    for i, name in enumerate(("Exit", "Depth", "Buffer", "Swing X", "Flow filter")):
        opts = (EXITS, DEPTHS, BUFFERS, XS, FLOWS)[i]
        for o in opts:
            g = tuple(o if j == i else BASE[j] for j in range(5))
            one.append({"Changed": name, "Value": o, **{f"2023-26 {k}": v for k, v in fmt(per(full[g], "2023-01-01", "2026-10-01")).items()},
                        **{f"2020-22 {k}": v for k, v in fmt(per(full[g], "2020-01-01", "2023-01-01")).items()}})
    one = pd.DataFrame(one)
    # 2. yearly walk-forward
    wf, picks = [], []
    for y in WF_YEARS:
        train = {g: per(full[g], "2020-01-01", f"{y}-01-01")["Sharpe"] for g in GRID}
        best = max(GRID, key=lambda g: train[g])
        picks.append({"Year": y, "Picked on earlier years": _label(best), "Train Sharpe": round(train[best], 2)})
        t = full[best]
        wf.append(t[(t.exit_time >= pd.Timestamp(f"{y}-01-01", tz="UTC")) & (t.exit_time < pd.Timestamp(f"{y + 1}-01-01", tz="UTC"))])
    wf = pd.concat(wf, ignore_index=True)
    wf_s = per(wf, "2021-01-01", "2026-10-01")
    base_s = per(full[BASE], "2021-01-01", "2026-10-01")
    # 3. coin split
    syms = sorted(coins)
    half_a, half_b = set(syms[0::2]), set(syms[1::2])
    pick_a = max(GRID, key=lambda g: per(_cat(trades[g], half_a), "2020-01-01", "2026-10-01")["Sharpe"])
    pick_b = max(GRID, key=lambda g: per(_cat(trades[g], half_b), "2020-01-01", "2026-10-01")["Sharpe"])
    split = []
    for nm, pick, other in (("picked on A, checked on B", pick_a, half_b), ("picked on B, checked on A", pick_b, half_a)):
        for lab, g in (("picked", pick), ("base", BASE)):
            split.append({"Check": nm, "Version": lab + ": " + _label(g),
                          **fmt(per(_cat(trades[g], other), "2020-01-01", "2026-10-01"))})
    table = pd.DataFrame([{"Version": _label(g), **{f"2023-26 {k}": v for k, v in fmt(per(full[g], "2023-01-01", "2026-10-01")).items()},
                           **{f"2020-22 {k}": v for k, v in fmt(per(full[g], "2020-01-01", "2023-01-01")).items()}} for g in GRID])
    table.sort_values("2023-26 Sharpe", ascending=False).to_csv(report_dir / "all-variants.csv", index=False)
    one.to_csv(report_dir / "one-change-at-a-time.csv", index=False)
    cols = ["Sharpe", "Return/yr %", "Max DD %", "By year %", "Trades", "Win %"]
    lines = ["# Refining the 2h move-1 pullback: results\n", "Plan: plan.md (written before running).\n",
             "## Yearly walk-forward (2021 to Sep 2026, each year picked on earlier years only)\n",
             pd.DataFrame([{"Version": "walk-forward picks", **{k: fmt(wf_s)[k] for k in cols}},
                           {"Version": "base, fixed", **{k: fmt(base_s)[k] for k in cols}}]).to_markdown(index=False),
             "\n", pd.DataFrame(picks).to_markdown(index=False),
             "\n## Coin split (2020 to Sep 2026)\n", f"Half A: {', '.join(sorted(half_a))}. Half B: {', '.join(sorted(half_b))}.\n",
             pd.DataFrame(split)[["Check", "Version"] + cols].to_markdown(index=False),
             "\n## One change at a time from the base\n",
             one[["Changed", "Value", "2023-26 Sharpe", "2023-26 Return/yr %", "2023-26 Max DD %", "2020-22 Sharpe",
                  "2020-22 Return/yr %", "2020-22 Max DD %"]].to_markdown(index=False)]
    path = report_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path
