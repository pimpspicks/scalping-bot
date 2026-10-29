"""The Understanding Crypto "three-move" method: move-1 pullbacks, move-2 shallow pullbacks, move-3 divergence fades.

Plan and pass rule: reports-2026-10-29-understanding-crypto/plan.md (written before running).

Moves come from a zigzag on closes (his "line chart"): a swing is confirmed once the close has moved X x ATR(14)
the other way, and is only used from the bar that confirms it. Entries fill at the next bar's open. Stop and
target are price levels from the swings; a bar touching both counts as the stop, a stop gapped through fills at
the open, and the trade is closed at the last allowed bar's close. Costs `COST` per side plus funding,
`RISK` of the account per trade.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .fresh import DESIGN, EXAM, Coin, load_coin
from .indicators import _rsi
from .intraday import COST, MAX_SIZE, RISK, _row

log = logging.getLogger(__name__)

SETUPS = ("A move-1 pullback", "B move-2 shallow pullback", "C move-3 RSI divergence", "C2 move-3 stochastic divergence")
TFS = {15: 20, 30: 10, 120: 36, 240: 18}  # bar minutes -> max hold in bars (5h same-day, 3 days multi-day)
GROUP = {15: "same-day", 30: "same-day", 120: "multi-day", 240: "multi-day"}
XS = (2.0, 3.0, 4.0)
TARGETS = ("structure", "2R")


def zigzag(c: np.ndarray, atr: np.ndarray, x: float) -> list[tuple[int, int, int]]:
    """Swings on closes as (bar of the swing, +1 high / -1 low, bar where it was confirmed)."""
    out, d, hi, lo, ext = [], 0, None, None, None
    for i in range(len(c)):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if d == 0:
            hi = i if hi is None or c[i] > c[hi] else hi
            lo = i if lo is None or c[i] < c[lo] else lo
            if c[hi] - c[i] >= x * a:
                out.append((hi, 1, i))
                d, ext = -1, i
            elif c[i] - c[lo] >= x * a:
                out.append((lo, -1, i))
                d, ext = 1, i
        elif d * c[i] > d * c[ext]:
            ext = i
        elif d * (c[ext] - c[i]) >= x * a:
            out.append((ext, d, i))
            d, ext = -d, i
    return out


def stoch_k(cn: Coin, n=14, smooth=3) -> np.ndarray:
    h = pd.Series(cn.h).rolling(n).max()
    l = pd.Series(cn.l).rolling(n).min()
    k = 100 * (pd.Series(cn.c) - l) / (h - l)
    return k.rolling(smooth).mean().to_numpy()


def signals(cn: Coin, x: float, setup: str, oi: np.ndarray | None = None):
    """Arrays over bars: direction at the confirming bar, stop price, structure-target price.
    oi (setup C only): open interest per bar; keep only moves 3 during which it rose."""
    n = len(cn.c)
    sig, stop, tgt = np.zeros(n, int), np.full(n, np.nan), np.full(n, np.nan)
    c, a = cn.c, cn.atr
    piv = zigzag(c, a, x)
    if setup.startswith("C"):
        osc = _rsi(pd.Series(c)).to_numpy() if setup.startswith("C ") else stoch_k(cn)
        for k in range(5, len(piv)):
            p = [q[0] for q in piv[k - 5:k + 1]]
            d = piv[k][1]  # +1: H5 is a high -> short
            i = piv[k][2]
            v = d * c[p]  # in direction space: odd indices are the swing extremes of the move
            if not (v[1] < v[3] < v[5] and v[0] < v[2] < v[4]):
                continue
            if not (d * osc[p[5]] < d * osc[p[3]]):
                continue
            if oi is not None and not oi[p[5]] > oi[p[4]]:
                continue
            if i + 1 >= n:
                continue
            ext = (cn.h[p[5]:i + 1].max() if d > 0 else cn.l[p[5]:i + 1].min())
            sig[i], stop[i], tgt[i] = -d, ext + d * 0.1 * a[i], c[p[4]]
        return sig, stop, tgt
    # A and B: swings H-1, L0, H1, L2 (long; mirrored for short)
    small = zigzag(c, a, x / 2) if setup.startswith("B") else None
    s_ptr = 0
    for k in range(4, len(piv)):
        pm2, pm1, p0, p1, p2 = (q[0] for q in piv[k - 4:k + 1])
        d = -piv[k][1]  # the last swing is L2 (a low) -> long
        i = piv[k][2]
        v = lambda j: d * c[j]
        if not (v(p0) < v(pm2) and v(p1) > v(pm1) and v(p2) > v(p0)):
            continue
        proj = c[p2] + (c[p1] - c[p0])
        if setup.startswith("A"):
            if i + 1 < n:
                ext = cn.l[p2:i + 1].min() if d > 0 else cn.h[p2:i + 1].max()
                sig[i], stop[i], tgt[i] = d, ext - d * 0.1 * a[i], proj
            continue
        # B: until the next big swing is confirmed, buy small pullbacks once price has closed beyond H1
        end = piv[k + 1][2] if k + 1 < len(piv) else n
        beyond = np.flatnonzero(d * c[i:end] > d * c[p1])
        if not len(beyond):
            continue
        b0 = i + beyond[0]
        while s_ptr < len(small) and small[s_ptr][2] <= b0:
            s_ptr += 1
        j = s_ptr
        while j < len(small) and small[j][2] < end:
            sp, sd, si = small[j]
            if sd == -d and sp > b0 and d * c[sp] > d * c[p2] and si + 1 < n:
                ext = cn.l[sp:si + 1].min() if d > 0 else cn.h[sp:si + 1].max()
                sig[si], stop[si], tgt[si] = d, ext - d * 0.1 * a[si], proj
            j += 1
    return sig, stop, tgt


def simulate(cn: Coin, sig, stop, tgt, target: str, hold: int | None, period) -> pd.DataFrame:
    inside = np.flatnonzero((cn.t_close >= period[0]) & (cn.t_close < period[1]))
    if not len(inside):
        return pd.DataFrame()
    i0, end = inside[0], inside[-1] + 1
    rows, free = [], 0
    for i in np.flatnonzero(sig[i0:end - 1]) + i0:
        if i < free:
            continue
        d, e = int(sig[i]), i + 1
        entry = cn.o[e]
        dist = d * (entry - stop[i])
        if not np.isfinite(dist) or dist <= 0:
            continue
        move = d * (tgt[i] - entry) if target == "structure" else 2 * dist
        if not np.isfinite(move) or move < dist:
            continue
        x_stop, x_tgt = d * entry - dist, d * entry + move
        limit = min(e + hold, end) if hold else end  # no hold limit: until stop, target or the period's end
        w = 256
        while True:
            last = min(e + w, limit) - 1
            x_lo = cn.l[e:last + 1] if d > 0 else -cn.h[e:last + 1]
            x_hi = cn.h[e:last + 1] if d > 0 else -cn.l[e:last + 1]
            s = np.flatnonzero(x_lo <= x_stop)
            t = np.flatnonzero(x_hi >= x_tgt)
            if len(s) or len(t) or last == limit - 1:
                break
            w *= 4
        s0 = s[0] if len(s) else np.inf
        t0 = t[0] if len(t) else np.inf
        if len(s) and s0 <= t0:
            k = int(s0)
            x_exit, kind = (min(d * cn.o[e + k], x_stop) if k > 0 else x_stop), "stop"
        elif len(t):
            k, x_exit, kind = int(t0), x_tgt, "target"
        else:
            k, x_exit, kind = last - e, d * cn.c[last], "time"
        j = e + k
        gross = (x_exit - d * entry) / entry
        funding = d * (cn.fcum[min(j + 1, len(cn.fcum) - 1)] - cn.fcum[e])
        net = gross - 2 * COST - funding
        size = min(RISK / (dist / entry), MAX_SIZE * 10)
        rows.append((cn.t_close[j], d, kind, gross, net, size, move / entry, (j - e + 1) * cn.tf / 60))
        free = j
    return pd.DataFrame(rows, columns=["exit_time", "dir", "exit", "gross", "net", "size", "target_move", "hours"])


GRID = [(s, tf, x, t) for s in SETUPS for tf in TFS for x in XS for t in TARGETS]


def _label(s, tf, x, t):
    tfs = f"{tf}m" if tf < 60 else f"{tf // 60}h"
    return f"{s}, {tfs} swings {x:.0f} ATR, target {t}"


def all_runs(coins, period, cap=True):
    out = {}
    for tf in TFS:
        for s in SETUPS:
            for x in XS:
                sigs = {sym: signals(cn, x, s) for sym, cn in coins[tf].items()}
                for t in TARGETS:
                    trades = []
                    for sym, cn in coins[tf].items():
                        tr = simulate(cn, *sigs[sym], t, TFS[tf] if cap else None, period)
                        if len(tr):
                            trades.append(tr.assign(coin=sym))
                    out[(s, tf, x, t)] = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
        log.info("done %sm", tf)
    return out


def run(symbols, root: Path, report_dir: Path, cap=True) -> Path:
    """cap=False: no time limit, trades end only at the stop or target."""
    report_dir.mkdir(parents=True, exist_ok=True)
    coins = {tf: {} for tf in TFS}
    for sym in symbols:
        for tf in TFS:
            try:
                coins[tf][sym] = load_coin(sym, tf, root)
            except SystemExit:
                pass
    design = all_runs(coins, DESIGN, cap)
    dsc = {g: _row(t, DESIGN) for g, t in design.items()}
    exam = all_runs(coins, EXAM, cap)
    esc = {g: _row(t, EXAM) for g, t in exam.items()}
    chosen = {}
    for grp in ("same-day", "multi-day"):
        best = None
        for g in GRID:
            s, tf, x, t = g
            if GROUP[tf] != grp:
                continue
            nbs = [(s, tf, x, tt) for tt in TARGETS if tt != t] + [(s, tf, xx, t) for xx in XS if abs(xx - x) == 1]
            ok = dsc[g]["Years positive"] >= 3 and all(dsc[nb]["Sharpe"] > 0 for nb in nbs)
            if ok and (best is None or dsc[g]["Sharpe"] > dsc[best]["Sharpe"]):
                best = g
        chosen[grp] = best
        log.info("chosen %s: %s", grp, best)

    def fmt(d):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}

    table = pd.DataFrame([{"Version": _label(*g), "Setup": g[0], "Timeframe": g[1], "Group": GROUP[g[1]], "X": g[2],
                           "Target": g[3], **{f"Design {k}": v for k, v in fmt(dsc[g]).items()},
                           **{f"Exam {k}": v for k, v in fmt(esc[g]).items()}} for g in GRID])
    for p, res in (("Design", design), ("Exam", exam)):
        table[f"{p} Avg hours held"] = [round(res[g].hours.mean(), 1) if len(res[g]) else np.nan for g in GRID]
    table = table.sort_values("Design Sharpe", ascending=False)
    table.to_csv(report_dir / "all-versions.csv", index=False)
    g1 = ["Design Avg gross (bps)", "Design Avg net per trade (bps of price)", "Exam Avg gross (bps)",
          "Exam Avg net per trade (bps of price)"]
    cols = ["Version", "Design Trades", "Design Win %", "Design Avg gross (bps)", "Design Avg net per trade (bps of price)",
            "Design Sharpe", "Design Return/yr %", "Design By year %", "Exam Sharpe", "Exam Return/yr %", "Exam By year %"]
    lines = ["# Understanding Crypto three-move method: results\n", "Plan: plan.md (written before running).\n"]
    for grp, best in chosen.items():
        lines.append(f"**Chosen, {grp}:** {_label(*best) if best else 'none eligible'}\n")
        if best:
            lines += [pd.DataFrame([{"Period": "Design 2023-26", **fmt(dsc[best])},
                                    {"Period": "Exam 2020-22", **fmt(esc[best])}]).to_markdown(index=False), ""]
    lines += [f"\nDesign: {(table['Design Sharpe'] > 0).sum()} of {len(table)} versions with Sharpe > 0. "
              f"Exam: {(table['Exam Sharpe'] > 0).sum()}.\n",
              "## Average per trade by setup and group (bps of price)\n",
              table.groupby(["Group", "Setup"])[g1].mean().round(1).to_markdown(),
              "\n## Average per trade by timeframe (bps of price)\n", table.groupby("Timeframe")[g1].mean().round(1).to_markdown(),
              "\n## Top 15 by design Sharpe\n", table.head(15)[cols].to_markdown(index=False)]
    path = report_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def oi_side_check(symbols, root: Path, metrics_root: Path) -> pd.DataFrame:
    """Design period only: setup C with and without the rising-open-interest condition."""
    from .binance_metrics import load as load_metrics
    rows = []
    for tf in TFS:
        for s in SETUPS[2:]:
            for x in XS:
                for t in TARGETS:
                    res = {"without": [], "with OI rising": []}
                    for sym in symbols:
                        try:
                            cn = load_coin(sym, tf, root)
                            m = load_metrics(sym, metrics_root)
                        except (SystemExit, FileNotFoundError, ValueError):
                            continue
                        oi = m["oi"].reindex(cn.t_close, method="ffill").to_numpy()
                        for k, o in (("without", None), ("with OI rising", oi)):
                            tr = simulate(cn, *signals(cn, x, s, o), t, TFS[tf], DESIGN)
                            if len(tr):
                                res[k].append(tr)
                    for k, v in res.items():
                        tr = pd.concat(v, ignore_index=True) if v else pd.DataFrame()
                        rows.append({"Version": _label(s, tf, x, t), "OI filter": k, **_row(tr, DESIGN)})
    return pd.DataFrame(rows)


# ---------- follow-up: no time limit, and the higher timeframe's move count as a filter ----------

HTF = {15: 1440, 30: 1440, 60: 1440, 120: 1440, 240: 1440}  # trading chart -> the chart he takes his bias from


def move_counts(cn: Coin, x: float = 3.0) -> pd.DataFrame:
    """Completed moves of the current trend on this chart, known from each swing's confirming bar: `up` counts
    consecutive higher highs, `dn` consecutive lower lows (a break the other way resets the count)."""
    piv = zigzag(cn.c, cn.atr, x)
    up = dn = 0
    last = {1: None, -1: None}
    rows = []
    for p, kind, ci in piv:
        prev = last[kind]
        if prev is not None:
            if kind == 1:
                up = up + 1 if cn.c[p] > prev else 0
                if cn.c[p] > prev:
                    dn = 0
            else:
                dn = dn + 1 if cn.c[p] < prev else 0
                if cn.c[p] < prev:
                    up = 0
        last[kind] = cn.c[p]
        rows.append((cn.t_close[ci], up, dn))
    return pd.DataFrame(rows, columns=["t", "up", "dn"]).drop_duplicates("t", keep="last").set_index("t")


def htf_filter(sig: np.ndarray, setup: str, up: np.ndarray, dn: np.ndarray) -> np.ndarray:
    """A/B: trade with the bigger chart's trend while it has done 1-2 moves. C: fade only when the bigger chart
    has done 3+ moves the same way, or is already trending the other way."""
    if setup.startswith("C"):
        ok_long = (dn >= 3) | ((up >= 1) & (dn == 0))
        ok_short = (up >= 3) | ((dn >= 1) & (up == 0))
    else:
        ok_long = (up >= 1) & (up <= 2) & (dn == 0)
        ok_short = (dn >= 1) & (dn <= 2) & (up == 0)
    return np.where(((sig > 0) & ok_long) | ((sig < 0) & ok_short), sig, 0)


TFS2 = (15, 30, 60, 120, 240)
GROUP2 = {**GROUP, 60: "same-day"}
GRID2 = [(s, tf, x, t, f) for s in SETUPS for tf in TFS2 for x in XS for t in TARGETS for f in (False, True)]


def _label2(s, tf, x, t, f):
    return _label(s, tf, x, t) + (", only with the daily bias" if f else "")


def all_runs2(coins, htf, period):
    out = {}
    for tf in TFS2:
        for s in SETUPS:
            for x in XS:
                sigs = {}
                for sym, cn in coins[tf].items():
                    sig, st, tg = signals(cn, x, s)
                    mc = htf[HTF[tf]][sym].reindex(cn.t_close, method="ffill").fillna(0)
                    sigs[sym] = {False: (sig, st, tg), True: (htf_filter(sig, s, mc.up.to_numpy(), mc.dn.to_numpy()), st, tg)}
                for t in TARGETS:
                    for f in (False, True):
                        trades = []
                        for sym, cn in coins[tf].items():
                            tr = simulate(cn, *sigs[sym][f], t, None, period)
                            if len(tr):
                                trades.append(tr.assign(coin=sym))
                        out[(s, tf, x, t, f)] = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
        log.info("done %sm", tf)
    return out


def run2(symbols, root: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    coins = {tf: {} for tf in TFS2}
    htf = {1440: {}}
    for sym in symbols:
        try:
            for tf in TFS2:
                coins[tf][sym] = load_coin(sym, tf, root)
            for tf in htf:
                htf[tf][sym] = move_counts(load_coin(sym, tf, root))
        except SystemExit:
            pass
    design = all_runs2(coins, htf, DESIGN)
    dsc = {g: _row(t, DESIGN) for g, t in design.items()}
    exam = all_runs2(coins, htf, EXAM)
    esc = {g: _row(t, EXAM) for g, t in exam.items()}
    chosen = {}
    for grp in ("same-day", "multi-day"):
        best = None
        for g in GRID2:
            s, tf, x, t, f = g
            if GROUP2[tf] != grp:
                continue
            nbs = [(s, tf, x, tt, f) for tt in TARGETS if tt != t] + [(s, tf, xx, t, f) for xx in XS if abs(xx - x) == 1]
            ok = dsc[g]["Years positive"] >= 3 and all(dsc[nb]["Sharpe"] > 0 for nb in nbs)
            if ok and (best is None or dsc[g]["Sharpe"] > dsc[best]["Sharpe"]):
                best = g
        chosen[grp] = best
        log.info("chosen %s: %s", grp, best)

    def fmt(d):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}

    table = pd.DataFrame([{"Version": _label2(*g), "Setup": g[0], "Timeframe": g[1], "Group": GROUP2[g[1]], "X": g[2],
                           "Target": g[3], "Bigger-chart filter": g[4],
                           **{f"Design {k}": v for k, v in fmt(dsc[g]).items()},
                           **{f"Exam {k}": v for k, v in fmt(esc[g]).items()}} for g in GRID2])
    for p, res in (("Design", design), ("Exam", exam)):
        table[f"{p} Avg hours held"] = [round(res[g].hours.mean(), 1) if len(res[g]) else np.nan for g in GRID2]
        table[f"{p} Median hours held"] = [round(res[g].hours.median(), 1) if len(res[g]) else np.nan for g in GRID2]
    table = table.sort_values("Design Sharpe", ascending=False)
    table.to_csv(report_dir / "all-versions.csv", index=False)
    g1 = ["Design Avg gross (bps)", "Design Avg net per trade (bps of price)", "Exam Avg gross (bps)",
          "Exam Avg net per trade (bps of price)", "Design Median hours held"]
    cols = ["Version", "Design Trades", "Design Win %", "Design Avg net per trade (bps of price)", "Design Sharpe",
            "Design Return/yr %", "Design Max DD %", "Design By year %", "Design Median hours held",
            "Exam Sharpe", "Exam Return/yr %", "Exam By year %"]
    lines = ["# Three-move method, no time limit, with and without the daily chart's bias: results\n"]
    for grp, best in chosen.items():
        lines.append(f"**Chosen, {grp}:** {_label2(*best) if best else 'none eligible'}\n")
        if best:
            lines += [pd.DataFrame([{"Period": "Design 2023-26", **fmt(dsc[best])},
                                    {"Period": "Exam 2020-22", **fmt(esc[best])}]).to_markdown(index=False), ""]
    pos = table.groupby(["Group", "Bigger-chart filter"]).apply(
        lambda d: pd.Series({"Versions": len(d), "Design Sharpe > 0": (d["Design Sharpe"] > 0).sum(),
                             "Exam Sharpe > 0": (d["Exam Sharpe"] > 0).sum()}))
    lines += ["\n## Versions with Sharpe > 0\n", pos.to_markdown(),
              "\n## Average per trade by setup, group and filter (bps of price)\n",
              table.groupby(["Group", "Setup", "Bigger-chart filter"])[g1].mean().round(1).to_markdown(),
              "\n## Top 20 by design Sharpe\n", table.head(20)[cols].to_markdown(index=False)]
    path = report_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path
