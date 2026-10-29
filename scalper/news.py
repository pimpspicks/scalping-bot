"""US macro releases (FOMC, CPI, jobs report) and crypto news volume as intraday setups or filters (max 5h hold).

Plan and pass rule: reports-2026-10-29-news/plan.md (written before running). Trade simulation, sizing and
costs are those of `intraday.simulate`; selection and exam rules as in `sessions`.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import econ_calendar, intraday, sessions
from .fresh import DESIGN, EXAM, Coin, load_coin
from .intraday import _row, ranking_filter, simulate

log = logging.getLogger(__name__)

EVENTS = ("FOMC", "CPI", "Jobs report", "Any release")
ENTRIES = ("Release range break", "Follow first move", "Fade first move", "One-hour continuation")
EXITS = {"chart stop, target 24h high/low": ("swing", "level"), "chart stop, target 2R": ("swing", 2.0),
         "stop 2 ATR, target 2R": (2.0, 2.0), "stop 3 ATR, target 2R": (3.0, 2.0)}
SIBLING = {"chart stop, target 24h high/low": "chart stop, target 2R", "chart stop, target 2R": "chart stop, target 24h high/low",
           "stop 2 ATR, target 2R": "stop 3 ATR, target 2R", "stop 3 ATR, target 2R": "stop 2 ATR, target 2R"}
DRIFT = "Pre-release drift (long 5h before, exit at release)"
GRID = [(e, ev, x) for e in ENTRIES for ev in EVENTS for x in EXITS] + [(DRIFT, ev, "stop 3 ATR, exit at release")
                                                                         for ev in EVENTS]


def _label(e, ev, x):
    return f"{ev}: {e}, {x}"


def release_signals(cn: Coin, times: pd.DatetimeIndex) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per entry: (signal array, stop level array) for release times `times` (UTC)."""
    n = len(cn.c)
    out = {e: (np.zeros(n, int), np.full(n, np.nan)) for e in ENTRIES + (DRIFT,)}
    o, h, l, c, atr = cn.o, cn.h, cn.l, cn.c, cn.atr
    for T in times:
        iT = int(np.searchsorted(cn.t_close, T))  # bar closing at T
        if iT >= n or cn.t_close[iT] != T or iT < 20 or iT + 13 >= n or not np.isfinite(atr[iT]):
            continue
        a = atr[iT]
        hi, lo = h[iT + 1:iT + 3].max(), l[iT + 1:iT + 3].min()
        for i in range(iT + 3, iT + 13):
            d = 1 if c[i] > hi else -1 if c[i] < lo else 0
            if d:
                sig, lvl = out["Release range break"]
                sig[i], lvl[i] = d, lo if d > 0 else hi
                break
        i1 = iT + 1
        mv = c[i1] - c[iT]
        if abs(mv) > a:
            d = int(np.sign(mv))
            sig, lvl = out["Follow first move"]
            sig[i1], lvl[i1] = d, l[i1] if d > 0 else h[i1]
            sig, lvl = out["Fade first move"]
            sig[i1], lvl[i1] = -d, h[i1] if d > 0 else l[i1]
        i4 = iT + 4
        mv = c[i4] - c[iT]
        if abs(mv) > a:
            d = int(np.sign(mv))
            sig, lvl = out["One-hour continuation"]
            sig[i4], lvl[i4] = d, l[iT + 1:i4 + 1].min() if d > 0 else h[iT + 1:i4 + 1].max()
        out[DRIFT][0][iT - intraday.HOLD] = 1
    return out


def event_times(cal: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
    t = {ev: pd.DatetimeIndex(cal.time[cal.event == ev]) for ev in EVENTS[:3]}
    t["Any release"] = pd.DatetimeIndex(cal.time).sort_values()
    return t


def release_runs(coins, times, period):
    prep = {ev: {s: release_signals(cn, ts) for s, cn in coins.items()} for ev, ts in times.items()}
    out = {}
    for e, ev, x in GRID:
        k, r = (3.0, 100.0) if e == DRIFT else EXITS[x]
        trades = []
        for s, cn in coins.items():
            sig, lvl = prep[ev][s][e]
            t = simulate(cn, sig, k, r, period, lvl)
            if len(t):
                trades.append(t.assign(coin=s))
        out[(e, ev, x)] = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return out


def event_t(t: pd.DataFrame) -> float:
    """t-stat of the average net P&L per release (trades grouped by the release they belong to)."""
    if len(t) < 3:
        return np.nan
    per = (t.net * t["size"]).groupby(t.signal_time.dt.floor("D")).sum()
    return per.mean() / per.std() * np.sqrt(len(per)) if per.std() > 0 else np.nan


def _pick(dsc, grid, sibling_of):
    best = None
    for g in grid:
        sib = sibling_of(g)
        ok = dsc[g]["Years positive"] >= 3 and (sib is None or dsc[sib]["Sharpe"] > 0)
        if ok and (best is None or dsc[g]["Sharpe"] > dsc[best]["Sharpe"]):
            best = g
    return best


def _exam_pass(s) -> bool:
    return s["Return/yr %"] > 0 and s["Sharpe"] >= 0.5 and s["Years positive"] >= 2 and s["Max DD %"] > -40


def _fmt(d):
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


def moves(coins, times, period) -> pd.DataFrame:
    """Price range in the 5 hours after each release vs the same clock time on other weekdays."""
    rows = []
    for ev in EVENTS[:3]:
        ts = times[ev][(times[ev] >= period[0]) & (times[ev] < period[1])]
        hh, mm = econ_calendar.TIMES[ev]
        ev_days = set(ts.tz_convert(econ_calendar.TZ).date)
        for s, cn in coins.items():
            hi = pd.Series(cn.h).rolling(intraday.HOLD).max().shift(-intraday.HOLD).to_numpy()
            lo = pd.Series(cn.l).rolling(intraday.HOLD).min().shift(-intraday.HOLD).to_numpy()
            ny = cn.t_close.tz_convert(econ_calendar.TZ)
            at = (ny.hour == hh) & (ny.minute == mm) & (ny.dayofweek < 5) & (cn.t_close >= period[0]) & (cn.t_close < period[1])
            on = pd.Index(ny.date).isin(ev_days)
            for tag, m in (("release", at & on), ("other weekdays", at & ~on)):
                rows += [(ev, tag, (hi[i] - lo[i]) / cn.c[i]) for i in np.flatnonzero(m) if np.isfinite(hi[i])]
    df = pd.DataFrame(rows, columns=["Event", "Days", "range"])
    g = df.groupby(["Event", "Days"]).range
    return pd.DataFrame({"Median 5h high-low range %": g.median() * 100, "Share >= 2% %": g.apply(lambda x: (x >= .02).mean() * 100),
                         "Share >= 4% %": g.apply(lambda x: (x >= .04).mean() * 100), "Windows": g.size()}).round(1)


def run(symbols, root: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    coins = {}
    for s in symbols:
        try:
            coins[s] = load_coin(s, 15, root)
        except SystemExit:
            pass
    cal = econ_calendar.load()
    times = event_times(cal)
    lines = ["# News and economic calendar: results\n", "Plan: plan.md (written before running).\n"]

    # Part A
    design = release_runs(coins, times, DESIGN)
    dsc = {g: _row(t, DESIGN) for g, t in design.items()}
    best = _pick(dsc, GRID, lambda g: None if g[0] == DRIFT else (g[0], g[1], SIBLING[g[2]]))
    exam = release_runs(coins, times, EXAM)
    esc = {g: _row(t, EXAM) for g, t in exam.items()}
    tA = pd.DataFrame([{"Version": _label(*g), "Entry": g[0], "Event": g[1], "Exit": g[2],
                        **{f"Design {k}": v for k, v in _fmt(dsc[g]).items()}, "Design t by release": round(event_t(design[g]), 2),
                        **{f"Exam {k}": v for k, v in _fmt(esc[g]).items()}, "Exam t by release": round(event_t(exam[g]), 2)}
                       for g in GRID]).sort_values("Design Sharpe", ascending=False)
    tA.to_csv(report_dir / "part-a-release-trades.csv", index=False)
    gcols = ["Design Avg gross (bps)", "Design Avg net per trade (bps of price)", "Exam Avg gross (bps)",
             "Exam Avg net per trade (bps of price)"]
    cols = ["Version", "Design Trades", "Design Win %", "Design Avg gross (bps)", "Design Avg net per trade (bps of price)",
            "Design Sharpe", "Design t by release", "Design By year %", "Exam Sharpe", "Exam By year %"]
    lines += ["## Part A: trading the release\n", f"Chosen on 2023-26: {_label(*best) if best else 'none eligible'}\n"]
    if best:
        lines += [pd.DataFrame([{"Period": "Design 2023-26", **_fmt(dsc[best])}, {"Period": "Exam 2020-22", **_fmt(esc[best])}]
                               ).to_markdown(index=False), f"\nExam passed: {_exam_pass(esc[best])}\n"]
    lines += [f"Design: {(tA['Design Sharpe'] > 0).sum()} of {len(tA)} with Sharpe > 0; exam: {(tA['Exam Sharpe'] > 0).sum()}.\n",
              "By entry (bps of price per trade):\n", tA.groupby("Entry")[gcols].mean().round(1).to_markdown(),
              "\nBy event:\n", tA.groupby("Event")[gcols].mean().round(1).to_markdown(),
              "\nTop 15 by design Sharpe:\n", tA.head(15)[cols].to_markdown(index=False)]

    # Part B
    allow = {}
    for s, cn in coins.items():
        ny_day = pd.Index(cn.t_close.tz_convert(econ_calendar.TZ).strftime("%Y%m%d"))
        allow[s] = ~ny_day.isin(set(cal.date)).__array__()
    filt = ranking_filter(coins)
    rowsB, bd, be = [], {}, {}
    for name, mod in (("Intraday", intraday), ("Sessions", sessions)):
        for tag, al in (("all days", None), ("skip release days", allow)):
            d = mod.all_runs(coins, filt, DESIGN, al)
            x = mod.all_runs(coins, filt, EXAM, al)
            for g in mod.GRID:
                key = (name, tag, g)
                bd[key], be[key] = _row(d[g], DESIGN), _row(x[g], EXAM)
                rowsB.append({"Test": name, "Days": tag, "Version": mod._label(*g),
                              **{f"Design {k}": v for k, v in _fmt(bd[key]).items()},
                              **{f"Exam {k}": v for k, v in _fmt(be[key]).items()}})
    tB = pd.DataFrame(rowsB)
    tB.to_csv(report_dir / "part-b-skip-release-days.csv", index=False)

    def sib_b(key):
        name, tag, g = key
        if name == "Sessions":
            return (name, tag, (g[0], g[1], g[2], sessions.SIBLING[g[3]]))
        e, f, k, r = g
        if k == "swing":
            return (name, tag, (e, f, k, "level" if r == 2.0 else 2.0))
        return (name, tag, (e, f, 3.0 if k != 3.0 else 2.0, r))
    keys_skip = [k for k in bd if k[1] == "skip release days"]
    bestB = _pick(bd, keys_skip, sib_b)
    piv = tB.groupby(["Test", "Days"])[["Design Avg gross (bps)", "Design Avg net per trade (bps of price)", "Exam Avg gross (bps)",
                                         "Exam Avg net per trade (bps of price)", "Design Sharpe"]].mean().round(2)
    pos = tB.assign(p=tB["Design Sharpe"] > 0).groupby(["Test", "Days"]).p.sum()
    lines += ["\n## Part B: skipping release days\n", piv.assign(**{"Versions Sharpe > 0": pos}).to_markdown(),
              f"\nChosen (skip release days): {bestB[0] + ': ' + (intraday if bestB[0] == 'Intraday' else sessions)._label(*bestB[2]) if bestB else 'none eligible'}\n"]
    if bestB:
        lines += [pd.DataFrame([{"Period": "Design 2023-26", **_fmt(bd[bestB])}, {"Period": "Exam 2020-22", **_fmt(be[bestB])}]
                               ).to_markdown(index=False), f"\nExam passed: {_exam_pass(be[bestB])}\n"]

    # Volatility around releases
    lines += ["\n## How far prices move in the 5 hours after a release (2023-26, 20 coins)\n",
              moves(coins, times, DESIGN).to_markdown()]
    path = report_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# Part C: GDELT daily crypto news volume and tone, acted on the next day

C_ENTRIES = ("Follow spike-day move", "Fade spike-day move", "Tone shock direction")


def load_gdelt(folder: Path) -> pd.DataFrame:
    """Daily share of all monitored articles that mention crypto (`share`) and their average tone (`tone`)."""
    import json
    vol, tone = {}, {}
    for f in sorted(folder.glob("*.json")):
        data = json.loads(f.read_text()).get("timeline", [])
        for p in (data[0]["data"] if data else []):
            day = pd.Timestamp(p["date"][:8], tz="UTC")
            if f.name.startswith("timelinevolraw"):
                vol[day] = p["value"] / p["norm"] if p.get("norm") else np.nan
            else:
                tone[day] = p["value"]
    df = pd.DataFrame({"share": pd.Series(vol), "tone": pd.Series(tone)}).sort_index()
    df["spike"] = df.share > 2 * df.share.rolling(7).median().shift()
    z = (df.tone - df.tone.rolling(30).mean().shift()) / df.tone.rolling(30).std().shift()
    df["tone_shock"] = np.sign(z.where(z.abs() > 2, 0)).fillna(0).astype(int)
    return df


def gdelt_signals(cn: Coin, g: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(cn.c)
    out = {e: np.zeros(n, int) for e in C_ENTRIES}
    pos = pd.Series(np.arange(n), index=cn.t_close)
    for day, row in g.iterrows():
        nxt = day + pd.Timedelta("1D")
        t_sig, t0, t1 = nxt + pd.Timedelta("13h30min"), day, nxt
        if t_sig not in pos.index or t0 not in pos.index or t1 not in pos.index:
            continue
        i = pos[t_sig]
        if row.spike:
            d = int(np.sign(cn.c[pos[t1]] - cn.c[pos[t0]]))
            out["Follow spike-day move"][i], out["Fade spike-day move"][i] = d, -d
        if row.tone_shock:
            out["Tone shock direction"][i] = row.tone_shock
    return out


def run_gdelt(symbols, root: Path, report_dir: Path, gdelt_dir: Path) -> Path:
    coins = {}
    for s in symbols:
        try:
            coins[s] = load_coin(s, 15, root)
        except SystemExit:
            pass
    g = load_gdelt(gdelt_dir)
    grid = [(e, x) for e in C_ENTRIES for x in EXITS]
    sigs = {s: gdelt_signals(cn, g) for s, cn in coins.items()}

    def runs(period):
        out = {}
        for e, x in grid:
            k, r = EXITS[x]
            ts = [simulate(cn, sigs[s][e], k, r, period).assign(coin=s) for s, cn in coins.items()]
            ts = [t for t in ts if len(t)]
            out[(e, x)] = pd.concat(ts, ignore_index=True) if ts else pd.DataFrame()
        return out
    d, x = runs(DESIGN), runs(EXAM)
    dsc, esc = {k: _row(v, DESIGN) for k, v in d.items()}, {k: _row(v, EXAM) for k, v in x.items()}
    best = _pick(dsc, grid, lambda k: (k[0], SIBLING[k[1]]))
    tC = pd.DataFrame([{"Version": f"{e}, {xx}", **{f"Design {k}": v for k, v in _fmt(dsc[(e, xx)]).items()},
                        "Design t by day": round(event_t(d[(e, xx)]), 2),
                        **{f"Exam {k}": v for k, v in _fmt(esc[(e, xx)]).items()}} for e, xx in grid]
                      ).sort_values("Design Sharpe", ascending=False)
    tC.to_csv(report_dir / "part-c-gdelt.csv", index=False)
    cover = f"{g.index.min():%Y-%m-%d} to {g.index.max():%Y-%m-%d}, {g.share.notna().sum()} days with counts, " \
            f"{g.tone.notna().sum()} with tone; {int(g.spike.sum())} spike days, {int((g.tone_shock != 0).sum())} tone shocks"
    lines = ["# Part C: GDELT crypto news volume and tone\n", f"Data: {cover}\n",
             f"Chosen on 2023-26: {' '.join(best) if best else 'none eligible'}\n"]
    if best:
        lines += [pd.DataFrame([{"Period": "Design 2023-26", **_fmt(dsc[best])}, {"Period": "Exam 2020-22", **_fmt(esc[best])}]
                               ).to_markdown(index=False), f"\nExam passed: {_exam_pass(esc[best])}\n"]
    cols = ["Version", "Design Trades", "Design Win %", "Design Avg gross (bps)", "Design Avg net per trade (bps of price)",
            "Design Sharpe", "Design t by day", "Design By year %", "Exam Avg gross (bps)", "Exam Sharpe", "Exam By year %"]
    lines += [tC[[c for c in cols if c in tC]].to_markdown(index=False)]

    # filter: no entries on the day after a spike
    spike_next = set((g.index[g.spike] + pd.Timedelta("1D")).strftime("%Y%m%d"))
    allow = {s: ~pd.Index(cn.t_close.strftime("%Y%m%d")).isin(spike_next).__array__() for s, cn in coins.items()}
    filt = ranking_filter(coins)
    rows = []
    for name, mod in (("Intraday", intraday), ("Sessions", sessions)):
        for tag, al in (("all days", None), ("skip day after news spike", allow)):
            dd = mod.all_runs(coins, filt, DESIGN, al)
            for gg in mod.GRID:
                s = _row(dd[gg], DESIGN)
                rows.append({"Test": name, "Days": tag, "Sharpe": s["Sharpe"], "Years positive": s["Years positive"],
                             "Avg net (bps)": s.get("Avg net per trade (bps of price)", np.nan)})
    tf = pd.DataFrame(rows)
    lines += ["\n## Filter: skip the day after a news spike (design 2023-26)\n",
              tf.groupby(["Test", "Days"]).agg(**{"Mean Sharpe": ("Sharpe", "mean"), "Versions Sharpe > 0": ("Sharpe", lambda v: (v > 0).sum()),
                                                  "Avg net (bps)": ("Avg net (bps)", "mean")}).round(2).to_markdown()]
    path = report_dir / "results-gdelt.md"
    path.write_text("\n".join(lines) + "\n")
    return path
