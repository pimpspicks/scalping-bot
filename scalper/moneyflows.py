"""Money moving in and out of crypto and exchanges (USDT dominance, stablecoin supply, BTC/ETH exchange balances)
as a bias filter on the three-move setups, and as a daily strategy on its own.

Plan and pass rule: reports-2026-10-29-money-flows/plan.md (written before running).

Daily data from Coin Metrics (community API) and DefiLlama, saved by `download`. A value dated day D is used only
from D+1 04:00 UTC. Each signal is +1 (money flowing in: bullish), -1 (flowing out) or 0 (no view).
"""
import csv
import json
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .fresh import DESIGN, EXAM, load_coin, score
from .intraday import COST, MAX_SIZE, RISK, _row
from .understanding import GROUP2, SETUPS, TARGETS, TFS2, XS, _label, signals, simulate

log = logging.getLogger(__name__)

CM = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
LLAMA = "https://stablecoins.llama.fi/stablecoincharts/all"
STABLES = ("usdt", "usdc", "dai", "busd", "tusd", "pax", "gusd", "husd", "usde_eth", "fdusd_eth", "pyusd_eth")
CAP_COINS = ("btc", "eth", "xrp", "ada", "doge", "ltc", "bch", "link", "xlm", "etc", "xmr", "dash", "zec", "algo",
             "cro", "leo_eth")
LAG = pd.Timedelta(hours=28)  # day D's value is used from D+1 04:00 UTC
NS = (7, 14, 30)
SIGNALS = ("USDT.D", "Stables", "BTC exchanges", "ETH exchanges", "Combined")


def _get(url):
    for i in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - network hiccups, retried
            log.warning("retry %s: %s", url, e)
            time.sleep(5 * (i + 1))
    raise SystemExit(f"download failed: {url}")


def download(out: Path, start="2019-06-01"):
    out.mkdir(parents=True, exist_ok=True)

    def cm(assets, metrics, fname):
        rows, url = [], (f"{CM}?assets={assets}&metrics={metrics}&frequency=1d&start_time={start}"
                         f"&page_size=10000&paging_from=start")
        while url:
            d = _get(url)
            rows += d["data"]
            url = d.get("next_page_url")
        keys = ["asset", "time"] + metrics.split(",")
        with open(out / fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            w.writerows([r.get(k, "") for k in keys] for r in rows)

    cm("btc,eth", "FlowInExUSD,FlowOutExUSD,FlowInExNtv,FlowOutExNtv,SplyExNtv,SplyExUSD,CapMrktCurUSD,PriceUSD",
       "exchange_flows_btc_eth.csv")
    cm(",".join(CAP_COINS), "CapMrktCurUSD", "caps_coins.csv")
    cm(",".join(STABLES), "CapMrktCurUSD", "caps_stables.csv")
    for q, name in (("?stablecoin=1", "usdt"), ("?stablecoin=2", "usdc"), ("", "all_stables")):
        d = _get(LLAMA + q)
        pd.DataFrame({"date": [r["date"] for r in d],
                      "usd": [r["totalCirculatingUSD"].get("peggedUSD", 0) for r in d]}).to_csv(
            out / f"llama_{name}.csv", index=False)


def load_flows(root: Path) -> pd.DataFrame:
    """Daily: USDT dominance (vs a big-coin + stablecoin total), stablecoin supply, BTC/ETH held on exchanges.
    Stablecoin caps come from Coin Metrics: DefiLlama's history misses most of USDT before 2020."""
    def caps(name, assets):
        f = pd.read_csv(root / name, parse_dates=["time"])
        return f.pivot(index="time", columns="asset", values="CapMrktCurUSD").reindex(columns=list(assets))

    coins = caps("caps_coins.csv", CAP_COINS).ffill()
    st = caps("caps_stables.csv", STABLES).ffill(limit=3)  # coins that ended (BUSD, HUSD...) drop to nothing
    ex = pd.read_csv(root / "exchange_flows_btc_eth.csv", parse_dates=["time"]).pivot(
        index="time", columns="asset", values="SplyExNtv")
    df = pd.DataFrame({"usdt": st.usdt, "stables": st.sum(axis=1), "coins": coins.sum(axis=1, min_count=len(CAP_COINS)),
                       "btc_ex": ex.btc, "eth_ex": ex.eth}).ffill().dropna()
    df["usdt_d"] = df.usdt / (df.coins + df.stables)
    return df


def flow_signals(df: pd.DataFrame) -> dict:
    """{(signal, N): Series of +1/-1/0 indexed by the time it becomes known}."""
    out = {}
    for n in NS:
        g = df.stables.pct_change(n)
        parts = {"USDT.D": np.where(df.usdt_d < df.usdt_d.shift(n), 1, -1),
                 "Stables": np.where(g > g.rolling(365, min_periods=180).median(), 1, -1),  # NaN median -> masked below
                 "BTC exchanges": np.where(df.btc_ex < df.btc_ex.shift(n), 1, -1),
                 "ETH exchanges": np.where(df.eth_ex < df.eth_ex.shift(n), 1, -1)}
        parts["Stables"] = np.where(g.rolling(365, min_periods=180).median().isna(), 0, parts["Stables"])
        tot = sum(parts.values())
        parts["Combined"] = np.where(tot >= 2, 1, np.where(tot <= -2, -1, 0))
        for k, v in parts.items():
            out[(k, n)] = pd.Series(v, index=df.index + LAG).iloc[n:]
    return out


def _at(s: pd.Series, t: pd.DatetimeIndex) -> np.ndarray:
    return s.reindex(s.index.union(t)).ffill().reindex(t).fillna(0).to_numpy()


# ---------- (a) filter on the three-move setups ----------

FILTERS = [None] + [(k, n) for k in SIGNALS for n in NS]


def _flabel(g):
    s, tf, x, t, f = g
    return _label(s, tf, x, t) + (f", only with {f[0]} {f[1]}d" if f else "")


def filter_runs(coins, sigs):
    out = {DESIGN: {}, EXAM: {}}
    for tf in TFS2:
        bias = {(sym, f): _at(sigs[f], cn.t_close) for sym, cn in coins[tf].items() for f in FILTERS if f}
        for s in SETUPS:
            for x in XS:
                base = {sym: signals(cn, x, s) for sym, cn in coins[tf].items()}
                for f in FILTERS:
                    for t in TARGETS:
                        for per in (DESIGN, EXAM):
                            trades = []
                            for sym, cn in coins[tf].items():
                                sig, st, tg = base[sym]
                                if f:
                                    sig = np.where(sig * bias[(sym, f)] > 0, sig, 0)
                                tr = simulate(cn, sig, st, tg, t, None, per)
                                if len(tr):
                                    trades.append(tr.assign(coin=sym))
                            out[per][(s, tf, x, t, f)] = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
        log.info("done %sm", tf)
    return out


def _nbs_a(g):
    s, tf, x, t, f = g
    nb = [(s, tf, x, tt, f) for tt in TARGETS if tt != t] + [(s, tf, xx, t, f) for xx in XS if abs(xx - x) == 1]
    i = NS.index(f[1])
    return nb + [(s, tf, x, t, (f[0], NS[j])) for j in (i - 1, i + 1) if 0 <= j < len(NS)]


# ---------- (b) on its own, daily ----------

STYLES = ("long/short", "long-only")
UNIVERSES = ("20 coins", "BTC only")


def standalone(daily, sig: pd.Series | None, style: str) -> pd.Series:
    """Daily account return. sig None: always long (context only)."""
    rets = []
    for sym, cn in daily.items():
        t_open = cn.t_close - pd.Timedelta(days=1)
        want = np.ones(len(t_open)) if sig is None else _at(sig, t_open)
        if style == "long-only":
            want = np.where(want > 0, 1.0, 0.0)
        atr_pct = np.r_[np.nan, (cn.atr / cn.c)[:-1]]  # yesterday's ATR, known at today's open
        n, r = len(cn.o), np.zeros(len(cn.o))
        pos = size = 0.0
        for i in range(n - 1):
            if want[i] != pos and np.isfinite(atr_pct[i]):
                new_size = min(RISK / (3 * atr_pct[i]), MAX_SIZE) if want[i] else 0.0
                r[i] -= COST * (size + new_size)
                pos, size = want[i], new_size
            if pos:
                r[i] += pos * size * ((cn.o[i + 1] / cn.o[i] - 1) - (cn.fcum[i + 1] - cn.fcum[i]))
        rets.append(pd.Series(r, index=t_open.floor("D")))
    return pd.concat(rets, axis=1).fillna(0).sum(axis=1)


def run(symbols, root: Path, flow_root: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    sigs = flow_signals(load_flows(flow_root))

    def fmt(d):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}

    lines = ["# Money flows (USDT dominance, stablecoins, exchange balances): results\n", "Plan: plan.md.\n"]

    # information: signal vs next BTC returns, design period
    btc = load_coin("BTC", 1440, root)
    t_open = btc.t_close - pd.Timedelta(days=1)
    o = pd.Series(btc.o, index=t_open)
    fwd = {h: (o.shift(-h) / o - 1) for h in (1, 7)}
    inside = (t_open >= DESIGN[0]) & (t_open < DESIGN[1])
    info = []
    for (k, n), s in sigs.items():
        v = pd.Series(_at(s, t_open), index=t_open)
        info.append({"Signal": k, "N days": n, "Share bullish %": round((v[inside] > 0).mean() * 100),
                     **{f"Corr with next {h}d BTC": round(v[inside].corr(fwd[h][inside]), 3) for h in fwd},
                     **{f"BTC next 7d % when +1/-1": f"{fwd[7][inside & (v > 0)].mean() * 100:+.2f} / "
                                                     f"{fwd[7][inside & (v < 0)].mean() * 100:+.2f}"}})
    lines += ["## Information: signal vs BTC's next returns, 2023-26\n", pd.DataFrame(info).to_markdown(index=False), ""]

    # (b) standalone
    daily = {}
    for sym in symbols:
        try:
            daily[sym] = load_coin(sym, 1440, root)
        except SystemExit:
            pass
    uni = {"20 coins": daily, "BTC only": {"BTC": daily["BTC"]}}
    sb = {}
    for k, n in sigs:
        for st in STYLES:
            for u in UNIVERSES:
                sb[(k, n, st, u)] = standalone(uni[u], sigs[(k, n)], st)
    base = {u: standalone(uni[u], None, "long-only") for u in UNIVERSES}
    bd = {g: score(r, DESIGN) for g, r in sb.items()}
    be = {g: score(r, EXAM) for g, r in sb.items()}
    best = None
    for g, s in bd.items():
        k, n, st, u = g
        i = NS.index(n)
        nb = [(k, NS[j], st, u) for j in (i - 1, i + 1) if 0 <= j < len(NS)]
        ok = (s["Years positive"] >= 3 and all(bd[x]["Sharpe"] > 0 for x in nb)
              and (st != "long-only" or s["Sharpe"] > score(base[u], DESIGN)["Sharpe"]))
        if ok and (best is None or s["Sharpe"] > bd[best]["Sharpe"]):
            best = g
    tb = pd.DataFrame([{"Version": f"{k} {n}d, {st}, {u}", **{f"Design {a}": b for a, b in fmt(bd[g]).items()},
                        **{f"Exam {a}": b for a, b in fmt(be[g]).items()}} for g in sb for k, n, st, u in [g]])
    tb = pd.concat([tb, pd.DataFrame([{"Version": f"Always long (context), {u}",
                                       **{f"Design {a}": b for a, b in fmt(score(base[u], DESIGN)).items()},
                                       **{f"Exam {a}": b for a, b in fmt(score(base[u], EXAM)).items()}}
                                      for u in UNIVERSES])]).sort_values("Design Sharpe", ascending=False)
    tb.to_csv(report_dir / "standalone-versions.csv", index=False)
    lines.append("## (b) On its own, daily\n")
    lines.append(f"**Chosen:** {' '.join(map(str, best)) if best else 'none eligible'}\n")
    if best:
        lines += [pd.DataFrame([{"Period": "Design 2023-26", **fmt(bd[best])},
                                {"Period": "Exam 2020-22", **fmt(be[best])}]).to_markdown(index=False), ""]
    c = ["Version", "Design Sharpe", "Design Return/yr %", "Design Max DD %", "Design By year %", "Exam Sharpe",
         "Exam Return/yr %", "Exam By year %"]
    lines += [f"Design: {(tb['Design Sharpe'] > 0).sum()} of {len(tb)} with Sharpe > 0.\n",
              tb[c].to_markdown(index=False), ""]

    # (a) filter on three-move setups
    coins = {tf: {} for tf in TFS2}
    for sym in symbols:
        for tf in TFS2:
            try:
                coins[tf][sym] = load_coin(sym, tf, root)
            except SystemExit:
                pass
    res = filter_runs(coins, sigs)
    dsc = {g: _row(t, DESIGN) for g, t in res[DESIGN].items()}
    esc = {g: _row(t, EXAM) for g, t in res[EXAM].items()}
    grid = list(dsc)
    chosen = {}
    for grp in ("same-day", "multi-day"):
        best = None
        for g in grid:
            if GROUP2[g[1]] != grp or g[4] is None:
                continue
            ok = dsc[g]["Years positive"] >= 3 and all(dsc[nb]["Sharpe"] > 0 for nb in _nbs_a(g))
            if ok and (best is None or dsc[g]["Sharpe"] > dsc[best]["Sharpe"]):
                best = g
        chosen[grp] = best
        log.info("chosen %s: %s", grp, best)
    ta = pd.DataFrame([{"Version": _flabel(g), "Setup": g[0], "Timeframe": g[1], "Group": GROUP2[g[1]], "X": g[2],
                        "Target": g[3], "Filter": g[4][0] if g[4] else "none", "N days": g[4][1] if g[4] else np.nan,
                        **{f"Design {k}": v for k, v in fmt(dsc[g]).items()},
                        **{f"Exam {k}": v for k, v in fmt(esc[g]).items()},
                        "Design Sharpe without filter": round(dsc[g[:4] + (None,)]["Sharpe"], 2),
                        "Exam Sharpe without filter": round(esc[g[:4] + (None,)]["Sharpe"], 2)} for g in grid])
    ta = ta.sort_values("Design Sharpe", ascending=False)
    ta.to_csv(report_dir / "filter-versions.csv", index=False)
    lines.append("## (a) As a filter on the three-move setups\n")
    for grp, g in chosen.items():
        lines.append(f"**Chosen, {grp}:** {_flabel(g) if g else 'none eligible'}\n")
        if g:
            lines += [pd.DataFrame([{"Period": "Design 2023-26", **fmt(dsc[g])}, {"Period": "Exam 2020-22", **fmt(esc[g])},
                                    {"Period": "Exam, same setup without filter", **fmt(esc[g[:4] + (None,)])}]
                                   ).to_markdown(index=False), ""]
    f = ta[ta.Filter != "none"]
    better = f.groupby(["Group", "Filter"]).apply(lambda d: pd.Series({
        "Versions": len(d), "Design Sharpe > 0": (d["Design Sharpe"] > 0).sum(),
        "Design beats no filter": (d["Design Sharpe"] > d["Design Sharpe without filter"]).sum(),
        "Exam Sharpe > 0": (d["Exam Sharpe"] > 0).sum(),
        "Exam beats no filter": (d["Exam Sharpe"] > d["Exam Sharpe without filter"]).sum(),
        "Design avg net bps": round(d["Design Avg net per trade (bps of price)"].mean(), 1),
        "Exam avg net bps": round(d["Exam Avg net per trade (bps of price)"].mean(), 1)}))
    nf = ta[ta.Filter == "none"].groupby("Group")[["Design Avg net per trade (bps of price)",
                                                   "Exam Avg net per trade (bps of price)"]].mean().round(1)
    cols = ["Version", "Design Trades", "Design Sharpe", "Design Sharpe without filter", "Design Return/yr %",
            "Design By year %", "Exam Sharpe", "Exam Sharpe without filter", "Exam Return/yr %", "Exam By year %"]
    lines += ["### Does each filter help? (counts of versions)\n", better.to_markdown(), "",
              "Without any filter, avg net bps per trade:\n", nf.to_markdown(), "",
              "### Top 20 by design Sharpe\n", ta.head(20)[cols].to_markdown(index=False)]
    path = report_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path
