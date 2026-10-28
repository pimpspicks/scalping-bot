"""Binance positioning signals (open interest, long/short ratios, taker volume) tested on perp prices.

The six tests and their settings were written down before running, in
reports-2026-10-28-positioning/plan.md. Nothing here is tuned: every number is out of sample.

Weekly tests rank the 20 coins every Monday with `xsection.rank_backtest`. Ladder tests run the paper-bot
engine (`ladder.simulate_ladder_limit`: 20x, limit entries, maker 1 / taker 6 bps) on 15-minute bars.
A metric row stamped T is only used from T + 5 minutes, so no signal sees data it could not have had.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import binance_metrics, xsection
from .ladder import simulate_ladder_limit
from .rules import _first
from .tune import donchian_hurst
from .swing import make_bars

log = logging.getLogger(__name__)

LAG = pd.Timedelta("5min")
DAY_15M = 96


def metrics_at(m: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """The latest metric row usable at each time (row stamp + 5 min <= time), NaN if over 30 min stale."""
    src = m.copy()
    src.index = src.index + LAG
    src = src.reset_index(names="avail")
    out = pd.merge_asof(pd.DataFrame({"t": times}), src, left_on="t", right_on="avail",
                        tolerance=pd.Timedelta("30min"), direction="backward")
    return out.set_index("t").drop(columns="avail")


# ---------- weekly scores (index = Monday 00:00 rebalance times) ----------

def _daily_log(m: pd.DataFrame, col: str) -> pd.Series:
    """Mean log ratio per UTC day, stamped at the end of the day (when it is fully known, after the lag)."""
    x = np.log(m[col].where(m[col] > 0))
    x.index = x.index + LAG
    d = x.resample("1D", label="right", closed="left").mean()
    return d


def _z7(d: pd.Series) -> pd.Series:
    """z-score of the past 7 days' mean against the 90 days before them."""
    return (d.rolling(7, min_periods=5).mean() - d.rolling(90, min_periods=60).mean().shift(7)) \
        / d.rolling(90, min_periods=60).std().shift(7)


def weekly_scores(metrics: dict[str, pd.DataFrame], times: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    retail, smart, oi = {}, {}, {}
    for s, m in metrics.items():
        zr = _z7(_daily_log(m, "all_acct_ls"))
        zs = _z7(_daily_log(m, "top_pos_ls"))
        retail[s] = zr.reindex(times)
        smart[s] = (zs - zr).reindex(times)
        now = metrics_at(m, times).oi
        before = metrics_at(m, times - pd.Timedelta("7D")).oi
        oi[s] = pd.Series(now.to_numpy() / before.to_numpy() - 1, index=times)
    return {"1. Retail contrarian": -pd.DataFrame(retail), "2. Smart vs retail": pd.DataFrame(smart),
            "3. OI crowding": -pd.DataFrame(oi)}


# ---------- per-coin ladder signals on 15-minute bars ----------

def _above(x: pd.Series, q: float) -> pd.Series:
    """x above its own q-quantile over the previous 30 days (current bar excluded)."""
    return x > x.rolling(30 * DAY_15M, min_periods=20 * DAY_15M).quantile(q).shift()


def _below(x: pd.Series, q: float) -> pd.Series:
    return x < x.rolling(30 * DAY_15M, min_periods=20 * DAY_15M).quantile(q).shift()


def ladder_signals(df: pd.DataFrame, m: pd.DataFrame) -> dict[str, np.ndarray]:
    close_t = df.index + pd.Timedelta("15min")
    mt = metrics_at(m, close_t)
    mt.index = df.index
    taker = np.log(m.taker_ls.where(m.taker_ls > 0)).rolling("1h", min_periods=10).mean().to_frame("tk")
    tk = metrics_at(taker, close_t).tk
    tk.index = df.index
    px = df.close.pct_change(4)
    doi = mt.oi.pct_change(4, fill_method=None)

    def sig(long, short):
        long, short = long.fillna(False).to_numpy(), short.fillna(False).to_numpy()
        return _first(np.where(long & ~short, 1, np.where(short & ~long, -1, 0)))

    oi_up = _above(doi, 0.98)
    oi_flush = _below(doi, 0.01)
    return {
        "4. OI-confirmed breakout": sig(_above(px, 0.98) & oi_up, _below(px, 0.02) & oi_up),
        "5. OI flush reversal": sig(oi_flush & _below(px, 0.05), oi_flush & _above(px, 0.95)),
        "6. Taker imbalance": sig(_above(tk, 0.99), _below(tk, 0.01)),
    }


def ladder_trades(symbols, price_root: Path, metrics_root: Path, start: pd.Timestamp, end: pd.Timestamp,
                  lev=20, stop_margin=0.25, maker_bps=1.0, taker_bps=6.0, max_hold_h=72, wait_bars=4,
                  survivor=True) -> pd.DataFrame:
    out = []
    for s in symbols:
        b = make_bars(s, 15, price_root)
        m = binance_metrics.load(s, metrics_root)
        sigs = ladder_signals(b.df, m)
        if survivor:
            sigs["Survivor: 15m Donchian 55 + Hurst>0.6"] = donchian_hurst(b.df, 55, 0.6)
        inside = np.asarray((b.df.index >= start) & (b.df.index < end))
        for name, sg in sigs.items():
            sg = np.where(inside, sg, 0)
            t = simulate_ladder_limit(b, sg, stop_margin / lev, maker_bps, taker_bps, int(max_hold_h * 4), wait_bars)
            t["test"], t["coin"] = name, s
            t["roi_pct"] = t.net_bps * lev / 100
            out.append(t)
        log.info("%s: ladder done", s)
    return pd.concat(out, ignore_index=True)


# ---------- summaries ----------

def ladder_summary(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in t.groupby("test"):
        r = g.roi_pct
        years = g.groupby(g.entry_time.dt.year).roi_pct.mean()
        coins = g.groupby("coin").roi_pct.mean()
        ok = r.mean() > 0 and r.mean() / r.std() * np.sqrt(len(r)) >= 2 and (years > 0).sum() >= 3 \
            and (coins > 0).sum() >= 12
        rows.append({"Test": name, "Trades": len(r), "Avg ROI/trade %": round(r.mean(), 2),
                     "t-stat": round(r.mean() / r.std() * np.sqrt(len(r)), 2),
                     "Win %": round((r > 0).mean() * 100), "Coins positive": f"{(coins > 0).sum()} of {len(coins)}",
                     "By year (avg ROI %)": ", ".join(f"{y} {v:+.1f}" for y, v in years.items()),
                     "Pass?": "n/a" if name.startswith("Survivor") else ("Yes" if ok else "No")})
    return pd.DataFrame(rows)


def weekly_summary(name: str, w: pd.DataFrame) -> dict:
    st = xsection.stats(w)
    years = w.groupby(pd.to_datetime(w.week).dt.year).ret.mean() * 100
    ok = st["avg_week_pct"] > 0 and st["t_stat"] >= 2 and (years > 0).sum() >= 3
    return {"Test": name, "Weeks": st["weeks"], "Avg per week %": round(st["avg_week_pct"], 2),
            "t-stat": round(st["t_stat"], 2), "Winning weeks %": round(st["win_weeks"] * 100),
            "Per year %": round(st["year_pct"], 1), "Max DD %": round(st["max_dd_pct"]),
            "By year (avg % per week)": ", ".join(f"{y} {v:+.2f}" for y, v in years.items()),
            "Pass?": "n/a" if name.startswith("Survivor") else ("Yes" if ok else "No")}


def weekly(symbols, price_root: Path, metrics_root: Path, start: pd.Timestamp, end: pd.Timestamp,
           cost_bps=2.0) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    closes, funding = xsection.weekly_panel({s: price_root for s in symbols})
    metrics = {s: binance_metrics.load(s, metrics_root) for s in symbols}
    scores = weekly_scores(metrics, closes.index)
    scores["Survivor: 3-week momentum"] = closes / closes.shift(3) - 1
    rows, weeks = [], {}
    for name, sc in scores.items():
        w = xsection.rank_backtest(closes, funding, sc, 4, cost_bps)
        w = w[(pd.to_datetime(w.week) >= start) & (pd.to_datetime(w.week) < end)]
        weeks[name] = w
        rows.append(weekly_summary(name, w))
    return pd.DataFrame(rows), weeks


def run(symbols, price_root: Path, metrics_root: Path, report_dir: Path, start: str, end: str,
        check_root: Path | None = None, check_symbols=("BTC", "ETH", "SOL")) -> Path:
    """Run the six pre-registered tests once (plus the two survivors) and write summary.md + trade files."""
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    report_dir.mkdir(parents=True, exist_ok=True)
    wk, weeks = weekly(symbols, price_root, metrics_root, t0, t1)
    pd.concat([w.assign(test=k) for k, w in weeks.items()]).to_csv(report_dir / "weekly.csv", index=False)
    tr = ladder_trades(symbols, price_root, metrics_root, t0, t1)
    tr.to_csv(report_dir / "ladder-trades.csv.gz", index=False, compression="gzip")
    ld = ladder_summary(tr)
    parts = [f"# Binance positioning tests: results ({start} to {end})\n",
             "Plan and settings: plan.md (written before running). Prices: Binance perp candles + funding.\n",
             "## Weekly coin-vs-coin (long top 4 / short bottom 4, 2 bps per side, unlevered)\n",
             wk.to_markdown(index=False), "\n## Per-coin 20x ladder (maker 1 / taker 6 bps, funding included)\n",
             ld.to_markdown(index=False)]
    if check_root is not None:
        ck = ladder_trades(list(check_symbols), check_root, metrics_root, t0, t1)
        ck.to_csv(report_dir / "ladder-trades-phemex-check.csv.gz", index=False, compression="gzip")
        bn = tr[tr.coin.isin(check_symbols)]
        cmp = ladder_summary(ck)[["Test", "Trades", "Avg ROI/trade %", "t-stat"]].merge(
            ladder_summary(bn)[["Test", "Trades", "Avg ROI/trade %", "t-stat"]], on="Test",
            suffixes=(" (Phemex)", " (Binance)"))
        parts += [f"\n## Price-source check: {', '.join(check_symbols)} on Phemex vs Binance candles\n",
                  cmp.to_markdown(index=False)]
    path = report_dir / "results.md"
    path.write_text("\n".join(parts) + "\n")
    return path
