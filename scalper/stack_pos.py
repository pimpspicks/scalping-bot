"""The stacked model again, with Binance positioning features added, plus a weekly coin-ranking model.

Plan and pass rule: reports-2026-10-28-stacked-positioning/plan.md (written before running).

A. Same rows, labels and walk-forward as `stack.py`; two versions trained on identical rows, one with the old
   features ("base") and one with ~30 positioning features added. Comparing them shows what the new data adds.
B. Each Monday a gradient-boosting model predicts every coin's next-week return rank from momentum and weekly
   positioning inputs, trained on all earlier weeks; long the top 4, short the bottom 4.
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import binance_metrics, stack, xsection
from .positioning import metrics_at, weekly_scores
from .swing import make_bars

log = logging.getLogger(__name__)
DAY = 96  # 15-minute bars


def _z(x: pd.Series, n: int) -> pd.Series:
    return (x - x.rolling(n, min_periods=n // 2).mean().shift()) / x.rolling(n, min_periods=n // 2).std().shift()


def pos_features(idx: pd.DatetimeIndex, close: pd.Series, volume_usd: pd.Series, m: pd.DataFrame) -> pd.DataFrame:
    """Positioning features known at each 15-minute bar's close."""
    lm = m.copy()
    for c in ("all_acct_ls", "top_acct_ls", "top_pos_ls", "taker_ls"):
        lm[c] = np.log(lm[c].where(lm[c] > 0))
    for h, n in (("1h", 12), ("4h", 48), ("24h", 288)):
        lm[f"tk_{h}"] = lm.taker_ls.rolling(n, min_periods=n // 2).mean()
    mt = metrics_at(lm, idx + pd.Timedelta("15min"))
    mt.index = idx
    f = pd.DataFrame(index=idx)
    for c, short in (("all_acct_ls", "retail"), ("top_acct_ls", "top_acct"), ("top_pos_ls", "top_pos")):
        f[f"pos_{short}"] = mt[c]
        f[f"pos_{short}_z1d"] = _z(mt[c], DAY)
        f[f"pos_{short}_z30d"] = _z(mt[c], 30 * DAY)
    f["pos_smart_minus_retail"] = f.pos_top_pos_z30d - f.pos_retail_z30d
    oi = mt.oi
    for h, n in (("1h", 4), ("4h", 16), ("24h", DAY), ("7d", 7 * DAY)):
        f[f"pos_oi_chg_{h}"] = oi.pct_change(n, fill_method=None)
    f["pos_oi_chg_1h_z"] = _z(f.pos_oi_chg_1h, 30 * DAY)
    f["pos_oi_chg_24h_z"] = _z(f.pos_oi_chg_24h, 30 * DAY)
    f["pos_oi_z30d"] = _z(np.log(oi), 30 * DAY)
    f["pos_oi_to_vol"] = mt.oi_usd / volume_usd.rolling(DAY, min_periods=DAY // 2).sum()
    f["pos_price_vs_oi_24h"] = close.pct_change(DAY) * np.sign(f.pos_oi_chg_24h)
    for h in ("1h", "4h", "24h"):
        f[f"pos_taker_{h}"] = mt[f"tk_{h}"]
    f["pos_taker_1h_z"] = _z(mt.tk_1h, 30 * DAY)
    f["pos_taker_24h_z"] = _z(mt.tk_24h, 30 * DAY)
    return f.astype(np.float32)


def build_rows(symbols, root: Path, metrics_root: Path, cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    parts, closes, xs = [], {}, {}
    for s in symbols:
        p = cache / f"{s}.pkl"
        if p.exists():
            r = pickle.loads(p.read_bytes())
        else:
            r = stack.coin_rows(s, root, root)
            b = make_bars(s, 15, root)
            pf = pos_features(b.df.index, b.df.close, b.df.volume * b.df.close, binance_metrics.load(s, metrics_root))
            r = r.join(pf)
            p.write_bytes(pickle.dumps(r))
        log.info("%s: %d rows", s, len(r))
        parts.append(r)
        closes[s] = make_bars(s, 15, root).df.close
    rows = pd.concat(parts).sort_index()
    closes = pd.DataFrame(closes)
    rows = stack.add_cross_section(rows, closes)
    # cross-sectional ranks of the positioning features at each bar
    for col in ("pos_retail_z30d", "pos_oi_chg_7d", "pos_taker_24h"):
        wide = rows.pivot_table(index=rows.index, columns="symbol", values=col)
        rk = wide.rank(axis=1, pct=True).stack()
        rows[f"{col}_xsrank"] = rk.reindex(pd.MultiIndex.from_arrays([rows.index, rows.symbol])).to_numpy(np.float32)
    return rows


def walk_forward(rows: pd.DataFrame, feats: list[str], first_quarter: str, last_quarter: str, thin: int = 4):
    q = rows.index.tz_localize(None).to_period("Q")
    out = []
    for quarter in sorted(set(q)):
        if quarter < pd.Period(first_quarter, "Q") or quarter > pd.Period(last_quarter, "Q"):
            continue
        start = quarter.start_time.tz_localize("UTC")
        train = rows[(rows.end_time < start) & rows.y_long.notna() & rows.y_short.notna()].iloc[::thin]
        test = rows[q == quarter]
        preds = {}
        for side in ("long", "short"):
            m = stack._model().fit(train[feats], train[f"y_{side}"])
            preds[f"p_{side}"] = m.predict(test[feats])
        out.append(pd.DataFrame(preds, index=test.index).assign(symbol=test.symbol.to_numpy(),
                                                               y_long=test.y_long.to_numpy(),
                                                               y_short=test.y_short.to_numpy()))
        log.info("%s: trained on %d rows, predicted %d", quarter, len(train), len(test))
    return pd.concat(out)


def _trade_stats(t: pd.DataFrame) -> dict:
    r = t.roi_pct
    q = t.groupby(t.entry_time.dt.tz_localize(None).dt.to_period("Q")).roi_pct.mean()
    c = t.groupby("symbol").roi_pct.mean()
    return {"Trades": len(r), "Avg ROI/trade %": round(r.mean(), 2),
            "t-stat": round(r.mean() / r.std() * np.sqrt(len(r)), 2),
            "Coins positive": f"{(c > 0).sum()} of {len(c)}", "Quarters positive": f"{(q > 0).sum()} of {len(q)}"}


def ladder_part(rows: pd.DataFrame, root: Path, first_q: str, last_q: str, report_dir: Path) -> list[str]:
    base = [c for c in rows.columns if c not in ("y_long", "y_short", "end_time", "symbol", "close")
            and not c.startswith("pos_")]
    plus = base + [c for c in rows.columns if c.startswith("pos_")]
    bars15 = {s: make_bars(s, 15, root) for s in rows.symbol.unique()}
    corr_rows, trade_rows = [], []
    for name, feats in (("Base", base), ("Plus positioning", plus)):
        p = walk_forward(rows, feats, first_q, last_q)
        p.to_pickle(report_dir / f"preds-{name.split()[0].lower()}.pkl")
        corr_rows.append({"Model": name, "Features": len(feats),
                          "Rank corr long": round(p.p_long.corr(p.y_long, method="spearman"), 3),
                          "Rank corr short": round(p.p_short.corr(p.y_short, method="spearman"), 3),
                          "Top-10% long: actual ROI %": round(p.y_long[p.p_long >= p.p_long.quantile(0.9)].mean(), 2),
                          "Top-10% short: actual ROI %": round(p.y_short[p.p_short >= p.p_short.quantile(0.9)].mean(), 2),
                          "All long: actual ROI %": round(p.y_long.mean(), 2),
                          "All short: actual ROI %": round(p.y_short.mean(), 2)})
        for thr in (0.0, 2.0, 5.0):
            t = stack.trade(p, bars15, thr)
            trade_rows.append({"Model": name, "Take trades predicted above": f"{thr:.0f}%", **_trade_stats(t)})
    return ["## A. 20x ladder model, out of sample " + f"{first_q} to {last_q}\n",
            "### Does the prediction relate to what happened?\n", pd.DataFrame(corr_rows).to_markdown(index=False),
            "\n### Trading the picks (one position per coin at a time)\n", pd.DataFrame(trade_rows).to_markdown(index=False)]


def weekly_part(symbols, root: Path, metrics_root: Path, start: pd.Timestamp, end: pd.Timestamp,
                cost_bps: float = 2.0) -> tuple[list[str], pd.DataFrame]:
    closes, funding = xsection.weekly_panel({s: root for s in symbols})
    metrics = {s: binance_metrics.load(s, metrics_root) for s in symbols}
    sc = weekly_scores(metrics, closes.index)
    tk = {}
    for s, m in metrics.items():
        x = np.log(m.taker_ls.where(m.taker_ls > 0))
        x.index = x.index + pd.Timedelta("5min")
        tk[s] = x.rolling("7D").mean().reindex(closes.index, method="ffill")
    rets = closes.pct_change(fill_method=None)
    feat = {"ret_1w": closes / closes.shift(1) - 1, "ret_2w": closes / closes.shift(2) - 1,
            "ret_3w": closes / closes.shift(3) - 1, "ret_4w": closes / closes.shift(4) - 1,
            "vol_8w": rets.rolling(8).std(), "funding_1w": funding.shift(1),
            "retail_z": -sc["1. Retail contrarian"], "smart_minus_retail": sc["2. Smart vs retail"],
            "oi_chg_7d": -sc["3. OI crowding"], "taker_7d": pd.DataFrame(tk)}
    # cross-sectional ranks, so the model sees "where this coin stands this week"
    long = pd.concat({k: v.rank(axis=1, pct=True).stack() for k, v in feat.items()}, axis=1)
    nxt = (closes.shift(-1) / closes - 1).rank(axis=1, pct=True).stack().rename("y")
    data = long.join(nxt, how="left")
    weeks = closes.index
    score = pd.DataFrame(np.nan, index=weeks, columns=closes.columns)
    for i, t in enumerate(weeks):
        if t < start or t >= end:
            continue
        hist = data[data.index.get_level_values(0) < t - pd.Timedelta("7D")].dropna(subset=["y"])
        if hist.index.get_level_values(0).nunique() < 52:
            continue
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=50,
                                          l2_regularization=1.0, random_state=0).fit(hist.drop(columns="y"), hist.y)
        now = data.xs(t, level=0).drop(columns="y")
        score.loc[t, now.index] = m.predict(now)
    w = xsection.rank_backtest(closes, funding, score, 4, cost_bps)
    mom = xsection.rank_backtest(closes, funding, closes / closes.shift(3) - 1, 4, cost_bps)
    mom = mom[mom.week.isin(w.week)]
    from .positioning import weekly_summary
    tbl = pd.DataFrame([weekly_summary("Model (momentum + positioning)", w),
                        weekly_summary("Survivor: 3-week momentum, same weeks", mom)])
    return ["\n## B. Weekly coin-ranking model (long top 4 / short bottom 4, 2 bps per side, unlevered)\n",
            tbl.to_markdown(index=False),
            f"\nWeekly returns correlation, model vs momentum: {w.set_index('week').ret.corr(mom.set_index('week').ret):.2f}"], w


def run(symbols, root: Path, metrics_root: Path, report_dir: Path, cache: Path, first_q="2024Q1", last_q="2026Q3"):
    report_dir.mkdir(parents=True, exist_ok=True)
    wk_lines, w = weekly_part(symbols, root, metrics_root, pd.Timestamp("2023-01-01", tz="UTC"),
                              pd.Timestamp("2026-10-01", tz="UTC"))
    w.to_csv(report_dir / "weekly-model.csv", index=False)
    (report_dir / "results.md").write_text("\n".join(wk_lines) + "\n")
    log.info("weekly part done")
    rows = build_rows(symbols, root, metrics_root, cache)
    rows = rows[rows.index < pd.Timestamp("2026-10-01", tz="UTC")]
    lines = ["# Combined model with Binance positioning: results\n", "Plan: plan.md (written before running).\n"]
    lines += ladder_part(rows, root, first_q, last_q, report_dir) + wk_lines
    path = report_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path
