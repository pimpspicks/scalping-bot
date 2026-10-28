"""Multi-indicator direction model traded with the leveraged ladder exits.

One gradient-boosting model, pooled over BTC/ETH/SOL, predicts whether the close `horizon` bars
ahead is above the current close. It is retrained at the start of every month on all earlier data
(purged by the horizon), so every prediction is out of sample. Trades: long when p_up > 0.5 + thr,
short when p_up < 0.5 - thr, managed with the ladder (stop, partial take profits, max hold).

The calendar is split in two. The development period chooses the horizon, threshold and exit
variant. The locked test period (from `test_start`) is evaluated once, with those choices fixed.
"""
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .candles import load_candles
from .indicators import base_tf, family_of, higher_tf
from .ladder import simulate_ladder
from .swing import make_bars

log = logging.getLogger(__name__)

THRESHOLDS = (0.02, 0.05, 0.08, 0.12)


def build(symbols, bar_min: int, root: Path):
    bars = {s: make_bars(s, bar_min, root) for s in symbols}
    step = pd.Timedelta(minutes=bar_min)
    feats = {}
    for s, b in bars.items():
        m = load_candles(s, root)
        f = base_tf(b.df, b.funding_cum)
        feats[s] = f.join(higher_tf(m, b.df.index, step, "1h")).join(higher_tf(m, b.df.index, step, "4h"))
        log.info("%s: %d bars x %d features", s, len(f), feats[s].shape[1])
    for s in symbols:  # what the other coins are doing
        for o in symbols:
            if o != s:
                for c in ("roc_1", "roc_4", "roc_16", "rsi_14", "supertrend"):
                    feats[s][f"x_{o}_{c}"] = feats[o][c].reindex(feats[s].index)
    return bars, feats


def _pooled(bars, feats, horizon):
    parts = []
    for i, (s, f) in enumerate(feats.items()):
        c = bars[s].df.close
        fut = c.shift(-horizon)
        parts.append(f.assign(sym_id=i, _y=(fut > c).astype(float).where(fut.notna()), _sym=s))
    df = pd.concat(parts).sort_index(kind="stable")
    # Cross-coin columns differ by symbol (x_BTC_* is missing on BTC's own rows); keep them all.
    return df, [c for c in df.columns if not c.startswith("_")]


def _model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=1000, l2_regularization=1.0, random_state=0)


def walk_forward_probs(bars, feats, horizon: int, bar_min: int, start: str) -> pd.DataFrame:
    df, cols = _pooled(bars, feats, horizon)
    purge = pd.Timedelta(minutes=bar_min * horizon)
    out = []
    for m in pd.period_range(start, df.index.max().tz_localize(None), freq="M"):
        lo, hi = m.start_time.tz_localize("UTC"), (m + 1).start_time.tz_localize("UTC")
        tr = df[(df.index < lo - purge) & df._y.notna()]
        te = df[(df.index >= lo) & (df.index < hi)]
        if te.empty:
            continue
        model = _model().fit(tr[cols], tr._y)
        out.append(pd.DataFrame({"sym": te._sym, "p_up": model.predict_proba(te[cols])[:, 1], "y": te._y}))
        log.info("horizon %d bars, %s: trained on %d rows", horizon, m, len(tr))
    return pd.concat(out)


def family_importance(bars, feats, horizon: int, bar_min: int, fit_end: str, eval_end: str) -> pd.DataFrame:
    """AUC lost when each theory's features are shuffled, on months after the model's training data."""
    df, cols = _pooled(bars, feats, horizon)
    fit_end_ts, eval_end_ts = pd.Timestamp(fit_end, tz="UTC"), pd.Timestamp(eval_end, tz="UTC")
    tr = df[(df.index < fit_end_ts - pd.Timedelta(minutes=bar_min * horizon)) & df._y.notna()]
    ev = df[(df.index >= fit_end_ts) & (df.index < eval_end_ts) & df._y.notna()]
    model = _model().fit(tr[cols], tr._y)
    base = roc_auc_score(ev._y, model.predict_proba(ev[cols])[:, 1])
    rng = np.random.default_rng(0)
    rows = [{"family": "all features (AUC)", "auc_drop": base}]
    fams = {}
    for c in cols:
        fams.setdefault(family_of(c), []).append(c)
    for fam, fc in sorted(fams.items()):
        x = ev[cols].copy()
        perm = rng.permutation(len(x))
        x[fc] = x[fc].to_numpy()[perm]
        rows.append({"family": fam, "features": len(fc),
                     "auc_drop": base - roc_auc_score(ev._y, model.predict_proba(x)[:, 1])})
    return pd.DataFrame(rows)


def trade(bars, probs: pd.DataFrame, thr: float, lev: float, stop_margin: float, breakeven: bool,
          fee_bps: float, max_hold_bars: int) -> pd.DataFrame:
    out = []
    for s, b in bars.items():
        p = probs[probs.sym == s].p_up.reindex(b.df.index)
        edge = (p - 0.5).fillna(0).to_numpy()
        sig = np.where(edge > thr, 1, np.where(edge < -thr, -1, 0))
        t = simulate_ladder(b, sig, stop_margin / lev, breakeven, fee_bps, max_hold_bars)
        out.append(t.assign(symbol=s, roi_pct=t.net_bps / 100 * lev))
    return pd.concat(out).sort_values("entry_time", ignore_index=True)


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    months = t.entry_time.dt.tz_localize(None).dt.to_period("M")
    acct = (1 + 0.1 * t.roi_pct / 100).cumprod()  # 10% of the account as margin per trade
    return {"trades": len(t), "trades_per_month": len(t) / months.nunique(),
            "hit_first_target": (t.targets_hit >= 1).mean(), "stopped_out": (t.exit == "stop").mean(),
            "avg_roi_pct": t.roi_pct.mean(), "roi_t_stat": t.roi_pct.mean() / (t.roi_pct.std() / np.sqrt(len(t))),
            "account_return": acct.iloc[-1] - 1, "account_max_dd": (acct / acct.cummax() - 1).min(),
            "positive_months": (t.roi_pct.groupby(months).sum() > 0).mean()}


def run(symbols, bar_min: int, horizons_h, lev: float, stop_margin: float, fee_bps: float, max_hold_h: float,
        first_month: str, test_start: str, root: Path, report_dir: Path) -> Path:
    bars, feats = build(symbols, bar_min, root)
    per_h = 60 // bar_min
    hold = int(max_hold_h * per_h)
    test_ts = pd.Timestamp(test_start, tz="UTC")
    probs, auc_rows, dev_rows = {}, [], []
    for hh in horizons_h:
        h = int(hh * per_h)
        p = walk_forward_probs(bars, feats, h, bar_min, first_month)
        probs[hh] = p
        for name, part in (("development", p[p.index < test_ts]), ("locked test", p[p.index >= test_ts])):
            part = part[part.y.notna()]
            auc_rows.append({"horizon_h": hh, "period": name, "auc": roc_auc_score(part.y, part.p_up),
                             "accuracy": ((part.p_up > 0.5) == (part.y > 0)).mean()})
        for thr, be in itertools.product(THRESHOLDS, (False, True)):
            t = trade(bars, p[p.index < test_ts], thr, lev, stop_margin, be, fee_bps, hold)
            dev_rows.append({"horizon_h": hh, "threshold": thr, "breakeven_stop": be, **stats(t)})
    dev = pd.DataFrame(dev_rows)
    ok = dev[dev.trades >= 100]
    best = (ok if not ok.empty else dev).sort_values("avg_roi_pct", ascending=False).iloc[0]
    log.info("chosen on development data: %s", best.to_dict())
    hh, thr, be = best.horizon_h, best.threshold, bool(best.breakeven_stop)
    p = probs[hh]
    test_trades = trade(bars, p[p.index >= test_ts], thr, lev, stop_margin, be, fee_bps, hold)
    dev_trades = trade(bars, p[p.index < test_ts], thr, lev, stop_margin, be, fee_bps, hold)
    # Trades that start before the test but would still be open are excluded by construction:
    # the test run only sees predictions from test_start on.
    final = pd.DataFrame([{"period": "development (chosen)", **stats(dev_trades)},
                          {"period": "locked test", **stats(test_trades)}])
    by_sym = pd.DataFrame([{"symbol": s, **stats(test_trades[test_trades.symbol == s])} for s in symbols])
    imp = family_importance(bars, feats, int(hh * per_h), bar_min, "2026-01-01", test_start)

    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"predict-phemex-{bar_min}m-{int(lev)}x.md"
    test_trades.to_csv(path.with_suffix(".test-trades.csv"), index=False)
    pd.concat({k: v for k, v in probs.items()}, names=["horizon_h"]).to_csv(path.with_suffix(".probs.csv.gz"))
    path.write_text(_render(final, by_sym, dev, pd.DataFrame(auc_rows), imp, best, bar_min, lev, stop_margin,
                            fee_bps, max_hold_h, first_month, test_start, feats))
    return path


def _fmt(df):
    d = df.copy()
    for c in d.columns:
        if c in ("hit_first_target", "stopped_out", "account_return", "account_max_dd", "positive_months", "accuracy"):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        elif d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return d.to_markdown(index=False)


def _render(final, by_sym, dev, auc, imp, best, bar_min, lev, stop_margin, fee_bps, max_hold_h, first_month,
            test_start, feats) -> str:
    n_feat = next(iter(feats.values())).shape[1]
    return f"""# Multi-indicator prediction bot: Phemex, {bar_min}-minute bars, {lev:g}x ladder

Model: gradient boosting on {n_feat} features (trend, momentum, volatility, volume, structure, ICT, funding,
cross-coin, time; 15 m + 1 h + 4 h), pooled over the coins, retrained monthly from {first_month}.
Exits: stop -{stop_margin:.0%} of margin, half off at +{stop_margin:.0%}, a quarter at +{2 * stop_margin:.0%}, rest at
+{3 * stop_margin:.0%}; max hold {max_hold_h:g} h. Fees {fee_bps} bps of position ({fee_bps / 100 * lev:.1f}% of margin) + funding.
Development = predictions before {test_start}; **locked test = {test_start} onwards, evaluated once**.
`avg_roi_pct` = result per trade in % of margin. `account_*` = 10% of the account as margin per trade.

Chosen on development data only: horizon {best.horizon_h:g} h, threshold {best.threshold}, break-even stop {bool(best.breakeven_stop)}.

## Result

{_fmt(final)}

### Locked test by coin

{_fmt(by_sym)}

## How well does it predict direction?

{_fmt(auc)}

## Which theories help (AUC lost when shuffled; model trained before 2026, scored Jan-Apr 2026)

{_fmt(imp)}

## Development grid (used for the choice)

{_fmt(dev)}
"""
