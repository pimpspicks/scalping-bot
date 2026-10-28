"""Stacked model: every rule signal at once, plus market conditions, deciding when to trust which.

For each coin, every rule of the classic, new and niche sets is computed on 15-minute, 1-hour and 4-hour
charts and placed on the 15-minute grid. Each becomes a feature: the direction of its latest signal if
that came within the last 4 bars of its own chart, else 0. Market-condition features are added (trend
strength, volatility, recent returns on 15m / 1h / 4h, Hurst, funding, hour and weekday, BTC's move, and
the coin's rank against the other coins). Rows are the 15-minute bars where at least one rule fired.

Each row is labelled with the 20x limit-order ladder outcome of a long and of a short entered there.
Two gradient-boosting models (long, short) learn the expected ROI. Walk-forward: before each quarter they
are trained only on rows whose trades had finished before it, then they choose the trades for that quarter:
the better direction, when its predicted ROI is above a threshold. Those trades are run through the same
ladder simulation (one position per coin at a time).
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .indicators import core, higher_tf
from .ladder import _limit_trade, simulate_ladder_limit
from .maker_test import to_15m
from .rules import rule_signals
from .rules2 import new_rule_signals
from .rules3 import hurst, niche_rule_signals
from .swing import make_bars

log = logging.getLogger(__name__)

LEV, STOP_MARGIN, MAKER, TAKER, HOLD, WAIT = 20, 0.25, 1.0, 6.0, 288, 4
CONTEXT = ["adx", "di_diff", "atr_pct", "rvol_16", "vol_ratio", "roc_4", "roc_16", "roc_64", "rsi_14",
           "range_pos", "bb_width", "vlm_z", "ema_50_dist", "ema_200_dist"]


def coin_rows(symbol: str, root: Path, btc_root: Path) -> pd.DataFrame:
    """Feature rows + long/short labels for one coin."""
    b15 = make_bars(symbol, 15, root)
    idx = b15.df.index
    feats, fired = {}, np.zeros(len(idx), bool)
    for tf in (15, 60, 240):
        b = b15 if tf == 15 else make_bars(symbol, tf, root)
        bars = {symbol: b}
        if symbol != "BTC":
            bars["BTC"] = make_bars("BTC", tf, btc_root)
        sigs = {**rule_signals(b.df, b.funding_cum), **new_rule_signals(symbol, bars),
                **niche_rule_signals(symbol, bars)}
        for (fam, name), sig in sigs.items():
            s15 = to_15m(np.asarray(sig), b.df.index, tf, idx)
            fired |= s15 != 0
            k = 4 * tf // 15
            feats[f"{tf}m|{name}"] = pd.Series(np.where(s15 != 0, s15, np.nan), index=idx).ffill(limit=k - 1) \
                .fillna(0).astype(np.int8).to_numpy()
    X = pd.DataFrame(feats, index=idx)
    ctx = core(b15.df)[CONTEXT].add_prefix("15m__")
    X = X.join(ctx.astype(np.float32))
    for tf in ("1h", "4h"):
        X = X.join(higher_tf(b15.df, idx, pd.Timedelta("15min"), tf).astype(np.float32))
    X["hurst_15m"] = hurst(b15.df.close).astype(np.float32).to_numpy()
    X["funding_24h"] = pd.Series(b15.funding_cum, index=idx).diff(96).astype(np.float32).to_numpy()
    t = idx + pd.Timedelta("15min")
    X["hour"], X["dow"] = t.hour.astype(np.int8), t.dayofweek.astype(np.int8)
    X["close"] = b15.df.close.to_numpy()
    # labels
    o, h, l, c = (b15.df[k].to_numpy() for k in ("open", "high", "low", "close"))
    rows = np.flatnonzero(fired)
    lab = {"y_long": np.full(len(rows), np.nan), "y_short": np.full(len(rows), np.nan),
           "end_long": np.full(len(rows), -1), "end_short": np.full(len(rows), -1)}
    for n_, i in enumerate(rows):
        for d, key in ((1, "long"), (-1, "short")):
            r = _limit_trade(o, h, l, c, b15.funding_cum, i, d, STOP_MARGIN / LEV, MAKER, TAKER, HOLD, WAIT)
            if r is None:
                lab[f"y_{key}"][n_] = 0.0  # no fill = no trade = 0
                lab[f"end_{key}"][n_] = min(i + WAIT, len(o) - 1)
            elif r[3] != "open":
                lab[f"y_{key}"][n_] = r[7] / 100 * LEV  # ROI in % of margin
                lab[f"end_{key}"][n_] = r[1]
    out = X.iloc[rows].copy()
    for k, v in lab.items():
        out[k] = v
    out["end_time"] = idx[np.maximum(np.maximum(lab["end_long"], lab["end_short"]), 0)]
    out["symbol"] = symbol
    return out


def add_cross_section(rows: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """The coin's 1-week and 4-week return rank among all coins (0 = worst, 1 = best), and BTC's 4h return."""
    for wk in (1, 4):
        rank = (closes / closes.shift(672 * wk) - 1).rank(axis=1, pct=True)
        rows[f"xs_rank_{wk}w"] = rank.stack().reindex(pd.MultiIndex.from_arrays([rows.index, rows.symbol])) \
            .to_numpy(np.float32)
    rows["btc_ret_4h"] = (closes.BTC / closes.BTC.shift(16) - 1).reindex(rows.index).to_numpy(np.float32)
    return rows


def _model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
                                         l2_regularization=1.0, random_state=0)


def walk_forward(rows: pd.DataFrame, first_quarter: str, thin: int = 4) -> pd.DataFrame:
    """Out-of-sample predicted ROI (long and short) for every row from `first_quarter` on. Training uses every
    `thin`-th row (neighbouring 15-minute rows are nearly the same trade)."""
    feats = [c for c in rows.columns if c not in ("y_long", "y_short", "end_time", "symbol", "close")]
    q = rows.index.tz_localize(None).to_period("Q")
    out = []
    for quarter in sorted(set(q)):
        if quarter < pd.Period(first_quarter, "Q"):
            continue
        start = quarter.start_time.tz_localize("UTC")
        train = rows[(rows.end_time < start) & rows.y_long.notna() & rows.y_short.notna()].iloc[::thin]
        test = rows[q == quarter]
        preds = {}
        for side in ("long", "short"):
            m = _model().fit(train[feats], train[f"y_{side}"])
            preds[f"p_{side}"] = m.predict(test[feats])
        out.append(pd.DataFrame(preds, index=test.index).assign(symbol=test.symbol.to_numpy()))
        log.info("%s: trained on %d rows, predicted %d", quarter, len(train), len(test))
    return pd.concat(out)


def trade(preds: pd.DataFrame, bars15: dict, thr: float) -> pd.DataFrame:
    """Take the better side where its predicted ROI is above `thr`, through the ladder, one position per coin."""
    trades = []
    for s, p in preds.groupby("symbol"):
        b = bars15[s]
        best = np.where(p.p_long >= p.p_short, 1, -1)
        val = np.maximum(p.p_long, p.p_short)
        sig = np.zeros(len(b.df), int)
        pos = b.df.index.get_indexer(p.index[val > thr])
        sig[pos[pos >= 0]] = best[val > thr][pos >= 0]
        t = simulate_ladder_limit(b, sig, STOP_MARGIN / LEV, MAKER, TAKER, HOLD, WAIT)
        trades.append(t.assign(symbol=s, roi_pct=t.net_bps / 100 * LEV))
    t = pd.concat(trades, ignore_index=True)
    return t[t.exit != "open"]
