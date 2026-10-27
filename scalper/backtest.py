"""Run the full research loop on stored data and write an honest report."""
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_DIR
from .features import build_dataset, feature_columns, make_bars
from .model import evaluate, fit, walk_forward
from .storage import load_raw

log = logging.getLogger(__name__)


def load_bars(exchange: str, symbols, bar_s: int, root: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        trades = load_raw(exchange, s, "trades", root)
        book = load_raw(exchange, s, "book", root)
        if trades.empty:
            raise SystemExit(f"no trade data for {exchange}/{s} under {root}/raw; run `collect` or `download` first")
        out[s] = make_bars(trades, book, bar_s)
        log.info("%s: %d trades, %d book updates -> %d bars", s, len(trades), len(book), len(out[s]))
    return out


def run(exchange: str, symbols, bar_s: int, horizons_s, model_kind: str, cost_bps: float,
        assumed_spread_bps: float, root: Path, report_dir: Path, save_models: bool) -> Path:
    bars = load_bars(exchange, symbols, bar_s, root)
    has_book = all(b.spread_bps.notna().any() for b in bars.values())
    summary, details = [], []
    for h_s in horizons_s:
        horizon = max(1, round(h_s / bar_s))
        data = build_dataset(bars, horizon)
        for s, df in data.items():
            cols = feature_columns(df)
            oos = walk_forward(df, cols, horizon, model_kind)
            stats, trades = evaluate(oos, cost_bps, assumed_spread_bps)
            best = trades.loc[trades.net_bps.idxmax()] if trades.net_bps.notna().any() else trades.iloc[0]
            summary.append({"symbol": s, "horizon_s": horizon * bar_s, **stats,
                            "best_thr": best.threshold, "best_trades": best.trades,
                            "best_gross_bps": best.gross_bps, "best_net_bps": best.net_bps})
            details.append(trades.assign(symbol=s, horizon_s=horizon * bar_s))
            if save_models:
                MODEL_DIR.mkdir(exist_ok=True)
                path = MODEL_DIR / f"{s}_{bar_s}s_h{horizon * bar_s}s_{model_kind}.joblib"
                joblib.dump({"model": fit(df, cols, model_kind), "features": cols, "bar_s": bar_s,
                             "horizon": horizon, "symbol": s, "symbols": list(symbols),
                             "trained_with_book": has_book}, path)
                log.info("saved %s", path)

    summary = pd.DataFrame(summary)
    details = pd.concat(details, ignore_index=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"backtest-{exchange}-{bar_s}s-{model_kind}-{stamp}.md"
    span = {s: (b.index.max() - b.index.min()) / 3.6e6 for s, b in bars.items()}
    path.write_text(_render(summary, details, exchange, bar_s, model_kind, cost_bps,
                            assumed_spread_bps, has_book, span))
    summary.to_csv(path.with_suffix(".summary.csv"), index=False)
    details.to_csv(path.with_suffix(".trades.csv"), index=False)
    return path


def _render(summary, details, exchange, bar_s, model_kind, cost_bps, assumed_spread_bps, has_book, span) -> str:
    fmt = {"flat_share": "{:.1%}", "accuracy": "{:.2%}", "baseline_majority": "{:.2%}",
           "baseline_reversal": "{:.2%}", "auc": "{:.3f}", "best_gross_bps": "{:+.2f}", "best_net_bps": "{:+.2f}"}
    s = summary.copy()
    for c, f in fmt.items():
        s[c] = s[c].map(lambda v, f=f: f.format(v))
    d = details.copy()
    for c in ("gross_bps", "spread_bps", "net_bps"):
        d[c] = d[c].map(lambda v: f"{v:+.2f}")
    d["hit_rate"] = d.hit_rate.map(lambda v: f"{v:.2%}")
    d["trade_share"] = d.trade_share.map(lambda v: f"{v:.1%}")
    hours = ", ".join(f"{k} {v:.1f}h" for k, v in span.items())
    caveat = ("Spread is taken from recorded best bid/ask." if has_book else
              f"**No order-book data**: prices are last trades and spread is an assumed {assumed_spread_bps} bps "
              "round trip. Bid-ask bounce in trade prices makes short-horizon direction look more predictable "
              "than it really is, so treat accuracy here as an upper bound.")
    return f"""# Backtest: {exchange}, {bar_s}s bars, {model_kind}

Data span: {hours}. Fees charged: {cost_bps} bps round trip. {caveat}

All numbers are out-of-sample: purged expanding-window walk-forward, 5 folds over the last 60% of the data.
Accuracy excludes bars where the price did not move. `baseline_reversal` predicts the opposite of the last bar's move.

## Summary

{s.to_markdown(index=False)}

## Trading simulation by confidence threshold

A trade is taken when |p_up - 0.5| > threshold, held for the horizon, one trade per signal (positions may overlap).
`net_bps` = gross - spread - fees, averaged per trade.

{d.to_markdown(index=False)}
"""
