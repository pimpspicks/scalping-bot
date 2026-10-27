"""Command line entry point: python -m scalper <command> ..."""
import argparse
import asyncio
import logging
from pathlib import Path

import joblib

from . import backtest, history, live, pine
from .config import COST_BPS, DATA_DIR, DEFAULT_EXCHANGE, SYMBOLS

EXCHANGES = ["phemex", "binance", "binanceus", "coinbase"]


def main():
    ap = argparse.ArgumentParser(prog="scalper", description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="record live trades + top of book to data/raw")
    c.add_argument("--exchange", choices=EXCHANGES, default=DEFAULT_EXCHANGE)
    c.add_argument("--symbols", nargs="+", default=SYMBOLS)
    c.add_argument("--minutes", type=float, help="stop after this long (default: run forever)")

    d = sub.add_parser("download", help="fetch historical Binance aggTrades (no order book)")
    d.add_argument("--symbols", nargs="+", default=SYMBOLS)
    d.add_argument("--days", type=int, default=3)

    b = sub.add_parser("backtest", help="walk-forward train/test and write a report")
    b.add_argument("--exchange", choices=EXCHANGES, default=DEFAULT_EXCHANGE)
    b.add_argument("--symbols", nargs="+", default=SYMBOLS)
    b.add_argument("--bar-seconds", type=int, default=5)
    b.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 15, 30], help="seconds ahead to predict")
    b.add_argument("--model", choices=["logreg", "gbm"], default="logreg")
    b.add_argument("--cost-bps", type=float, help="round-trip fees in bps (default: the venue's base taker rate)")
    b.add_argument("--assumed-spread-bps", type=float, default=1.0,
                   help="round-trip spread to charge when no order-book data was recorded")
    b.add_argument("--reports", type=Path, default=Path("reports"))
    b.add_argument("--save-models", action="store_true", help="also fit on all data and save for paper trading")
    b.add_argument("--feature-set", choices=["full", "tv"], default="full",
                   help="tv = only chart OHLCV features, so the model can be exported to TradingView")

    p = sub.add_parser("paper", help="live paper trading with saved models (no orders are sent)")
    p.add_argument("--exchange", choices=EXCHANGES, default=DEFAULT_EXCHANGE)
    p.add_argument("--models", type=Path, nargs="+", required=True)
    p.add_argument("--threshold", type=float, default=0.05, help="trade only when |p_up - 0.5| exceeds this")
    p.add_argument("--cost-bps", type=float, help="round-trip fees in bps (default: the venue's base taker rate)")
    p.add_argument("--out", type=Path, default=Path("reports/paper-trades.csv"))

    x = sub.add_parser("export-pine", help="turn a saved logreg/tv model into a TradingView Pine strategy")
    x.add_argument("model", type=Path)
    x.add_argument("--threshold", type=float, default=0.05)
    x.add_argument("--exchange", choices=EXCHANGES, help="TradingView ticker prefix (default: the one trained on)")
    x.add_argument("--out", type=Path, help="default: pine/<model name>.pine")

    a = ap.parse_args()
    if getattr(a, "cost_bps", "unset") is None:
        a.cost_bps = COST_BPS[a.exchange]
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if a.cmd == "collect":
        asyncio.run(live.collect(a.exchange, a.symbols, a.data, a.minutes))
    elif a.cmd == "download":
        history.download(a.symbols, a.days, root=a.data)
    elif a.cmd == "backtest":
        path = backtest.run(a.exchange, a.symbols, a.bar_seconds, a.horizons, a.model, a.cost_bps,
                            a.assumed_spread_bps, a.data, a.reports, a.save_models, a.feature_set)
        print(path.read_text())
        print(f"report written to {path}")
    elif a.cmd == "export-pine":
        out = a.out or Path("pine") / f"{a.model.stem}.pine"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(pine.export(joblib.load(a.model), a.threshold, a.exchange))
        print(f"wrote {out}; paste it into TradingView's Pine Editor and click 'Add to chart'")
    elif a.cmd == "paper":
        asyncio.run(live.PaperTrader(a.models, a.threshold, a.cost_bps, a.out).run(a.exchange))


if __name__ == "__main__":
    main()
