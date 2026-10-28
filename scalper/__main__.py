"""Command line entry point: python -m scalper <command> ..."""
import argparse
import asyncio
import logging
import threading
from pathlib import Path

import joblib

from . import backtest, binance_metrics, build_up, candles, flow, history, ladder, live, maker_test, paper_ladder, pine, positioning, predict, recorder, rules, swing
from .config import COST_BPS, DATA_DIR, DEFAULT_EXCHANGE, SYMBOLS, WATCH_SYMBOLS

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

    dc = sub.add_parser("download-candles", help="fetch Phemex 1-minute candles + funding history")
    dc.add_argument("--symbols", nargs="+", default=SYMBOLS)
    dc.add_argument("--days", type=int, default=365)
    dc.add_argument("--start", help="YYYY-MM-DD; overrides --days")
    dc.add_argument("--end", help="YYYY-MM-DD, exclusive (default: now)")
    dc.add_argument("--minutes", type=int, choices=[1, 15], default=1, help="candle size (15 = much faster)")

    bm = sub.add_parser("download-binance-metrics",
                        help="fetch Binance futures open interest, long/short ratios and taker volume (5 min)")
    bm.add_argument("--symbols", nargs="+", default=WATCH_SYMBOLS)
    bm.add_argument("--start", default="2022-12-01", help="YYYY-MM-DD")
    bm.add_argument("--end", help="YYYY-MM-DD, exclusive (default: today)")
    bm.add_argument("--candles", action="store_true",
                    help="instead fetch Binance 15-minute perp candles + funding into the Phemex candle layout")

    po = sub.add_parser("positioning", help="pre-registered Binance OI / long-short / taker-volume tests")
    po.add_argument("--symbols", nargs="+", default=WATCH_SYMBOLS)
    po.add_argument("--prices", type=Path, default=Path("data-binance"), help="candle root used to price trades")
    po.add_argument("--check-prices", type=Path, help="second candle root (e.g. Phemex) to re-run BTC/ETH/SOL on")
    po.add_argument("--start", default="2023-01-01")
    po.add_argument("--end", default="2026-10-01")
    po.add_argument("--reports", type=Path, default=Path("reports/positioning"))

    sw = sub.add_parser("swing", help="walk-forward swing-trade backtest (minutes to hours) on Phemex candles")
    sw.add_argument("--symbols", nargs="+", default=SYMBOLS)
    sw.add_argument("--bar-minutes", type=int, default=15)
    sw.add_argument("--cost-bps", type=float, default=COST_BPS["phemex"], help="round-trip fees in bps")
    sw.add_argument("--train-months", type=int, default=6)
    sw.add_argument("--strategies", choices=["classic", "ict"], default="classic",
                    help="classic = breakout/trend/mean reversion/model; ict = FVG, sweeps, MSS, silver bullet")
    sw.add_argument("--reports", type=Path, default=Path("reports"))

    ld = sub.add_parser("ladder", help="leveraged trades with a stop and partial take profits, in % of margin")
    ld.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ld.add_argument("--bar-minutes", type=int, default=15)
    ld.add_argument("--leverage", type=float, nargs="+", default=[10, 20])
    ld.add_argument("--stop-margin", type=float, default=0.25, help="stop and each target step, as a fraction of margin")
    ld.add_argument("--cost-bps", type=float, default=COST_BPS["phemex"], help="round-trip fees in bps of position size")
    ld.add_argument("--train-months", type=int, default=6)
    ld.add_argument("--max-hold-hours", type=float, default=72)
    ld.add_argument("--reports", type=Path, default=Path("reports"))

    pr = sub.add_parser("predict", help="multi-indicator direction model traded with the ladder; locked final test")
    pr.add_argument("--symbols", nargs="+", default=SYMBOLS)
    pr.add_argument("--bar-minutes", type=int, default=15)
    pr.add_argument("--horizons-hours", type=float, nargs="+", default=[1, 4, 12])
    pr.add_argument("--leverage", type=float, default=20)
    pr.add_argument("--stop-margin", type=float, default=0.25)
    pr.add_argument("--cost-bps", type=float, default=COST_BPS["phemex"])
    pr.add_argument("--max-hold-hours", type=float, default=72)
    pr.add_argument("--first-month", default="2025-04", help="first month with out-of-sample predictions")
    pr.add_argument("--test-start", default="2026-05-01", help="start of the locked test period")
    pr.add_argument("--reports", type=Path, default=Path("reports"))

    rk = sub.add_parser("rank-indicators", help="test each textbook indicator rule alone with the ladder (development data)")
    rk.add_argument("--symbols", nargs="+", default=SYMBOLS)
    rk.add_argument("--bar-minutes", type=int, nargs="+", default=[15, 60])
    rk.add_argument("--leverage", type=float, default=20)
    rk.add_argument("--stop-margin", type=float, default=0.25)
    rk.add_argument("--cost-bps", type=float, default=COST_BPS["phemex"])
    rk.add_argument("--max-hold-hours", type=float, default=72)
    rk.add_argument("--start", default="2024-10-01")
    rk.add_argument("--test-start", default="2026-05-01")
    rk.add_argument("--rule-set", choices=["classic", "new"], default="classic")
    rk.add_argument("--reports", type=Path, default=Path("reports"))

    mk = sub.add_parser("maker-test", help="every rule traded with limit entries/take profits and market stops")
    mk.add_argument("--symbols", nargs="+", default=SYMBOLS)
    mk.add_argument("--bar-minutes", type=int, nargs="+", default=[15, 60, 240])
    mk.add_argument("--leverage", type=float, default=20)
    mk.add_argument("--stop-margin", type=float, default=0.25)
    mk.add_argument("--maker-bps", type=float, default=1.0)
    mk.add_argument("--taker-bps", type=float, default=6.0)
    mk.add_argument("--max-hold-hours", type=float, default=72)
    mk.add_argument("--wait-bars", type=int, default=4, help="15-minute bars a limit entry waits before it is cancelled")
    mk.add_argument("--start", default="2024-10-01")
    mk.add_argument("--end", default="2026-11-01")
    mk.add_argument("--label", default="dev")
    mk.add_argument("--rule-set", choices=["all", "niche", "papers"], default="all")
    mk.add_argument("--only", type=Path, help="survivors CSV from an earlier run: test just those rules")
    mk.add_argument("--reports", type=Path, default=Path("reports"))

    bu = sub.add_parser("build-up", help="start from the 1h Ichimoku cloud break and add pieces one at a time")
    bu.add_argument("--symbols", nargs="+", default=SYMBOLS)
    bu.add_argument("--leverage", type=float, default=20)
    bu.add_argument("--stop-margin", type=float, default=0.25)
    bu.add_argument("--cost-bps", type=float, default=COST_BPS["phemex"])
    bu.add_argument("--max-hold-hours", type=float, default=72)
    bu.add_argument("--start", default="2024-10-01")
    bu.add_argument("--test-start", default="2026-05-01")
    bu.add_argument("--min-trades", type=int, default=150)
    bu.add_argument("--min-gain", type=float, default=0.1, help="ROI points a piece must add to be kept")
    bu.add_argument("--reports", type=Path, default=Path("reports"))

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

    ra = sub.add_parser("record-all", help="run record-market and record-flow together, until stopped")
    ra.add_argument("--symbols", nargs="+", default=WATCH_SYMBOLS)
    ra.add_argument("--minutes", type=float, help="stop after this long (default: run forever)")

    rf = sub.add_parser("record-flow", help="10-second order-flow and book-depth summaries for many perps")
    rf.add_argument("--symbols", nargs="+", default=WATCH_SYMBOLS)
    rf.add_argument("--minutes", type=float, help="stop after this long (default: run forever)")

    rm = sub.add_parser("record-market", help="every minute, save open interest, funding, basis and quotes of all Phemex perps")
    rm.add_argument("--once", action="store_true", help="take one snapshot and exit")

    pl = sub.add_parser("paper-ladder", help="live paper trading of the ladder strategies on Phemex candles (no orders)")
    pl.add_argument("--symbols", nargs="+", default=SYMBOLS)
    pl.add_argument("--out", type=Path, default=Path("paper"), help="folder for state.json, trades.csv, signals.csv")
    pl.add_argument("--leverage", type=float, default=20)
    pl.add_argument("--stop-margin", type=float, default=0.25)
    pl.add_argument("--maker-bps", type=float, default=1.0)
    pl.add_argument("--taker-bps", type=float, default=6.0)
    pl.add_argument("--max-hold-hours", type=float, default=72)
    pl.add_argument("--wait-bars", type=int, default=4)
    pl.add_argument("--history-days", type=int, default=60)
    pl.add_argument("--once", action="store_true", help="update once and exit")

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
    elif a.cmd == "download-candles":
        candles.download(a.symbols, a.days, root=a.data, start=a.start, end=a.end, minutes=a.minutes)
    elif a.cmd == "download-binance-metrics":
        if a.candles:
            binance_metrics.download_candles(a.symbols, a.start, a.end, a.data)
        else:
            binance_metrics.download(a.symbols, a.start, a.end, a.data)
    elif a.cmd == "positioning":
        path = positioning.run(a.symbols, a.prices, a.data, a.reports, a.start, a.end, a.check_prices)
        print(path.read_text())
    elif a.cmd == "swing":
        path = swing.run(a.symbols, a.bar_minutes, a.cost_bps, a.train_months, a.data, a.reports, a.strategies)
        print(path.read_text())
        print(f"report written to {path}")
    elif a.cmd == "ladder":
        path = ladder.run(a.symbols, a.bar_minutes, a.leverage, a.stop_margin, a.cost_bps, a.train_months,
                          a.max_hold_hours, a.data, a.reports)
        print(path.read_text())
        print(f"report written to {path}")
    elif a.cmd == "predict":
        path = predict.run(a.symbols, a.bar_minutes, a.horizons_hours, a.leverage, a.stop_margin, a.cost_bps,
                           a.max_hold_hours, a.first_month, a.test_start, a.data, a.reports)
        print(path.read_text())
        print(f"report written to {path}")
    elif a.cmd == "rank-indicators":
        path = rules.run(a.symbols, a.bar_minutes, a.leverage, a.stop_margin, a.cost_bps, a.max_hold_hours,
                         a.start, a.test_start, a.data, a.reports, a.rule_set)
        print(path.read_text())
        print(f"report written to {path}")
    elif a.cmd == "maker-test":
        path = maker_test.run(a.symbols, a.bar_minutes, a.leverage, a.stop_margin, a.maker_bps, a.taker_bps,
                              a.max_hold_hours, a.wait_bars, a.start, a.end, a.data, a.reports, a.label, a.only,
                              rule_set=a.rule_set)
        print(path.read_text()[:3000])
        print(f"report written to {path}")
    elif a.cmd == "build-up":
        path = build_up.run(a.symbols, a.leverage, a.stop_margin, a.cost_bps, a.max_hold_hours, a.start,
                            a.test_start, a.min_trades, a.min_gain, a.data, a.reports)
        print(path.read_text())
        print(f"report written to {path}")
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
    elif a.cmd == "record-flow":
        asyncio.run(flow.record(a.symbols, a.data, a.minutes))
    elif a.cmd == "record-all":
        async def both():
            threading.Thread(target=recorder.run, args=(a.data,), daemon=True).start()
            await flow.record(a.symbols, a.data, a.minutes)
        asyncio.run(both())
    elif a.cmd == "record-market":
        recorder.run(a.data, once=a.once)
    elif a.cmd == "paper-ladder":
        paper_ladder.run(a.symbols, a.data, a.out, a.leverage, a.stop_margin, a.maker_bps, a.taker_bps,
                         a.max_hold_hours, a.wait_bars, a.history_days, a.once)
    elif a.cmd == "paper":
        asyncio.run(live.PaperTrader(a.models, a.threshold, a.cost_bps, a.out).run(a.exchange))


if __name__ == "__main__":
    main()
