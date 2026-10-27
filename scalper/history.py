"""Download historical Binance spot aggTrades (free daily dumps) so you can backtest without
waiting days for the live collector.

Caveat: these dumps have trades only, no order book. Bars built from them use last-trade
prices, which bounce between bid and ask. That bounce looks like predictable mean reversion
but cannot be captured, so accuracy on trade-only data is inflated. Prefer live-collected
data (which includes best bid/ask) for any conclusion you act on.
"""
import io
import logging
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from .config import BINANCE_SYMBOLS, DATA_DIR, TRADE_COLUMNS
from .storage import raw_path

log = logging.getLogger(__name__)

BASE = "https://data.binance.vision/data/spot/daily/aggTrades"


def download_day(symbol: str, day: date, root: Path = DATA_DIR) -> Path | None:
    pair = BINANCE_SYMBOLS[symbol].upper()
    out = raw_path("binance", symbol, "trades", day.strftime("%Y%m%d"), root)
    if out.exists():
        return out
    url = f"{BASE}/{pair}/{pair}-aggTrades-{day:%Y-%m-%d}.zip"
    r = requests.get(url, timeout=120)
    if r.status_code == 404:
        log.warning("not published yet: %s", url)
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None)
    if not str(df.iloc[0, 0]).isdigit():  # some dumps ship a header row
        df = df.iloc[1:].astype({0: "int64"})
    # columns: agg_id, price, qty, first_id, last_id, time, is_buyer_maker, is_best_match
    ts = pd.to_numeric(df[5])
    ts = ts // 1000 if ts.iloc[0] > 1e14 else ts  # spot dumps switched to microseconds in 2025
    maker = df[6].astype(str).str.lower() == "true"
    out_df = pd.DataFrame({
        "ts_ms": ts.astype("int64"),
        "price": pd.to_numeric(df[1]),
        "qty": pd.to_numeric(df[2]),
        "side": (~maker).astype(int) * 2 - 1,
    })[TRADE_COLUMNS]
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, compression="gzip")
    log.info("saved %s (%d trades)", out, len(out_df))
    return out


def download(symbols, days: int, end: date | None = None, root: Path = DATA_DIR):
    end = end or date.today() - timedelta(days=1)
    for symbol in symbols:
        for i in range(days):
            download_day(symbol, end - timedelta(days=i), root)
