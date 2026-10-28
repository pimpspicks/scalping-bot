"""Binance USDT-M futures positioning metrics from the public archive (data.binance.vision).

One zip per symbol per day, 5-minute rows: open interest, the top-trader and all-account long/short
ratios, and the taker buy/sell volume ratio. Saved as one gzip CSV per symbol per month under
<root>/binance-metrics/<SYMBOL>/. Binance prices lead Phemex's closely, so these are used as signals for
trades priced on Phemex candles.

Column meanings (Binance's names are confusing):
  oi            sum_open_interest, in coins
  oi_usd        sum_open_interest_value, in USDT
  top_acct_ls   count_toptrader_long_short_ratio  (top 20% of traders by margin: accounts long / short)
  top_pos_ls    sum_toptrader_long_short_ratio    (top traders: position size long / short)
  all_acct_ls   count_long_short_ratio            (all accounts long / short; "retail")
  taker_ls      sum_taker_long_short_vol_ratio    (taker buy volume / taker sell volume in the 5 minutes)
"""
import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import requests.adapters

log = logging.getLogger(__name__)

BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
RENAME = {"sum_open_interest": "oi", "sum_open_interest_value": "oi_usd",
          "count_toptrader_long_short_ratio": "top_acct_ls", "sum_toptrader_long_short_ratio": "top_pos_ls",
          "count_long_short_ratio": "all_acct_ls", "sum_taker_long_short_vol_ratio": "taker_ls"}
COLUMNS = ["ts"] + list(RENAME.values())


def binance_pair(symbol: str) -> str:
    """Project symbol -> Binance USDT-M symbol (u1000PEPE -> 1000PEPEUSDT)."""
    return f"{symbol.removeprefix('u')}USDT"


def _day(session: requests.Session, pair: str, d: date) -> pd.DataFrame | None:
    url = f"{BASE}/{pair}/{pair}-metrics-{d:%Y-%m-%d}.zip"
    for attempt in range(5):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                df = pd.read_csv(z.open(z.namelist()[0]))
            df = df.rename(columns=RENAME)
            df["ts"] = (pd.to_datetime(df.create_time, utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")
            return df[COLUMNS]
        except (requests.RequestException, zipfile.BadZipFile) as e:
            log.debug("%s %s: %s", pair, d, e)
    log.warning("giving up on %s", url)
    return None


def _session(workers: int) -> requests.Session:
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers))
    return s


def download(symbols, start: str, end: str | None, root: Path, workers: int = 32):
    """Fetch every day from `start` up to (not including) `end` (default: today). Finished months that
    already have a file are skipped."""
    first = date.fromisoformat(start).replace(day=1)
    stop = date.fromisoformat(end) if end else date.today()
    session = _session(workers)
    for s in symbols:
        pair, out = binance_pair(s), root / "binance-metrics" / s
        out.mkdir(parents=True, exist_ok=True)
        month = first
        while month < stop:
            nxt = (month + timedelta(days=32)).replace(day=1)
            path = out / f"{month:%Y%m}.csv.gz"
            if not path.exists() or nxt > stop:
                days = [month + timedelta(n) for n in range((min(nxt, stop) - month).days)]
                with ThreadPoolExecutor(workers) as ex:
                    parts = [p for p in ex.map(lambda d: _day(session, pair, d), days) if p is not None]
                if parts:
                    pd.concat(parts).to_csv(path, index=False, compression="gzip")
                log.info("%s: %d days for %s", s, len(parts), f"{month:%Y-%m}")
            month = nxt


def load(symbol: str, root: Path) -> pd.DataFrame:
    """5-minute metrics indexed by UTC time, duplicates dropped, empty ratio cells as NaN."""
    files = sorted((root / "binance-metrics" / symbol).glob("*.csv.gz"))
    if not files:
        raise SystemExit(f"no Binance metrics for {symbol} under {root}; run `download-binance-metrics` first")
    df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("ts").sort_values("ts")
    df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True)
    return df.apply(pd.to_numeric, errors="coerce")


def _zip_csv(session: requests.Session, url: str) -> pd.DataFrame | None:
    for attempt in range(5):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                return pd.read_csv(z.open(z.namelist()[0]))
        except (requests.RequestException, zipfile.BadZipFile) as e:
            log.debug("%s: %s", url, e)
    log.warning("giving up on %s", url)
    return None


def download_candles(symbols, start: str, end: str | None, root: Path, minutes: int = 15):
    """Binance USDT-M perp candles + funding history, saved in the same layout as Phemex candles
    (<root>/candles/phemex/<SYMBOL>/), so every backtest can run on them unchanged with --data <root>.
    Used where Phemex's own history can't be fetched. Monthly files only (the archive publishes a month
    after it ends)."""
    first = date.fromisoformat(start).replace(day=1)
    stop = date.fromisoformat(end) if end else date.today()
    months, m = [], first
    while m < stop:
        months.append(m)
        m = (m + timedelta(days=32)).replace(day=1)
    session = _session(16)
    base = "https://data.binance.vision/data/futures/um/monthly"
    for s in symbols:
        pair, out = binance_pair(s), root / "candles" / "phemex" / s
        out.mkdir(parents=True, exist_ok=True)
        urls = [f"{base}/klines/{pair}/{minutes}m/{pair}-{minutes}m-{m:%Y-%m}.zip" for m in months]
        with ThreadPoolExecutor(16) as ex:
            kl = list(ex.map(lambda u: _zip_csv(session, u), urls))
        for m, k in zip(months, kl):
            if k is None:
                continue
            if "open_time" not in k.columns:  # some files have no header row
                k = pd.DataFrame(np.vstack([k.columns.to_numpy(), k.to_numpy()])[:, :8],
                                 columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                          "quote_volume"])
            k = k.apply(pd.to_numeric, errors="coerce").dropna(subset=["open_time"])
            df = pd.DataFrame({"ts": (k.open_time // 1000).astype("int64"), "open": k.open, "high": k.high,
                               "low": k.low, "close": k.close, "volume": k.volume, "turnover": k.quote_volume})
            df.to_csv(out / f"{minutes}m-{m:%Y%m}.csv.gz", index=False, compression="gzip")
        furls = [f"{base}/fundingRate/{pair}/{pair}-fundingRate-{m:%Y-%m}.zip" for m in months]
        with ThreadPoolExecutor(16) as ex:
            fr = [f for f in ex.map(lambda u: _zip_csv(session, u), furls) if f is not None]
        if fr:
            f = pd.concat(fr)
            f = pd.DataFrame({"ts_ms": f.calc_time // 1000 * 1000, "rate": f.last_funding_rate})
            f.drop_duplicates("ts_ms").sort_values("ts_ms").to_csv(out / "funding.csv.gz", index=False,
                                                                     compression="gzip")
        log.info("%s: %d months of candles, %d of funding", s, sum(k is not None for k in kl), len(fr))
