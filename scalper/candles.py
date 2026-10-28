"""Download Phemex USDT-perpetual history from the public REST API: 1-minute candles and 8-hour
funding rates. Saved as data/candles/phemex/<symbol>/1m-YYYYMM.csv.gz and funding.csv.gz.

Finished months are skipped on re-runs; the current month is always refreshed.
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_DIR, PHEMEX_SYMBOLS

log = logging.getLogger(__name__)

API = "https://api.phemex.com"
PAGE = 1000  # max candles per kline request
CANDLE_COLUMNS = ["ts", "open", "high", "low", "close", "volume", "turnover"]


def _get(path: str, params: dict) -> dict:
    for attempt in range(5):
        try:
            r = requests.get(API + path, params=params, timeout=30)
            if r.status_code == 429:
                raise requests.RequestException("rate limited")
            r.raise_for_status()
            d = r.json()
            if d.get("code", 0) != 0:
                raise requests.RequestException(f"{d.get('code')} {d.get('msg')}")
            return d["data"]
        except requests.RequestException as e:
            log.warning("%s %s failed (%s); retrying", path, params, e)
            time.sleep(2 ** attempt)
    raise SystemExit(f"giving up on {path} {params}")


def candles_dir(symbol: str, root: Path = DATA_DIR) -> Path:
    return root / "candles" / "phemex" / symbol


def _fetch_minutes(pair: str, start: int, end: int) -> pd.DataFrame:
    rows = []
    t = start
    while t < end:
        page_end = min(t + PAGE * 60, end)
        data = _get("/exchange/public/md/v2/kline/list",
                    {"symbol": pair, "resolution": 60, "from": t, "to": page_end})
        # row: [ts, interval, last_close, open, high, low, close, volume, turnover, symbol]
        rows += [[r[0], r[3], r[4], r[5], r[6], r[7], r[8]] for r in data["rows"]]
        t = page_end
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=CANDLE_COLUMNS).apply(pd.to_numeric)
    return df[(df.ts >= start) & (df.ts < end)].drop_duplicates("ts").sort_values("ts")


def download(symbols, days: int, root: Path = DATA_DIR, start: str | None = None, end: str | None = None):
    """Fetch the last `days` days, or the months from `start` up to (not including) `end` (YYYY-MM-DD)."""
    now = datetime.now(timezone.utc)
    stop = min(now, datetime.fromisoformat(end).replace(tzinfo=timezone.utc)) if end else now
    first = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start else now - timedelta(days=days)
    first = first.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    for s in symbols:
        out_dir = candles_dir(s, root)
        out_dir.mkdir(parents=True, exist_ok=True)
        month = first
        while month < stop:
            nxt = (month + timedelta(days=32)).replace(day=1)
            path = out_dir / f"1m-{month:%Y%m}.csv.gz"
            if not path.exists() or nxt > now:
                upto = min(nxt, stop.replace(second=0, microsecond=0))
                frm, old = int(month.timestamp()), None
                if path.exists():  # current month: fetch only what is new
                    old = pd.read_csv(path)
                    frm = int(old.ts.max()) - 60  # refetch the last minute, it may have been partial
                df = _fetch_minutes(PHEMEX_SYMBOLS[s], frm, int(upto.timestamp()))
                if old is not None:
                    df = pd.concat([old[old.ts < frm], df]).sort_values("ts")
                if len(df):
                    df.to_csv(path, index=False, compression="gzip")
                log.info("%s: %d candles for %s", s, len(df), f"{month:%Y-%m}")
            month = nxt
        _download_funding(s, first, stop, out_dir)


def _download_funding(symbol: str, start: datetime, end: datetime, out_dir: Path):
    rows = []
    t = int(start.timestamp() * 1000)
    stop = int(end.timestamp() * 1000)
    step = 90 * 8 * 3600 * 1000  # 90 funding intervals per request
    while t < stop:
        data = _get("/api-data/public/data/funding-rate-history",
                    {"symbol": f".{PHEMEX_SYMBOLS[symbol]}FR8H", "start": t, "end": min(t + step, stop), "limit": 100})
        rows += [[r["fundingTime"], r["fundingRate"]] for r in data["rows"]]
        t += step
    df = pd.DataFrame(rows, columns=["ts_ms", "rate"]).apply(pd.to_numeric).drop_duplicates("ts_ms").sort_values("ts_ms")
    df.to_csv(out_dir / "funding.csv.gz", index=False, compression="gzip")
    log.info("%s: %d funding rates", symbol, len(df))


def load_candles(symbol: str, root: Path = DATA_DIR) -> pd.DataFrame:
    """1-minute candles indexed by UTC open time, with missing minutes filled flat."""
    files = sorted(candles_dir(symbol, root).glob("1m-*.csv.gz"))
    if not files:
        raise SystemExit(f"no candles for {symbol} under {root}/candles; run `download-candles` first")
    df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("ts").sort_values("ts")
    df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True)
    df = df.asfreq("1min")
    df["close"] = df.close.ffill()
    for c in ("open", "high", "low"):
        df[c] = df[c].fillna(df.close)
    df[["volume", "turnover"]] = df[["volume", "turnover"]].fillna(0.0)
    return df


def load_funding(symbol: str, root: Path = DATA_DIR) -> pd.Series:
    path = candles_dir(symbol, root) / "funding.csv.gz"
    if not path.exists():
        return pd.Series(dtype=float)
    f = pd.read_csv(path)
    return pd.Series(f.rate.values, index=pd.to_datetime(f.ts_ms, unit="ms", utc=True))
