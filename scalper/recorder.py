"""Record a snapshot of every Phemex USDT perpetual once a minute: price, mark and index price (basis),
best bid/ask, open interest, current and predicted funding, 24 h turnover.

Phemex keeps no history of open interest or predicted funding, so these can only be studied from data
recorded going forward. One CSV per UTC day in `<data>/market/YYYYMMDD.csv`. Public data only; no API key.
"""
import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)

URL = "https://api.phemex.com/md/v3/ticker/24hr/all"
FIELDS = {"lastRp": "last", "markRp": "mark", "indexRp": "index", "bidRp": "bid", "askRp": "ask",
          "openInterestRv": "open_interest", "fundingRateRr": "funding", "predFundingRateRr": "pred_funding",
          "turnoverRv": "turnover_24h"}
COLUMNS = ["ts", "symbol", *FIELDS.values()]


def snapshot() -> list[dict]:
    r = requests.get(URL, timeout=20)
    r.raise_for_status()
    rows = r.json()["result"]
    now = int(time.time())
    return [{"ts": now, "symbol": x["symbol"], **{v: x.get(k, "") for k, v in FIELDS.items()}}
            for x in rows if str(x.get("symbol", "")).endswith("USDT")]


def write(rows: list[dict], root: Path) -> Path:
    out = root / "market" / f"{datetime.now(timezone.utc):%Y%m%d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)
    return out


def run(root: Path, every_s: int = 60, once: bool = False):
    while True:
        try:
            rows = snapshot()
            path = write(rows, root)
            log.info("%d perps -> %s", len(rows), path)
        except (requests.RequestException, KeyError, ValueError) as e:
            log.warning("snapshot failed (%s); trying again next minute", e)
        if once:
            return
        time.sleep(every_s - time.time() % every_s)  # on the minute
