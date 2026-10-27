"""Append-only daily CSV storage for raw ticks: data/raw/<exchange>/<symbol>/<kind>-YYYYMMDD.csv.gz"""
import csv
import gzip
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import BOOK_COLUMNS, DATA_DIR, TRADE_COLUMNS

COLUMNS = {"trades": TRADE_COLUMNS, "book": BOOK_COLUMNS}


def raw_path(exchange: str, symbol: str, kind: str, day: str, root: Path = DATA_DIR) -> Path:
    return root / "raw" / exchange / symbol / f"{kind}-{day}.csv.gz"


class TickWriter:
    """Buffers rows in memory and flushes them to the right daily file."""

    def __init__(self, exchange: str, root: Path = DATA_DIR, flush_every: int = 500):
        self.exchange = exchange
        self.root = root
        self.flush_every = flush_every
        self.buffers: dict[tuple[str, str], list] = {}

    def add(self, symbol: str, kind: str, row: list):
        buf = self.buffers.setdefault((symbol, kind), [])
        buf.append(row)
        if len(buf) >= self.flush_every:
            self.flush(symbol, kind)

    def flush(self, symbol: str | None = None, kind: str | None = None):
        for (sym, k), rows in list(self.buffers.items()):
            if (symbol and sym != symbol) or (kind and k != kind) or not rows:
                continue
            by_day: dict[str, list] = {}
            for r in rows:
                day = datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc).strftime("%Y%m%d")
                by_day.setdefault(day, []).append(r)
            for day, day_rows in by_day.items():
                path = raw_path(self.exchange, sym, k, day, self.root)
                path.parent.mkdir(parents=True, exist_ok=True)
                new = not path.exists()
                with gzip.open(path, "at", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(COLUMNS[k])
                    w.writerows(day_rows)
            self.buffers[(sym, k)] = []


def _read_tolerant(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a gzip CSV, keeping every complete row even if the file was cut off mid-write."""
    rows = []
    try:
        with gzip.open(path, "rt", newline="") as f:
            for row in csv.reader(f):
                if len(row) == len(columns) and row[0] != columns[0]:
                    rows.append(row)
    except (EOFError, gzip.BadGzipFile, OSError):
        pass  # collector killed mid-write; keep what was readable
    return pd.DataFrame(rows, columns=columns).apply(pd.to_numeric, errors="coerce").dropna()


def load_raw(exchange: str, symbol: str, kind: str, root: Path = DATA_DIR) -> pd.DataFrame:
    files = sorted((root / "raw" / exchange / symbol).glob(f"{kind}-*.csv.gz"))
    if not files:
        return pd.DataFrame(columns=COLUMNS[kind])
    frames = [_read_tolerant(f, COLUMNS[kind]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("ts_ms", kind="stable").reset_index(drop=True)
