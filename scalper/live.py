"""Live collection and paper trading. Neither places orders."""
import asyncio
import csv
import logging
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import BOOK_COLUMNS, DATA_DIR, TRADE_COLUMNS
from .feeds import binance, coinbase, phemex
from .features import build_dataset, make_bars
from .storage import TickWriter

log = logging.getLogger(__name__)

BOOK_SAMPLE_MS = 100  # keep at most one top-of-book snapshot per symbol per 100 ms


def open_feed(exchange: str, symbols):
    if exchange in ("binance", "binanceus"):
        return binance.binance_stream(symbols, venue=exchange)
    if exchange == "coinbase":
        return coinbase.coinbase_stream(symbols)
    if exchange == "phemex":
        return phemex.phemex_stream(symbols)
    raise SystemExit(f"unknown exchange {exchange!r}")


class BookSampler:
    """Top-of-book feeds can push hundreds of updates a second; keep the latest per time slot."""

    def __init__(self):
        self.last_slot: dict[str, int] = {}

    def is_new_slot(self, symbol: str, ts_ms: int) -> bool:
        slot = ts_ms // BOOK_SAMPLE_MS
        new = self.last_slot.get(symbol) != slot
        self.last_slot[symbol] = slot
        return new


async def collect(exchange: str, symbols, root: Path = DATA_DIR, minutes: float | None = None):
    writer, sampler = TickWriter(exchange, root), BookSampler()
    pending_book: dict[str, list] = {}
    counts = {"trade": 0, "book": 0}
    deadline = time.time() + minutes * 60 if minutes else None
    last_report = time.time()
    try:
        async for kind, sym, row in open_feed(exchange, symbols):
            if kind == "trade":
                writer.add(sym, "trades", row)
            else:
                # Write the last snapshot of each finished slot.
                if sampler.is_new_slot(sym, row[0]) and sym in pending_book:
                    writer.add(sym, "book", pending_book[sym])
                pending_book[sym] = row
            counts[kind] += 1
            if time.time() - last_report > 30:
                log.info("received %d trades, %d book updates", counts["trade"], counts["book"])
                last_report = time.time()
            if deadline and time.time() > deadline:
                break
    finally:
        writer.flush()


class PaperTrader:
    def __init__(self, model_paths, threshold: float, cost_bps: float, out: Path):
        self.models = [joblib.load(p) for p in model_paths]
        bar_sizes = {m["bar_s"] for m in self.models}
        if len(bar_sizes) != 1:
            raise SystemExit("all models must use the same bar size")
        self.bar_s = bar_sizes.pop()
        self.symbols = sorted({s for m in self.models for s in m["symbols"]})
        self.threshold, self.cost_bps = threshold, cost_bps
        keep_ms = 150 * self.bar_s * 1000
        self.keep_ms = keep_ms
        self.trades = {s: deque() for s in self.symbols}
        self.book = {s: deque() for s in self.symbols}
        self.sampler = BookSampler()
        self.open: list[dict] = []
        self.closed: list[dict] = []
        self.out = out
        out.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(out, "a", newline="")
        self._csv = csv.writer(self._fh)
        if out.stat().st_size == 0:
            self._csv.writerow(["symbol", "horizon_s", "entry_ms", "exit_ms", "side", "p_up",
                                "entry_mid", "exit_mid", "gross_bps", "spread_bps", "net_bps"])

    def on_event(self, kind, sym, row):
        if sym not in self.trades:
            return
        if kind == "trade":
            self.trades[sym].append(row)
        elif self.sampler.is_new_slot(sym, row[0]) or not self.book[sym]:
            self.book[sym].append(row)
        else:
            self.book[sym][-1] = row

    def _bars(self, now_ms):
        step = self.bar_s * 1000
        current = (now_ms // step) * step
        out = {}
        for s in self.symbols:
            for q in (self.trades[s], self.book[s]):
                while q and q[0][0] < now_ms - self.keep_ms:
                    q.popleft()
            if not self.trades[s]:
                return None
            t = pd.DataFrame(list(self.trades[s]), columns=TRADE_COLUMNS)
            b = pd.DataFrame(list(self.book[s]), columns=BOOK_COLUMNS)
            bars = make_bars(t, b, self.bar_s)
            out[s] = bars[bars.index < current]  # only finished bars
        return out

    def on_bar(self, now_ms):
        bars = self._bars(now_ms)
        if bars is None or any(len(b) < 70 for b in bars.values()):
            log.info("warming up...")
            return
        last = min(b.index.max() for b in bars.values())
        self._close_positions(bars, last)
        for m in self.models:
            data = build_dataset(bars, m["horizon"], m.get("feature_set", "full"))[m["symbol"]]
            if last not in data.index:
                continue
            x = data.loc[[last]].reindex(columns=m["features"])
            p_up = float(m["model"].predict_proba(x)[0, 1])
            if abs(p_up - 0.5) <= self.threshold:
                continue
            b = bars[m["symbol"]].loc[last]
            self.open.append({"symbol": m["symbol"], "horizon": m["horizon"], "entry_bar": last,
                              "exit_bar": last + m["horizon"] * self.bar_s * 1000,
                              "side": 1 if p_up > 0.5 else -1, "p_up": p_up,
                              "entry_mid": b.mid, "entry_spread": b.spread_bps})

    def _close_positions(self, bars, last):
        still_open = []
        for pos in self.open:
            if pos["exit_bar"] > last:
                still_open.append(pos)
                continue
            b = bars[pos["symbol"]].loc[pos["exit_bar"]]
            gross = pos["side"] * np.log(b.mid / pos["entry_mid"]) * 1e4
            # Half the spread at entry plus half at exit; zero if the feed gave no book.
            spread = np.nan_to_num((pos["entry_spread"] + b.spread_bps) / 2)
            net = gross - spread - self.cost_bps
            self._csv.writerow([pos["symbol"], pos["horizon"] * self.bar_s, pos["entry_bar"], pos["exit_bar"],
                                pos["side"], round(pos["p_up"], 4), pos["entry_mid"], b.mid,
                                round(gross, 3), round(spread, 3), round(net, 3)])
            self.closed.append({"symbol": pos["symbol"], "gross": gross, "net": net})
        self._fh.flush()
        self.open = still_open
        if self.closed:
            c = pd.DataFrame(self.closed)
            summary = c.groupby("symbol").agg(n=("net", "size"), hit=("gross", lambda g: (g > 0).mean()),
                                               gross_bps=("gross", "mean"), net_bps=("net", "mean"))
            log.info("paper results so far:\n%s", summary.round(3).to_string())

    async def run(self, exchange: str):
        async def consume():
            async for kind, sym, row in open_feed(exchange, self.symbols):
                self.on_event(kind, sym, row)

        task = asyncio.create_task(consume())
        step = self.bar_s
        try:
            while True:
                now = time.time()
                # Wake shortly after each bar closes so late trades from that bar have arrived.
                await asyncio.sleep(step - (now % step) + 0.25)
                self.on_bar(int(time.time() * 1000))
        finally:
            task.cancel()
            self._fh.close()
