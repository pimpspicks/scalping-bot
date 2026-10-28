"""Record order flow for many Phemex perps as 10-second summaries (small enough to run for months).

Per coin and 10-second slot: taker buy and sell volume (coins and USDT), trade count, the largest trade,
last price, best bid/ask at the slot's end, and resting order-book depth within 0.1% and 0.5% of the
mid price on each side. Phemex sends only the best 30 levels, so for BTC/ETH (tight ticks) both bands
usually hold the whole 30-level book; the bid/ask imbalance is still comparable over time. One CSV per coin per UTC day under
`<data>/flow/<SYMBOL>/YYYYMMDD.csv`. Public data only; no API key.
"""
import asyncio
import csv
import itertools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import websockets

from .candles import _pair
from .feeds.phemex import URL, _apply

log = logging.getLogger(__name__)

SLOT_S = 10
PER_CONNECTION = 5  # coins per websocket connection
COLUMNS = ["ts", "buy_qty", "sell_qty", "buy_usdt", "sell_usdt", "trades", "max_trade_usdt", "last", "bid", "ask",
           "bid_depth_10bp", "ask_depth_10bp", "bid_depth_50bp", "ask_depth_50bp"]


class Slot:
    def __init__(self):
        self.buy_qty = self.sell_qty = self.buy_usdt = self.sell_usdt = self.max_trade = 0.0
        self.trades = 0
        self.last = None

    def add_trade(self, price, qty, side):
        usdt = price * qty
        if side > 0:
            self.buy_qty, self.buy_usdt = self.buy_qty + qty, self.buy_usdt + usdt
        else:
            self.sell_qty, self.sell_usdt = self.sell_qty + qty, self.sell_usdt + usdt
        self.trades += 1
        self.max_trade = max(self.max_trade, usdt)
        self.last = price


def depth(bids: dict, asks: dict):
    """(bid, ask, bid/ask depth in USDT within 0.1% and 0.5% of mid)."""
    if not bids or not asks:
        return [None] * 6
    bb, ba = max(bids), min(asks)
    mid = (bb + ba) / 2
    out = [bb, ba]
    for bp in (0.001, 0.005):
        out.append(sum(p * q for p, q in bids.items() if p >= mid * (1 - bp)))
        out.append(sum(p * q for p, q in asks.items() if p <= mid * (1 + bp)))
    return out


class FlowWriter:
    def __init__(self, root: Path):
        self.root = root

    def write(self, symbol: str, row: list):
        day = datetime.fromtimestamp(row[0], tz=timezone.utc).strftime("%Y%m%d")
        path = self.root / "flow" / symbol / f"{day}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(COLUMNS)
            w.writerow(row)


async def _connection(symbols, writer: FlowWriter, stop: asyncio.Event):
    reverse = {_pair(s): s for s in symbols}
    ids = itertools.count(1)
    backoff = 1
    while not stop.is_set():
        books = {s: ({}, {}) for s in symbols}
        slots = {s: Slot() for s in symbols}
        current = None
        try:
            async with websockets.connect(URL, ping_interval=10, max_size=2**22) as ws:
                for sym in reverse:
                    for channel in ("trade_p", "orderbook_p"):
                        await ws.send(json.dumps({"id": next(ids), "method": f"{channel}.subscribe", "params": [sym]}))
                log.info("flow: connected for %s", ", ".join(symbols))
                backoff = 1
                while not stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1)
                    except asyncio.TimeoutError:
                        raw = None
                    now_slot = int(datetime.now(timezone.utc).timestamp()) // SLOT_S * SLOT_S
                    if current is None:
                        current = now_slot
                    if now_slot > current:  # slot finished: write one row per coin
                        for s in symbols:
                            sl = slots[s]
                            writer.write(s, [current, sl.buy_qty, sl.sell_qty, round(sl.buy_usdt, 2),
                                             round(sl.sell_usdt, 2), sl.trades, round(sl.max_trade, 2), sl.last,
                                             *[None if x is None else round(x, 8) for x in depth(*books[s])]])
                            slots[s] = Slot()
                        current = now_slot
                    if raw is None:
                        continue
                    d = json.loads(raw)
                    s = reverse.get(d.get("symbol"))
                    if s is None:
                        continue
                    trades = d.get("trades_p")
                    if trades is not None:
                        if d.get("type") != "snapshot":
                            for _, side, price, qty in trades:
                                slots[s].add_trade(float(price), float(qty), 1 if side.lower() in ("buy", "bid") else -1)
                        continue
                    book = d.get("orderbook_p")
                    if book is not None:
                        bids, asks = books[s]
                        if d.get("type") == "snapshot":
                            bids.clear()
                            asks.clear()
                        _apply(bids, book.get("bids", []))
                        _apply(asks, book.get("asks", []))
        except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError) as e:
            log.warning("flow: disconnected (%s); reconnecting in %ss", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def record(symbols, root: Path, minutes: float | None = None):
    stop = asyncio.Event()
    writer = FlowWriter(root)
    groups = [symbols[i:i + PER_CONNECTION] for i in range(0, len(symbols), PER_CONNECTION)]
    tasks = [asyncio.create_task(_connection(g, writer, stop)) for g in groups]
    if minutes:
        await asyncio.sleep(minutes * 60)
        stop.set()
    await asyncio.gather(*tasks)
