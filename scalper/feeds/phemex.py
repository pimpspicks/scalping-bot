"""Phemex USDT-margined perpetuals public websocket: trade_p + orderbook_p.

Docs: https://github.com/phemex/phemex-api-docs/blob/master/Public-Hedged-Perpetual-API.md
The order book channel sends a snapshot then 20 ms incremental diffs; we keep a local book and
emit its top level after every message.
"""
import asyncio
import itertools
import json
import logging

import websockets

from ..config import PHEMEX_SYMBOLS

log = logging.getLogger(__name__)

URL = "wss://phemex.com/ws"


def _apply(levels: dict[float, float], updates):
    for price, qty in updates:
        p, q = float(price), float(qty)
        if q == 0:
            levels.pop(p, None)
        else:
            levels[p] = q


async def phemex_stream(symbols, venue: str = "phemex"):
    reverse = {PHEMEX_SYMBOLS[s]: s for s in symbols}
    ids = itertools.count(1)
    backoff = 1
    while True:
        books = {s: ({}, {}) for s in symbols}  # (bids, asks) price -> qty
        try:
            # Phemex drops connections without a heartbeat inside 30 s; protocol pings count.
            async with websockets.connect(URL, ping_interval=10, max_size=2**22) as ws:
                for sym in reverse:
                    for channel in ("trade_p", "orderbook_p"):
                        await ws.send(json.dumps({"id": next(ids), "method": f"{channel}.subscribe", "params": [sym]}))
                log.info("connected to %s", URL)
                backoff = 1
                async for raw in ws:
                    d = json.loads(raw)
                    sym = reverse.get(d.get("symbol"))
                    if sym is None:
                        continue
                    trades = d.get("trades_p") or d.get("trades")
                    if trades is not None:
                        if d.get("type") == "snapshot":
                            continue  # replay of the last 200 trades on (re)connect; already seen or too old
                        for ts_ns, side, price, qty in trades:
                            # "side" is the taker's side.
                            yield "trade", sym, [int(ts_ns) // 1_000_000, float(price), float(qty),
                                                 1 if side.lower() in ("buy", "bid") else -1]
                        continue
                    book = d.get("orderbook_p") or d.get("book")
                    if book is not None:
                        bids, asks = books[sym]
                        if d.get("type") == "snapshot":
                            bids.clear()
                            asks.clear()
                        _apply(bids, book.get("bids", []))
                        _apply(asks, book.get("asks", []))
                        if bids and asks:
                            bb, ba = max(bids), min(asks)
                            yield "book", sym, [int(d["timestamp"]) // 1_000_000, bb, bids[bb], ba, asks[ba]]
        except (websockets.ConnectionClosed, OSError) as e:
            log.warning("phemex disconnected (%s); reconnecting in %ss", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
