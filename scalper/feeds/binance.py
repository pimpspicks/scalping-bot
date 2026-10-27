"""Binance spot public websocket: aggTrade + bookTicker for each symbol.

Docs: https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
Note: binance.com is geo-blocked in some regions (e.g. the US); use binance.us or coinbase there.
"""
import asyncio
import json
import logging
import time

import websockets

from ..config import BINANCE_SYMBOLS

log = logging.getLogger(__name__)

URLS = {
    "binance": "wss://stream.binance.com:9443/stream?streams=",
    "binanceus": "wss://stream.binance.us:9443/stream?streams=",
}


async def binance_stream(symbols, venue: str = "binance"):
    reverse = {BINANCE_SYMBOLS[s]: s for s in symbols}
    streams = "/".join(f"{BINANCE_SYMBOLS[s]}@{kind}" for s in symbols for kind in ("aggTrade", "bookTicker"))
    url = URLS[venue] + streams
    backoff = 1
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, max_size=2**22) as ws:
                log.info("connected to %s", url)
                backoff = 1
                async for raw in ws:
                    msg = json.loads(raw)
                    stream, d = msg.get("stream", ""), msg.get("data", {})
                    sym = reverse.get(stream.split("@")[0])
                    if sym is None:
                        continue
                    if stream.endswith("@aggTrade"):
                        # m=True means the buyer was the maker, so the aggressor sold.
                        side = -1 if d["m"] else 1
                        yield "trade", sym, [int(d["T"]), float(d["p"]), float(d["q"]), side]
                    elif stream.endswith("@bookTicker"):
                        # Spot bookTicker carries no timestamp; stamp with local receive time.
                        yield "book", sym, [int(time.time() * 1000), float(d["b"]), float(d["B"]),
                                            float(d["a"]), float(d["A"])]
        except (websockets.ConnectionClosed, OSError) as e:
            log.warning("binance disconnected (%s); reconnecting in %ss", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
