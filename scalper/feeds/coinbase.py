"""Coinbase Exchange public websocket: matches (trades) + ticker (best bid/ask).

Docs: https://docs.cdp.coinbase.com/exchange/websocket-feed/overview
"""
import asyncio
import json
import logging
from datetime import datetime

import websockets

from ..config import COINBASE_SYMBOLS

log = logging.getLogger(__name__)

URL = "wss://ws-feed.exchange.coinbase.com"


def _ts_ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


async def coinbase_stream(symbols, venue: str = "coinbase"):
    reverse = {COINBASE_SYMBOLS[s]: s for s in symbols}
    sub = {"type": "subscribe", "product_ids": list(reverse), "channels": ["matches", "ticker"]}
    backoff = 1
    while True:
        try:
            async with websockets.connect(URL, ping_interval=20, max_size=2**22) as ws:
                await ws.send(json.dumps(sub))
                log.info("connected to %s", URL)
                backoff = 1
                async for raw in ws:
                    d = json.loads(raw)
                    sym = reverse.get(d.get("product_id"))
                    if sym is None or "time" not in d:
                        continue
                    if d["type"] in ("match", "last_match"):
                        # "side" is the maker's side, so a sell-side maker means a buy aggressor.
                        side = 1 if d["side"] == "sell" else -1
                        yield "trade", sym, [_ts_ms(d["time"]), float(d["price"]), float(d["size"]), side]
                    elif d["type"] == "ticker" and d.get("best_bid"):
                        yield "book", sym, [_ts_ms(d["time"]), float(d["best_bid"]), float(d.get("best_bid_size") or 0),
                                            float(d["best_ask"]), float(d.get("best_ask_size") or 0)]
        except (websockets.ConnectionClosed, OSError) as e:
            log.warning("coinbase disconnected (%s); reconnecting in %ss", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
