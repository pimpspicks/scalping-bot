"""Live public market-data feeds. No API keys, no order placement.

Each feed is an async generator yielding normalized events:
    ("trade", symbol, [ts_ms, price, qty, side])
    ("book",  symbol, [ts_ms, bid, bid_qty, ask, ask_qty])
"""
from .binance import binance_stream
from .coinbase import coinbase_stream

FEEDS = {"binance": binance_stream, "coinbase": coinbase_stream}
