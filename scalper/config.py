"""Shared settings. Everything here can be overridden from the CLI."""
from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

# Canonical symbols used everywhere in the code; each feed maps them to its own names.
SYMBOLS = ["BTC", "ETH", "SOL"]

BINANCE_SYMBOLS = {"BTC": "btcusdt", "ETH": "ethusdt", "SOL": "solusdt"}
COINBASE_SYMBOLS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}

# Round-trip cost in basis points charged on every simulated trade (entry + exit).
# Binance spot base taker fee is 10 bps per side -> 20 bps round trip, before
# spread and slippage. Set this to what you actually pay.
DEFAULT_COST_BPS = 20.0

TRADE_COLUMNS = ["ts_ms", "price", "qty", "side"]  # side: +1 buyer aggressor, -1 seller aggressor
BOOK_COLUMNS = ["ts_ms", "bid", "bid_qty", "ask", "ask_qty"]
