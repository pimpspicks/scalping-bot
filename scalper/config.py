"""Shared settings. Everything here can be overridden from the CLI."""
from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

# Canonical symbols used everywhere in the code; each feed maps them to its own names.
SYMBOLS = ["BTC", "ETH", "SOL"]

BINANCE_SYMBOLS = {"BTC": "btcusdt", "ETH": "ethusdt", "SOL": "solusdt"}
COINBASE_SYMBOLS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}
PHEMEX_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}  # USDT perpetuals

# Round-trip fees in basis points charged on every simulated trade (entry + exit), at each
# venue's base taker rate, before spread and slippage. Override with --cost-bps to match your tier.
#   phemex: USDT perpetuals 0.06% taker per side (0.01% maker)
#   binance: spot 0.10% taker per side
COST_BPS = {"phemex": 12.0, "binance": 20.0, "binanceus": 20.0, "coinbase": 20.0}
DEFAULT_EXCHANGE = "phemex"

TRADE_COLUMNS = ["ts_ms", "price", "qty", "side"]  # side: +1 buyer aggressor, -1 seller aggressor
BOOK_COLUMNS = ["ts_ms", "bid", "bid_qty", "ask", "ask_qty"]
