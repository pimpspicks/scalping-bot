# Scalping bot: BTC / ETH / SOL short-horizon predictor

Predicts the direction of the mid price 5 to 30 seconds ahead from live trades and top of book,
evaluates it honestly out of sample, and paper trades it live. **It never places orders.**

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                      # sanity checks on synthetic data

# 1. Record live data (trades + best bid/ask). Leave it running for a day or more.
python -m scalper collect                       # Phemex USDT perpetuals by default; also binance / binanceus / coinbase

# 2. Walk-forward backtest, 5s bars, predicting 5/10/15/30s ahead
python -m scalper backtest --save-models        # fees default to Phemex taker (0.06%/side); override with --cost-bps

# 3. Paper trade the saved models live (writes reports/paper-trades.csv)
python -m scalper paper --models models/*_h10s_logreg_full.joblib --threshold 0.05
```

`python -m scalper download --days 3` fetches Binance historical trades for a quick start (use
`--exchange binance` when backtesting them). Those files have no order book, so see the bounce
warning below before trusting the results.

Fees: Phemex perpetuals are 0.06% taker / 0.01% maker per side at the base tier, so the default
charge is 12 bps round trip. If you can get filled as a maker, pass `--cost-bps 2`, but limit
orders don't always fill and the sim doesn't model that.

## TradingView

TradingView can't receive data from a bot, so the bot goes to TradingView instead: it exports a
trained model as a **Pine Script strategy** that recomputes the features on the chart and shows
the model's signal, backtests in TradingView's strategy tester, and fires alerts.

```bash
python -m scalper backtest --feature-set tv --save-models     # model using chart-visible data only
python -m scalper export-pine models/BTC_5s_h10s_logreg_tv.joblib
# -> pine/BTC_5s_h10s_logreg_tv.pine: paste into the Pine Editor, add to the PHEMEX:BTCUSDT.P 5S chart
```

Tradeoffs to know:
* Pine Script only sees each bar's OHLCV, not the order book or the buy/sell split, so the `tv`
  feature set drops order-flow, spread and book features. Compare its report with the `full` one
  to see what that costs. The `full` model only runs in Python (`paper`).
* Seconds charts (5S, 10S...) need a paid TradingView plan.
* TradingView's strategy tester fills at bar closes with no spread, so it looks better than the
  Python backtest, which charges spread. Trust the Python numbers.
* TradingView skips seconds bars with no trades, while the Python side fills them in. On
  BTC/ETH/SOL perps that is rare at 5s, but signals can differ slightly on quiet bars.
* Alerts: create one alert on the strategy with "alert() function calls only". Each signal sends
  JSON (`symbol`, `side`, `p_up`, `horizon_s`), which can go to a webhook URL.
* Only logistic regression exports (gradient boosting can't be written as a short formula).
  A test checks the exported formula gives the same probabilities as the Python model.

## How it works

| Step | File | What it does |
|---|---|---|
| Feeds | `scalper/feeds/` | Public websockets (Phemex trade_p + orderbook_p, Binance aggTrade + bookTicker, Coinbase matches + ticker), auto-reconnect |
| Storage | `scalper/storage.py` | Daily gzip CSVs in `data/raw/<exchange>/<symbol>/` |
| Bars | `scalper/features.py` | Fixed-time bars (default 5s): OHLC, buy/sell volume, trade count, VWAP, last bid/ask |
| Features | `scalper/features.py` | Returns over 1-30 bars, volatility, order-flow imbalance, volume z-score, VWAP deviation, spread, book imbalance, microprice, plus the other coins' recent moves (lead-lag) |
| Model | `scalper/model.py` | Logistic regression (default) or gradient boosting; label = sign of mid return over the horizon |
| Evaluation | `scalper/model.py` | Purged expanding-window walk-forward: each test block is predicted only by a model trained on earlier data, with a gap so no label overlaps |
| Report | `scalper/backtest.py` | Accuracy vs. naive baselines, AUC, and a trading sim per confidence threshold with gross, spread and fee costs |
| Paper | `scalper/live.py` | Rebuilds features every bar from the live feed and logs simulated trades and PnL |
| TradingView | `scalper/pine.py` | Exports a logreg model on chart-only features as a Pine Script strategy with alerts |

## Read this before trusting any number

* **Fees dominate.** A 5-30s move in BTC is typically a few basis points. Phemex's base taker fee
  is 6 bps per side (12 bps round trip). A model that is right 55% of the time on a 3 bps move earns
  about 0.3 bps gross per trade, so it loses money at almost any retail taker fee. The report shows
  `gross_bps` next to `net_bps` so you can see the breakeven fee directly.
* **Bid-ask bounce fakes an edge.** Last-trade prices flip between bid and ask, so "price went down,
  it will go up" looks ~65% accurate even on pure noise (see `test_trade_prices_fake_an_edge`).
  That is why labels use the mid price, which needs recorded book data from `collect`.
* **Overlapping trades.** The sim takes one trade per signal, so with a 30s horizon on 5s bars
  positions overlap. Treat `net_bps` as expectancy per trade, not as account PnL.
* **Latency and queue position are not modeled.** Live fills will be worse than the sim.
* **Regimes shift.** Hours of data prove little. Collect at least several days across quiet and
  volatile sessions before concluding anything, and re-check with `paper` before risking money.

Ways to make an edge more plausible: maker (limit) orders and a fee tier with rebates, longer
horizons, trading only when spread is tight and the signal is strong, and richer order-book data
(depth, not just top of book).
