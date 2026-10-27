# Scalping bot: BTC / ETH / SOL short-horizon predictor

Predicts the direction of the mid price 5 to 30 seconds ahead from live trades and top of book,
evaluates it honestly out of sample, and paper trades it live. **It never places orders.**

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                      # sanity checks on synthetic data

# 1. Record live data (trades + best bid/ask). Leave it running for a day or more.
python -m scalper collect --exchange binance        # or binanceus / coinbase

# 2. Walk-forward backtest, 5s bars, predicting 5/10/15/30s ahead
python -m scalper backtest --exchange binance --cost-bps 20 --save-models

# 3. Paper trade the saved models live (writes reports/paper-trades.csv)
python -m scalper paper --exchange binance --models models/*_h10s_logreg.joblib --threshold 0.05
```

`python -m scalper download --days 3` fetches Binance historical trades for a quick start, but
those files have no order book, so see the bounce warning below before trusting the results.

## How it works

| Step | File | What it does |
|---|---|---|
| Feeds | `scalper/feeds/` | Public websockets (Binance aggTrade + bookTicker, Coinbase matches + ticker), auto-reconnect |
| Storage | `scalper/storage.py` | Daily gzip CSVs in `data/raw/<exchange>/<symbol>/` |
| Bars | `scalper/features.py` | Fixed-time bars (default 5s): OHLC, buy/sell volume, trade count, VWAP, last bid/ask |
| Features | `scalper/features.py` | Returns over 1-30 bars, volatility, order-flow imbalance, volume z-score, VWAP deviation, spread, book imbalance, microprice, plus the other coins' recent moves (lead-lag) |
| Model | `scalper/model.py` | Logistic regression (default) or gradient boosting; label = sign of mid return over the horizon |
| Evaluation | `scalper/model.py` | Purged expanding-window walk-forward: each test block is predicted only by a model trained on earlier data, with a gap so no label overlaps |
| Report | `scalper/backtest.py` | Accuracy vs. naive baselines, AUC, and a trading sim per confidence threshold with gross, spread and fee costs |
| Paper | `scalper/live.py` | Rebuilds features every bar from the live feed and logs simulated trades and PnL |

## Read this before trusting any number

* **Fees dominate.** A 5-30s move in BTC is typically a few basis points. Binance's base taker fee
  is 10 bps per side (20 bps round trip). A model that is right 55% of the time on a 3 bps move earns
  about 0.3 bps gross per trade, so it loses money at almost any retail fee tier. The report shows
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
