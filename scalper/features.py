"""Turn raw ticks into fixed-time bars and model features.

Conventions
- A bar is indexed by its START time; every feature on row t only uses data up to the END of bar t.
- The target on row t is the log return of the reference price from the end of bar t to the
  end of bar t+H. So features never see the future, and train/test splits must leave a gap of
  H bars (see model.walk_forward).
- Reference price is the mid (best bid/ask) when book data exists, else the last trade.
"""
import numpy as np
import pandas as pd


def make_bars(trades: pd.DataFrame, book: pd.DataFrame | None, bar_s: int) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    step = bar_s * 1000
    t = trades.assign(
        bar=(trades.ts_ms // step) * step,
        notional=trades.price * trades.qty,
        buy_vol=trades.qty.where(trades.side > 0, 0.0),
        sell_vol=trades.qty.where(trades.side < 0, 0.0),
    )
    bars = t.groupby("bar").agg(
        open=("price", "first"), high=("price", "max"), low=("price", "min"), close=("price", "last"),
        volume=("qty", "sum"), buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
        n_trades=("price", "size"), notional=("notional", "sum"),
    )
    full = pd.RangeIndex(bars.index.min(), bars.index.max() + step, step)
    bars = bars.reindex(full)
    bars["close"] = bars.close.ffill()
    for c in ("open", "high", "low"):
        bars[c] = bars[c].fillna(bars.close)
    for c in ("volume", "buy_vol", "sell_vol", "n_trades", "notional"):
        bars[c] = bars[c].fillna(0.0)
    bars["vwap"] = (bars.notional / bars.volume).where(bars.volume > 0, bars.close)

    if book is not None and not book.empty:
        b = book.assign(bar=(book.ts_ms // step) * step).groupby("bar")[["bid", "bid_qty", "ask", "ask_qty"]].last()
        b = b.reindex(full).ffill()
        bars = bars.join(b)
        bars["mid"] = (bars.bid + bars.ask) / 2
        bars["spread_bps"] = (bars.ask - bars.bid) / bars.mid * 1e4
        # Before the first book update arrives there is no mid; fall back to trades.
        bars["mid"] = bars.mid.fillna(bars.close)
    else:
        bars["mid"] = bars.close
        bars["spread_bps"] = np.nan
    bars.index.name = "bar_start_ms"
    return bars


def _safe_ratio(num, den):
    return (num / den).where(den > 0, 0.0)


def symbol_features(bars: pd.DataFrame) -> pd.DataFrame:
    p = bars.mid
    lp = np.log(p)
    lr = lp.diff()
    f = pd.DataFrame(index=bars.index)
    f["ret_1"] = lr
    for k in (3, 6, 12, 30):
        f[f"ret_{k}"] = lp - lp.shift(k)
    f["vol_12"] = lr.rolling(12).std()
    f["vol_60"] = lr.rolling(60).std()
    f["range_1"] = np.log(bars.high / bars.low)

    flow = bars.buy_vol - bars.sell_vol
    tot = bars.buy_vol + bars.sell_vol
    f["ofi_1"] = _safe_ratio(flow, tot)
    for k in (6, 12, 30):
        f[f"ofi_{k}"] = _safe_ratio(flow.rolling(k).sum(), tot.rolling(k).sum())

    f["log_trades"] = np.log1p(bars.n_trades)
    lv = np.log1p(bars.volume)
    f["volume_z"] = (lv - lv.rolling(60).mean()) / lv.rolling(60).std()
    rvwap = _safe_ratio(bars.notional.rolling(12).sum(), bars.volume.rolling(12).sum()).replace(0, np.nan)
    f["vwap_dev"] = np.log(bars.close / rvwap)

    if bars.spread_bps.notna().any():
        f["spread_bps"] = bars.spread_bps
        depth = bars.bid_qty + bars.ask_qty
        f["book_imb"] = _safe_ratio(bars.bid_qty - bars.ask_qty, depth)
        micro = _safe_ratio(bars.bid * bars.ask_qty + bars.ask * bars.bid_qty, depth).replace(0, np.nan)
        f["micro_dev_bps"] = (micro - bars.mid) / bars.mid * 1e4
    return f


CROSS_COLS = ("ret_1", "ret_3", "ofi_6")


def build_dataset(bars_by_symbol: dict[str, pd.DataFrame], horizon: int) -> dict[str, pd.DataFrame]:
    """Features + targets per symbol, including the other symbols' recent moves (lead-lag)."""
    feats = {s: symbol_features(b) for s, b in bars_by_symbol.items()}
    common = None
    for f in feats.values():
        common = f.index if common is None else common.intersection(f.index)
    out = {}
    for s, f in feats.items():
        f = f.loc[common].copy()
        for other, g in feats.items():
            if other != s:
                for c in CROSS_COLS:
                    f[f"x_{other}_{c}"] = g.loc[common, c]
        b = bars_by_symbol[s].loc[common]
        lp = np.log(b.mid)
        f["fwd_ret"] = lp.shift(-horizon) - lp
        # Spread paid on the round trip, in bps: half at entry, half at exit.
        f["rt_spread_bps"] = (b.spread_bps + b.spread_bps.shift(-horizon)) / 2
        out[s] = f
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("fwd_ret", "rt_spread_bps")]
