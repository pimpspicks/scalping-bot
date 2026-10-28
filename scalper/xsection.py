"""Coin-vs-coin strategies (momentum, funding crowding) and per-coin funding extremes.

Coin-vs-coin momentum: each week, long the coins that rose most and short the ones that rose least.

Liu, Tsyvinski & Wu (2022, "Common Risk Factors in Cryptocurrency") find that the past 1 to 4 weeks'
return predicts the next week's, across coins. Here: every Monday 00:00 UTC, rank the coins by their
return over the last `lookback` weeks, go long the top `k` and short the bottom `k` in equal size, and
hold one week. Costs: `cost_bps` per side on every position opened or closed (unchanged positions are
kept, not traded), plus the funding each side paid or received. Returns are unlevered, in % of the
capital on each side.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .swing import make_bars


def panel(symbols_roots: dict[str, Path], freq: str = "W-MON") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Closes at each rebalance time (00:00 UTC; weekly on Mondays or daily) and the funding paid by a long
    over the period that starts there."""
    closes, funding = {}, {}
    for s, root in symbols_roots.items():
        b = make_bars(s, 60, root)
        idx = b.df.index + pd.Timedelta("1h")  # close times
        c = pd.Series(b.df.close.to_numpy(), index=idx)
        f = pd.Series(b.funding_cum, index=b.df.index)  # cumulative funding up to each bar's open
        at = c.index[c.index.hour == 0]
        if freq == "W-MON":
            at = at[at.dayofweek == 0]
        closes[s] = c.reindex(at)
        funding[s] = f.reindex(at).diff().shift(-1)
    return pd.DataFrame(closes), pd.DataFrame(funding)


def weekly_panel(symbols_roots: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return panel(symbols_roots, "W-MON")


def backtest(closes: pd.DataFrame, funding: pd.DataFrame, lookback: int, k: int, cost_bps: float,
             reverse: bool = False) -> pd.DataFrame:
    """Momentum: rank by the return over the last `lookback` periods, long the top k, short the bottom k."""
    return rank_backtest(closes, funding, closes / closes.shift(lookback) - 1, k, cost_bps, reverse)


def rank_backtest(closes: pd.DataFrame, funding: pd.DataFrame, score: pd.DataFrame, k: int, cost_bps: float,
                  reverse: bool = False) -> pd.DataFrame:
    """Each period: long the k coins with the highest `score`, short the k lowest (`reverse` swaps), hold one period."""
    nxt = closes.shift(-1) / closes - 1
    rows, prev_long, prev_short = [], set(), set()
    for t in closes.index:
        p = score.loc[t].dropna()
        p = p[nxt.loc[t].reindex(p.index).notna()]
        if len(p) < 2 * k + 2:
            continue
        order = p.sort_values()
        long, short = set(order.index[-k:]), set(order.index[:k])
        if reverse:
            long, short = short, long
        r_long = nxt.loc[t, list(long)].mean() - funding.loc[t, list(long)].fillna(0).mean()
        r_short = -nxt.loc[t, list(short)].mean() + funding.loc[t, list(short)].fillna(0).mean()
        turnover = (len(long ^ prev_long) + len(short ^ prev_short)) / k  # positions opened + closed, per side
        cost = turnover / 2 * cost_bps / 1e4
        rows.append({"week": t, "long": ",".join(sorted(long)), "short": ",".join(sorted(short)),
                     "ret": (r_long + r_short) / 2 - cost, "gross": (r_long + r_short) / 2, "cost": cost})
        prev_long, prev_short = long, short
    return pd.DataFrame(rows)


def signal_backtest(closes: pd.DataFrame, funding: pd.DataFrame, signal: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Per-coin positions (+1 / -1 / 0) held one period, equal size across the open ones; flat periods return 0."""
    nxt = closes.shift(-1) / closes - 1
    rows, prev = [], pd.Series(0, index=closes.columns)
    for t in closes.index[:-1]:
        d = signal.loc[t].fillna(0).where(nxt.loc[t].notna(), 0)
        active = d[d != 0]
        changed = (d != prev).sum() + ((d != prev) & (d != 0) & (prev != 0)).sum()  # a flip = close + open
        n = max(len(active), 1)
        gross = (active * (nxt.loc[t, active.index] - funding.loc[t, active.index].fillna(0))).sum() / n
        cost = changed / n * cost_bps / 1e4 if len(active) else (changed > 0) * cost_bps / 1e4
        rows.append({"week": t, "positions": len(active), "ret": gross - cost, "gross": gross, "cost": cost})
        prev = d
    return pd.DataFrame(rows)


def stats(w: pd.DataFrame) -> dict:
    r = w.ret
    return {"weeks": len(r), "avg_week_pct": r.mean() * 100, "t_stat": r.mean() / r.std() * np.sqrt(len(r)),
            "win_weeks": (r > 0).mean(), "year_pct": ((1 + r).prod() ** (52 / len(r)) - 1) * 100,
            "max_dd_pct": ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100,
            "avg_cost_week_pct": w.cost.mean() * 100}
