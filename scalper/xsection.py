"""Coin-vs-coin momentum: each week, long the coins that rose most and short the ones that rose least.

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


def weekly_panel(symbols_roots: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly closes and weekly funding paid by a long, Monday 00:00 UTC to Monday 00:00 UTC."""
    closes, funding = {}, {}
    for s, root in symbols_roots.items():
        b = make_bars(s, 60, root)
        idx = b.df.index + pd.Timedelta("1h")  # close times
        c = pd.Series(b.df.close.to_numpy(), index=idx)
        f = pd.Series(b.funding_cum, index=b.df.index)  # cumulative funding up to each bar's open
        mondays = c.index[(c.index.dayofweek == 0) & (c.index.hour == 0)]
        closes[s] = c.reindex(mondays)
        funding[s] = f.reindex(mondays).diff().shift(-1)  # paid over the week that starts there
    return pd.DataFrame(closes), pd.DataFrame(funding)


def backtest(closes: pd.DataFrame, funding: pd.DataFrame, lookback: int, k: int, cost_bps: float,
             reverse: bool = False) -> pd.DataFrame:
    past = closes / closes.shift(lookback) - 1
    nxt = closes.shift(-1) / closes - 1
    rows, prev_long, prev_short = [], set(), set()
    for t in closes.index:
        p = past.loc[t].dropna()
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


def stats(w: pd.DataFrame) -> dict:
    r = w.ret
    return {"weeks": len(r), "avg_week_pct": r.mean() * 100, "t_stat": r.mean() / r.std() * np.sqrt(len(r)),
            "win_weeks": (r > 0).mean(), "year_pct": ((1 + r).prod() ** (52 / len(r)) - 1) * 100,
            "max_dd_pct": ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100,
            "avg_cost_week_pct": w.cost.mean() * 100}
