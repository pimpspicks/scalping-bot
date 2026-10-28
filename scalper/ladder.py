"""Leveraged trades with a fixed stop and a ladder of partial take profits, sized in % of margin.

Default plan: stop at -25% of margin; take profit at +25%, +50% and +75% of margin, selling half
the position at the first two steps and the rest at the last. With leverage L, a step of s% of
margin is a price move of s / L %. Optionally the stop moves to the entry price once the first
target is hit ("break-even").

Bars are checked in order. Within one bar the stop is assumed to come before any target (the
conservative choice), and a gap through the stop fills at the open. Fees (taker, both sides) and
funding are charged on the full position size. Entry signals come from the swing strategies
(classic and ICT) and are picked walk-forward, so every reported trade is out of sample.
"""
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .swing import Bars, make_bars, strategy_grid, walk_forward

log = logging.getLogger(__name__)

SELL_FRACTIONS = (0.5, 0.25, 0.25)  # of the original position at +1, +2, +3 steps


def simulate_ladder(b: Bars, signal: np.ndarray, step: float, breakeven: bool, fee_bps: float,
                    max_hold_bars: int) -> pd.DataFrame:
    """`step` is the price move (as a fraction, e.g. 0.0125) for the stop and each target step."""
    o, h, l, c = (b.df[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(o)
    rows = []
    free = 0
    for i in np.flatnonzero(signal):
        if i < free:
            continue
        e = i + 1
        if e >= n:
            break
        d = signal[i]
        entry = o[e]
        stop = entry * (1 - d * step)
        targets = [entry * (1 + d * step * k) for k in (1, 2, 3)]
        last = min(e + max_hold_bars, n) - 1
        hi, lo = h[e:last + 1], l[e:last + 1]
        m = len(hi)

        def first(mask, frm=0):
            k = np.argmax(mask[frm:]) if mask[frm:].any() else None
            return m if k is None else frm + k

        crossed = (lambda px: hi >= px) if d > 0 else (lambda px: lo <= px)
        stopped = (lambda px: lo <= px) if d > 0 else (lambda px: hi >= px)
        t_idx = [first(crossed(t)) for t in targets]
        s_idx = first(stopped(stop))  # a target in the same bar as the stop counts as missed
        hit = 0
        while hit < 3 and t_idx[hit] < s_idx:
            hit += 1
            if breakeven and hit == 1:
                # From the bar after the first target, the stop sits at the entry price.
                s_idx = min(s_idx, first(stopped(entry), t_idx[0] + 1))
                stop = entry
        pnl = sum(SELL_FRACTIONS[k] * d * (targets[k] / entry - 1) for k in range(hit))
        left = 1.0 - sum(SELL_FRACTIONS[:hit])
        if hit == 3:
            k, exit_kind = t_idx[2], "t3"
        elif s_idx < m:
            k = s_idx
            px = min(o[e + k], stop) if d > 0 else max(o[e + k], stop)
            pnl += left * d * (px / entry - 1)
            exit_kind = "stop" if hit == 0 else f"stop_after_t{hit}"
        else:
            k = m - 1
            pnl += left * d * (c[e + k] / entry - 1)
            exit_kind = "time"
        j = e + k
        free = j + 1
        exit_bar = min(j + 1, n - 1)
        funding = d * (b.funding_cum[exit_bar] - b.funding_cum[e]) * 1e4
        gross = pnl * 1e4
        rows.append((b.df.index[e], d, hit, exit_kind, gross, funding, gross - funding - fee_bps,
                     (j - e + 1) * (b.df.index[1] - b.df.index[0]).total_seconds() / 3600))
    t = pd.DataFrame(rows, columns=["entry_time", "dir", "targets_hit", "exit", "gross_bps", "funding_bps",
                                    "net_bps", "hours_held"])
    t["entry_time"] = pd.to_datetime(t.entry_time, utc=True)
    return t


def run(symbols, bar_min: int, leverages, stop_margin: float, fee_bps: float, train_months: int,
        max_hold_h: float, root: Path, report_dir: Path) -> Path:
    bars = {s: make_bars(s, bar_min, root) for s in symbols}
    hold = int(max_hold_h * 60 / bar_min)
    rows, all_oos = [], []
    for s, b in bars.items():
        others = [o for k, o in bars.items() if k != s]
        grids = {**strategy_grid(b, others, bar_min, "classic"), **strategy_grid(b, others, bar_min, "ict")}
        for lev, be in itertools.product(leverages, (False, True)):
            step = stop_margin / lev
            for strat, entries in grids.items():
                combos = {k: simulate_ladder(b, sig, step, be, fee_bps, hold) for k, sig in entries.items()}
                oos, _ = walk_forward(combos, train_months, purge_hours=max_hold_h)
                if oos.empty:
                    continue
                oos = oos.assign(symbol=s, strategy=strat, leverage=lev, breakeven=be)
                all_oos.append(oos)
                rows.append(_summary(oos, lev) | {"symbol": s, "strategy": strat, "leverage": lev,
                                                  "plan": "break-even stop" if be else "as asked"})
                log.info("%s %s %sx be=%s: %d trades", s, strat, lev, be, len(oos))
    summary = pd.DataFrame(rows)
    trades = pd.concat(all_oos, ignore_index=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"ladder-phemex-{bar_min}m.md"
    trades.to_csv(path.with_suffix(".trades.csv"), index=False)
    path.write_text(_render(summary, bar_min, leverages, stop_margin, fee_bps, max_hold_h, bars))
    return path


def _summary(t: pd.DataFrame, lev: float) -> dict:
    margin = t.net_bps / 100 * lev  # % of margin per trade
    acct = (1 + 0.1 * margin / 100).cumprod()  # 10% of the account as margin on each trade
    months = t.entry_time.dt.tz_localize(None).dt.to_period("M")
    return {
        "trades": len(t),
        "trades_per_month": len(t) / max(months.nunique(), 1),
        "hit_t1": (t.targets_hit >= 1).mean(),
        "hit_t3": (t.targets_hit >= 3).mean(),
        "stopped_out": (t.exit == "stop").mean(),
        "win_rate": (t.net_bps > 0).mean(),
        "avg_hours": t.hours_held.mean(),
        "avg_margin_pct": margin.mean(),
        "account_return": acct.iloc[-1] - 1,
        "account_max_dd": (acct / acct.cummax() - 1).min(),
        "positive_months": (margin.groupby(months).sum() > 0).mean(),
    }


def _render(summary, bar_min, leverages, stop_margin, fee_bps, max_hold_h, bars) -> str:
    s = summary.sort_values("avg_margin_pct", ascending=False).copy()
    cols = ["symbol", "strategy", "leverage", "plan"] + [c for c in s.columns if c not in
                                                         ("symbol", "strategy", "leverage", "plan")]
    s = s[cols]
    for c in ("hit_t1", "hit_t3", "stopped_out", "win_rate", "account_return", "account_max_dd", "positive_months"):
        s[c] = s[c].map(lambda v: f"{v:.1%}")
    for c in ("trades_per_month", "avg_hours", "avg_margin_pct"):
        s[c] = s[c].map(lambda v: f"{v:.2f}")
    span = "; ".join(f"{k} {b.df.index.min():%Y-%m-%d} to {b.df.index.max():%Y-%m-%d}" for k, b in bars.items())
    by_plan = summary.groupby(["leverage", "plan"]).agg(
        setups=("trades", "size"), profitable=("avg_margin_pct", lambda v: int((v > 0).sum())),
        median_margin_pct=("avg_margin_pct", "median"), median_hit_t1=("hit_t1", "median")).reset_index()
    return f"""# Ladder exits: stop -{stop_margin:.0%} of margin, take half off at +{stop_margin:.0%} / +{2 * stop_margin:.0%}, rest at +{3 * stop_margin:.0%}

Data: {span}. {bar_min}-minute bars for entry signals. Fees {fee_bps} bps round trip on the full position
(= {fee_bps / 100:.2f}% x leverage of margin) plus funding. Max hold {max_hold_h:g} h.
Entry settings re-picked monthly from the previous months only (out of sample).
`avg_margin_pct` = average result per trade as % of the margin used. `account_*` assumes 10% of the
account is used as margin on every trade. `hit_t1` / `hit_t3` = share of trades reaching the first / last target.

## By leverage and plan

{by_plan.to_markdown(index=False)}

## Every setup (best first)

{s.to_markdown(index=False)}
"""


def simulate_ladder_limit(b: Bars, signal: np.ndarray, step: float, maker_bps: float, taker_bps: float,
                          max_hold_bars: int, wait_bars: int, offset: float = 0.0) -> pd.DataFrame:
    """Ladder trades entered and taken profit with limit orders (maker), stopped with market orders (taker).

    At a signal bar's close a limit order is placed at close x (1 - d * offset). It fills only if a later
    bar trades through it (strictly beyond the price) within `wait_bars`; otherwise there is no trade.
    Take-profit limits also need price to trade through them. In the bar the entry fills, only the stop
    is checked (the worst case). Stops and time exits pay the taker fee on what is left of the position.
    Fees are per side, in bps of the position size."""
    o, h, l, c = (b.df[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(o)
    rows, free, unfilled = [], 0, 0
    for i in np.flatnonzero(signal):
        if i < free or i + 1 >= n:
            continue
        d = signal[i]
        limit = c[i] * (1 - d * offset)
        w = slice(i + 1, min(i + 1 + wait_bars, n))
        through = (l[w] < limit) if d > 0 else (h[w] > limit)
        if not through.any():
            if i + wait_bars < n:  # still waiting at the end of the data: not counted yet
                unfilled += 1
            continue
        e = i + 1 + int(np.argmax(through))
        entry = min(limit, o[e]) if d > 0 else max(limit, o[e])  # a gap fills at the better open
        stop = entry * (1 - d * step)
        targets = [entry * (1 + d * step * k) for k in (1, 2, 3)]
        last = min(e + max_hold_bars, n) - 1
        hi, lo = h[e:last + 1], l[e:last + 1]
        m = len(hi)
        stop_mask = (lo <= stop) if d > 0 else (hi >= stop)
        s_idx = int(np.argmax(stop_mask)) if stop_mask.any() else m
        t_idx = []
        for t in targets:
            mask = (hi[1:] > t) if d > 0 else (lo[1:] < t)  # not in the fill bar
            t_idx.append(1 + int(np.argmax(mask)) if mask.any() else m)
        hit = 0
        while hit < 3 and t_idx[hit] < s_idx:
            hit += 1
        pnl = sum(SELL_FRACTIONS[k] * d * (targets[k] / entry - 1) for k in range(hit))
        left = 1.0 - sum(SELL_FRACTIONS[:hit])
        fees = maker_bps * (1 + (1 - left))  # entry + take-profit fills, as maker
        if hit == 3:
            k, kind = t_idx[2], "t3"
        elif s_idx < m:
            k = s_idx
            px = min(o[e + k], stop) if (d > 0 and k > 0) else (max(o[e + k], stop) if k > 0 else stop)
            pnl += left * d * (px / entry - 1)
            fees += taker_bps * left
            kind = "stop" if hit == 0 else f"stop_after_t{hit}"
        else:
            k = m - 1
            pnl += left * d * (c[e + k] / entry - 1)
            fees += taker_bps * left
            kind = "time" if e + max_hold_bars <= n else "open"  # "open": data ended first, marked at the last close
        j = e + k
        free = j + 1
        funding = d * (b.funding_cum[min(j + 1, n - 1)] - b.funding_cum[e]) * 1e4
        rows.append((b.df.index[e], d, hit, kind, pnl * 1e4, funding, fees, pnl * 1e4 - funding - fees))
    t = pd.DataFrame(rows, columns=["entry_time", "dir", "targets_hit", "exit", "gross_bps", "funding_bps",
                                    "fee_bps", "net_bps"])
    t["entry_time"] = pd.to_datetime(t.entry_time, utc=True)
    t.attrs["unfilled"] = unfilled
    return t
