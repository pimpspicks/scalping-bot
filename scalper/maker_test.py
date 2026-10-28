"""Every single-indicator rule (classic and new sets) traded with limit orders instead of market orders.

Entries and take profits are limit orders that only fill when price trades through them (maker fee);
stops and time exits are market orders (taker fee). Signals from 1 h / 4 h charts are placed on the
15-minute bar that closes with them, and every trade is simulated on 15-minute candles.

`rule_set` "all" = the classic and new sets, "niche" = rules3, "papers" = rules4.

`run` ranks rules on one period and writes the survivors (positive ROI, t >= `min_t`, positive on
every coin) to `survivors.csv`. Passing `only=` that file runs just those rules, which is how the
survivors are checked once on the locked holdout data.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .ladder import simulate_ladder_limit
from .rules import _stats, rule_signals
from .rules2 import new_rule_signals
from .rules3 import niche_rule_signals
from .rules4 import paper_rule_signals
from .swing import make_bars

log = logging.getLogger(__name__)


def to_15m(sig: np.ndarray, index: pd.DatetimeIndex, bar_min: int, idx15: pd.DatetimeIndex) -> np.ndarray:
    """Move signals from bars opening at t (bar_min long) to the 15 m bar that closes at the same time."""
    out = np.zeros(len(idx15), int)
    nz = np.flatnonzero(sig)
    pos = idx15.get_indexer(index[nz] + pd.Timedelta(minutes=bar_min - 15))
    ok = pos >= 0
    out[pos[ok]] = sig[nz][ok]
    return out


def run(symbols, bar_mins, lev, stop_margin, maker_bps, taker_bps, max_hold_h, wait_bars, start, end,
        root: Path, report_dir: Path, label: str, only: Path | None = None, min_t: float = 2.0,
        rule_set: str = "all") -> Path:
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    keep = None
    if only is not None:
        o = pd.read_csv(only)
        keep = set(zip(o.chart, o.rule, o.way))
    b15 = {s: make_bars(s, 15, root) for s in symbols}
    hold = int(max_hold_h * 4)
    per_rule, signals, unfilled = {}, {}, {}
    for bm in bar_mins:
        bars = b15 if bm == 15 else {s: make_bars(s, bm, root) for s in symbols}
        for s, b in bars.items():
            if rule_set == "niche":
                sigs = niche_rule_signals(s, bars)
            elif rule_set == "papers":
                sigs = paper_rule_signals(s, bars)
            else:
                sigs = {**rule_signals(b.df, b.funding_cum), **new_rule_signals(s, bars)}
            in_period = np.asarray((b.df.index >= lo) & (b.df.index < hi))
            for (fam, name), sig in sigs.items():
                for way, d in (("as rule", 1), ("fade", -1)):
                    key = (f"{bm}m", fam, name, way)
                    if keep is not None and (key[0], name, way) not in keep:
                        continue
                    sg = to_15m(np.where(in_period, sig, 0) * d, b.df.index, bm, b15[s].df.index)
                    t = simulate_ladder_limit(b15[s], sg, stop_margin / lev, maker_bps, taker_bps, hold, wait_bars)
                    signals[key] = signals.get(key, 0) + int((sg != 0).sum())
                    unfilled[key] = unfilled.get(key, 0) + t.attrs["unfilled"]
                    per_rule.setdefault(key, []).append(t.assign(symbol=s, roi_pct=t.net_bps / 100 * lev))
            log.info("%dm %s done", bm, s)
    rows, all_trades = [], []
    for key, ts in per_rule.items():
        t = pd.concat(ts)
        if len(t) < 2:
            continue
        chart, fam, name, way = key
        all_trades.append(t.assign(chart=chart, family=fam, rule=name, way=way))
        filled = 1 - unfilled[key] / max(signals[key], 1)
        rows.append({"chart": chart, "family": fam, "rule": name, "way": way, "fill_rate": filled, **_stats(t)})
    res = pd.DataFrame(rows).sort_values("avg_roi_pct", ascending=False)
    res["survives"] = (res.avg_roi_pct > 0) & (res.t_stat >= min_t) & (res.coins_positive == len(symbols))
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"maker-{label}.md"
    res.to_csv(path.with_suffix(".csv"), index=False)
    res[res.survives].to_csv(report_dir / f"survivors-{label}.csv", index=False)
    pd.concat(all_trades).to_csv(path.with_suffix(".trades.csv.gz"), index=False)
    path.write_text(_render(res, label, lev, stop_margin, maker_bps, taker_bps, max_hold_h, wait_bars, start, end,
                            min_t, len(symbols)))
    return path


def _render(res, label, lev, stop_margin, maker_bps, taker_bps, max_hold_h, wait_bars, start, end, min_t, n_sym):
    r = res.drop(columns="survives").copy()
    for c in ("fill_rate", "hit_first_target", "positive_months"):
        r[c] = r[c].map(lambda v: f"{v:.1%}")
    for c in ("avg_gross_move_bps", "avg_roi_pct", "t_stat"):
        r[c] = r[c].map(lambda v: f"{v:.2f}")
    surv = r[res.survives.to_numpy()]
    return f"""# Limit-order (maker) test: {label}, {lev:g}x ladder

Period {start} to {end}. BTC, ETH, SOL pooled. Entry: limit at the signal close, filled only if price trades
through it within {wait_bars} 15-minute bars, else no trade. Take profits: limit orders, also need a trade-through.
Stops and time exits: market orders. Fees {maker_bps} bps (maker) / {taker_bps} bps (taker) per side + funding.
Exits: stop -{stop_margin:.0%} of margin, half off at +{stop_margin:.0%}, a quarter at +{2 * stop_margin:.0%}, rest at
+{3 * stop_margin:.0%}, max hold {max_hold_h:g} h, simulated on 15-minute candles.
"Survives" = average ROI above 0, t-stat at least {min_t:g}, positive on all {n_sym} coins.
{len(res)} rule variants tested, so a few will pass by luck; that is what the holdout check is for.

## Survivors ({len(surv)})

{surv.to_markdown(index=False) if len(surv) else "None."}

## All rules, best first

{r.to_markdown(index=False)}
"""
