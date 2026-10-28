"""Cross-market check: the paper bot's three strategies, unchanged, on non-crypto Phemex perpetuals.

Phemex lists gold (PAXG since Dec 2022; XAU, XAG since Jan 2026), index perps (NAS100, SPYX, QQQ)
and single stocks (NVDA, TSLA, AAPL). The signals, the limit-order ladder simulation, the leverage,
the exits and the fees are exactly those of `paper-ladder` (STRATEGIES there); nothing is tuned here.
BTC/ETH/SOL can be run over the same dates for comparison.

Writes cross-market.md (per strategy and market), cross-market.csv and cross-market.trades.csv.gz.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .ladder import simulate_ladder_limit
from .maker_test import to_15m
from .paper_ladder import STRATEGIES
from .rules import rule_signals
from .rules3 import niche_rule_signals
from .swing import make_bars

log = logging.getLogger(__name__)


def _stats(t: pd.DataFrame) -> dict:
    n = len(t)
    return {"trades": n, "hit_first_target": (t.targets_hit >= 1).mean() if n else np.nan,
            "avg_roi_pct": t.roi_pct.mean() if n else np.nan,
            "t_stat": t.roi_pct.mean() / (t.roi_pct.std() / np.sqrt(n)) if n > 1 else np.nan,
            "total_roi_pct": t.roi_pct.sum()}


def trades_for(symbols, root: Path, start, end, lev, stop_margin, maker_bps, taker_bps, max_hold_h,
               wait_bars) -> pd.DataFrame:
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    out = []
    for s in symbols:
        b15 = make_bars(s, 15, root)
        cache = {}
        for name, (bm, rule_set, key) in STRATEGIES.items():
            if (bm, rule_set) not in cache:
                b = b15 if bm == 15 else make_bars(s, bm, root)
                sigs = rule_signals(b.df, b.funding_cum) if rule_set == "classic" else niche_rule_signals(s, {s: b})
                cache[(bm, rule_set)] = (b.df.index, sigs)
            idx, sigs = cache[(bm, rule_set)]
            sig = np.where(np.asarray((idx >= lo) & (idx < hi)), sigs[key], 0)
            sg = to_15m(sig, idx, bm, b15.df.index)
            t = simulate_ladder_limit(b15, sg, stop_margin / lev, maker_bps, taker_bps, int(max_hold_h * 4), wait_bars)
            out.append(t.assign(strategy=name, symbol=s, roi_pct=t.net_bps / 100 * lev))
        log.info("%s done (%s to %s)", s, b15.df.index[0].date(), b15.df.index[-1].date())
    return pd.concat(out, ignore_index=True)


def summarize(t: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for strat, ts in t.groupby("strategy", sort=False):
        for g, syms in groups.items():
            rows.append({"strategy": strat, "market": g, **_stats(ts[ts.symbol.isin(syms)])})
        for s, x in ts.groupby("symbol"):
            rows.append({"strategy": strat, "market": s, **_stats(x)})
    return pd.DataFrame(rows)


GROUPS = {"gold, silver": ["PAXG_IDX", "XAU_IDX", "XAG_IDX"], "indices": ["NAS100_IDX", "SPYX_IDX", "QQQ_IDX"],
          "stocks": ["NVDA_IDX", "TSLA_IDX", "AAPL_IDX"]}


def run(root: Path, report_dir: Path, lev, stop_margin, maker_bps, taker_bps, max_hold_h, wait_bars,
        gold_start="2023-01-01", start="2026-01-01", end="2026-11-01") -> Path:
    kw = dict(lev=lev, stop_margin=stop_margin, maker_bps=maker_bps, taker_bps=taker_bps, max_hold_h=max_hold_h,
              wait_bars=wait_bars)
    rest = [s for g in GROUPS.values() for s in g if s != "PAXG_IDX"]
    t = pd.concat([trades_for(["PAXG_IDX"], root, gold_start, end, **kw), trades_for(rest, root, start, end, **kw)])
    res = summarize(t, {"all": t.symbol.unique().tolist(), **GROUPS})
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "cross-market.md"
    res.to_csv(path.with_suffix(".csv"), index=False)
    t.to_csv(path.with_suffix(".trades.csv.gz"), index=False)
    r = res.copy()
    r["hit_first_target"] = r.hit_first_target.map(lambda v: f"{v:.0%}")
    path.write_text(f"""# Paper-bot strategies on gold, indices and stocks (Phemex index prices), {lev:g}x ladder

Signals, fills, exits and fees exactly as `paper-ladder` (maker {maker_bps} / taker {taker_bps} bps per side + perp
funding), simulated on 15-minute bars of Phemex's index price for each perp. PAXG (gold) from {gold_start},
the others from their Phemex listing (Jan to Apr 2026), all to {end}. `avg_roi_pct` is per trade in % of margin.

{r.round(2).to_markdown(index=False)}
""")
    return path
