"""Live paper trading of the ladder strategies on Phemex candles. No orders are ever sent.

Every 15 minutes, just after a candle closes, it downloads the new 1-minute Phemex candles and
funding, rebuilds the bars, computes each strategy's signals and replays them through the same
limit-order ladder simulation the backtests use (`simulate_ladder_limit`), counting only signals
after the bot was first started. So the paper results are exactly what the backtest would say
about the new data: same signals, same fills, same exits and fees.

Files in `out_dir`:
  state.json   the start time (kept across restarts, so a restart continues the same record)
  trades.csv   every paper trade so far; exit "open" = still running, marked at the last close
  signals.csv  every signal since the start, with whether it led to a trade
The candles themselves are kept in `data_dir` (data collection).
"""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .candles import download
from .ladder import simulate_ladder_limit
from .maker_test import to_15m
from .rules import rule_signals
from .rules3 import niche_rule_signals
from .swing import make_bars

log = logging.getLogger(__name__)

# name -> (chart minutes, rule set, rule key). All traded as the rule says.
STRATEGIES = {
    "4h EMA 21/50 cross": (240, "classic", ("trend", "EMA 21/50 cross")),
    "1h Friday post-expiry range break": (60, "niche", ("options expiry", "Friday post-expiry range break")),
    "15m Donchian 20 breakout when trending": (15, "niche", ("hurst switch", "Donchian 20 breakout when trending")),
}


def _signals(symbols, bar_min, rule_set, root, until):
    bars = {s: make_bars(s, bar_min, root, until.floor(f"{bar_min}min")) for s in symbols}
    out = {}
    for s, b in bars.items():
        sigs = rule_signals(b.df, b.funding_cum) if rule_set == "classic" else niche_rule_signals(s, bars)
        out[s] = (b.df.index, sigs)
    return out


def step(symbols, start: pd.Timestamp, data_dir: Path, lev, stop_margin, maker_bps, taker_bps, max_hold_h,
         wait_bars, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One update: (trades, signals) since `start`, using every 15-minute candle closed before `now`."""
    now = now or pd.Timestamp.now(tz="UTC")
    cutoff = now.floor("15min")
    b15 = {s: make_bars(s, 15, data_dir, cutoff) for s in symbols}
    trades, signals = [], []
    by_chart = {}
    for name, (bm, rule_set, key) in STRATEGIES.items():
        if (bm, rule_set) not in by_chart:
            by_chart[(bm, rule_set)] = _signals(symbols, bm, rule_set, data_dir, cutoff)
        for s in symbols:
            idx, sigs = by_chart[(bm, rule_set)][s]
            sig = np.where(np.asarray(idx >= start), sigs[key], 0)
            sg = to_15m(sig, idx, bm, b15[s].df.index)
            t = simulate_ladder_limit(b15[s], sg, stop_margin / lev, maker_bps, taker_bps, int(max_hold_h * 4),
                                      wait_bars)
            trades.append(t.assign(strategy=name, symbol=s, roi_pct=t.net_bps / 100 * lev))
            at = b15[s].df.index[sg != 0] + pd.Timedelta("15min")  # decided when that candle closed
            signals.append(pd.DataFrame({"time": at, "strategy": name, "symbol": s, "dir": sg[sg != 0],
                                         "price": b15[s].df.close.to_numpy()[sg != 0]}))
    return pd.concat(trades, ignore_index=True), pd.concat(signals, ignore_index=True)


def summary(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "no paper trades yet"
    done = trades[trades.exit != "open"]
    lines = []
    for name, t in trades.groupby("strategy"):
        d = t[t.exit != "open"]
        lines.append(f"{name}: {len(d)} closed, avg ROI {d.roi_pct.mean() if len(d) else 0:+.2f}% of margin, "
                     f"{(t.exit == 'open').sum()} open")
    lines.append(f"all: {len(done)} closed, total {done.roi_pct.sum():+.1f}% of margin "
                 f"(sum of per-trade ROI), {(trades.exit == 'open').sum()} open")
    for _, r in trades[trades.exit == "open"].iterrows():
        lines.append(f"  open: {r.strategy} {r.symbol} {'long' if r.dir > 0 else 'short'} since {r.entry_time:%m-%d %H:%M} "
                     f"UTC, {r.targets_hit} targets hit, now {r.roi_pct:+.1f}%")
    return "\n".join(lines)


def run(symbols, data_dir: Path, out_dir: Path, lev, stop_margin, maker_bps, taker_bps, max_hold_h, wait_bars,
        history_days: int = 60, once: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    if state_path.exists():
        start = pd.Timestamp(json.loads(state_path.read_text())["start"])
    else:
        start = pd.Timestamp.now(tz="UTC").floor("15min")
        state_path.write_text(json.dumps({"start": start.isoformat()}))
    log.info("paper trading since %s (no orders are sent)", start)
    while True:
        download(symbols, history_days, data_dir)
        trades, signals = step(symbols, start, data_dir, lev, stop_margin, maker_bps, taker_bps, max_hold_h,
                               wait_bars)
        trades.sort_values("entry_time").to_csv(out_dir / "trades.csv", index=False)
        signals.sort_values("time").to_csv(out_dir / "signals.csv", index=False)
        print(f"\n{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n{summary(trades)}", flush=True)
        if once:
            return trades
        # wake 20 s after the next 15-minute candle closes
        nxt = (pd.Timestamp.now(tz="UTC").floor("15min") + pd.Timedelta(minutes=15, seconds=20))
        time.sleep(max(5.0, (nxt - pd.Timestamp.now(tz="UTC")).total_seconds()))
