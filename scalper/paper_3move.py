"""Live paper trading of the two three-move setups on Phemex candles. No orders are ever sent.

  Move-1 pullback: 2h chart, zigzag 2 ATR, stop 0.1 ATR beyond the pullback, 2R target, only in the direction of
                   the stablecoin-supply signal (30 days).
  Move-3 fade:     2h chart, zigzag 3 ATR, stochastic divergence at the end of a 3-move run, 2R target, only in the
                   direction of the USDT dominance signal (30 days).
Both use a limit entry at the signal candle's close (good for 2 candles), a limit target and a market stop, and
risk `risk` of the account per trade (backtest: reports-2026-10-29-efficiency/).

Every 2 hours, just after a candle closes, it downloads new 15-minute Phemex candles (and the money-flow data
once every 6 hours), rebuilds the 2h bars and replays every signal since the bot was first started through the
same simulation as the backtest. So the paper record is exactly what the backtest says about the new data.

Files in `out_dir`:
  state.json   the start time (kept across restarts, so a restart continues the same record)
  trades.csv   every paper trade; exit "open" = still running (P&L marked at the last close)
"""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import moneyflows
from .candles import download
from .efficient import trades_2r
from .fresh import Coin
from .moneyflows import _at, flow_signals, load_flows
from .refine import a_signals
from .swing import make_bars
from .understanding import signals as u_signals

log = logging.getLogger(__name__)

TF = 120
FLOW_REFRESH = pd.Timedelta(hours=6)


def _coin(sym: str, root: Path, until: pd.Timestamp) -> Coin:
    b = make_bars(sym, TF, root, until)
    df = b.df
    return Coin(sym, TF, df.index + pd.Timedelta(minutes=TF), *(df[k].to_numpy() for k in ("open", "high", "low", "close")),
                b.atr, b.funding_cum)


def step(symbols, start: pd.Timestamp, data_dir: Path, flow_dir: Path, risk: float,
         now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = now or pd.Timestamp.now(tz="UTC")
    cutoff = now.floor(f"{TF}min")  # only fully closed 2h candles
    fs = flow_signals(load_flows(flow_dir))
    period = (start, cutoff + pd.Timedelta(minutes=TF))
    out = []
    for sym in symbols:
        try:
            cn = _coin(sym, data_dir, cutoff)
        except SystemExit:
            log.warning("no candles for %s", sym)
            continue
        if len(cn.c) < 200 or cn.t_close[-1] < start:
            continue
        sig, stop = a_signals(cn, 2.0, "any", 0.1)
        sig = np.where(sig * _at(fs[("Stables", 30)], cn.t_close) > 0, sig, 0)
        out.append(trades_2r(cn, sig, stop, "limit entry + limit target", risk, period).assign(
            strategy="Move-1 pullback", symbol=sym))
        sig, stop, _ = u_signals(cn, 3.0, "C2 move-3 stochastic divergence")
        sig = np.where(sig * _at(fs[("USDT.D", 30)], cn.t_close) > 0, sig, 0)
        out.append(trades_2r(cn, sig, stop, "limit entry + limit target", risk, period).assign(
            strategy="Move-3 fade", symbol=sym))
    t = pd.concat([x for x in out if len(x)], ignore_index=True) if any(len(x) for x in out) else pd.DataFrame()
    if len(t):
        t["exit"] = t.exit.replace({"open at end": "open"})
        t["account_pct"] = t.net * t["size"] * 100
        t = t.rename(columns={"entry_time": "signal_time"})
        t = t[["strategy", "symbol", "dir", "signal_time", "fill_time", "entry_price", "stop_price", "target_price",
               "exit", "exit_time", "net", "size", "account_pct"]].sort_values("signal_time")
    return t


def summary(t: pd.DataFrame) -> str:
    if t.empty:
        return "no paper trades yet"
    lines = []
    for name, g in t.groupby("strategy"):
        d = g[g.exit != "open"]
        lines.append(f"{name}: {len(d)} closed ({(d.net > 0).sum()} won), {d.account_pct.sum():+.2f}% of account, "
                     f"{(g.exit == 'open').sum()} open")
    for _, r in t[t.exit == "open"].iterrows():
        lines.append(f"  open: {r.strategy} {r.symbol} {'long' if r.dir > 0 else 'short'} @ {r.entry_price:.6g}, "
                     f"stop {r.stop_price:.6g}, target {r.target_price:.6g}, now {r.account_pct:+.2f}% of account")
    return "\n".join(lines)


def run(symbols, data_dir: Path, out_dir: Path, risk=0.005, history_days=150, once=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    flow_dir = data_dir / "money-flows"
    state_path = out_dir / "state.json"
    if state_path.exists():
        start = pd.Timestamp(json.loads(state_path.read_text())["start"])
    else:
        start = pd.Timestamp.now(tz="UTC").floor(f"{TF}min")
        state_path.write_text(json.dumps({"start": start.isoformat()}))
    log.info("paper trading the three-move setups since %s (no orders are sent)", start)
    last_flow = None
    while True:
        if last_flow is None or pd.Timestamp.now(tz="UTC") - last_flow > FLOW_REFRESH:
            moneyflows.download(flow_dir, start="2019-06-01")
            last_flow = pd.Timestamp.now(tz="UTC")
        download(symbols, history_days, data_dir, minutes=15)
        trades = step(symbols, start, data_dir, flow_dir, risk)
        trades.to_csv(out_dir / "trades.csv", index=False)
        print(f"\n{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n{summary(trades)}", flush=True)
        if once:
            return trades
        nxt = pd.Timestamp.now(tz="UTC").floor(f"{TF}min") + pd.Timedelta(minutes=TF, seconds=30)
        time.sleep(max(5.0, (nxt - pd.Timestamp.now(tz="UTC")).total_seconds()))
