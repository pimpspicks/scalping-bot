"""Our own signals, built from published fund and research ideas, combined into one composite.

Plan and pass rule: reports-2026-10-29-own-signals/plan.md (written before running).

Daily panel at 00:00 UTC closes. Each component gives a raw position per coin at each close (+/-1 picks or a
continuous forecast). Positions are scaled by 1 / the coin's 30-day volatility, held over the next day, and
charged `COST` per side on every change plus funding. Each component's return series is then scaled to 20%
annual volatility with its own trailing 90-day volatility (past data only), so components and the composite
can be compared and averaged at the same risk.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .fresh import DESIGN, EXAM, load_coin, score

log = logging.getLogger(__name__)

COST, TARGET_VOL = 0.0008, 0.20


def panel(symbols, root: Path):
    closes, fund = {}, {}
    for s in symbols:
        try:
            cn = load_coin(s, 1440, root)
        except SystemExit:
            continue
        closes[s] = pd.Series(cn.c, index=cn.t_close)
        fc = np.r_[cn.fcum[1:], cn.fcum[-1]]  # cumulative funding at each bar's close (= next bar's open)
        fund[s] = pd.Series(fc, index=cn.t_close)
    C = pd.DataFrame(closes)
    FC = pd.DataFrame(fund).reindex(C.index).ffill()
    return C, FC


def _picks(score_df: pd.DataFrame, k=4, reverse=False) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=score_df.index, columns=score_df.columns)
    for t, r in score_df.iterrows():
        r = r.dropna()
        if len(r) < 2 * k + 2:
            continue
        o = r.sort_values()
        lo, hi = o.index[:k], o.index[-k:]
        out.loc[t, hi], out.loc[t, lo] = (-1.0, 1.0) if reverse else (1.0, -1.0)
    return out


def _weekly(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep Monday 00:00 values and hold them through the week."""
    mon = raw.index.dayofweek == 0
    return raw.where(pd.Series(mon, index=raw.index), np.nan).ffill().fillna(0.0)


def components(C: pd.DataFrame) -> dict[str, pd.DataFrame]:
    r = C.pct_change(fill_method=None)
    S = {}
    # Carver-style trend forecast
    sig = r.ewm(span=25, min_periods=20).std()
    fc = []
    for f in (8, 16, 32, 64):
        raw = (C.ewm(span=f, adjust=False).mean() - C.ewm(span=4 * f, adjust=False).mean()) / (C * sig)
        scale = raw.abs().expanding(min_periods=60).mean().shift()
        fc.append((raw / scale).clip(-2, 2))
    S["Trend forecast (Carver)"] = (sum(fc) / len(fc)).clip(-2, 2)
    # weekly ranking (3-week momentum)
    S["Weekly ranking (3w momentum)"] = _weekly(_picks(C / C.shift(21) - 1))
    # MAX effect: highest single-day return over the past 28 days, long the highest
    S["MAX effect"] = _weekly(_picks(r.rolling(28, min_periods=20).max()))
    # residual momentum vs BTC
    rb = r["BTC"]
    beta = r.rolling(90, min_periods=60).cov(rb).div(rb.rolling(90, min_periods=60).var(), axis=0)
    m3 = C / C.shift(21) - 1
    S["Residual momentum"] = _weekly(_picks(m3.sub(beta.mul(m3["BTC"], axis=0)).drop(columns="BTC")).reindex(columns=C.columns, fill_value=0))
    # daily reversal
    S["Daily reversal"] = _picks(r, reverse=True)
    return S


def backtest(C: pd.DataFrame, FC: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Unscaled daily returns of a raw position panel (positions decided at close t, earning t -> t+1)."""
    r = C.pct_change(fill_method=None)
    vol = r.rolling(30, min_periods=20).std() * np.sqrt(365)
    W = (raw.fillna(0) * (TARGET_VOL / vol)).where(C.notna(), 0).fillna(0)
    W = W.where(C.shift(-1).notna(), 0)  # can't hold a coin with no next price
    nxt = (C.shift(-1) / C - 1).fillna(0)
    fnd = (FC.shift(-1) - FC).fillna(0)
    pnl = (W * (nxt - fnd)).sum(axis=1) - COST * (W - W.shift().fillna(0)).abs().sum(axis=1)
    return pnl.shift(1).fillna(0), W  # booked on the day it is earned


def vol_scale(p: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Scale a return series to TARGET_VOL using its trailing 90-day volatility (known the day before)."""
    v = p.rolling(90, min_periods=60).std().shift() * np.sqrt(365)
    k = (TARGET_VOL / v).clip(upper=10).fillna(0)
    return p * k, k


def _days(p: pd.Series, period) -> pd.Series:
    x = p[(p.index >= period[0]) & (p.index < period[1])]
    x.index = x.index.floor("D")
    return x


def run(symbols, root: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    C, FC = panel(symbols, root)
    S = components(C)
    scaled, kfac = {}, {}
    for name, raw in S.items():
        p, _ = backtest(C, FC, raw)
        scaled[name], kfac[name] = vol_scale(p)
    dsc = {n: score(_days(p, DESIGN), DESIGN) for n, p in scaled.items()}
    chosen = [n for n in S if dsc[n]["Sharpe"] > 0]
    log.info("composite uses: %s", chosen)
    # composite: average the components' risk-scaled positions, so offsetting trades net out before costs
    comp_raw = sum(S[n].fillna(0) * kfac[n].reindex(S[n].index).fillna(0).to_numpy()[:, None] for n in chosen) / max(len(chosen), 1)
    p, _ = backtest(C, FC, comp_raw)
    scaled["Composite (our signal)"], _ = vol_scale(p)
    rows = []
    for n, p in scaled.items():
        for per_name, per in (("Design 2023-26", DESIGN), ("Exam 2020-22", EXAM)):
            s = score(_days(p, per), per)
            rows.append({"Signal": n, "Period": per_name, "Sharpe": round(s["Sharpe"], 2),
                         "Return/yr % (at 20% vol)": round(s["Return/yr %"], 1), "Max DD %": round(s["Max DD %"], 1),
                         "By year %": s["By year %"]})
    tab = pd.DataFrame(rows)
    pd.DataFrame(scaled).to_csv(report_dir / "daily-returns.csv.gz", compression="gzip")
    corr = pd.DataFrame({n: _days(p, DESIGN) for n, p in scaled.items()}).corr().round(2)
    lines = ["# Our own signals: results\n", "Plan: plan.md (written before running).\n",
             f"Components in the composite (design Sharpe > 0): {', '.join(chosen)}\n",
             tab.sort_values(["Period", "Sharpe"], ascending=[True, False]).to_markdown(index=False),
             "\n## Correlation of daily returns (design period)\n", corr.to_markdown()]
    path = report_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path
