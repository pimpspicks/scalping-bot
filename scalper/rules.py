"""Textbook trading rules for single indicators, each traded alone with the leveraged ladder.

Every rule fires on the bar where its condition starts (a cross), not on every bar it holds, and
uses the standard settings (RSI 14 with 30/70, MACD 12/26/9, Bollinger 20/2, ...). Each rule is
also tested reversed ("fade"), because some indicators work better read the other way.

Only the development period (before `test_start`) is reported here: these runs are for ranking
and choosing, and the locked test period stays unseen until the final check.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .ict import fvg_retrace, market_structure_shift, sweep_reversal, window_mask, SILVER_BULLET
from .indicators import _atr, base_tf
from .ladder import simulate_ladder
from .swing import donchian, make_bars

log = logging.getLogger(__name__)


def _up(x: pd.Series, level) -> pd.Series:
    return (x > level) & (x.shift() <= level)


def _down(x: pd.Series, level) -> pd.Series:
    return (x < level) & (x.shift() >= level)


def _sig(long, short) -> np.ndarray:
    long, short = np.asarray(long, bool), np.asarray(short, bool)
    return np.where(long & ~short, 1, np.where(short & ~long, -1, 0))


def rule_signals(df: pd.DataFrame, funding_cum: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    """{(family, rule): signal array}"""
    f = base_tf(df, funding_cum)
    c, lo, hi = df.close, df.low, df.high
    atr = _atr(df).to_numpy()
    e = {n: c / (1 + f[f"ema_{n}_dist"]) for n in (9, 21, 50, 200)}
    s1 = c / (1 + f.pivot_s1)
    r1 = c / (1 + f.pivot_r1)
    pdh, pdl = c / (1 + f.pdh_dist), c / (1 + f.pdl_dist)
    fund_hi = f.funding_24h.rolling(96 * 30, min_periods=96 * 7).quantile(0.9)
    fund_lo = f.funding_24h.rolling(96 * 30, min_periods=96 * 7).quantile(0.1)
    sb = window_mask(df, SILVER_BULLET)
    R = {
        ("trend", "EMA 9/21 cross"): _sig(_up(e[9] - e[21], 0), _down(e[9] - e[21], 0)),
        ("trend", "EMA 21/50 cross"): _sig(_up(e[21] - e[50], 0), _down(e[21] - e[50], 0)),
        ("trend", "EMA 50/200 cross"): _sig(_up(e[50] - e[200], 0), _down(e[50] - e[200], 0)),
        ("trend", "MACD signal cross"): _sig(_up(f.macd_hist, 0), _down(f.macd_hist, 0)),
        ("trend", "ADX>25 + DI cross"): _sig(_up(f.di_diff, 0) & (f.adx > 25), _down(f.di_diff, 0) & (f.adx > 25)),
        ("trend", "Ichimoku TK cross"): _sig(_up(f.ichi_tk, 0), _down(f.ichi_tk, 0)),
        ("trend", "Ichimoku cloud break"): _sig(_up(f.ichi_cloud, 0), _down(f.ichi_cloud, 0)),
        ("trend", "Supertrend flip"): _sig(_up(f.supertrend, 0), _down(f.supertrend, 0)),
        ("momentum", "RSI 30/70 reversal"): _sig(_up(f.rsi_14, 30), _down(f.rsi_14, 70)),
        ("momentum", "RSI 50 cross"): _sig(_up(f.rsi_14, 50), _down(f.rsi_14, 50)),
        ("momentum", "Stochastic cross in 20/80"): _sig(_up(f.stoch_k - f.stoch_d, 0) & (f.stoch_k < 20),
                                                       _down(f.stoch_k - f.stoch_d, 0) & (f.stoch_k > 80)),
        ("momentum", "CCI -100/+100 reversal"): _sig(_up(f.cci, -100), _down(f.cci, 100)),
        ("momentum", "CCI +100/-100 breakout"): _sig(_up(f.cci, 100), _down(f.cci, -100)),
        ("momentum", "ROC(16) zero cross"): _sig(_up(f.roc_16, 0), _down(f.roc_16, 0)),
        ("volatility", "Bollinger band reversal"): _sig(_up(f.bb_pctb, 0), _down(f.bb_pctb, 1)),
        ("volatility", "Bollinger band breakout"): _sig(_up(f.bb_pctb, 1), _down(f.bb_pctb, 0)),
        ("volatility", "Keltner breakout"): _sig(_up(f.kc_pos, 1), _down(f.kc_pos, -1)),
        ("volatility", "Donchian 20 breakout"): _first(donchian(_Bars(df), 20)),
        ("volume", "VWAP cross"): _sig(_up(f.vwap_dist, 0), _down(f.vwap_dist, 0)),
        ("volume", "OBV trend cross"): _sig(_up(f.obv_slope, 0), _down(f.obv_slope, 0)),
        ("volume", "MFI 20/80 reversal"): _sig(_up(f.mfi, 20), _down(f.mfi, 80)),
        ("volume", "Chaikin money flow cross"): _sig(_up(f.cmf, 0), _down(f.cmf, 0)),
        ("structure", "Pivot point cross"): _sig(_up(f.pivot_p, 0), _down(f.pivot_p, 0)),
        ("structure", "Pivot S1/R1 bounce"): _sig((lo <= s1) & (c > s1), (hi >= r1) & (c < r1)),
        ("structure", "Previous day high/low breakout"): _sig(_up(c - pdh, 0), _down(c - pdl, 0)),
        ("structure", "Fibonacci 61.8% reclaim"): _sig(_up(f.fib_pos, 0.382), _down(f.fib_pos, 0.618)),
        ("ict", "Fair value gap retrace"): fvg_retrace(df, atr, 4, 0.0),
        ("ict", "Previous day liquidity sweep"): sweep_reversal(df, "prev_day"),
        ("ict", "Asian range sweep"): sweep_reversal(df, "asia"),
        ("ict", "Market structure shift"): market_structure_shift(df, 4),
        ("ict", "Silver bullet"): fvg_retrace(df, atr, 2, 0.0, sb, form_window=sb),
        ("funding", "Funding extreme (contrarian)"): _sig(_down(f.funding_24h - fund_lo, 0),
                                                         _up(f.funding_24h - fund_hi, 0)),
    }
    return R


def _first(sig: np.ndarray) -> np.ndarray:
    """Keep only the first bar of each run of the same signal."""
    prev = np.r_[0, sig[:-1]]
    return np.where(sig != prev, sig, 0)


class _Bars:
    """Minimal stand-in so swing.donchian can be reused on a plain DataFrame."""

    def __init__(self, df):
        self.df = df


def run(symbols, bar_mins, lev: float, stop_margin: float, fee_bps: float, max_hold_h: float, start: str,
        test_start: str, root: Path, report_dir: Path, rule_set: str = "classic") -> Path:
    """Rank every rule of `rule_set` ("classic" or "new") on the bars between `start` and `test_start`."""
    from .rules2 import new_rule_signals
    rows, all_trades = [], []
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(test_start, tz="UTC")
    for bm in bar_mins:
        hold = int(max_hold_h * 60 / bm)
        per_rule: dict = {}
        all_bars = {s: make_bars(s, bm, root) for s in symbols}
        for s, b in all_bars.items():
            in_dev = np.asarray((b.df.index >= lo) & (b.df.index < hi))
            sigs = (rule_signals(b.df, b.funding_cum) if rule_set == "classic"
                    else new_rule_signals(s, all_bars))
            for (fam, name), sig in sigs.items():
                sig = np.where(in_dev, sig, 0)
                for way, sg in (("as rule", sig), ("fade", -sig)):
                    t = simulate_ladder(b, sg, stop_margin / lev, False, fee_bps, hold)
                    t = t[t.entry_time < hi].assign(symbol=s, roi_pct=lambda x: x.net_bps / 100 * lev)
                    per_rule.setdefault((fam, name, way), []).append(t)
            log.info("%dm %s done", bm, s)
        for (fam, name, way), ts in per_rule.items():
            t = pd.concat(ts)
            if t.empty:
                continue
            all_trades.append(t.assign(chart=f"{bm}m", family=fam, rule=name, way=way))
            rows.append({"chart": f"{bm}m", "family": fam, "rule": name, "way": way, **_stats(t)})
    res = pd.DataFrame(rows).sort_values("avg_roi_pct", ascending=False)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / ("indicator-ranking.md" if rule_set == "classic" else f"{rule_set}-ranking.md")
    res.to_csv(path.with_suffix(".csv"), index=False)
    pd.concat(all_trades).to_csv(path.with_suffix(".trades.csv.gz"), index=False)
    path.write_text(_render(res, lev, stop_margin, fee_bps, max_hold_h, start, test_start))
    return path


def _stats(t: pd.DataFrame) -> dict:
    months = t.entry_time.dt.tz_localize(None).dt.to_period("M")
    by_coin = t.groupby("symbol").roi_pct.mean()
    return {"trades": len(t), "hit_first_target": (t.targets_hit >= 1).mean(),
            "avg_gross_move_bps": t.gross_bps.mean(), "avg_roi_pct": t.roi_pct.mean(),
            "t_stat": t.roi_pct.mean() / (t.roi_pct.std() / np.sqrt(len(t))) if len(t) > 1 else np.nan,
            "coins_positive": int((by_coin > 0).sum()),
            "positive_months": (t.roi_pct.groupby(months).sum() > 0).mean()}


def _render(res, lev, stop_margin, fee_bps, max_hold_h, start, test_start) -> str:
    r = res.copy()
    for c in ("hit_first_target", "positive_months"):
        r[c] = r[c].map(lambda v: f"{v:.1%}")
    for c in ("avg_gross_move_bps", "avg_roi_pct", "t_stat"):
        r[c] = r[c].map(lambda v: f"{v:.2f}")
    fam = res[res.way == "as rule"].groupby("family").avg_roi_pct.max().sort_values(ascending=False)
    return f"""# Single-indicator ranking: each rule alone, {lev:g}x ladder

Development data only: {start} to {test_start} (locked test not used). BTC, ETH, SOL pooled.
Exits: stop -{stop_margin:.0%} of margin, half off at +{stop_margin:.0%}, a quarter at +{2 * stop_margin:.0%}, rest at
+{3 * stop_margin:.0%}, max hold {max_hold_h:g} h. Fees {fee_bps} bps of position ({fee_bps / 100 * lev:.1f}% of margin) + funding.
`avg_roi_pct` = average result per trade in % of margin. `t_stat` above ~2 is unlikely to be luck for a single
test, but with {len(res)} tests a few will pass 2 by chance. "fade" = the rule traded in the opposite direction.

## Best rule per family (traded as the rule says)

{fam.round(2).to_frame("best avg_roi_pct").to_markdown()}

## All rules, best first

{r.to_markdown(index=False)}
"""
