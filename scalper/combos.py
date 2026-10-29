"""Combination search: entries at many settings, paired with filters and with each other, chosen walk-forward.

Plan and pass rule: reports-2026-10-29-combos/plan.md (written before running).

Every 15-minute bar gets the 20x ladder outcome of a long and of a short entered there (`labels`). Entries fire
on the bar where they trigger; filters are states (+1 / -1 agree with a direction, 2 = condition holds for
both directions, 0 = no). A combination = one entry + nothing / one filter / one other entry's recent signal /
two filters. Its trades are pooled over all coins and summed per quarter (by the quarter the trade ended in, so
a quarter's selection only uses finished trades). Each quarter the best 10 by past t-stat are traded on that
quarter with the real ladder simulation.
"""
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import binance_metrics
from .indicators import _atr, _rsi, _supertrend
from .ladder import _limit_trade, simulate_ladder_limit
from .maker_test import to_15m
from .rules3 import hurst
from .stack_pos import pos_features
from .swing import make_bars

log = logging.getLogger(__name__)

LEV, STOP_MARGIN, MAKER, TAKER, HOLD, WAIT = 20, 0.25, 1.0, 6.0, 288, 4
TFS = (15, 60, 240)
FIRST_Q, LAST_Q = pd.Period("2023Q1", "Q"), pd.Period("2026Q3", "Q")
START, END = pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC")


# ---------- entries on one chart (events: +1 / -1 on the trigger bar) ----------

def _x_up(a, b):
    """a crosses above b on this bar (b may be a number)."""
    b_prev = b.shift() if isinstance(b, pd.Series) else b
    return (a > b) & (a.shift() <= b_prev)


def _ev(up, dn):
    up, dn = np.asarray(up.fillna(False), bool), np.asarray(dn.fillna(False), bool)
    return np.where(up & ~dn, 1, np.where(dn & ~up, -1, 0)).astype(np.int8)


def entries(df: pd.DataFrame) -> dict[str, np.ndarray]:
    o, h, l, c = df.open, df.high, df.low, df.close
    E = {}
    for n in (10, 20, 55, 100):
        hi, lo = h.rolling(n).max().shift(), l.rolling(n).min().shift()
        E[f"Donchian {n}"] = _ev(_x_up(c, hi), _x_up(lo, c))
    for f, s in ((9, 21), (12, 26), (21, 50), (50, 200)):
        d = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
        E[f"EMA {f}/{s} cross"] = _ev(_x_up(d, 0), _x_up(-d, 0))
    for f, s, g in ((12, 26, 9), (8, 21, 5)):
        m = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
        d = m - m.ewm(span=g, adjust=False).mean()
        E[f"MACD {f}/{s}/{g} cross"] = _ev(_x_up(d, 0), _x_up(-d, 0))
    for n in (7, 14):
        r = _rsi(c, n)
        for lo_, hi_ in ((30, 70), (20, 80)):
            E[f"RSI {n} reversal {lo_}/{hi_}"] = _ev(_x_up(r, lo_), _x_up(-r, -hi_))
    r = _rsi(c, 14)
    E["RSI 14 cross 50"] = _ev(_x_up(r, 50), _x_up(-r, -50))
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    for k in (2.0, 2.5):
        up, lo = m20 + k * s20, m20 - k * s20
        E[f"Bollinger {k} breakout"] = _ev(_x_up(c, up), _x_up(lo, c))
        E[f"Bollinger {k} re-entry"] = _ev(_x_up(c, lo), _x_up(up, c))
    atr = _atr(df)
    mid = c.ewm(span=20, adjust=False).mean()
    for k in (1.5, 2.0, 2.5):
        E[f"Keltner {k} breakout"] = _ev(_x_up(c, mid + k * atr), _x_up(mid - k * atr, c))
    for k in (2.0, 3.0):
        st = _supertrend(df, atr, k)
        E[f"Supertrend {k:.0f} flip"] = _ev((st > 0) & (st.shift() < 0), (st < 0) & (st.shift() > 0))
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    sk = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    sd = sk.rolling(3).mean()
    E["Stochastic cross (OS/OB)"] = _ev(_x_up(sk, sd) & (sd < 20), _x_up(sd, sk) & (sd > 80))
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    E["Ichimoku TK cross"] = _ev(_x_up(tenkan, kijun), _x_up(kijun, tenkan))
    return E


def _resample(df15: pd.DataFrame, tf: int) -> pd.DataFrame:
    if tf == 15:
        return df15
    return df15.resample(f"{tf}min").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                          "volume": "sum"}).dropna()


def _align(x: pd.Series, tf: int, idx15: pd.DatetimeIndex) -> np.ndarray:
    """A higher-chart state, known from its bar's close, on the 15-minute grid (by close time)."""
    s = x.copy()
    s.index = s.index + pd.Timedelta(minutes=tf)
    closes = idx15 + pd.Timedelta("15min")
    return s.reindex(s.index.union(closes)).ffill().reindex(closes).to_numpy()


# ---------- filters ----------

def _dir(cond_up, cond_dn) -> np.ndarray:
    up, dn = np.nan_to_num(np.asarray(cond_up, float)) > 0, np.nan_to_num(np.asarray(cond_dn, float)) > 0
    return np.where(up & ~dn, 1, np.where(dn & ~up, -1, 0)).astype(np.int8)


def _nd(cond) -> np.ndarray:
    return np.where(np.nan_to_num(np.asarray(cond, float)) > 0, 2, 0).astype(np.int8)


def filters(df15: pd.DataFrame, funding_cum: np.ndarray, pf: pd.DataFrame, btc_up: np.ndarray,
            ranks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    idx, c = df15.index, df15.close
    h1, h4 = _resample(df15, 60), _resample(df15, 240)
    F = {}
    e200 = _align(h1.close / h1.close.ewm(span=200, adjust=False).mean() - 1, 60, idx)
    F["Trend: 1h EMA 200"] = _dir(e200 > 0, e200 < 0)
    e50 = _align(h4.close / h4.close.ewm(span=50, adjust=False).mean() - 1, 240, idx)
    F["Trend: 4h EMA 50"] = _dir(e50 > 0, e50 < 0)
    a1 = _atr(h1)
    st = _align(_supertrend(h1, a1), 60, idx)
    F["Trend: 1h Supertrend"] = _dir(st > 0, st < 0)
    r1 = _align(_rsi(h1.close), 60, idx)
    F["Trend: 1h RSI vs 50"] = _dir(r1 > 50, r1 < 50)
    F["Trend: BTC vs 4h EMA 50"] = _dir(btc_up > 0, btc_up < 0)
    up, dn = h1.high.diff(), -h1.low.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h1.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h1.index)
    pdi, mdi = 100 * plus.ewm(alpha=1 / 14, adjust=False).mean() / a1, 100 * minus.ewm(alpha=1 / 14, adjust=False).mean() / a1
    adx = _align((100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).ewm(alpha=1 / 14, adjust=False).mean(), 60, idx)
    F["Regime: 1h ADX > 25"] = _nd(adx > 25)
    F["Regime: 1h ADX < 20"] = _nd(adx < 20)
    hu = hurst(c).to_numpy()
    F["Regime: Hurst > 0.55"] = _nd(hu > 0.55)
    F["Regime: Hurst > 0.6"] = _nd(hu > 0.6)
    F["Regime: Hurst < 0.45"] = _nd(hu < 0.45)
    r = np.log(c).diff()
    vr = (r.rolling(16).std() / r.rolling(128).std()).to_numpy()
    F["Regime: volatility high"] = _nd(vr > 1.2)
    F["Regime: volatility low"] = _nd(vr < 0.8)
    lv = np.log1p(df15.volume)
    F["Regime: volume spike"] = _nd(((lv - lv.rolling(96).mean()) / lv.rolling(96).std()).to_numpy() > 1.5)
    for name, rk in ranks.items():
        F[f"Cross-coin: {name}"] = _dir(rk >= 0.7, rk <= 0.3) if name != "retail long/short rank" \
            else _dir(rk <= 0.3, rk >= 0.7)
    fd = pd.Series(funding_cum, index=idx).diff(96)
    fz = ((fd - fd.rolling(30 * 96, min_periods=10 * 96).mean()) / fd.rolling(30 * 96, min_periods=10 * 96).std()).to_numpy()
    F["Funding: unusual (contrarian)"] = _dir(fz < -1, fz > 1)
    F["Positioning: OI 1h jump"] = _nd(pf.pos_oi_chg_1h_z.to_numpy() > 2)
    ret24 = c.pct_change(96).to_numpy()
    oi_up = pf.pos_oi_chg_24h.to_numpy() > 0
    F["Positioning: OI confirms 24h move"] = _dir(oi_up & (ret24 > 0), oi_up & (ret24 < 0))
    F["Positioning: OI 24h drop"] = _nd(pf.pos_oi_chg_24h_z.to_numpy() < -1.5)
    rz = pf.pos_retail_z30d.to_numpy()
    F["Positioning: retail extreme (fade)"] = _dir(rz < -1, rz > 1)
    tz_ = pf.pos_top_pos_z30d.to_numpy()
    F["Positioning: top traders leaning"] = _dir(tz_ > 1, tz_ < -1)
    sm = pf.pos_smart_minus_retail.to_numpy()
    F["Positioning: smart minus retail"] = _dir(sm > 1, sm < -1)
    for h in ("1h", "24h"):
        z = pf[f"pos_taker_{h}_z"].to_numpy()
        F[f"Positioning: taker flow {h}"] = _dir(z > 1, z < -1)
    ct = idx + pd.Timedelta("15min")
    F["Time: US session"] = _nd((ct.hour >= 13) & (ct.hour < 21))
    F["Time: Asia session"] = _nd(ct.hour < 8)
    F["Time: weekend"] = _nd(ct.dayofweek >= 5)
    return F


# ---------- per-coin data ----------

def labels(b) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ROI (% margin) of a long and of a short from every bar, and the bar each one ended (NaN / -1: no trade)."""
    o, h, l, c = (b.df[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(o)
    y = np.full((2, n), np.nan, np.float32)
    end = np.full((2, n), -1, np.int32)
    for i in range(n - 1):
        for k, d in ((0, 1), (1, -1)):
            r = _limit_trade(o, h, l, c, b.funding_cum, i, d, STOP_MARGIN / LEV, MAKER, TAKER, HOLD, WAIT)
            if r is not None and r[3] != "open":
                y[k, i], end[k, i] = r[7] / 100 * LEV, r[1]
    return y[0], y[1], end[0], end[1]


def coin_data(s: str, root: Path, metrics_root: Path, btc_up: np.ndarray, ranks: dict, cache: Path) -> dict:
    b = make_bars(s, 15, root)
    idx = b.df.index
    lp = cache / f"labels-{s}.npz"
    if lp.exists():
        z = np.load(lp)
        yl, ys, el, es = z["yl"], z["ys"], z["el"], z["es"]
    else:
        yl, ys, el, es = labels(b)
        np.savez(lp, yl=yl, ys=ys, el=el, es=es)
    ent, state = {}, {}
    for tf in TFS:
        dft = _resample(b.df, tf)
        for name, sig in entries(dft).items():
            s15 = to_15m(sig.astype(int), dft.index, tf, idx).astype(np.int8)
            key = f"{tf}m {name}"
            ent[key] = s15
            k = 4 * tf // 15
            state[key] = pd.Series(np.where(s15 != 0, s15, np.nan)).ffill(limit=k - 1).fillna(0).to_numpy(np.int8)
    m = binance_metrics.load(s, metrics_root)
    pf = pos_features(idx, b.df.close, b.df.volume * b.df.close, m)
    F = filters(b.df, b.funding_cum, pf, btc_up, ranks)
    return {"bars": b, "yl": yl, "ys": ys, "el": el, "es": es, "entries": ent, "states": state, "filters": F}


def _quarter_ids(times: pd.DatetimeIndex) -> np.ndarray:
    q = times.tz_localize(None).to_period("Q")
    return np.asarray([(p - FIRST_Q).n for p in q]) if len(q) else np.zeros(0, int)


def combo_names(entry_names, filter_names) -> list[tuple]:
    out = []
    pairs = list(itertools.combinations(filter_names, 2))
    for e in entry_names:
        out.append((e,))
        out += [(e, f) for f in filter_names]
        out += [(e, "+" + o) for o in entry_names if o != e]
        out += [(e, a, b) for a, b in pairs]
    return out


def accumulate(cd: dict, combos_index: dict, n_q: int, S, SS, N):
    """Add one coin's trades to the per-combination, per-quarter sums (bucketed by the quarter a trade ended)."""
    idx = cd["bars"].df.index
    qid_all = _quarter_ids(idx)
    in_range = np.asarray((idx >= START) & (idx < END))
    fnames = list(cd["filters"])
    pairs = list(itertools.combinations(range(len(fnames)), 2))
    enames = list(cd["entries"])
    for e in enames:
        sig = cd["entries"][e]
        ev = np.flatnonzero((sig != 0) & in_range)
        d = sig[ev]
        y = np.where(d > 0, cd["yl"][ev], cd["ys"][ev])
        endb = np.where(d > 0, cd["el"][ev], cd["es"][ev])
        ok = ~np.isnan(y)
        ev, d, y, endb = ev[ok], d[ok], y[ok].astype(np.float64), endb[ok]
        q = qid_all[endb]
        q = np.clip(q, 0, n_q - 1)
        P = np.stack([(cd["filters"][f][ev] == d) | (cd["filters"][f][ev] == 2) for f in fnames], axis=1)
        A = np.stack([cd["states"][o][ev] == d for o in enames if o != e], axis=1)

        def add(name, mask):
            i = combos_index[name]
            qq, yy = q[mask], y[mask]
            S[i] += np.bincount(qq, yy, n_q)
            SS[i] += np.bincount(qq, yy * yy, n_q)
            N[i] += np.bincount(qq, minlength=n_q)

        add((e,), np.ones(len(y), bool))
        for j, f in enumerate(fnames):
            add((e, f), P[:, j])
        for j, o in enumerate([o for o in enames if o != e]):
            add((e, "+" + o), A[:, j])
        for a, b in pairs:
            add((e, fnames[a], fnames[b]), P[:, a] & P[:, b])


def combo_signal(cd: dict, combo: tuple) -> np.ndarray:
    e = combo[0]
    sig = cd["entries"][e].astype(int).copy()
    for part in combo[1:]:
        if part.startswith("+"):
            sig = np.where(cd["states"][part[1:]] == sig, sig, 0)
        else:
            f = cd["filters"][part]
            sig = np.where((f == sig) | (f == 2), sig, 0)
    return sig


# ---------- run ----------

def _panels(symbols, root, metrics_root):
    closes, retail, oi7 = {}, {}, {}
    for s in symbols:
        b = make_bars(s, 15, root)
        m = binance_metrics.load(s, metrics_root)
        pf = pos_features(b.df.index, b.df.close, b.df.volume * b.df.close, m)
        closes[s], retail[s], oi7[s] = b.df.close, pf.pos_retail_z30d, pf.pos_oi_chg_7d
    closes, retail, oi7 = pd.DataFrame(closes), pd.DataFrame(retail), pd.DataFrame(oi7)
    mom = (closes / closes.shift(3 * 7 * 96) - 1).rank(axis=1, pct=True)
    return closes, {"3-week return rank": mom, "OI 7-day change rank": oi7.rank(axis=1, pct=True),
                    "retail long/short rank": retail.rank(axis=1, pct=True)}


def run(symbols, root: Path, metrics_root: Path, report_dir: Path, cache: Path, top=(1, 10, 50),
        min_past=150) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    closes, rank_panels = _panels(symbols, root, metrics_root)
    btc = make_bars("BTC", 15, root)
    h4 = _resample(btc.df, 240)
    btc_state = pd.Series(_align(h4.close / h4.close.ewm(span=50, adjust=False).mean() - 1, 240, btc.df.index),
                          index=btc.df.index)
    n_q = (LAST_Q - FIRST_Q).n + 1
    data, combos, S = {}, None, None
    for s in symbols:
        idx = make_bars(s, 15, root).df.index
        ranks = {k: v[s].reindex(idx).to_numpy() for k, v in rank_panels.items()}
        cd = coin_data(s, root, metrics_root, btc_state.reindex(idx).to_numpy(), ranks, cache)
        if combos is None:
            combos = combo_names(list(cd["entries"]), list(cd["filters"]))
            cindex = {c: i for i, c in enumerate(combos)}
            S, SS, N = (np.zeros((len(combos), n_q)) for _ in range(3))
            log.info("%d combinations", len(combos))
        accumulate(cd, cindex, n_q, S, SS, N)
        data[s] = cd
        log.info("%s: accumulated", s)
    np.savez(report_dir / "combo-sums.npz", S=S, SS=SS, N=N)
    pd.Series([" | ".join(c) for c in combos]).to_csv(report_dir / "combo-names.csv.gz", index=False,
                                                       compression="gzip")
    return evaluate(data, combos, S, SS, N, report_dir, top, min_past)


def _t(s, ss, n):
    with np.errstate(divide="ignore", invalid="ignore"):
        m = s / n
        sd = np.sqrt(np.maximum(ss / n - m * m, 0) * n / np.maximum(n - 1, 1))
        return m / sd * np.sqrt(n)


def evaluate(data, combos, S, SS, N, report_dir: Path, top, min_past) -> Path:
    first_test = (pd.Period("2024Q1", "Q") - FIRST_Q).n
    n_q = S.shape[1]
    trades, picks, persist, deciles = {k: [] for k in top}, [], [], []
    for k in range(first_test, n_q):
        ps, pss, pn = S[:, :k].sum(1), SS[:, :k].sum(1), N[:, :k].sum(1)
        t = np.where(pn >= min_past, _t(ps, pss, pn), np.nan)
        nxt = np.where(N[:, k] >= 30, S[:, k] / np.maximum(N[:, k], 1), np.nan)
        ok = ~np.isnan(t) & ~np.isnan(nxt)
        persist.append({"Quarter": str(FIRST_Q + k), "Combos scored": int(ok.sum()),
                        "Rank corr (past t-stat vs next-quarter ROI)":
                            round(pd.Series(t[ok]).corr(pd.Series(nxt[ok]), method="spearman"), 3)})
        dec = pd.qcut(pd.Series(t[ok]).rank(method="first"), 10, labels=False)
        deciles.append(pd.Series(nxt[ok]).groupby(dec.to_numpy()).mean())
        order = np.argsort(-np.nan_to_num(t, nan=-np.inf))
        q0 = (FIRST_Q + k).start_time.tz_localize("UTC")
        q1 = (FIRST_Q + k + 1).start_time.tz_localize("UTC")
        for rank, ci in enumerate(order[:max(top)]):
            if rank < 10:
                picks.append({"Quarter": str(FIRST_Q + k), "Rank": rank + 1, "Combination": " | ".join(combos[ci]),
                              "Past trades": int(pn[ci]), "Past avg ROI %": round(ps[ci] / pn[ci], 2),
                              "Past t-stat": round(t[ci], 2)})
            for s, cd in data.items():
                idx = cd["bars"].df.index
                sig = combo_signal(cd, combos[ci])
                sig = np.where(np.asarray((idx >= q0) & (idx < q1)), sig, 0)
                if not sig.any():
                    continue
                tr = simulate_ladder_limit(cd["bars"], sig, STOP_MARGIN / LEV, MAKER, TAKER, HOLD, WAIT)
                tr = tr[tr.exit != "open"].assign(coin=s, combo=" | ".join(combos[ci]), quarter=str(FIRST_Q + k),
                                                  roi_pct=lambda x: x.net_bps / 100 * LEV)
                for kk in top:
                    if rank < kk:
                        trades[kk].append(tr)
        log.info("%s: selected and traded", FIRST_Q + k)
    rows = []
    for kk in top:
        t_ = pd.concat(trades[kk], ignore_index=True)
        t_.to_csv(report_dir / f"oos-trades-top{kk}.csv.gz", index=False, compression="gzip")
        r = t_.roi_pct
        qs = t_.groupby("quarter").roi_pct.mean()
        cs = t_.groupby("coin").roi_pct.mean()
        rows.append({"Portfolio": f"Top {kk}", "Trades": len(r), "Avg ROI/trade %": round(r.mean(), 2),
                     "t-stat": round(r.mean() / r.std() * np.sqrt(len(r)), 2), "Win %": round((r > 0).mean() * 100),
                     "Quarters positive": f"{(qs > 0).sum()} of {len(qs)}", "Coins positive": f"{(cs > 0).sum()} of {len(cs)}",
                     "By quarter": ", ".join(f"{q} {v:+.1f}" for q, v in qs.items())})
    allc = N.sum(1) > 0
    base = (S[allc].sum() / N[allc].sum())
    dtab = pd.DataFrame(deciles).mean().round(2)
    lines = ["# Combination search: results\n", "Plan: plan.md (written before running).\n",
             f"{len(combos)} combinations, 20 coins, out of sample 2024 Q1 to 2026 Q3.\n",
             "## Out-of-sample portfolios (chosen each quarter on past data only)\n",
             pd.DataFrame(rows).to_markdown(index=False),
             "\n## Does a combination's past score predict its next quarter?\n",
             pd.DataFrame(persist).to_markdown(index=False),
             "\nNext-quarter average ROI per trade by past t-stat decile (0 = worst past, 9 = best past), averaged "
             "over quarters:\n", dtab.to_frame("Next-quarter ROI %").T.to_markdown(),
             f"\nAverage over every combination and trade: {base:+.2f}% ROI per trade.\n",
             "## What was picked (top 10 each quarter)\n", pd.DataFrame(picks).to_markdown(index=False)]
    path = report_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path
