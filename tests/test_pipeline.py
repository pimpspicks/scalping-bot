"""Sanity checks on synthetic ticks. These verify the plumbing, not any market edge:
- pure noise must score ~50% out of sample (catches look-ahead leaks)
- a planted order-flow signal must be found (catches a pipeline that learns nothing)
"""
import numpy as np
import pandas as pd

from scalper.features import build_dataset, feature_columns, make_bars
from scalper.model import evaluate, walk_forward
from scalper.storage import TickWriter, load_raw


def synth_ticks(n_seconds=6 * 3600, signal=0.0, seed=0, start_ms=1_760_000_000_000):
    """One trade per second-ish; next-second mid move is `signal` * current flow plus noise."""
    rng = np.random.default_rng(seed)
    side = rng.choice([-1, 1], n_seconds)
    noise = rng.normal(0, 1e-4, n_seconds)
    drift = np.zeros(n_seconds)
    drift[1:] = signal * 1e-4 * side[:-1]
    mid = 100 * np.exp(np.cumsum(noise + drift))
    ts = start_ms + np.arange(n_seconds) * 1000 + rng.integers(0, 999, n_seconds)
    half_spread = mid * 0.5e-4
    trades = pd.DataFrame({"ts_ms": ts, "price": mid + side * half_spread, "qty": rng.exponential(1, n_seconds), "side": side})
    book = pd.DataFrame({"ts_ms": ts - 1, "bid": mid - half_spread, "bid_qty": 1.0, "ask": mid + half_spread, "ask_qty": 1.0})
    return trades, book


def run_oos(signal, seed=0, bar_s=1, horizon=1):
    bars = {}
    for i, s in enumerate(["BTC", "ETH"]):
        t, b = synth_ticks(signal=signal, seed=seed + i)
        bars[s] = make_bars(t, b, bar_s)
    df = build_dataset(bars, horizon)["BTC"]
    oos = walk_forward(df, feature_columns(df), horizon)
    return evaluate(oos, cost_bps=0, assumed_spread_bps=0)


def test_noise_is_unpredictable():
    stats, _ = run_oos(signal=0.0)
    assert abs(stats["accuracy"] - 0.5) < 0.015, stats


def test_planted_signal_is_found():
    stats, trades = run_oos(signal=1.0)
    assert stats["accuracy"] > 0.6, stats
    assert trades.gross_bps.iloc[0] > 0


def test_features_do_not_use_future():
    t, b = synth_ticks(n_seconds=2000)
    full = build_dataset({"BTC": make_bars(t, b, 5)}, 2)["BTC"]
    cut = 1000 * 1000 + t.ts_ms.min()
    part = build_dataset({"BTC": make_bars(t[t.ts_ms < cut], b[b.ts_ms < cut], 5)}, 2)["BTC"]
    cols = feature_columns(full)
    common = part.index[:-1]  # last partial bar may differ
    pd.testing.assert_frame_equal(full.loc[common, cols], part.loc[common, cols])


def test_storage_roundtrip(tmp_path):
    t, b = synth_ticks(n_seconds=1200)
    w = TickWriter("test", tmp_path, flush_every=100)
    for row in t.itertuples(index=False):
        w.add("BTC", "trades", list(row))
    w.flush()
    back = load_raw("test", "BTC", "trades", tmp_path)
    assert len(back) == len(t)
    np.testing.assert_allclose(back.price.values, t.sort_values("ts_ms").price.values)


def test_trade_prices_fake_an_edge():
    """Documents the bid-ask bounce trap: on pure noise, last-trade prices look ~65% predictable
    even though mids are unpredictable. This is why the backtest prefers recorded mids."""
    bars = {s: make_bars(synth_ticks(seed=i)[0], None, 1) for i, s in enumerate(["BTC", "ETH"])}
    df = build_dataset(bars, 1)["BTC"]
    stats, _ = evaluate(walk_forward(df, feature_columns(df), 1), cost_bps=0, assumed_spread_bps=0)
    assert stats["accuracy"] > 0.6


def test_pine_export_matches_python_model(tmp_path):
    from scalper.model import fit
    from scalper.pine import export, predict_like_pine

    bars = {s: make_bars(*synth_ticks(n_seconds=4 * 3600, signal=0.3, seed=i), 5) for i, s in enumerate(["BTC", "ETH", "SOL"])}
    df = build_dataset(bars, 2, "tv")["BTC"]
    cols = feature_columns(df)
    assert not any(c.startswith(("ofi", "spread", "book", "micro", "log_trades")) for c in cols)
    bundle = {"model": fit(df, cols, "logreg"), "features": cols, "bar_s": 5, "horizon": 2, "symbol": "BTC",
              "symbols": ["BTC", "ETH", "SOL"], "feature_set": "tv", "exchange": "binance", "cost_bps": 20.0}
    rows = df[cols].iloc[::97]
    expected = bundle["model"].predict_proba(rows)[:, 1]
    got = [predict_like_pine(bundle, r.to_dict()) for _, r in rows.iterrows()]
    np.testing.assert_allclose(got, expected, rtol=1e-9)

    script = export(bundle)
    assert script.startswith("//@version=6")
    for c in cols:
        assert f"nz({c}," in script
    assert 'request.security("BINANCE:ETHUSDT"' in script
