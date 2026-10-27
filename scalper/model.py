"""Direction classifier with purged walk-forward evaluation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
        )
    raise ValueError(f"unknown model {kind!r}")


def _trainable(df: pd.DataFrame) -> pd.DataFrame:
    # Flat moves carry no direction; leave them out of training.
    return df[df.fwd_ret.notna() & (df.fwd_ret != 0)]


def fit(df: pd.DataFrame, cols: list[str], kind: str):
    tr = _trainable(df)
    return make_model(kind).fit(tr[cols], (tr.fwd_ret > 0).astype(int))


def walk_forward(df: pd.DataFrame, cols: list[str], horizon: int, kind: str = "logreg",
                 n_folds: int = 5, min_train_frac: float = 0.4) -> pd.DataFrame:
    """Expanding-window walk-forward. Each test block is predicted by a model trained only on
    earlier rows, with a purge gap of `horizon` bars so no training label overlaps the test block.
    Returns the test rows with an added `p_up` column (out-of-sample probability of an up move)."""
    df = df[df.fwd_ret.notna()]
    n = len(df)
    start = int(n * min_train_frac)
    size = (n - start) // n_folds
    if size < 50:
        raise ValueError(f"not enough data for walk-forward: {n} usable bars")
    parts = []
    for i in range(n_folds):
        lo, hi = start + i * size, (start + (i + 1) * size if i < n_folds - 1 else n)
        train = df.iloc[: max(lo - horizon, 0)]
        test = df.iloc[lo:hi].copy()
        model = fit(train, cols, kind)
        test["p_up"] = model.predict_proba(test[cols])[:, 1]
        test["fold"] = i
        test["train_up_rate"] = (_trainable(train).fwd_ret > 0).mean()
        parts.append(test)
    return pd.concat(parts)


def evaluate(oos: pd.DataFrame, cost_bps: float, assumed_spread_bps: float,
             thresholds=(0.0, 0.02, 0.05, 0.10)) -> tuple[dict, pd.DataFrame]:
    """Accuracy stats plus a per-threshold trading simulation (one trade per signal, held H bars)."""
    from sklearn.metrics import roc_auc_score

    moved = oos[oos.fwd_ret != 0]
    y = (moved.fwd_ret > 0).astype(int)
    pred = (moved.p_up > 0.5).astype(int)
    majority = (moved.train_up_rate > 0.5).astype(int)
    stats = {
        "test_bars": len(oos),
        "flat_share": 1 - len(moved) / max(len(oos), 1),
        "accuracy": (pred == y).mean(),
        "baseline_majority": (majority == y).mean(),
        "baseline_reversal": ((moved.ret_1 < 0).astype(int) == y)[moved.ret_1 != 0].mean(),
        "auc": roc_auc_score(y, moved.p_up) if y.nunique() == 2 else np.nan,
    }
    spread = oos.rt_spread_bps.fillna(assumed_spread_bps)
    rows = []
    for thr in thresholds:
        sig = np.sign(oos.p_up - 0.5).where((oos.p_up - 0.5).abs() > thr, 0)
        on = sig != 0
        gross = (sig * oos.fwd_ret * 1e4)[on]
        net = gross - spread[on] - cost_bps
        rows.append({
            "threshold": thr, "trades": int(on.sum()), "trade_share": on.mean(),
            "hit_rate": (gross > 0).mean() if on.any() else np.nan,
            "gross_bps": gross.mean() if on.any() else np.nan,
            "spread_bps": spread[on].mean() if on.any() else np.nan,
            "net_bps": net.mean() if on.any() else np.nan,
        })
    return stats, pd.DataFrame(rows)
