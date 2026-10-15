"""Does the index measure stress? Compare it with realised volatility and MASI drawdowns.

Targets are computed on the full MASI trading calendar. Forward-looking targets
(next 20 trading days) are *outcomes* used only to evaluate the index, never as inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def validation_targets(masi: pd.Series, window: int = 20) -> pd.DataFrame:
    """Per trading date: drawdown, trailing and forward realised volatility, forward return.

    - drawdown: MASI / running peak - 1 (<= 0)
    - realised_vol: annualised std of the last `window` daily log returns (ending today)
    - fwd_realised_vol: same over the next `window` returns (t+1 .. t+window)
    - fwd_return: MASI return from t to t+window
    - fwd_max_drawdown: worst close over t+1 .. t+window relative to the close at t
    - masi_log_return: daily log return (lets the MASI path be rebuilt up to scale)
    """
    masi = masi.dropna().astype(float)
    r = np.log(masi / masi.shift(1))
    vol = r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    fwd_min = masi[::-1].rolling(window).min()[::-1].shift(-1)
    return pd.DataFrame({
        "masi_log_return": r,
        "drawdown": masi / masi.cummax() - 1.0,
        "realised_vol": vol,
        "fwd_realised_vol": vol.shift(-window),
        "fwd_return": masi.shift(-window) / masi - 1.0,
        "fwd_max_drawdown": (fwd_min / masi - 1.0).clip(upper=0.0),
    })


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation on common non-missing dates (Pearson correlation of ranks)."""
    pair = pd.concat([a, b], axis=1, sort=True).dropna()
    return pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())


def rebased_masi(targets: pd.DataFrame, start: pd.Timestamp | str) -> pd.Series:
    """MASI rebased to 100 on `start`, rebuilt from the published daily log returns."""
    r = targets["masi_log_return"].loc[start:].fillna(0.0)
    r.iloc[0] = 0.0
    return 100 * np.exp(r.cumsum())


def partial_spearman(x: pd.Series, y: pd.Series, control: pd.Series) -> float:
    """Rank correlation of x and y after removing the linear effect of `control` on both ranks."""
    d = pd.concat([x, y, control], axis=1, sort=True).dropna().rank()
    r = d.corr().to_numpy()
    rxy, rxz, ryz = r[0, 1], r[0, 2], r[1, 2]
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def regime_table(regime: pd.Series, targets: pd.DataFrame) -> pd.DataFrame:
    """Forward outcomes by real-time regime (calm / elevated / stress)."""
    d = pd.concat([regime.rename("regime"), targets[["fwd_realised_vol", "fwd_max_drawdown"]]],
                  axis=1, sort=True).dropna()
    out = d.groupby("regime").agg(n_days=("fwd_realised_vol", "size"),
                                  fwd_realised_vol=("fwd_realised_vol", "mean"),
                                  fwd_max_drawdown=("fwd_max_drawdown", "mean"))
    return out.reindex([r for r in ["calm", "elevated", "stress"] if r in out.index])


def correlation_table(index: pd.Series, targets: pd.DataFrame) -> pd.DataFrame:
    """Pearson and Spearman correlation of the index with each target (common dates)."""
    rows = []
    for col in targets.columns:
        pair = pd.concat([index, targets[col]], axis=1, sort=True).dropna()
        rows.append({"target": col, "n_days": len(pair),
                     "pearson": pair.iloc[:, 0].corr(pair.iloc[:, 1]),
                     "spearman": spearman(pair.iloc[:, 0], pair.iloc[:, 1])})
    return pd.DataFrame(rows).set_index("target")


def bucket_table(index: pd.Series, targets: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """Average target values per index quantile bucket (Q1 = calmest days)."""
    df = pd.concat([index.rename("index"), targets], axis=1, sort=True).dropna()
    df["bucket"] = pd.qcut(df["index"], q, labels=[f"Q{i}" for i in range(1, q + 1)])
    out = df.groupby("bucket", observed=True).mean()
    out.insert(0, "n_days", df.groupby("bucket", observed=True).size())
    return out


def drawdown_episodes(masi: pd.Series, threshold: float = -0.10) -> pd.DataFrame:
    """MASI peak-to-trough episodes deeper than `threshold` (e.g. -0.10 = 10%).

    An episode starts at a running peak, reaches its trough, and ends when the
    previous peak is recovered (or at the end of the sample).
    """
    masi = masi.dropna()
    peak_val, peak_date = masi.iloc[0], masi.index[0]
    trough_val, trough_date = peak_val, peak_date
    episodes = []
    for d, v in masi.items():
        if v >= peak_val:
            if trough_val / peak_val - 1 <= threshold:
                episodes.append((peak_date, trough_date, d, trough_val / peak_val - 1))
            peak_val, peak_date, trough_val, trough_date = v, d, v, d
        elif v < trough_val:
            trough_val, trough_date = v, d
    if trough_val / peak_val - 1 <= threshold:
        episodes.append((peak_date, trough_date, pd.NaT, trough_val / peak_val - 1))
    return pd.DataFrame(episodes, columns=["peak", "trough", "recovery", "depth"])


def episode_table(index: pd.Series, episodes: pd.DataFrame, after_days: int = 60) -> pd.DataFrame:
    """Index behaviour from each episode's MASI peak to its trough, and in the
    `after_days` trading days that follow the trough."""
    rows = []
    for e in episodes.itertuples(index=False):
        window = index.loc[e.peak:e.trough]
        before = index.loc[:e.peak].iloc[-60:]           # ~3 months before the peak
        if window.empty:
            continue
        after = index.loc[e.trough:].iloc[:after_days + 1]
        rows.append({"peak": e.peak.date(), "trough": e.trough.date(),
                     "masi_depth_pct": round(100 * e.depth, 1),
                     "index_before_mean": round(before.mean(), 1) if len(before) else np.nan,
                     "index_mean": round(window.mean(), 1),
                     "index_max": round(window.max(), 1),
                     "index_max_date": window.idxmax().date(),
                     "index_max_after_trough": round(after.max(), 1),
                     "index_max_after_trough_date": after.idxmax().date()})
    return pd.DataFrame(rows)
