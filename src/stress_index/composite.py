"""Composite stress index following the ECB CISS method (Hollo, Kremer & Lo Duca, 2012).

    CISS_t = (w o s_t) C_t (w o s_t)'

s_t: components mapped to [0, 1] by a rolling percentile rank; w: weights;
o: element-wise product; C_t: time-varying correlation matrix estimated by EWMA.
The quadratic form makes the index rise more when components are stressed
*together* (systemic stress) than when one channel alone is stressed.

Every step uses only data available on the date computed: backward-looking
windows, a recursion started without future data, and a fixed 0-100 scale
(100 x CISS, since CISS is bounded in [0, 1]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BandsConfig, CompositeConfig


def rolling_percentile_rank(series: pd.Series, window: int, min_obs: int) -> pd.Series:
    """Share of the last `window` observations (current one included) that are <= today."""
    v = series.to_numpy(float)
    out = np.full(len(v), np.nan)
    for t in range(len(v)):
        if np.isnan(v[t]):
            continue
        w = v[max(0, t - window + 1):t + 1]
        w = w[~np.isnan(w)]
        if len(w) >= min_obs:
            out[t] = (w <= v[t]).sum() / len(w)
    return pd.Series(out, index=series.index)


def ciss_aggregate(ranks: pd.DataFrame, weights: np.ndarray,
                   lam: float) -> tuple[pd.Series, pd.DataFrame]:
    """CISS value and each component's additive contribution, per date.

    Covariances of the de-meaned ranks (s - 0.5) follow
    cov_t = lam * cov_{t-1} + (1 - lam) * x_t x_t'. The recursion starts from the
    covariance of independent uniform ranks (variance 1/12, no correlation), so
    no future observation is used; the first days are a warm-up dropped by the
    caller. The contribution of component i is (w_i s_i) * (C_t (w o s_t))_i, and
    contributions sum to the CISS value. CISS lies in [0, 1] by construction.
    """
    s = ranks.to_numpy(float)
    x = s - 0.5
    cov = np.eye(s.shape[1]) / 12.0

    value = np.full(len(s), np.nan)
    contrib = np.full(s.shape, np.nan)
    for t in range(len(s)):
        cov = lam * cov + (1 - lam) * np.outer(x[t], x[t])
        sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = cov / np.outer(sd, sd)
        np.fill_diagonal(corr, 1.0)
        ws = weights * s[t]
        value[t] = float(ws @ corr @ ws.T)
        contrib[t] = ws * (corr @ ws)
    return (pd.Series(value, index=ranks.index),
            pd.DataFrame(contrib, index=ranks.index, columns=ranks.columns))


def realtime_regimes(index: pd.Series, bands: BandsConfig) -> pd.DataFrame:
    """Calm / elevated / stress label per day, from thresholds known on that day.

    Thresholds are percentiles of the index over all days up to and including
    the current one (expanding window), available after `realtime_min_obs` days.
    """
    past = index.expanding(min_periods=bands.realtime_min_obs)
    elevated = past.quantile(bands.elevated_percentile / 100)
    stress = past.quantile(bands.stress_percentile / 100)
    regime = pd.Series(np.where(index >= stress, "stress",
                                np.where(index >= elevated, "elevated", "calm")), index=index.index)
    regime[stress.isna()] = None
    return pd.DataFrame({"regime_realtime": regime, "elevated_threshold_realtime": elevated,
                         "stress_threshold_realtime": stress})


def build_stress_index(raw: pd.DataFrame, cfg: CompositeConfig, bands: BandsConfig) -> pd.DataFrame:
    """From raw components to the 0-100 stress index (100 x smoothed CISS).

    Returns, indexed by date: `stress_index` (0-100, high = stress), real-time
    regime and thresholds, one rank column per component in [0, 1], and
    `contrib_<component>`, each component's contribution in index points (they
    sum to `stress_index`). The first `corr_warmup_obs` days are not returned.
    """
    ranks = raw.apply(rolling_percentile_rank, window=cfg.rank_window,
                      min_obs=cfg.rank_min_obs).dropna(how="any")
    ciss_raw, contrib = ciss_aggregate(ranks, cfg.weights, cfg.corr_lambda)

    # EWMA smoothing is linear, so smoothed contributions still sum to the smoothed index
    index = 100 * ciss_raw.ewm(span=cfg.smoothing_span, adjust=False).mean()
    contrib = 100 * contrib.ewm(span=cfg.smoothing_span, adjust=False).mean()

    keep = ranks.index[cfg.corr_warmup_obs:]
    index, ranks, contrib = index.loc[keep], ranks.loc[keep], contrib.loc[keep]
    out = (index.rename("stress_index").to_frame()
           .join(realtime_regimes(index, bands)).join(ranks).join(contrib.add_prefix("contrib_")))
    out.index.name = "date"
    return out
