"""Read and write the published datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_config


def save_frame(df: pd.DataFrame, path: Path, decimals: int | None = 6) -> None:
    """Write a date-indexed frame as CSV with ISO dates (`decimals=None`: full precision)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (df if decimals is None else df.round(decimals)).reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, encoding="utf-8")


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def load_stress_index(path: str | Path | None = None,
                      include_contributions: bool = False) -> pd.DataFrame:
    """The Morocco market stress index as a DataFrame indexed by trading date.

    Columns:
        stress_index                 0-100, high = stress (100 x CISS; fixed scale,
                                     past values are never revised)
        regime_realtime              "calm" / "elevated" / "stress", from thresholds
                                     known on that day (empty during the first year)
        elevated_threshold_realtime  expanding 75th percentile of the index
        stress_threshold_realtime    expanding 95th percentile of the index
        equity_momentum, equity_volatility, bond_volatility, market_breadth,
        news_sentiment, oil_volatility
                                     component percentile ranks in [0, 1], high = stress
        contrib_<component>          (optional) contribution in index points; the six
                                     contributions sum to `stress_index`

    Timing: the value dated t uses closing prices of day t and articles published
    on day t, some after the market close. Use it for decisions from day t+1 on
    (e.g. `df.shift(1)` before joining with day-t returns).

    Parameters
    ----------
    path : CSV to read; defaults to `data/processed/stress_index.csv` of this repository.
    include_contributions : keep the `contrib_*` columns.
    """
    df = read_frame(Path(path) if path else load_config().paths["stress_index"])
    if not include_contributions:
        df = df.loc[:, ~df.columns.str.startswith("contrib_")]
    return df
