"""Market inputs and the raw (un-normalised) stress components.

Inputs (kept locally, not published):
- a workbook of daily closes: sheet 'cours' (one column per listed stock + MASI)
  and sheet 'benchmarks' (date, MASI, MBI);
- Brent daily prices (CSV with `Date` in MM/DD/YYYY and `Price`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CompositeConfig


def load_equity_workbook(path: Path) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """(MASI, stock closes, MBI) indexed by trading date.

    MASI comes from the stock sheet and is completed by the benchmark sheet where
    missing. MBI is re-indexed on MASI dates without filling.
    """
    stocks = pd.read_excel(path, sheet_name="cours").rename(columns={"Unnamed: 0": "date"})
    stocks["date"] = pd.to_datetime(stocks["date"], errors="coerce")
    stocks = stocks.dropna(subset=["date"]).sort_values("date").set_index("date")

    bench = pd.read_excel(path, sheet_name="benchmarks")
    bench["date"] = pd.to_datetime(bench["date"], errors="coerce")
    bench = bench.dropna(subset=["date"]).sort_values("date").set_index("date")

    masi = stocks["MASI"].astype(float)
    if "MASI" in bench.columns:
        masi = masi.combine_first(bench["MASI"].reindex(masi.index))
    stocks = stocks.drop(columns=["MASI"])
    mbi = bench["MBI"].reindex(masi.index)
    return masi, stocks, mbi


def load_brent(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    return df["Price"].astype(float)


def align_inputs(masi: pd.Series, stocks: pd.DataFrame, mbi: pd.Series, news: pd.Series,
                 brent: pd.Series, news_max_ffill: int) -> pd.DataFrame:
    """Put every input on the Casablanca trading calendar.

    News sentiment is carried forward at most `news_max_ffill` trading days. Any
    date where MASI, MBI, news or Brent is missing is dropped (inner join, no filling).
    Returns a frame with columns masi, mbi, news, brent and the stock closes.
    """
    news = news.reindex(masi.index).ffill(limit=news_max_ffill)
    base = pd.DataFrame({"masi": masi, "mbi": mbi, "news": news,
                         "brent": brent.reindex(masi.index)}).dropna(how="any")
    return base.join(stocks.reindex(base.index))


def raw_components(aligned: pd.DataFrame, cfg: CompositeConfig) -> pd.DataFrame:
    """The six stress components, all oriented HIGH = STRESS, before normalisation."""
    masi, mbi, brent = aligned["masi"], aligned["mbi"], aligned["brent"]
    stocks = aligned.drop(columns=["masi", "mbi", "news", "brent"])

    def realised_vol(price: pd.Series) -> pd.Series:
        return np.log(price / price.shift(1)).rolling(cfg.volatility_window).std()

    ma = stocks.rolling(cfg.breadth_window).mean()
    valid = stocks.notna() & ma.notna()
    share_above = (stocks > ma).where(valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)

    out = pd.DataFrame({
        "equity_momentum": -(masi / masi.rolling(cfg.momentum_window).mean() - 1.0),
        "equity_volatility": realised_vol(masi),
        "bond_volatility": realised_vol(mbi),
        "market_breadth": -share_above,
        "news_sentiment": aligned["news"],
        "oil_volatility": realised_vol(brent),
    })
    return out[cfg.components]


def data_quality(masi: pd.Series, mbi: pd.Series, brent: pd.Series, news: pd.Series,
                 aligned: pd.DataFrame, hole_days: int = 10) -> dict:
    """Facts about calendar alignment, reported in the limits section.

    Counts trading days (from the first sentiment date) dropped because a
    series is missing, returns that span more than one trading day after the
    drop, and holes longer than `hole_days` calendar days in the market data.
    """
    trading = masi.dropna().index
    since = trading[trading >= news.dropna().index.min()]
    missing = {"bond_index": int(mbi.reindex(since).isna().sum()),
               "brent": int(brent.reindex(since).isna().sum())}
    pos = trading.get_indexer(aligned.index)
    gaps = trading.to_series().diff().dt.days
    holes = [{"from": str(trading[i - 1].date()), "to": str(trading[i].date())}
             for i in np.flatnonzero(gaps.to_numpy() > hole_days)]
    return {"dropped_trading_days_since_sentiment_start": missing,
            "returns_spanning_two_or_more_days": int((np.diff(pos) > 1).sum()),
            "market_data_holes": holes}
