"""Look-ahead test: a value computed with data up to date T must not change when
later data is added. Runs on the published (processed) data only."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stress_index.composite import build_stress_index  # noqa: E402
from stress_index.config import load_config  # noqa: E402
from stress_index.io import read_frame  # noqa: E402
from stress_index.sentiment import daily_sentiment, load_article_scores, sentiment_index  # noqa: E402

CFG = load_config()
CUTS = ["2020-03-31", "2021-06-30", "2023-01-31", "2025-10-15"]


def _same_prefix(full: pd.DataFrame, part: pd.DataFrame) -> None:
    common = part.index
    num = full.loc[common].select_dtypes("number")
    diff = (num - part[num.columns]).abs().to_numpy()
    assert np.nanmax(diff) < 1e-9, f"values change when later data is added (max diff {np.nanmax(diff)})"
    pd.testing.assert_series_equal(full.loc[common].isna().sum(), part.isna().sum())


def test_composite_has_no_lookahead():
    raw = read_frame(CFG.paths["market_features"])[CFG.composite.components]
    full = build_stress_index(raw, CFG.composite, CFG.bands)
    for cut in CUTS:
        part = build_stress_index(raw.loc[:cut], CFG.composite, CFG.bands)
        _same_prefix(full, part)
        assert (full.loc[part.index, "regime_realtime"].fillna("") == part["regime_realtime"].fillna("")).all()


def test_sentiment_has_no_lookahead():
    scores = load_article_scores(CFG.paths["article_scores"])
    full = sentiment_index(scores, CFG.sentiment).set_index("date")
    for cut in CUTS:
        part = sentiment_index(scores[scores["date"] <= cut], CFG.sentiment).set_index("date")
        _same_prefix(full, part)


def test_market_components_have_no_lookahead():
    """Needs the local raw price data; skipped when it is absent."""
    if not CFG.paths["market_workbook"].exists():
        print("  (raw market data absent: skipped)")
        return
    from stress_index import market as mk
    masi, stocks, mbi = mk.load_equity_workbook(CFG.paths["market_workbook"])
    brent = mk.load_brent(CFG.paths["brent_csv"])
    news = read_frame(CFG.paths["sentiment_index"])["sentiment_raw"]
    full = mk.raw_components(mk.align_inputs(masi, stocks, mbi, news, brent,
                                             CFG.composite.news_max_ffill), CFG.composite)
    for cut in CUTS:
        part = mk.raw_components(mk.align_inputs(masi.loc[:cut], stocks.loc[:cut], mbi.loc[:cut],
                                                 news.loc[:cut], brent.loc[:cut],
                                                 CFG.composite.news_max_ffill), CFG.composite)
        _same_prefix(full, part)


def test_contributions_sum_to_index():
    raw = read_frame(CFG.paths["market_features"])[CFG.composite.components]
    out = build_stress_index(raw, CFG.composite, CFG.bands)
    total = out.filter(like="contrib_").sum(axis=1)
    assert (total - out["stress_index"]).abs().max() < 1e-9


def test_index_is_bounded():
    df = read_frame(CFG.paths["stress_index"])
    assert df["stress_index"].between(0, 100).all()
    ranks = df[CFG.composite.components]
    assert ((ranks > 0) & (ranks <= 1)).all().all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
