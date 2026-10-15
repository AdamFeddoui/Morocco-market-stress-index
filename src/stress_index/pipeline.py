"""Command-line pipeline.

    python -m stress_index collect     # update local article files (needs sources.local.toml)
    python -m stress_index score       # score new articles -> data/processed/article_scores.csv
    python -m stress_index sentiment   # daily sentiment     -> data/processed/sentiment_index.csv
    python -m stress_index market      # raw components      -> data/processed/market_features.csv
                                       # + validation data   -> data/processed/validation_targets.csv
    python -m stress_index composite   # stress index        -> data/processed/stress_index.csv
    python -m stress_index all         # every step above, in order

`sentiment` and `composite` run from published data only; `collect`, `score` and
`market` need the local raw data (articles, prices).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from . import market as mk
from .collect import collect_all
from .composite import build_stress_index
from .config import Config, load_config, load_sites
from .io import read_frame, save_frame
from .sentiment import (export_article_scores, headline_topic_counts, load_article_scores,
                        load_articles, score_articles, sentiment_index)
from .validation import validation_targets


def step_collect(cfg: Config, since: str | None = None, until: str | None = None) -> None:
    collect_all(load_sites(), since=since, until=until)


def step_score(cfg: Config, rebuild: bool = False) -> None:
    articles = load_articles(load_sites())
    scored = score_articles(articles, cfg, rebuild=rebuild)
    pub = export_article_scores(scored, cfg.paths["article_scores"])
    print(f"article scores: {len(pub)} rows, {pub['date'].min()} -> {pub['date'].max()}")
    headline_topic_counts(articles, cfg.episodes.topics).to_csv(
        cfg.paths["headline_topics"], index=False, encoding="utf-8")


def step_sentiment(cfg: Config) -> pd.DataFrame:
    idx = sentiment_index(load_article_scores(cfg.paths["article_scores"]), cfg.sentiment)
    save_frame(idx.set_index("date"), cfg.paths["sentiment_index"], decimals=None)
    print(f"sentiment index: {len(idx)} days, {idx['date'].min().date()} -> {idx['date'].max().date()}")
    return idx


def step_market(cfg: Config) -> pd.DataFrame:
    masi, stocks, mbi = mk.load_equity_workbook(cfg.paths["market_workbook"])
    brent = mk.load_brent(cfg.paths["brent_csv"])
    news = read_frame(cfg.paths["sentiment_index"])["sentiment_raw"]
    aligned = mk.align_inputs(masi, stocks, mbi, news, brent, cfg.composite.news_max_ffill)
    print(f"market: {masi.notna().sum()} trading days, {stocks.shape[1]} stocks; "
          f"{len(masi) - len(aligned)} sheet dates dropped as incomplete, {len(aligned)} kept")
    feats = mk.raw_components(aligned, cfg.composite)
    save_frame(feats, cfg.paths["market_features"], decimals=None)
    save_frame(validation_targets(masi), cfg.paths["validation_targets"], decimals=8)
    meta = {"n_stocks": int(stocks.shape[1]), "n_trading_days": int(masi.notna().sum()),
            "first_trading_day": str(masi.dropna().index.min().date()),
            "last_trading_day": str(masi.dropna().index.max().date()),
            "aligned_days": len(aligned), "brent_last_day": str(brent.index.max().date()),
            **mk.data_quality(masi, mbi, brent, news, aligned)}
    cfg.paths["market_metadata"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return feats


def step_composite(cfg: Config) -> pd.DataFrame:
    raw = read_frame(cfg.paths["market_features"])[cfg.composite.components]
    out = build_stress_index(raw, cfg.composite, cfg.bands)
    save_frame(out, cfg.paths["stress_index"], decimals=4)
    print(f"stress index: {len(out)} days, {out.index.min().date()} -> {out.index.max().date()}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="stress_index", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["collect", "score", "sentiment", "market", "composite", "all"])
    ap.add_argument("--since", help="collect: first date (YYYY-MM-DD); default = resume point")
    ap.add_argument("--until", help="collect: last date (YYYY-MM-DD); default = today")
    ap.add_argument("--rebuild", action="store_true", help="score: ignore the cache")
    args = ap.parse_args(argv)
    cfg = load_config()

    if args.step in ("collect", "all"):
        step_collect(cfg, args.since, args.until)
    if args.step in ("score", "all"):
        step_score(cfg, args.rebuild)
    if args.step in ("sentiment", "all"):
        step_sentiment(cfg)
    if args.step in ("market", "all"):
        step_market(cfg)
    if args.step in ("composite", "all"):
        step_composite(cfg)


if __name__ == "__main__":
    main()
