"""Load `config.toml` (public parameters) and `sources.local.toml` (private scraping setup)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "config.toml"
SOURCES_FILE = REPO_ROOT / "sources.local.toml"


@dataclass(frozen=True)
class SentimentConfig:
    model_name: str
    max_chars: int
    batch_size: int
    window_days: int
    half_life_days: float
    min_daily_weight: float
    publish_start: str | None
    zscore_min_days: int
    round_decimals: int
    source_weights: dict[str, float]


@dataclass(frozen=True)
class CompositeConfig:
    momentum_window: int
    volatility_window: int
    breadth_window: int
    news_max_ffill: int
    rank_window: int
    rank_min_obs: int
    corr_lambda: float
    corr_warmup_obs: int
    smoothing_span: int
    components: list[str]
    weights: np.ndarray


@dataclass(frozen=True)
class BandsConfig:
    elevated_percentile: float
    stress_percentile: float
    realtime_min_obs: int


@dataclass(frozen=True)
class EpisodesConfig:
    drawdown_threshold: float
    min_headlines: int
    topics: dict[str, str]          # topic name -> regular expression on lower-case titles


@dataclass(frozen=True)
class Config:
    paths: dict[str, Path]
    sentiment: SentimentConfig
    composite: CompositeConfig
    bands: BandsConfig
    episodes: EpisodesConfig


@dataclass(frozen=True)
class SiteConfig:
    """How to collect one news source. Values come from the private sources file."""

    label: str                      # anonymised source label, e.g. "A"
    output: Path                    # CSV kept up to date by the collector
    history_files: list[Path]       # older CSVs of the same source (read-only)
    first_page_url: str             # listing page 1; may contain {section}
    page_url: str                   # listing page n; contains {page}, may contain {section}
    sections: list[str]             # sub-listings to walk (empty = single listing)
    link_mode: str                  # "url_regex" (date in URL) or "item_blocks" (date in listing)
    exclude_sections: list[str] = field(default_factory=list)  # sections left out of scoring
    article_url_regex: str = ""     # url_regex mode: groups (year, month, day)
    item_selector: str = ""         # item_blocks mode: CSS selector of one listing entry
    link_prefix: str = ""           # item_blocks mode: href prefix of article links
    base_url: str = ""              # prepended to relative links
    start_page: str = "first"       # "first" or "bisect" (binary search for --until)
    max_page: int = 3000
    max_empty_pages: int = 1
    title_from: list[str] = field(default_factory=lambda: ["h1"])
    date_from: list[str] = field(default_factory=list)
    body_selector: str = ""
    body_tags: list[str] = field(default_factory=lambda: ["p"])
    delay: float = 0.7


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: Path = CONFIG_FILE) -> Config:
    """Read the public parameter file."""
    with open(path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    s = dict(raw["sentiment"])
    s["source_weights"] = {k: float(v) for k, v in s["source_weights"].items()}
    s["publish_start"] = s.get("publish_start") or None

    c = dict(raw["composite"])
    w = np.asarray(c["weights"], dtype=float)
    if len(w) != len(c["components"]):
        raise ValueError("composite.weights and composite.components differ in length")
    c["weights"] = w / w.sum()

    return Config(
        paths={k: _resolve(v) for k, v in raw["paths"].items()},
        sentiment=SentimentConfig(**s),
        composite=CompositeConfig(**c),
        bands=BandsConfig(**raw["bands"]),
        episodes=EpisodesConfig(**raw["episodes"]),
    )


def load_sites(path: Path = SOURCES_FILE) -> list[SiteConfig]:
    """Read the private scraping setup. Raises if the file does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Copy sources.example.toml to sources.local.toml "
            "and fill in the sites you want to collect.")
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    sites = []
    for s in raw.get("source", []):
        s = dict(s)
        s["output"] = _resolve(s["output"])
        s["history_files"] = [_resolve(p) for p in s.get("history_files", [])]
        sites.append(SiteConfig(**s))
    return sites
