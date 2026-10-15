"""Press sentiment: score each article with a French transformer, then aggregate daily.

Pipeline:  article text -> stress score in [-1, +1] -> daily weighted average over a
rolling window with exponential time decay (+ a real-time z-score for reading).

Scores are oriented towards *stress*: a negative-toned article scores close to +1,
a reassuring one close to -1.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, SentimentConfig, SiteConfig

# ----------------------------------------------------------------------------- text


def clean_text(txt: object) -> str:
    """Remove separators, URLs and leftover HTML entities; collapse whitespace."""
    if not isinstance(txt, str):
        return ""
    t = txt.replace("-oOo-", " ").replace("-ooo-", " ")
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"&#\d+;", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_model_input(title: str, text: str, max_chars: int) -> str:
    """Title then body, truncated. The title is not up-weighted, only placed first."""
    return (clean_text(title) + ". " + clean_text(text)).strip()[:max_chars]


def content_hash(title: str, text: str) -> str:
    """Fingerprint of an article, so an edited article (same URL) gets re-scored."""
    h = hashlib.md5()
    h.update((title or "").encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update((text or "").encode("utf-8", "ignore"))
    return h.hexdigest()


def model_signature(cfg: SentimentConfig) -> str:
    """Fingerprint of the scoring setup; a change invalidates every cached score."""
    return hashlib.md5(repr((cfg.model_name, cfg.max_chars)).encode()).hexdigest()


# ----------------------------------------------------------------------------- model
class SentimentModel:
    """Lazy wrapper around the Hugging Face model (GPU + fp16 when available)."""

    def __init__(self, cfg: SentimentConfig):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"loading {cfg.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name).to(self.device)
        self.model.eval()
        if self.device == "cuda":
            self.model.half()
        labels = self.model.config.id2label
        # labels look like "1 star" ... "5 stars": keep the number of stars
        self.stars = np.array([int(re.search(r"\d", str(labels[i])).group())
                               for i in range(len(labels))], dtype=float)

    def score(self, texts: list[str]) -> np.ndarray:
        """Stress score in [-1, +1] for each text.

        The model outputs a probability for 1..5 stars; the expected number of
        stars is mapped linearly to [-1, +1] (sentiment) and its sign flipped (stress).
        """
        lo, hi = self.stars.min(), self.stars.max()
        out = np.empty(len(texts), dtype=np.float32)
        bs = self.cfg.batch_size
        with self.torch.no_grad():
            for b in range(0, len(texts), bs):
                enc = self.tokenizer(texts[b:b + bs], return_tensors="pt", truncation=True,
                                     max_length=512, padding=True).to(self.device)
                probs = self.torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
                expected_stars = probs @ self.stars
                sentiment = 2.0 * (expected_stars - lo) / (hi - lo) - 1.0
                out[b:b + bs] = -sentiment
                done = min(b + bs, len(texts))
                if done % (bs * 20) == 0 or done == len(texts):
                    print(f"  {done}/{len(texts)} articles scored")
        return out


# ----------------------------------------------------------------------------- raw articles
def load_articles(sites: list[SiteConfig]) -> pd.DataFrame:
    """All local article CSVs as one frame, de-duplicated by URL.

    Columns: url, date, title, text, source, section, content_hash.
    """
    frames = []
    for site in sites:
        for path in [site.output, *site.history_files]:
            if not path.exists():
                print(f"  [!] missing, skipped: {path.name}")
                continue
            d = pd.read_csv(path, encoding="utf-8-sig")
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d["title"] = d["title"].fillna("") if "title" in d else ""
            d["text"] = d["text"].fillna("") if "text" in d else ""
            d = d.dropna(subset=["date"])
            d = d[(d["title"].str.len() + d["text"].str.len()) > 0].copy()
            d["source"] = site.label
            d["section"] = d["section"].fillna("").astype(str) if "section" in d else ""
            excluded = d["section"].isin(site.exclude_sections)
            d = d[~excluded]
            frames.append(d[["url", "date", "title", "text", "source", "section"]])
            print(f"  source {site.label} {path.name}: {len(d)} articles"
                  + (f" ({int(excluded.sum())} in excluded sections)" if excluded.any() else ""))
    if not frames:
        raise FileNotFoundError("no article file found; run the collector first")
    df = pd.concat(frames, ignore_index=True)
    n = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    if n != len(df):
        print(f"  {n - len(df)} duplicate URLs removed")
    df["content_hash"] = [content_hash(t, a) for t, a in zip(df["title"], df["text"])]
    return df


# ----------------------------------------------------------------------------- scoring with cache
CACHE_COLUMNS = ["url", "source", "date", "content_hash", "score_stress"]


def score_articles(articles: pd.DataFrame, cfg: Config, rebuild: bool = False) -> pd.DataFrame:
    """Add `score_stress` to `articles`, scoring only new or edited articles.

    The cache (private: it is keyed by URL) is rewritten after each run. If the
    model or text settings changed since the cache was written, everything is
    re-scored.
    """
    cache_path, meta_path = cfg.paths["score_cache"], cfg.paths["score_cache_meta"]
    sig = model_signature(cfg.sentiment)
    if meta_path.exists() and json.loads(meta_path.read_text("utf-8")).get("model_signature") != sig:
        print("  scoring setup changed -> full re-score")
        rebuild = True

    cache: dict[str, tuple[str, float]] = {}
    if cache_path.exists() and not rebuild:
        c = pd.read_csv(cache_path, encoding="utf-8")
        cache = dict(zip(c["url"], zip(c["content_hash"].fillna("").astype(str), c["score_stress"])))

    scores = np.full(len(articles), np.nan)
    todo: list[int] = []
    for i, (url, h) in enumerate(zip(articles["url"], articles["content_hash"])):
        hit = cache.get(url)
        if hit is not None and hit[0] in ("", h):
            scores[i] = hit[1]
        else:
            todo.append(i)
    print(f"  {len(articles) - len(todo)} scores from cache, {len(todo)} to compute")
    if todo:
        model = SentimentModel(cfg.sentiment)
        texts = [build_model_input(articles["title"].iat[i], articles["text"].iat[i],
                                   cfg.sentiment.max_chars) for i in todo]
        scores[todo] = model.score(texts)

    out = articles.copy()
    out["score_stress"] = scores
    cache_df = out[CACHE_COLUMNS].copy()
    cache_df["date"] = cache_df["date"].dt.strftime("%Y-%m-%d")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(cache_path, index=False, encoding="utf-8")
    meta_path.write_text(json.dumps({"model_signature": sig, "model": cfg.sentiment.model_name}),
                         encoding="utf-8")
    return out


def export_article_scores(scored: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Public, anonymised per-article scores: date, source label, score (no URL, no text)."""
    pub = scored.dropna(subset=["score_stress"])[["date", "source", "score_stress"]].copy()
    pub["date"] = pd.to_datetime(pub["date"]).dt.strftime("%Y-%m-%d")
    pub = pub.sort_values(["date", "source", "score_stress"], kind="mergesort")
    path.parent.mkdir(parents=True, exist_ok=True)
    pub.to_csv(path, index=False, encoding="utf-8")
    return pub


def headline_topic_counts(articles: pd.DataFrame, topics: dict[str, str]) -> pd.DataFrame:
    """Daily number of headlines matching each topic (public: counts only, no text)."""
    titles = articles["title"].str.lower()
    rows = []
    for topic, pattern in topics.items():
        hits = articles.loc[titles.str.contains(pattern, regex=True), "date"].dt.normalize()
        counts = hits.value_counts().sort_index()
        rows.append(pd.DataFrame({"date": counts.index.strftime("%Y-%m-%d"), "topic": topic,
                                  "n_headlines": counts.values}))
    return pd.concat(rows, ignore_index=True).sort_values(["date", "topic"]).reset_index(drop=True)


def load_article_scores(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8")
    d["date"] = pd.to_datetime(d["date"])
    return d


# ----------------------------------------------------------------------------- daily index
def daily_sentiment(scores: pd.DataFrame, cfg: SentimentConfig) -> pd.DataFrame:
    """Daily stress sentiment: weighted mean of the articles of the last `window_days`.

    weight(article) = source_weight * 0.5 ** (age_in_days / half_life_days).
    A day is published only if its summed weight reaches `min_daily_weight`.
    Articles before `publish_start` feed the window but no day before it is published.

    Returns columns: date, sentiment_raw, window_weight.
    """
    art = scores.dropna(subset=["score_stress"]).copy()
    art["day"] = art["date"].dt.normalize()
    first, last = art["day"].min(), art["day"].max()
    start = first if cfg.publish_start is None else max(first, pd.Timestamp(cfg.publish_start))
    calendar = pd.date_range(start, last, freq="D")

    day_num = (art["day"] - first).dt.days.to_numpy().astype(int)
    order = np.argsort(day_num)
    d_sorted = day_num[order]
    s_sorted = art["score_stress"].to_numpy(float)[order]
    w_sorted = art["source"].map(cfg.source_weights).to_numpy(float)[order]

    decay = 0.5 ** (1.0 / cfg.half_life_days)
    value = np.full(len(calendar), np.nan)
    weight = np.zeros(len(calendar))
    for i, t in enumerate((calendar - first).days.to_numpy()):
        j0 = np.searchsorted(d_sorted, t - cfg.window_days + 1, side="left")
        j1 = np.searchsorted(d_sorted, t, side="right")
        if j1 <= j0:
            continue
        w = w_sorted[j0:j1] * decay ** (t - d_sorted[j0:j1])
        weight[i] = w.sum()
        if weight[i] >= cfg.min_daily_weight:
            value[i] = float(np.dot(w, s_sorted[j0:j1]) / weight[i])
    return pd.DataFrame({"date": calendar, "sentiment_raw": value, "window_weight": weight})


def sentiment_index(scores: pd.DataFrame, cfg: SentimentConfig) -> pd.DataFrame:
    """Daily sentiment series: `date`, `sentiment_raw`, `sentiment_z`.

    `sentiment_raw` feeds the composite (which ranks it, so its scale does not
    matter). `sentiment_z` is a real-time z-score for standalone reading: each day
    is standardised with the mean and std of the days up to and including it,
    once `zscore_min_days` of history exist.
    """
    daily = daily_sentiment(scores, cfg).dropna(subset=["sentiment_raw"])
    x = daily["sentiment_raw"]
    past = x.expanding(min_periods=cfg.zscore_min_days)
    z = (x - past.mean()) / past.std(ddof=0)
    d = cfg.round_decimals
    out = pd.DataFrame({"date": daily["date"], "sentiment_raw": x.round(d), "sentiment_z": z.round(d)})
    return out.reset_index(drop=True)
