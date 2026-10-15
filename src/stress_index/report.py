"""Key numbers and figures, computed from the published data only.

    python -m stress_index.report     # -> reports/key_numbers.json + figures/*.png

Every number quoted in the README, the PDF summary and the notebook comes from
`key_numbers()`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plotting
from . import validation as val
from .config import Config, load_config
from .io import load_stress_index, read_frame
from .sentiment import load_article_scores

TARGETS = ["drawdown", "realised_vol", "fwd_realised_vol", "fwd_max_drawdown", "fwd_return"]


def load_all(cfg: Config) -> dict[str, pd.DataFrame | pd.Series | dict]:
    """Every published dataset the analysis needs."""
    index = load_stress_index(cfg.paths["stress_index"], include_contributions=True)
    targets = read_frame(cfg.paths["validation_targets"])
    topics = pd.read_csv(cfg.paths["headline_topics"], parse_dates=["date"])
    return {
        "index": index,
        "targets": targets,
        "masi": val.rebased_masi(targets, index.index.min()),
        "scores": load_article_scores(cfg.paths["article_scores"]),
        "sentiment": read_frame(cfg.paths["sentiment_index"]),
        "topics": topics,
        "market_meta": json.loads(cfg.paths["market_metadata"].read_text("utf-8")),
    }


def named_episodes(masi: pd.Series, topics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """MASI drawdown episodes, named after a headline topic only when the topic is clear.

    For each topic, compare its number of headlines between the MASI peak and
    trough with the number expected at its average daily rate over the whole
    article sample. A topic qualifies if it has at least `min_headlines` and at
    least 3x the expected count; the episode takes the qualifying topic with the
    highest ratio, otherwise it stays unnamed.
    """
    eps = val.drawdown_episodes(masi, cfg.episodes.drawdown_threshold)
    span_days = (topics["date"].max() - topics["date"].min()).days + 1
    rate = topics.groupby("topic")["n_headlines"].sum() / span_days
    names, counts, ratios = [], [], []
    for e in eps.itertuples(index=False):
        days = (e.trough - e.peak).days + 1
        window = topics[(topics["date"] >= e.peak) & (topics["date"] <= e.trough)]
        count = window.groupby("topic")["n_headlines"].sum().reindex(rate.index, fill_value=0)
        ratio = count / (rate * days)
        ok = ratio[(count >= cfg.episodes.min_headlines) & (ratio >= 3)]
        top = ok.idxmax() if len(ok) else None
        names.append(top)
        counts.append(int(count[top]) if top else 0)
        ratios.append(round(float(ratio[top]), 1) if top else None)
    eps["topic"], eps["topic_headlines"], eps["topic_excess_ratio"] = names, counts, ratios
    return eps


def expost_thresholds(index: pd.Series, cfg: Config) -> tuple[float, float]:
    """Full-sample percentiles: for describing history only (they use future data)."""
    return (float(np.percentile(index, cfg.bands.elevated_percentile)),
            float(np.percentile(index, cfg.bands.stress_percentile)))


def key_numbers(cfg: Config, d: dict) -> dict:
    df, t, scores = d["index"], d["targets"], d["scores"]
    idx = df["stress_index"]
    comps = cfg.composite.components
    p75, p95 = expost_thresholds(idx, cfg)
    eps = named_episodes(d["masi"], d["topics"], cfg)
    ep_tab = val.episode_table(idx, eps).assign(
        topic=eps["topic"].values, topic_headlines=eps["topic_headlines"].values,
        topic_excess_ratio=eps["topic_excess_ratio"].values)
    buckets = val.bucket_table(idx, t[TARGETS])
    corr = val.correlation_table(idx, t[TARGETS])
    regimes = val.regime_table(df["regime_realtime"], t)
    peaks_by_year = {str(y): {"date": str(g.idxmax().date()), "value": round(float(g.max()), 1)}
                     for y, g in idx.groupby(idx.index.year)}

    def r(x, n=3):
        return None if pd.isna(x) else round(float(x), n)

    return {
        "data": {
            "n_articles": len(scores),
            "n_sources": int(scores["source"].nunique()),
            "articles_by_source": {k: int(v) for k, v in scores["source"].value_counts().sort_index().items()},
            "articles_first": str(scores["date"].min().date()),
            "articles_last": str(scores["date"].max().date()),
            "sentiment_first": str(d["sentiment"].index.min().date()),
            **d["market_meta"],
        },
        "index": {
            "first": str(idx.index.min().date()), "last": str(idx.index.max().date()),
            "n_days": len(idx),
            "max": r(idx.max(), 1), "max_date": str(idx.idxmax().date()),
            "min": r(idx.min(), 1), "min_date": str(idx.idxmin().date()),
            "median": r(idx.median(), 1), "mean": r(idx.mean(), 1),
            "last_value": r(idx.iloc[-1], 1),
            "last_regime": df["regime_realtime"].iloc[-1],
            "peak_by_year": peaks_by_year,
            "expost_p75": r(p75, 1), "expost_p95": r(p95, 1),
            "expost_stress_days": int((idx >= p95).sum()),
            "realtime_first": str(df["regime_realtime"].first_valid_index().date()),
            "realtime_regime_days": {k: int(v) for k, v in df["regime_realtime"].value_counts().items()},
            "mean_contribution_share": {c: r(df[f"contrib_{c}"].sum() / idx.sum(), 3) for c in comps},
        },
        "validation": {
            "correlations": {k: {"n_days": int(v.n_days), "pearson": r(v.pearson), "spearman": r(v.spearman)}
                             for k, v in corr.iterrows()},
            "quintiles": {k: {"n_days": int(v.n_days), "index_mean": r(v["index"], 1),
                              "fwd_realised_vol_pct": r(100 * v.fwd_realised_vol, 1),
                              "fwd_max_drawdown_pct": r(100 * v.fwd_max_drawdown, 2),
                              "realised_vol_pct": r(100 * v.realised_vol, 1)}
                          for k, v in buckets.iterrows()},
            "partial_spearman_fwd_vol_given_trailing_vol": {
                name: r(val.partial_spearman(s, t["fwd_realised_vol"], t["realised_vol"]))
                for name, s in [("stress_index", idx), *[(c, df[c]) for c in comps]]},
            "partial_spearman_fwd_max_drawdown_given_trailing_vol": {
                name: r(val.partial_spearman(s, t["fwd_max_drawdown"], t["realised_vol"]))
                for name, s in [("stress_index", idx), *[(c, df[c]) for c in comps]]},
            "component_spearman_fwd_vol": {c: r(val.spearman(df[c], t["fwd_realised_vol"])) for c in comps},
            "realtime_regimes": {k: {"n_days": int(v.n_days), "fwd_realised_vol_pct": r(100 * v.fwd_realised_vol, 1),
                                     "fwd_max_drawdown_pct": r(100 * v.fwd_max_drawdown, 2)}
                                 for k, v in regimes.iterrows()},
            "episodes": json.loads(ep_tab.to_json(orient="records", date_format="iso")),
        },
    }


def make_figures(cfg: Config, d: dict, k: dict) -> list[Path]:
    fig_dir = cfg.paths["figures_dir"]
    df, idx = d["index"], d["index"]["stress_index"]
    first, last = k["index"]["first"][:4], k["index"]["last"][:4]
    p95 = k["index"]["expost_p95"]
    comps = cfg.composite.components
    paths = [
        plotting.plot_thumbnail(
            idx, d["masi"], idx >= p95,
            title="Morocco Market Stress Index",
            subtitle=f"Daily, {first}-{last}. News sentiment + market signals. "
                     f"Shaded: top 5% most stressed days (index >= {p95:.0f}).",
            masi_label="MASI (rebased to 100)", path=fig_dir / "thumbnail.png"),
        plotting.plot_method_diagram(k["data"]["n_articles"], k["data"]["n_stocks"],
                                     fig_dir / "method_diagram.png"),
        plotting.plot_components(
            df[[f"contrib_{c}" for c in comps]].rename(columns=lambda c: c.removeprefix("contrib_")),
            title="What drives the index",
            subtitle="Monthly average contribution of each component, in index points "
                     "(contributions add up to the index)",
            path=fig_dir / "components.png"),
        plotting.plot_validation(
            idx, d["targets"], val.bucket_table(idx, d["targets"][TARGETS]),
            masi_drawdown=d["masi"] / d["masi"].cummax() - 1,
            title="Does the index track market stress?",
            subtitle="Higher index readings are followed by more volatile markets; "
                     "the link with future losses is weaker",
            path=fig_dir / "validation.png"),
    ]
    return paths


def main() -> None:
    cfg = load_config()
    d = load_all(cfg)
    k = key_numbers(cfg, d)
    cfg.paths["key_numbers"].parent.mkdir(parents=True, exist_ok=True)
    cfg.paths["key_numbers"].write_text(json.dumps(k, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {cfg.paths['key_numbers']}")
    for p in make_figures(cfg, d, k):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
