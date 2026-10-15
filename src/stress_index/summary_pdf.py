"""Two-page plain-language PDF summary, built from reports/key_numbers.json and figures/.

    python -m stress_index.summary_pdf     # -> reports/stress_index_summary.pdf
"""

from __future__ import annotations

import json
from datetime import date

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from .config import load_config

INK = colors.HexColor("#1b1f24")
MUTED = colors.HexColor("#6b7280")
ACCENT = colors.HexColor("#0b3c5d")
PAGE_W = A4[0] - 3.2 * cm

STYLES = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=10, leading=13, textColor=MUTED),
    "h": ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=ACCENT,
                        spaceBefore=7, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="Helvetica", fontSize=9.3, leading=12.4, textColor=INK,
                        alignment=TA_LEFT),
    "b": ParagraphStyle("b", fontName="Helvetica", fontSize=9.3, leading=12.2, textColor=INK,
                        leftIndent=10, bulletIndent=0),
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8, leading=10, textColor=MUTED),
}


def _fmt_date(iso: str) -> str:
    return date.fromisoformat(iso[:10]).strftime("%d %b %Y").lstrip("0")


def _img(path, width: float) -> Image:
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def _bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(t, STYLES["b"], bulletText="•") for t in items]


def build(out_path=None) -> str:
    cfg = load_config()
    k = json.loads(cfg.paths["key_numbers"].read_text("utf-8"))
    d, ix, v = k["data"], k["index"], k["validation"]
    fig = cfg.paths["figures_dir"]
    q, corr, reg = v["quintiles"], v["correlations"], v["realtime_regimes"]
    part = v["partial_spearman_fwd_vol_given_trailing_vol"]
    named = [e for e in v["episodes"] if e["topic"]]
    out_path = out_path or cfg.paths["reports_dir"] / "stress_index_summary.pdf"
    S = STYLES

    story = [
        Paragraph("Morocco Market Stress Index", S["title"]),
        Paragraph(f"A daily 0-100 gauge of stress on the Casablanca Stock Exchange, built from financial "
                  f"news and market data &nbsp;|&nbsp; {_fmt_date(ix['first'])} to {_fmt_date(ix['last'])}",
                  S["sub"]),
        Spacer(1, 4),
        Paragraph("The problem", S["h"]),
        Paragraph("Investors in Moroccan equities have no widely used daily stress gauge comparable to the "
                  "VIX for US stocks. Market signals such as volatility react only once prices have moved, "
                  "while the news flow often turns first. The goal is one daily number that rises when "
                  "several sources of stress flare up at the same time, built only from information "
                  "available on each day, so that it can feed allocation and risk decisions.", S["p"]),
        Paragraph("The method", S["h"]),
        *_bullets([
            f"<b>News tone.</b> {d['n_articles']:,} articles from {d['n_sources']} Moroccan financial news "
            "websites are scored by a French language model, from reassuring to alarming. The daily press "
            "tone averages the last 90 days of articles, recent ones counting more.",
            f"<b>Market signals.</b> MASI trend, volatility of equities (MASI), bonds (MBI) and oil (Brent), "
            f"and the share of the {d['n_stocks']} listed stocks above their recent average.",
            "<b>One scale.</b> Each of the six signals is ranked against its own past year (0 = calmest, "
            "1 = most stressed).",
            "<b>Systemic aggregation.</b> The CISS formula of the European Central Bank combines the six "
            "ranks and their correlations: stress that hits several channels at once counts much more "
            "than stress in one channel. The result is a fixed 0-100 scale; past values are never revised.",
            "<b>No look-ahead.</b> Every step uses only past data; an automated test checks that adding "
            "new data never changes earlier values.",
        ]),
        Spacer(1, 4),
        _img(fig / "method_diagram.png", PAGE_W * 0.92),
        Spacer(1, 2),
        _img(fig / "thumbnail.png", PAGE_W * 0.86),
        Paragraph("Shading: the 5% most stressed days of the sample (descriptive, uses the full sample).",
                  S["cap"]),
        PageBreak(),
        Paragraph("Results", S["h"]),
        *_bullets([
            "<b>It flags the market's crises.</b> " + "; ".join(
                f"{e['topic']}: MASI {e['masi_depth_pct']}% from {_fmt_date(e['peak'])} to "
                f"{_fmt_date(e['trough'])}, index peak {max(e['index_max'], e['index_max_after_trough'])}"
                for e in named) + f". Highest reading: {ix['max']} on {_fmt_date(ix['max_date'])}; "
            f"median {ix['median']}.",
            f"<b>It separates calm from turbulent markets.</b> In the calmest fifth of days, MASI volatility "
            f"over the next 20 trading days averaged {q['Q1']['fwd_realised_vol_pct']}% (annualised), "
            f"against {q['Q5']['fwd_realised_vol_pct']}% in the most stressed fifth. Correlation with "
            f"current volatility: {corr['realised_vol']['pearson']:.2f}.",
            f"<b>Real-time labels work in the expected direction.</b> Days labelled stress were followed "
            f"by {reg['stress']['fwd_realised_vol_pct']}% volatility, calm days by "
            f"{reg['calm']['fwd_realised_vol_pct']}% (only {reg['stress']['n_days']} stress days).",
            f"<b>News adds information that prices do not yet show.</b> Once current volatility is taken "
            f"into account, the composite keeps little forward information (partial rank correlation "
            f"{part['stress_index']:.2f}), but the press-sentiment component keeps {part['news_sentiment']:.2f}.",
        ]),
        Spacer(1, 4),
        _img(fig / "components.png", PAGE_W * 0.80),
        _img(fig / "validation.png", PAGE_W * 0.86),
        KeepTogether([
            Paragraph("Limits", S["h"]),
            *_bullets([
                f"<b>Not a crash predictor.</b> The link with losses over the next 20 days is weak (rank "
                f"correlation {corr['fwd_max_drawdown']['spearman']:.2f}), and the index peaked after the "
                "MASI trough in the sharpest crises: it confirms stress more than it anticipates it.",
                f"<b>In-sample.</b> Parameters were set on the validation sample. Real-time labels start "
                f"on {_fmt_date(ix['realtime_first'])}, after the COVID-19 shock.",
                "<b>Timing.</b> Day t uses that day's closes and articles; use it from day t+1.",
                f"<b>Data.</b> Two news sources and a general-purpose language model; breadth uses the "
                f"stocks listed today (no delisted firms); days with a missing input are dropped "
                f"({d['returns_spanning_two_or_more_days']} returns span two trading days).",
            ]),
        ]),
    ]

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.2 * cm,
                            title="Morocco Market Stress Index", author="", subject="Two-page summary")
    doc.build(story)
    return str(out_path)


if __name__ == "__main__":
    print("wrote", build())
