"""Generic, incremental collector for news websites with a paginated article listing.

Each site is described by a `SiteConfig` (private `sources.local.toml`): listing URLs,
how to find article links and their dates on a listing page, and where the title,
date and body sit on an article page. The collector:

1. finds the first listing page inside the requested date window (optionally by
   binary search on reverse-chronological listings);
2. walks listing pages until it passes the start of the window;
3. downloads only articles that are inside the window and not already saved;
4. appends them to the source's CSV (`date, title, text, section, url`), so an
   interrupted run resumes where it stopped.

Raw article text is kept locally for scoring and is never published.
"""

from __future__ import annotations

import csv
import html as html_lib
import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import SiteConfig

ARTICLE_COLUMNS = ["date", "title", "text", "section", "url"]

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

FRENCH_MONTHS = {
    "janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "aout": 8, "août": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "decembre": 12, "décembre": 12,
}

# Days of overlap when resuming (articles published late), and how recent a
# section's last article must be for the section to count when choosing --since.
RESUME_OVERLAP_DAYS = 3
ACTIVE_SECTION_DAYS = 90
DEFAULT_SINCE = "2020-01-01"


# ----------------------------------------------------------------------------- HTTP
def make_session() -> requests.Session:
    """HTTP session with retries on throttling and server errors."""
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "fr-FR,fr;q=0.9"})
    retry = Retry(total=5, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def fetch(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


# ----------------------------------------------------------------------------- dates
def parse_french_date(txt: str | None) -> date | None:
    """'Mardi 07 Juillet 2026' -> date(2026, 7, 7); None when not recognised."""
    if not txt:
        return None
    m = re.search(r"(\d{1,2})\s+([a-zàâçéèêëîïôûùü]+)\s+(\d{4})", txt.strip().lower())
    if not m or m.group(2) not in FRENCH_MONTHS:
        return None
    try:
        return date(int(m.group(3)), FRENCH_MONTHS[m.group(2)], int(m.group(1)))
    except ValueError:
        return None


def parse_iso_date(txt: str | None) -> date | None:
    if not txt:
        return None
    try:
        return datetime.fromisoformat(txt.replace("Z", "+00:00")).date()
    except ValueError:
        return None


# ----------------------------------------------------------------------------- listing pages
def listing_url(site: SiteConfig, page: int, section: str | None) -> str:
    tpl = site.first_page_url if page == 1 else site.page_url
    return tpl.format(page=page, section=section or "")


def extract_listing(site: SiteConfig, html: str) -> list[tuple[str, date | None]]:
    """Article links of one listing page with the date shown for each."""
    soup = BeautifulSoup(html, "lxml")
    out: list[tuple[str, date | None]] = []
    seen: set[str] = set()

    if site.link_mode == "url_regex":
        pattern = re.compile(site.article_url_regex)
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0].split("?")[0]
            m = pattern.match(href)
            if m and href not in seen:
                seen.add(href)
                out.append((href, date(int(m.group(1)), int(m.group(2)), int(m.group(3)))))
    elif site.link_mode == "item_blocks":
        for block in soup.select(site.item_selector):
            a = block.find("a", href=True)
            if not a or not a["href"].startswith(site.link_prefix):
                continue
            url = site.base_url + a["href"].split("?")[0].split("#")[0]
            if url not in seen:
                seen.add(url)
                out.append((url, parse_french_date(block.get_text(" ", strip=True))))
    else:
        raise ValueError(f"unknown link_mode: {site.link_mode}")
    return out


def newest_date_on_page(session, site: SiteConfig, page: int, section: str | None) -> date | None:
    try:
        listing = extract_listing(site, fetch(session, listing_url(site, page, section)))
    except requests.HTTPError:
        return None
    dates = [d for _, d in listing if d]
    return max(dates) if dates else None


def find_start_page(session, site: SiteConfig, section: str | None, until: date) -> int:
    """First listing page whose newest article is on or before `until` (binary search)."""
    if site.start_page != "bisect":
        return 1
    first = newest_date_on_page(session, site, 1, section)
    if first is None or first <= until:
        return 1
    lo, hi = 1, 2
    while hi <= site.max_page:                       # exponential search for an upper bound
        d = newest_date_on_page(session, site, hi, section)
        if d is None or d <= until:
            break
        lo, hi = hi, hi * 2
        time.sleep(site.delay)
    hi = min(hi, site.max_page)
    while lo + 1 < hi:                               # bisection between lo (too new) and hi
        mid = (lo + hi) // 2
        d = newest_date_on_page(session, site, mid, section)
        time.sleep(site.delay)
        if d is None or d <= until:
            hi = mid
        else:
            lo = mid
    return hi


# ----------------------------------------------------------------------------- article pages
def _iter_jsonld(raw: str) -> Iterator[dict]:
    try:
        data = json.loads(raw, strict=False)
    except Exception:
        return
    stack = [data]
    while stack:
        cur = stack.pop()
        if isinstance(cur, list):
            stack.extend(cur)
        elif isinstance(cur, dict):
            if "@graph" in cur:
                stack.extend(cur["@graph"])
            yield cur


def _jsonld_fields(soup: BeautifulSoup) -> tuple[str, str]:
    """(headline, datePublished) from JSON-LD blocks, with a regex fallback for broken JSON."""
    headline, published = "", ""
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text()
        if not raw:
            continue
        for obj in _iter_jsonld(raw):
            if obj.get("@type") in ("NewsArticle", "Article", "WebPage"):
                headline = headline or obj.get("headline", "")
                published = published or obj.get("datePublished", "")
        if not published and (m := re.search(r'"datePublished"\s*:\s*"([^"]+)"', raw)):
            published = m.group(1)
        if not headline and (m := re.search(r'"headline"\s*:\s*"([^"]+)"', raw)):
            headline = m.group(1)
    return headline, published


def parse_article(site: SiteConfig, html: str) -> tuple[date | None, str, str]:
    """(publication date or None, title, body text) of one article page.

    `site.title_from` / `site.date_from` list strategies tried in order:
    titles: "jsonld", "title" (<title>), "h1"; dates: "jsonld_french",
    "meta_iso:<property>".
    """
    soup = BeautifulSoup(html, "lxml")
    headline, jsonld_date = _jsonld_fields(soup) if (
        "jsonld" in site.title_from or "jsonld_french" in site.date_from) else ("", "")

    title = ""
    for how in site.title_from:
        if title:
            break
        if how == "jsonld" and headline:
            title = html_lib.unescape(headline)
        elif how == "title" and soup.title:
            title = soup.title.get_text(" ", strip=True)
        elif how == "h1" and (h1 := soup.find("h1")):
            title = h1.get_text(" ", strip=True)

    published = None
    for how in site.date_from:
        if published:
            break
        if how == "jsonld_french":
            published = parse_french_date(jsonld_date)
        elif how.startswith("meta_iso:"):
            meta = soup.find("meta", attrs={"property": how.split(":", 1)[1]})
            published = parse_iso_date(meta.get("content") if meta else None)

    text = ""
    container = soup.select_one(site.body_selector) if site.body_selector else None
    if container:
        for t in container(["script", "style"]):
            t.decompose()
        paras = [p.get_text(" ", strip=True) for p in container.find_all(site.body_tags)]
        paras = [p for p in paras if p]
        text = "\n".join(paras) if paras else container.get_text("\n", strip=True)
    return published, title, text


# ----------------------------------------------------------------------------- incremental run
def saved_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["url"] for row in csv.DictReader(f)}


def resume_since(path: Path) -> str:
    """Start date for an incremental run: last saved date minus a small overlap.

    With several sections, use the *minimum* of the per-section last dates (over
    sections active in the last ACTIVE_SECTION_DAYS days), so a section that lags
    behind is not cut short. Duplicates from the overlap are skipped by URL.
    """
    try:
        d = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: c in ("date", "section"))
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError):
        return DEFAULT_SINCE
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if "section" in d.columns and d["section"].notna().any():
        last = d.groupby("section")["date"].max()
        active = last[last >= last.max() - pd.Timedelta(days=ACTIVE_SECTION_DAYS)]
        newest = active.min() if len(active) else last.max()
    else:
        newest = d["date"].max()
    if pd.isna(newest):
        return DEFAULT_SINCE
    return (newest.date() - timedelta(days=RESUME_OVERLAP_DAYS)).isoformat()


def collect_site(site: SiteConfig, since: date, until: date) -> int:
    """Download articles of one site published in [since, until]. Returns the number added."""
    session = make_session()
    done = saved_urls(site.output) | set().union(*(saved_urls(p) for p in site.history_files))
    site.output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not site.output.exists()
    added = 0
    with open(site.output, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ARTICLE_COLUMNS)
        if new_file:
            writer.writeheader()
        for section in (site.sections or [None]):
            print(f"[{site.label}] section {section or '-'}: {since} -> {until}")
            page, empty = find_start_page(session, site, section, until), 0
            while True:
                try:
                    listing = extract_listing(site, fetch(session, listing_url(site, page, section)))
                except requests.HTTPError:
                    break
                if not listing:
                    empty += 1
                    if empty >= site.max_empty_pages:
                        break
                    page += 1
                    continue
                empty = 0
                dates = [d for _, d in listing if d]
                for url, listed in listing:
                    if not listed or not (since <= listed <= until) or url in done:
                        continue
                    try:
                        published, title, text = parse_article(site, fetch(session, url))
                    except Exception as ex:          # one bad page must not stop the run
                        print(f"  [error] {url}: {ex}")
                        time.sleep(site.delay)
                        continue
                    published = published or listed
                    writer.writerow({"date": published.isoformat(), "title": title, "text": text,
                                     "section": section or "", "url": url})
                    f.flush()
                    done.add(url)
                    added += 1
                    print(f"  + {published}  {title[:60]}")
                    time.sleep(site.delay)
                if dates and min(dates) < since:     # reverse-chronological: window passed
                    break
                page += 1
                time.sleep(site.delay)
    print(f"[{site.label}] {added} new articles")
    return added


def collect_all(sites: list[SiteConfig], since: str | None = None, until: str | None = None) -> None:
    """Incremental update of every configured site (since = resume point unless given)."""
    end = date.fromisoformat(until) if until else date.today()
    for site in sites:
        start = date.fromisoformat(since or resume_since(site.output))
        collect_site(site, start, end)
