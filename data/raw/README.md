# Raw data (not published)

This folder is git-ignored except for this note. It holds inputs that cannot be
redistributed:

| Path | Content | How to obtain |
|---|---|---|
| `articles/source_*.csv` | Press articles (`date, title, text, section, url`) | `python -m stress_index collect` with your own `sources.local.toml` |
| `market/casablanca_prices.xlsx` | Daily closes. Sheet `cours`: first column = date, one column per listed stock, plus `MASI`. Sheet `benchmarks`: `date, MASI, MBI` | Public end-of-day data of the Casablanca Stock Exchange (MASI equity index, MBI bond index, listed stocks) |
| `market/brent.csv` | Brent crude daily prices: `Date` (MM/DD/YYYY), `Price` | Any public historical-price download |

Article text is copyrighted, so only the derived per-article sentiment scores are
published (`data/processed/article_scores.csv`). From there, the sentiment and the
composite index can be rebuilt without any raw file.
