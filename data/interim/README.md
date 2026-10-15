# Interim data (not published)

`article_score_cache.csv` stores one sentiment score per article, keyed by URL and a
hash of the text, so that only new or edited articles are re-scored. It is
git-ignored because the URLs identify the sources; its anonymised version is
`data/processed/article_scores.csv`.
