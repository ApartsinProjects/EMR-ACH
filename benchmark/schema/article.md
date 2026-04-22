# Article Schema

One row in `data/{cutoff}/articles.jsonl`. The shipped `articles.jsonl` contains only the subset of the unified article pool referenced by at least one accepted FD; the full pool is regenerable by rerunning `build.py`.

Rows are produced by the per-track fetchers (`scripts/gdelt_cameo/fetch_text.py`, `scripts/forecastbench/fetch_article_text.py`, optionally rescued by `scripts/common/retry_article_text.py`) and merged by `scripts/common/unify_articles.py` using `sha1(url)[:12]` as the dedup key.

## Shipped fields

| Field | Type | Description |
|---|---|---|
| `id` | string | `art_` + first 12 hex chars of `sha1(url)`. Referenced by `forecast.article_ids`. |
| `url` | string | Canonical URL. Primary dedup key. |
| `title` | string | Article headline. May be empty if trafilatura extraction failed. |
| `text` | string | Full article body as extracted by trafilatura. Empty string on extraction failure (such articles are typically dropped by the quality filter's `min_chars` rule). |
| `title_text` | string | Convenience field for embedding models: `title + "\n" + text`. |
| `publish_date` | string (YYYY-MM-DD) | Publication date. Sourced from GDELT `DATE`, trafilatura metadata, or HTTP `Last-Modified`. Guaranteed present. |
| `source_domain` | string | Lowercased netloc with leading `www.` stripped. |
| `char_count` | int | `len(text)`. Used by the quality filter's minimum-evidence-characters guard. |
| `provenance` | list[string] | Tags recording which pipeline(s) surfaced the row. Elements drawn from `forecastbench`, `gdelt_cameo`, `earnings`. A single article can carry multiple tags after dedup. |

## Example

```json
{
  "id": "art_1e7cb2b2f8c8",
  "url": "https://www.reuters.com/sports/olympics/paris-2024-opening-ceremony-2024-07-26/",
  "title": "Paris 2024 opens with rain-soaked river parade",
  "text": "PARIS, July 26 (Reuters) - The Paris 2024 Olympic Games opened ...",
  "title_text": "Paris 2024 opens with rain-soaked river parade\nPARIS, July 26 (Reuters) - The Paris 2024 Olympic Games opened ...",
  "publish_date": "2024-07-26",
  "source_domain": "reuters.com",
  "char_count": 4821,
  "provenance": ["forecastbench"]
}
```

## Notes

- Articles whose `text` is below the per-source minimum-character threshold are usually dropped from downstream FDs by the quality filter's `min_chars` rule (aggregated across all articles of a given FD), even though the row itself may still appear in the pool.
- The retrieval-leakage condition `article.publish_date >= forecast.forecast_point` is enforced by the quality filter: violating articles are removed from the offending FD's `article_ids` rather than being kept in a separate file. The count of pruned articles per build is recorded in `meta/quality_meta.json` as `leakage_articles_pruned`.
- `publish_date` source precedence: trafilatura-extracted metadata.date → GDELT DATE column → HTTP Last-Modified. When the article is a GDELT-CAMEO oracle article, this date is the GDELT *event* date, which coincides with the publication date by construction of the same-date filter in the KG builder, but is not guaranteed to equal the true article publish date reported by the publisher.
