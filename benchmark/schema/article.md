# Article Schema

One row in `data/{cutoff}/articles.jsonl`. The shipped `articles.jsonl` contains only the subset of the unified article pool referenced by at least one accepted FD; the full pool is regenerable by rerunning `scripts/build_benchmark.py`.

This page is the per-benchmark facing schema reference. The machine-readable spec lives at [`docs/article.schema.json`](../../docs/article.schema.json); cross-reference [`docs/FORECAST_DOSSIER.md`](../../docs/FORECAST_DOSSIER.md) for how `article_ids` resolves into FDs.

Rows are produced by the per-track fetchers (`scripts/gdelt_cameo/fetch_text.py`, `scripts/forecastbench/fetch_article_text.py`, the Earnings 5-source cascade including SEC EDGAR 8-K, optionally rescued by `scripts/common/retry_article_text.py`) and merged by `scripts/common/unify_articles.py` using `sha1(url)[:12]` as the dedup key.

## Shipped fields

| Field | Type | Description |
|---|---|---|
| `id` | string | `art_` + first 12 hex chars of `sha1(url)`. Referenced by `forecast.article_ids`. |
| `url` | string | Canonical URL. Primary dedup key. |
| `title` | string | Article headline. May be empty if trafilatura extraction failed. |
| `text` | string | Full article body as extracted by trafilatura (or by SEC EDGAR / Finnhub parsers for non-HTML sources). Empty string on extraction failure (such articles are typically dropped by the quality filter's `min_chars` rule). |
| `title_text` | string | Convenience field for embedding models: `title + "\n" + text`. |
| `publish_date` | string (YYYY-MM-DD) | Publication date. Sourced from GDELT `DATE`, trafilatura metadata, SEC EDGAR filing date, or HTTP `Last-Modified`. Guaranteed present. |
| `source_domain` | string | Lowercased netloc with leading `www.` stripped. |
| `char_count` | int | `len(text)`. Used by the quality filter's minimum-evidence-characters guard. |
| `source_type` | enum string | One of `news`, `filing`, `social`, `blog`, `other`. Drives downstream retrieval logic; in particular SEC EDGAR 8-K rows carry `filing` and are subject to an asymmetric pre-event filter in the Earnings track. ForecastBench and GDELT-CAMEO editorial-news rows carry `news`. |
| `language` | string | ISO 639-1 code (default `en`). Used by language-aware downstream filters. |
| `provenance` | string | Pipe-separated multi-source tag string recording which pipeline(s) surfaced the row. Tokens drawn from `forecastbench`, `gdelt_cameo`, `earnings`. A single article surfaced by multiple tracks after dedup carries multiple tokens, e.g. `gdelt_cameo|earnings`. |

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
  "source_type": "news",
  "language": "en",
  "provenance": "forecastbench"
}
```

A second example, an SEC EDGAR 8-K filing surfaced by the Earnings track:

```json
{
  "id": "art_a8c9012f4d61",
  "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240802.htm",
  "title": "Apple Inc. 8-K Current Report (2024-08-02)",
  "text": "Item 2.02 Results of Operations and Financial Condition. ...",
  "title_text": "Apple Inc. 8-K Current Report (2024-08-02)\nItem 2.02 Results of Operations and Financial Condition. ...",
  "publish_date": "2024-08-02",
  "source_domain": "sec.gov",
  "char_count": 12044,
  "source_type": "filing",
  "language": "en",
  "provenance": "earnings"
}
```

## Notes

- Articles whose `text` is below the per-source minimum-character threshold are usually dropped from downstream FDs by the quality filter's `min_chars` rule (aggregated across all articles of a given FD), even though the row itself may still appear in the pool.
- The retrieval-leakage condition `article.publish_date >= forecast.forecast_point` is enforced by the quality filter: violating articles are removed from the offending FD's `article_ids` rather than being kept in a separate file. The count of pruned articles per build is recorded in `meta/quality_meta.json` as `leakage_articles_pruned`. SEC EDGAR 8-K rows (`source_type: "filing"`) additionally enforce an asymmetric pre-event filter in the Earnings retrieval stage.
- `publish_date` source precedence: trafilatura-extracted metadata.date, then GDELT DATE column, then SEC EDGAR filing date (for `source_type: "filing"`), then HTTP `Last-Modified`. When the article is a GDELT-CAMEO oracle article, this date is the GDELT *event* date, which coincides with the publication date by construction of the same-date filter in the KG builder, but is not guaranteed to equal the true article publish date reported by the publisher. Oracle articles are replaced with editorial-news context by `scripts/gdelt_cameo/relink_context.py` before publication.
- `provenance` is a string (pipe-separated), not a list. Older revisions of this schema used `list[string]`; downstream consumers should split on `|` to recover individual source tags.

## Schema invariants enforced in CI

`tests/test_article_schema.py` (8 tests) enforces presence and type of every field above, that `source_type` is in the documented enum, that `publish_date` parses as YYYY-MM-DD, that `id` matches `art_` + 12 hex chars, that `provenance` is a non-empty pipe-separated string, and that `char_count == len(text)`. See also `tests/test_pipeline_invariants.py` (16 tests) for the FD-to-article cross-reference invariants and the machine-readable spec at [`docs/article.schema.json`](../../docs/article.schema.json).
