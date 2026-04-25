# v2.2 CC-News Index Build Log

## 2025-12 SQLite index (B6.1B-1)

Built by `scripts/build_cc_news_sqlite.py` on the 616-shard CC-News
fetch at `data/cc_news/2025-12/`.

```json
{
  "build_timestamp": "2026-04-25T05:10:22+00:00",
  "in_dir": "data/cc_news/2025-12",
  "out_path": "data/cc_news/index_2025-12.sqlite",
  "shards_total": 616,
  "shards_skipped_no_done": 0,
  "rows_inserted": 108861,
  "rows_with_publish_date": 108861,
  "distinct_hosts": 10,
  "duplicate_urls_ignored": 0,
  "duration_s": 29.42
}
```

### Host distribution

| host | rows |
| --- | ---: |
| timesofindia.indiatimes.com | 60,118 |
| www.hindustantimes.com | 22,779 |
| www.independent.co.uk | 12,595 |
| www.straitstimes.com | 5,904 |
| www.aljazeera.com | 3,600 |
| www.latimes.com | 2,927 |
| www.fool.com | 589 |
| auto.hindustantimes.com | 330 |
| tech.hindustantimes.com | 16 |
| www.pbs.org | 3 |

### Date coverage

Full month: `2025-12-01` through `2025-12-31`. Daily counts range from
2,325 (2025-12-28) to 10,638 (2025-12-02), median ~3,400/day.

### Schema

`articles(rowid, url UNIQUE, host, publish_date, publish_ts, title,
text, shard, meta_date)` plus a content-table FTS5 virtual table
`articles_fts(title, text)` with porter+unicode61 tokenizer. B-tree
indices on `publish_date` and `host`.

`text` is truncated to 8,000 chars at index time (FB question retrieval
matches mostly on lede + first paragraphs; truncation keeps SQLite at
~250 MB).

### Reproduction

```
/c/Python314/python scripts/build_cc_news_sqlite.py \
    --in data/cc_news/2025-12/ \
    --out data/cc_news/index_2025-12.sqlite
```

The output `.sqlite` is gitignored (per `.gitignore` `data/*.sqlite`
rule). The shard root `data/cc_news/2025-12/` is now also gitignored
(per the new `data/cc_news/` rule added in this commit) — re-fetch via
`scripts/fetch_cc_news_archive.py` if needed.
