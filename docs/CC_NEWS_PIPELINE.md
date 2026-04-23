# CC-News Pipeline (v2.2 scaffolding)

**Status**: scaffolded 2026-04-23. Scripts shipped; no full bulk download yet.
**Sibling**: [`V2_2_ARCHITECTURE.md`](V2_2_ARCHITECTURE.md) Section 3 (GDELT DOC).

## Why a second bulk source

GDELT DOC is the v2.2 primary article channel, but it indexes metadata
plus snippets and not raw body text. Common Crawl News (CC-News) is the
complement: it is the canonical public archive of raw HTML for news
articles worldwide, published as monthly WARC shards under
`s3://commoncrawl/crawl-data/CC-NEWS/{YYYY}/{MM}/` and mirrored on HTTPS
at `https://data.commoncrawl.org/`.

The combined picture for v2.2:

| Source | Content | Resolution | Use case |
|---|---|---|---|
| GDELT DOC index | URL + metadata + tone + themes | Global, dense, multilingual | Primary retrieval; fast |
| CC-News archive | Full HTML body | English editorial (post-whitelist) | Body text for survivors; offline |

CC-News is never used for v2.1 deliverables. It pays back on the NEXT
rebuild where we want bodies without per-URL trafilatura fetches.

## Storage and cost estimates

A monthly CC-News release is ~2-5 TB of raw WARC across 300-600 shards
of 3-5 GB each. The domain whitelist filter typically keeps < 1% of
records; trafilatura-extracted bodies compress ~5x under zstd level 10.

For a 90-day window (2026-01..2026-03):

- **Raw network transfer**: ~15 TB if every shard is streamed end to end.
  In practice we can stop each shard early when the record rate drops
  (CC-News within a shard is roughly time-ordered, but not tightly).
- **Kept records**: ~100k-300k articles depending on how broad the
  whitelist is.
- **On-disk footprint** (zstd JSONL): ~2-6 GB total across all three
  months.
- **Wall clock**: download-bound; ~40-60 min per shard at 50 MB/s; ~24-48
  hours for a 90-day full pull on a home connection. Use `--max-shards`
  to cap.

## Scripts

### 1. Download and filter

    python scripts/fetch_cc_news_archive.py \
        --start 2026-01 --end 2026-03 \
        --out data/cc_news/

Resume-safe: each shard gets a `.jsonl.zst.done` sidecar with stats;
subsequent runs skip completed shards. Each shard is capped at 5 GB
(abort otherwise, configurable via `--max-shard-bytes`). Sample-only
runs can pass `--max-shards 1` to prove the plumbing.

### 2. Build the FAISS index

    python scripts/build_cc_news_index.py \
        --in data/cc_news/ --out data/cc_news/index/ \
        --embedder openai

Defaults to the v2.2 OpenAI Batch backend (see
[`V2_2_ARCHITECTURE.md`](V2_2_ARCHITECTURE.md) Section 6). The SBERT
backend is available via `--embedder sbert` for offline builds.

### 3. Query per-FD

    python scripts/query_cc_news_index.py \
        --index data/cc_news/index/ \
        --fds data/unified/forecasts.jsonl \
        --top-k 10 \
        --out data/cc_news/enrichments.jsonl

Applies a date-window prefilter (`--date-window-days`) that matches the
v2.1 `lookback_days` semantics. `--host-filter` restricts to a comma
separated host suffix allowlist if the caller wants a tighter set than
`src/common/cc_news_domains.py`.

## Domain whitelist

`src/common/cc_news_domains.py` ships ~50 outlets covering US editorial,
business/markets, wire services, UK/Commonwealth, and pan-regional
sources. Extra domains can be layered in via
`--domain-whitelist path/to/extras.txt` (one domain per line, `#` for
comments). Host matching is parent-domain aware: `edition.cnn.com`
matches a `cnn.com` whitelist entry.

## Trade-offs versus GDELT DOC

1. **Coverage**: GDELT DOC is global and multilingual; CC-News plus our
   whitelist is English-editorial-only. CC-News is not a replacement,
   only a complement.
2. **Freshness**: GDELT DOC updates every 15 minutes; CC-News shards are
   assembled over a month and published in chunks. For a build cycle
   running near the cutoff, GDELT DOC is the only option.
3. **Body text**: GDELT DOC offers snippets only (~250 chars); CC-News
   has full body. For the ETD Stage-1 fact extraction pipeline this
   matters.
4. **Cost profile**: GDELT DOC raw download is ~3 minutes at ~300 MB;
   CC-News 90-day window is ~24-48 hours at ~15 TB pre-filter. CC-News
   is emphatically a one-time cost per cutoff window.
5. **Dedup**: GDELT DOC already deduplicates at ingest; CC-News requires
   URL canonicalization and dedup at our layer. `src/common/spam_domains.py`
   applies but CC-News can surface same-story reprints across outlets.

## Open questions

- Do we want to dedup CC-News against GDELT DOC before FAISS encoding?
  Probably yes, but URL canonicalization across both sources needs more
  thought.
- Should the `--embedder openai` default here match the one in
  `compute_relevance.py`? The v2.2 r4 decision says yes (§6); the index
  manifest records the choice so the query script can match at runtime.
- How aggressive should the per-shard early-exit be? Today we download
  every shard fully; a date-based heuristic could trim ~40%.
