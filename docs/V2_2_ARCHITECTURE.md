# EMR-ACH Pipeline v2.2 Architecture

**Status**: design proposal, drafted 2026-04-22 from v2.1 build-cycle findings.
**Predecessor**: v2.1 (shipped 2026-04-23). See [`docs/PIPELINE.md`](PIPELINE.md) for the v2.1 stage map and [`docs/FORECAST_DOSSIER.md`](FORECAST_DOSSIER.md) for the FD contract; this document does not duplicate either.
**Scope**: this document is design only; no v2.2 code lives in the tree yet. The v2.1 entrypoint at [`scripts/build_benchmark.py`](../scripts/build_benchmark.py) and its dependents (`compute_relevance.py`, `annotate_prior_state.py`, `configs/default_config.yaml`) remain authoritative until v2.2 lands.

---

## 1. Motivation

v2.1 ships a working three-benchmark, leakage-guarded, audit-hardened pipeline. Today's marathon build cycle (eight build attempts before a clean run) surfaced three structural pain points that v2.2 must close.

### 1.1 News fetch is the dominant wall-clock cost

Per-FD HTTP scraping owns the cold rebuild budget. Concrete numbers from the 2026-04-23 cycle:

- Cold rebuild on the v2.1 1,172-FD ForecastBench pool: 5 to 7 hours at 24 workers.
- Root cause: serial source cascade per FD (GDELT DOC, Google News RSS, Guardian, NYT) with Google News rate-limiting the slowest hop, and trafilatura body fetch single-threaded per URL inside [`scripts/fetch_article_text.py`](../scripts/fetch_article_text.py).
- GDELT-CAMEO and earnings fetchers ([`scripts/fetch_gdelt_cameo_news.py`](../scripts/fetch_gdelt_cameo_news.py), [`scripts/fetch_earnings_news.py`](../scripts/fetch_earnings_news.py)) replicate the same cascade with the same per-host throttling profile.
- The GDELT KG raw download at [`scripts/build_gdelt_cameo.py`](../scripts/build_gdelt_cameo.py) is one-time at roughly 3 minutes; not a hot-path concern.

### 1.2 SBERT scoring repeats work across benchmarks

[`scripts/compute_relevance.py`](../scripts/compute_relevance.py) is invoked once per benchmark in v2.1 (`step_relevance_parallel` at `scripts/build_benchmark.py:276`). Today's `batch_size=256` plus FP16 speedups make the encoder roughly 8 to 16 times faster, but the script still re-encodes the full unified pool on every benchmark pass. With three benchmarks that is three times the encoder work for an article pool that is already physically unified at `data/unified/articles.jsonl`.

The per-row MD5 fingerprint cache at `compute_relevance.py:90` (`_per_row_fingerprints` and `load_or_embed`) prevents re-encoding articles whose text did not change, but it does not prevent re-encoding the same article three times in one build.

### 1.3 Architectural drift recurs

v2.0 to v2.1 collapsed two parallel script trees ([`benchmark/build.py`](../benchmark/build.py) was deprecated to a forwarding shim in commit `d60a4ba`; canonical entrypoint moved to [`scripts/build_benchmark.py`](../scripts/build_benchmark.py)). The drift cost roughly six hours of debugging on 2026-04-23. The same drift pattern is latent in three other places:

- `unify_articles.py` and `unify_forecasts.py` are tightly coupled (same intermediate, same caller, same call site) but live as two top-level scripts.
- The three news fetchers share roughly 70% of their code (URL canonicalization, spam filter, dedup, append loop) but each maintains its own copy.
- Output paths are split across `data/unified/` (staging) and `data/{benchmark}/` (per-benchmark intermediates) with no single owner; some scripts assume one, some the other.

### 1.4 What v2.2 explicitly does not change

To keep the migration tractable, v2.2 keeps the v2.1 FD schema (no changes to `forecasts.jsonl` or `articles.jsonl` records), the `Comply` vs `Surprise` primary target, the three-benchmark partition, the prior-state oracle definitions, and the two-cutoff leakage-probe protocol. v2.2 is a build-system refactor, not a benchmark redesign.

---

## 2. Pipeline shape

```
                         ┌─────────────────────────────────────────────────┐
                         │  STAGE 0: raw build (unchanged from v2.1)        │
                         │  build_gdelt_cameo │ build_earnings │ download_fb│
                         └──────────────────┬───────────────────────────────┘
                                            │ FD seeds per benchmark
                                            ▼
       ┌──────────────────────────────┐
       │  STAGE 1: GDELT DOC ARCHIVE  │  one-time bulk download per cutoff window
       │  fetch_gdelt_doc_archive.py  │  → data/gdelt_doc/raw/{YYYY-MM}.jsonl.zst
       └──────────────┬───────────────┘
                      ▼
       ┌──────────────────────────────┐
       │  STAGE 2: GDELT DOC INDEX    │  encode bodies once with SBERT
       │  build_gdelt_doc_index.py    │  → data/gdelt_doc/index/{shard}.faiss + meta.parquet
       └──────────────┬───────────────┘
                      │
                      │  primary article channel
                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: HYBRID RETRIEVAL (per benchmark)                                  │
│                                                                             │
│   ForecastBench   →  GDELT DOC index lookup  →  Google News residual        │
│   GDELT-CAMEO     →  GDELT DOC index lookup  (editorial-only filter)        │
│   Earnings        →  SEC EDGAR + Finnhub     →  GDELT DOC index residual    │
│                                                                             │
│   query_gdelt_doc_index.py   (replaces ~70% of per-FD HTTP scraping)        │
│   fetch_*_news.py (residual) (kept as fallback for niche FDs)               │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
       ┌──────────────────────────────────────────┐
       │  STAGE 4: UNIFY (one package, two faces) │
       │  unify.articles + unify.forecasts        │
       └──────────────┬───────────────────────────┘
                      ▼
       ┌──────────────────────────────────────────────────────────────────┐
       │  STAGE 5: SCORE-MANY (one encode pass, three benchmark filters)  │
       │  compute_relevance.py --benchmarks fb,gdelt-cameo,earnings       │
       │  reads cross-benchmark embedding cache                           │
       └──────────────┬───────────────────────────────────────────────────┘
                      ▼
       ┌──────────────────────────────────────────┐
       │  STAGE 6+: annotate_prior_state, quality,│
       │  diagnostics, publish (unchanged)        │
       └──────────────────────────────────────────┘
```

Stages 0 and 6 onward are v2.1 code. Stages 1, 2, 3, 5 are the v2.2 deltas. Stage 4 is a refactor of existing scripts into a package.

---

## 3. GDELT DOC bulk pipeline architecture

The single largest wall-clock win in v2.2 comes from replacing per-FD HTTP scraping with a one-time bulk download of GDELT DOC, plus a local SBERT index that every FD queries against. This section specifies the three new scripts.

### 3.1 `scripts/fetch_gdelt_doc_archive.py`

**Role**: pull GDELT DOC's monthly archives for the `[context_start - lookback_days, all_end]` window into local zstd-compressed JSONL shards.

**Inputs**:
- `--start YYYY-MM`, `--end YYYY-MM` (defaults derived from `configs/default_config.yaml` `benchmarks.gdelt_cameo.context_start`/`all_end`).
- `--workers N` (default 8; GDELT DOC is more permissive than its DOC API, so we can run higher concurrency than per-FD scraping).
- `--filter-language en` (optional ISO 639-1; default keeps multilingual).

**Outputs**: `data/gdelt_doc/raw/{YYYY-MM}.jsonl.zst`. One JSON object per article, schema:

```
{"url": "...", "title": "...", "publish_date": "YYYY-MM-DDTHH:MM:SSZ",
 "source_domain": "reuters.com", "language": "en", "tone": -1.2,
 "themes": ["ECON", "POL"], "v2tone_doc": [...], "raw_doc_id": "..."}
```

The body is **not** in this file. Body fetching is deferred to Stage 2 (so we only fetch bodies for articles that survive the per-benchmark relevance filter). This is the key inversion versus v2.1: v2.1 pulls bodies first, then filters; v2.2 filters first, then pulls bodies for the survivors.

**Idempotency**: a shard is considered done if its `.zst` exists and its `meta.json` sidecar records the expected article count. `--force` overrides; default skips done shards.

**Estimated cost**: 6 to 8 GB on disk for a 12-month window with English filter; 30 to 50 minutes on a warm cache, dominated by network throughput.

### 3.2 `scripts/build_gdelt_doc_index.py`

**Role**: shard the raw archive, encode title plus snippet with SBERT, build a per-shard FAISS index plus aligned parquet metadata.

**Inputs**:
- `--model sentence-transformers/all-mpnet-base-v2` (must match `configs/default_config.yaml` `relevance.embedding_model`; otherwise the index is incompatible with the score-many stage).
- `--shard-size 100000` (number of articles per FAISS shard; smaller shards are faster to query and can be sharded across workers).
- `--device cuda|cpu`.

**Outputs**: under `data/gdelt_doc/index/`:
- `{YYYY-MM}.faiss`, FAISS flat-IP index (cosine via L2-normalized vectors).
- `{YYYY-MM}.parquet`, aligned metadata: row index, url, publish_date, source_domain, language, themes.
- `manifest.json`, model name, embedding dimension, total rows, per-shard row counts, build timestamp, `git_sha`. Score-many refuses to run if the manifest's model name does not match the active config.

**Idempotency**: `(shard_path, model_name, model_revision)` triple. Re-running with the same triple is a no-op; changing the model invalidates the entire index.

**Estimated cost**: GPU encode of ~5M articles at batch 256 plus FP16 is roughly 35 to 50 minutes on RTX 2060. CPU is roughly 8 to 10 hours; recommend GPU for the cold build.

### 3.3 `scripts/query_gdelt_doc_index.py`

**Role**: per-FD lookup against the index. Returns top-K candidate URLs plus metadata, with date-range filter applied at the FAISS-postscore stage and language plus theme filters applied at the parquet-postscore stage.

**Inputs**:
- `--forecasts data/unified/forecasts.jsonl` (same as `compute_relevance.py`).
- `--top-k 50` (over-pulls; the score-many stage will tighten with per-source `top_k` from config).
- `--date-window-days 90` (matches `lookback_days`).
- `--benchmark-filter forecastbench,earnings,gdelt_cameo` (optional; default all). Per-benchmark application of the editorial-only filter for GDELT-CAMEO is wired through `--source-blocklist gdelt-doc-aggregator` (configurable).
- `--fetch-bodies` (boolean; if set, follow up with bulk trafilatura fetch on the survivors only).

**Outputs**: per benchmark, appends to `data/{benchmark}/{benchmark}_articles.jsonl` in the v2.1 schema (so the rest of the pipeline is unchanged). When `--fetch-bodies` is set, body text is filled by an internal `concurrent.futures.ThreadPoolExecutor` over trafilatura (replaces the single-threaded path in [`scripts/fetch_article_text.py`](../scripts/fetch_article_text.py) for these articles).

**Lookup flow (per FD)**:

```
1. Build query text:
   - forecastbench: extract_keywords(question + background)  (existing logic)
   - gdelt-cameo: actor country names  (existing logic)
   - earnings: company name + ticker  (existing logic)
2. Encode query (one SBERT forward; cached per FD by question hash).
3. For each shard whose [start_month, end_month] intersects
   [forecast_point - lookback_days, forecast_point):
     a. faiss.search(query_emb, top_k * 3) over the shard.
     b. Join to parquet metadata; drop rows outside the exact date window,
        wrong language, blocked source_domain (from spam_domains.py),
        or aggregator domains (for gdelt-cameo).
4. Merge per-shard top-K, re-sort, take overall top-K.
5. (Optional) bulk-fetch bodies for the survivors.
```

This replaces the v2.1 cascade of NYT plus Guardian plus Google News plus GDELT DOC API per FD with a single in-memory FAISS query plus, in some cases, a residual Google News pull (see Section 4).

### 3.4 File formats, why these choices

- **zstd JSONL** for raw shards: compressible (~5x over plain JSONL), streamable, language-agnostic. Avoids parquet for raw because GDELT DOC's theme arrays are awkward to nest in arrow schemas.
- **FAISS flat-IP plus parquet** for the index: flat-IP is exact (no recall trade-off versus the v2.1 brute-force cosine in `compute_relevance.py`), parquet gives random-access metadata join without loading everything in RAM.
- **Per-month sharding**: aligns with GDELT's natural distribution chunking; lets us prune entire shards by date intersection before any FAISS work.

---

## 4. Hybrid retrieval contract

Not every FD's evidence lives in GDELT DOC. The v2.2 contract specifies which source is primary for which benchmark and how fallbacks fire.

| Benchmark | Primary | Secondary | Tertiary | Forbidden |
|---|---|---|---|---|
| `forecastbench` | GDELT DOC index (Stage 2/3) | Google News residual (only if survivors < `min_articles + 2`) | NYT, Guardian (kept as opt-in `--source nyt,guardian`) | none |
| `gdelt_cameo` | GDELT DOC index, `--source-blocklist` filtered to editorial outlets | Google News residual | NYT, Guardian (opt-in) | GDELT DOC API live (the label channel; using the bulk archive is acceptable because it is filtered to editorial domains and queried independently of the KG event row) |
| `earnings` | SEC EDGAR (filings) | Finnhub (curated finance news) | GDELT DOC index residual | yfinance live (kept opt-in only; weak provenance) |

**Fallback rule**: a fallback fires if and only if the primary plus already-fired secondaries return fewer than `quality.min_articles + 2` survivors after date and spam filtering. The `+2` provides headroom for downstream leakage pruning. Each fallback is invoked once and then the cascade stops; no infinite-tier escalation.

**Editorial-only filter for GDELT-CAMEO**: the v2.1 design (`docs/PIPELINE.md` §1) decouples retrieval from labels by forbidding the GDELT DOC live API for this benchmark. v2.2 keeps the spirit but allows the bulk archive provided the filter at `query_gdelt_doc_index.py:domain_filter` removes anything that looks like a re-syndication of the GDELT DOC ranking itself (aggregator domains, GDELT-affiliated outlets). The list lives in `src/common/gdelt_aggregator_domains.py` (new file).

**Why not push everything through GDELT DOC**: SEC EDGAR is the canonical primary source for earnings; replacing it with a news article is a regression in evidence quality. Finnhub provides curated finance editorial that GDELT DOC under-indexes. The two together cover the "before this earnings report, what did analysts believe" question better than any general news index.

---

## 5. Cross-benchmark embedding cache (encode-once, score-many)

### 5.1 The problem

In v2.1, `step_relevance_parallel` (at `scripts/build_benchmark.py:276`) calls `compute_relevance.py --benchmark-filter X` once per benchmark. Each call:

1. Loads the full unified articles file.
2. Computes per-row fingerprints, loads cached embeddings, encodes deltas.
3. Filters articles by benchmark.
4. Filters forecasts by benchmark.
5. Scores.

Step 2's "encode deltas" reads the article cache that was just written by the prior benchmark's call, so most articles are cached after the first call. But:

- Each call still loads roughly 220k articles into RAM and re-runs the cosine top-K computation against the full pool.
- The "filter articles by benchmark" step (4) is misaligned: an article tagged `provenance: ["forecastbench", "gdelt_cameo"]` is scored once per benchmark even though the embedding is identical.
- The fingerprint cache key is `MD5(text)`, invariant across benchmarks, so the cache is structurally re-usable but the call structure does not exploit it.

### 5.2 The design

Replace `--benchmark-filter` with a single multi-benchmark invocation. New CLI:

```
compute_relevance.py --benchmarks forecastbench,gdelt_cameo,earnings
                     [--rebuild]
                     [--device cuda]
```

Internal flow:

1. Load articles.jsonl once. Compute per-row fingerprints. Load `.fp.txt` plus `.npy` cache. Encode only the deltas. Persist new cache. (Same as v2.1, but invoked once.)
2. Load forecasts.jsonl. Compute per-row question fingerprints. Load forecast cache. Encode deltas. Persist.
3. Build a per-benchmark mask over the article axis: `mask[bench] = is_provenance_eligible(article, bench) AND date_window_ok(article, fd, bench)`.
4. For each benchmark, score forecasts in that benchmark against the masked article slice. Apply per-source filters (`embedding_threshold`, `keyword_overlap_min`, `top_k`, `recency_weight`, `actor_match_required`) from config.
5. Write `article_ids` back per benchmark; write per-benchmark `relevance_meta.json`.

**Memory**: at 768-dim FP32 the article matrix is roughly 660 MB for 220k articles, fits comfortably in 16 GB. The per-benchmark mask is a boolean vector at roughly 27 KB.

**Compute**: cosine score is a single batched `articles @ forecasts.T` GEMM per benchmark, on float32. For 220k articles times ~12k FDs (largest benchmark), one matmul is roughly 6 GFLOPs, sub-second on GPU, sub-minute on CPU. The bottleneck moves from encode to filter logic, which is fine.

**Backwards compatibility**: the on-disk cache files (`.npy`, `.fp.txt`) are unchanged; v2.2's encode-once writes the same cache that v2.1's per-benchmark calls would have written. A user can downgrade to v2.1 between builds without invalidating the cache.

### 5.3 Per-benchmark provenance eligibility

The mask in step 3 needs a clear rule for which articles a benchmark is allowed to see. Today this is implicit in the per-benchmark fetcher (each writes to its own JSONL). v2.2 makes it explicit:

| Benchmark | Eligible provenance tags |
|---|---|
| `forecastbench` | `forecastbench`, `gdelt-doc` (general index), residual `google-news` |
| `gdelt_cameo` | `gdelt-doc-editorial` (filtered subset), `nyt`, `guardian`, `google-news`, but never `gdelt-cameo` (the KG oracle channel) |
| `earnings` | `sec-edgar`, `finnhub`, `gdelt-doc`, residual `google-news`, opt-in `yfinance` |

This contract is enforced at score time (mask-out) and re-enforced at publish time by `quality_filter.py`'s leakage audit. Two layers of enforcement is intentional; the score-time mask is for performance, the publish-time audit is for correctness.

---

## 6. Optional OpenAI Batch embeddings path

Some users have no GPU. v2.2 ships an optional drop-in: replace local SBERT with OpenAI's `text-embedding-3-small` via the Batch API.

### 6.1 When to choose it

- **Choose Batch**: no local GPU; rebuild cadence is weekly or slower; OK with a 24-hour latency floor; cost-sensitive (Batch is half-price of sync).
- **Choose local SBERT**: GPU available; rebuild cadence is hourly during a build cycle; need sub-minute encode for the per-row fingerprint delta loop.

### 6.2 Schema-compatible drop-in

New module `src/common/embeddings_backend.py` exposes:

```
encode(texts: list[str], model: str, backend: Literal["sbert", "openai_batch"],
       dim: int = 768) -> np.ndarray  # shape (N, dim), L2-normalized, float32
```

The OpenAI backend:

1. Submits Batch jobs in chunks of 50k inputs (well under the 50k-line per-batch limit).
2. Polls until done (or returns immediately if `--async`, leaving a job-id sidecar for a later resume).
3. Pulls results, normalizes to dim=768 by truncate-then-renormalize (text-embedding-3-small is 1536-dim native; we project to 768 to stay schema-compatible with the SBERT cache).

Cost estimate: 220k articles times 500 char trunc plus 12k FDs at 100 chars equals ~110M tokens at $0.013/M Batch equals roughly $1.50 per cold rebuild. The earlier $0.30 estimate in the design discussion was for the FD-only encode; the article side dominates.

**Cache compatibility**: cache files are tagged with backend in their manifest; switching backends invalidates the cache (encoded vectors are not interchangeable across model families). Mid-flight switches require `--rebuild`.

### 6.3 What the OpenAI Batch path does not change

- The fingerprint protocol (`MD5(text)`) is unchanged.
- The per-source filter logic in `compute_relevance.py` is unchanged; it operates on the embedding matrix without caring about the backend.
- The on-disk index format (`.npy`, `.fp.txt`) is unchanged; only the backend tag in the sidecar manifest differs.

---

## 7. Resumability and cache invariants

v2.1's pipeline is mostly resumable but the invariants are subtle. v2.2 makes them explicit, with one snapshot per stage and a clear "what invalidates what" matrix.

### 7.1 Snapshot semantics

Every stage that writes a file pair (`X.jsonl`, `X.meta.json`) must satisfy: `X.meta.json` is written **last and atomically** (write to `.tmp`, rename). Resume logic is "if `X.meta.json` exists and matches the current config-hash, skip the stage; otherwise rerun."

**Config-hash**: `MD5(canonical_yaml(effective_config_for_this_stage))`. Per-stage config slices live in `src/common/config_slices.py` (new module): each stage declares which config keys it depends on; the slice is hashed.

### 7.2 Cache invalidation matrix

| Cache | Invalidates when... |
|---|---|
| `data/gdelt_doc/raw/*.jsonl.zst` | window changes, language filter changes |
| `data/gdelt_doc/index/*.faiss` | embedding model name or revision changes |
| `data/gdelt_doc/index/*.parquet` | shard size changes (rebuild triggers reshard) |
| `data/unified/article_embeddings.npy` plus `.fp.txt` | per-row fingerprint mismatch (text changed); embedding model changed; backend changed |
| `data/unified/forecast_embeddings.npy` plus `.fp.txt` | same as above for forecasts |
| `data/{bench}/{bench}_articles.jsonl` | source list changes; lookback window changes; primary-source switch in the hybrid contract |
| Per-FD relevance results in `forecasts.jsonl[article_ids]` | upstream embeddings invalidated; per-source threshold changed |

### 7.3 Mid-stage kill and restart

Each long-running stage (`fetch_gdelt_doc_archive`, `build_gdelt_doc_index`, body fetch in `query_gdelt_doc_index`) must support `Ctrl-C` mid-loop and resume from the last completed shard or batch. The protocol:

- Process work in fixed-size batches (e.g. 1k articles per body-fetch batch).
- Append-only writes; never truncate.
- Per-batch sidecar `_progress.json` records last completed batch index.
- On startup, scan the sidecar; resume from `last + 1`.

This is already partially done by `--skip-completed` in v2.1's fetchers (see `scripts/fetch_forecastbench_news.py`), but it is per-FD rather than per-batch and so has worse restart granularity.

---

## 8. Refactor backlog

This section names specific files and functions to consolidate. The full prioritized list lives in [`docs/V2_2_REFACTOR_BACKLOG.md`](V2_2_REFACTOR_BACKLOG.md); here we group the architecturally-significant items.

### 8.1 Extract `NewsFetcher` base class

Source files: [`scripts/fetch_forecastbench_news.py`](../scripts/fetch_forecastbench_news.py) (504 lines), [`scripts/fetch_gdelt_cameo_news.py`](../scripts/fetch_gdelt_cameo_news.py) (418 lines), [`scripts/fetch_earnings_news.py`](../scripts/fetch_earnings_news.py) (678 lines).

Shared logic to extract into `src/common/news_fetcher.py`:

- `art_id(url)`, different prefixes per fetcher (`fbn_`, `gdc_`, `earn_`) but same SHA1 logic.
- `domain_of(url)`, identical across all three.
- `is_spam_url(url)`, already shared via `src.common.spam_domains` but each fetcher re-imports.
- The append-loop with `--skip-completed` resume.
- The cross-source dedup by `(fd_id, title_prefix[:80].lower())`.
- The HEADERS plus TIMEOUT constants.

Per-fetcher subclasses override:
- `build_query(fd) -> str`
- `eligible_sources -> list[Source]`
- `art_id_prefix -> str`

Estimated reduction: roughly 400 lines of net-net duplicated code becomes one base class plus three thin subclasses. Net delta: -400 lines, +1 base file.

### 8.2 Collapse `unify_articles` plus `unify_forecasts` into a `unify` package

Source files: [`scripts/unify_articles.py`](../scripts/unify_articles.py) (329 lines), [`scripts/unify_forecasts.py`](../scripts/unify_forecasts.py) (379 lines).

These two are run as a pair, share intermediates, and have parallel "load from N sources, dedup, write" structure. Move to `src/unify/__init__.py` exposing `unify_articles()` and `unify_forecasts()`. Keep [`scripts/unify_articles.py`](../scripts/unify_articles.py) and [`scripts/unify_forecasts.py`](../scripts/unify_forecasts.py) as thin CLI shims that import from the package, same pattern as the v2.1 `benchmark/build.py` deprecation.

Benefits: shared loader for the per-source JSONL files; shared schema-version handling; tests can exercise the package without subprocess invocation.

### 8.3 Stabilize date-validator surface in `articles_to_facts.py`

[`scripts/articles_to_facts.py`](../scripts/articles_to_facts.py) line 230 had the `>` versus `>=` bug recovered in commit `7237553`. The validator surface has grown organically across Phase A (recovery), Phase C (v3 prompt), and the optional `--strict-dates` flag. Consolidate into `src/etd/date_validators.py` with named, individually-testable validator functions: `is_post_publish`, `is_within_window`, `is_iso_format`, `is_calendar_valid`. Each is a pure function, exhaustively tested.

### 8.4 Single owner for output paths

Today some scripts write to `data/unified/`, some to `data/{benchmark}/`, and some both. Create `src/common/paths.py` with named constants and a single `Layout` dataclass. Every script imports from this; no `Path("data/...")` literals outside `paths.py`.

### 8.5 Move SBERT speedup config into the embeddings backend

Today's `batch_size=256` plus FP16 lives as inline arguments at `compute_relevance.py:embed`. Move these into `embeddings_backend.encode()` so both SBERT and OpenAI Batch backends pick up the same plumbing.

---

## 9. Migration plan

Goal: ship v2.2 without breaking the v2.1 deliverable already published at `benchmark/data/2026-01-01/` and without invalidating the cache.

### 9.1 Phase A, additive infrastructure (no behavior change)

- Add `src/common/embeddings_backend.py` with the SBERT backend mirroring current `compute_relevance.py` behavior. No code calls it yet.
- Add `src/common/paths.py`. Migrate one script as proof-of-life.
- Add `src/unify/__init__.py` mirroring current `unify_articles.py` plus `unify_forecasts.py`. Both scripts become two-line CLI shims.
- Add `src/common/news_fetcher.py` base class. Migrate `fetch_earnings_news.py` first (largest, most isolated).
- Tests: 46 v2.1 tests stay green; add roughly 20 new tests for the new modules.

**Acceptance**: a v2.1-style build (using `--no-gdelt-doc-index`) produces a byte-identical `forecasts.jsonl` and `articles.jsonl` to the prior build at the same cutoff.

### 9.2 Phase B, GDELT DOC index, opt-in

- Ship `fetch_gdelt_doc_archive.py`, `build_gdelt_doc_index.py`, `query_gdelt_doc_index.py`.
- Wire into `build_benchmark.py` behind `--use-gdelt-doc-index` flag (default off).
- Document in `docs/PIPELINE.md` §3 as an opt-in alternative path.
- Run a side-by-side build at `--cutoff 2026-01-01` with both paths; compare `forecasts.jsonl` per-FD article ID overlap. Acceptance: ≥ 80% Jaccard overlap on `article_ids` per FD; total survivors within ±5%.

### 9.3 Phase C, encode-once-score-many

- Refactor `compute_relevance.py` to accept `--benchmarks fb,gdelt-cameo,earnings`. Internal restructure per Section 5.
- `step_relevance_parallel` becomes `step_relevance_unified` with one call.
- Acceptance: same per-benchmark `relevance_meta.json` outputs (modulo float32 noise floor).

### 9.4 Phase D, flip defaults, deprecate old paths

- `--use-gdelt-doc-index` becomes default-on; add `--no-gdelt-doc-index` for the legacy fallback.
- The per-FD HTTP scraping cascade in fetchers becomes the residual fallback per Section 4.
- Retire `--benchmark-filter` from `compute_relevance.py`.
- Bump pipeline version to 2.2 in `build_manifest.json`.

### 9.5 Rollback plan

Each phase is independently revertible. The cache files written by v2.1 remain readable by v2.2 (Section 7.2 invariants). If the GDELT DOC index path produces a regressed deliverable, `--no-gdelt-doc-index` returns to v2.1 behavior with no cache rebuild.

---

## 10. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GDELT DOC bulk archive download is rate-limited harder than the per-FD API | Medium | Could push Stage 1 from 30 min to 4+ hours | Stage 1 is one-time per cutoff window; even at 4 hours it amortizes over 10+ rebuilds. Mitigation: shard download by month with `--workers 4` initial floor; backoff on 429. |
| FAISS index recall on 5M-article shards is worse than brute-force cosine | Low | Article-ID coverage drops in evaluation | Use FAISS flat-IP (exact, not ANN); shard sizes chosen to fit in RAM. ANN (HNSW, IVF) is explicitly out of scope per Section 11. |
| Editorial-only filter for GDELT-CAMEO leaks aggregator domains we forgot | Medium | Subtle leakage; could surface in two-cutoff probe | Explicit `gdelt_aggregator_domains.py` allow/block list; audit at publish time as a second layer. |
| OpenAI Batch backend produces different rankings than SBERT, breaking baseline-comparison continuity | High | Apples-to-oranges across published cutoffs | Treat backend choice as part of the cutoff identity; store `embedding_backend` in `build_manifest.json`; do not mix backends within a published cutoff directory. |
| Encode-once-score-many regresses per-benchmark filter logic (e.g. `actor_match_required` for GDELT) | Medium | Silent quality regression in one benchmark | Keep per-benchmark filter logic intact; only the encode step is unified. Side-by-side acceptance test in Phase C. |
| Refactor of news fetcher base class changes a subtle behavior | Medium | Article counts shift between v2.1 and v2.2 | Phase A acceptance: byte-identical deliverable when GDELT DOC index is off. |
| Resume protocol's per-batch sidecar conflicts with parallel workers | Low | Lost progress on `Ctrl-C` mid-build | Lock-file around sidecar writes; document that parallel workers must use disjoint shard sets. |

---

## 11. Out of scope for v2.2

Explicitly not in this design:

- **ANN over the GDELT DOC index** (HNSW, IVF, ScaNN). Flat-IP is exact and fast enough at the corpus size we operate on. Revisit at 50M+ articles.
- **Multi-cutoff stratification of the deliverable**. v2.2 still ships one `{cutoff}/` directory per build; multi-cutoff comparison is an evaluation-time concern, not a build-time concern.
- **ETD Stages 2 plus 3 redesign**. The `etd_dedup.py` plus `etd_link.py` scripts shipped in v2.1 (commit `7b11d7a`) stay as-is. v2.2 only touches the build pipeline.
- **New benchmark tracks**. Sports, weather, election outcomes were discussed; not in v2.2.
- **FD schema changes**. `forecasts.jsonl` records remain v2.1-compatible. The `Comply` plus `Surprise` target, the `fd_type` partition, the `x_multiclass_*` ablation slots are all unchanged.
- **ACLED wiring**. Excluded permanently per the v2.1 decision (algorithmic pseudo-probabilities, Brier > 0.45).
- **Replacing trafilatura**. Body extraction stays on trafilatura with the v2.1 retry fallback; the parallelization win in Section 3.3 is what matters, not the extractor.
- **Streaming or incremental publish**. Each build still publishes a single self-contained `{cutoff}/` directory atomically.

---

## 12. References

- v2.1 pipeline: [`docs/PIPELINE.md`](PIPELINE.md)
- FD canonical schema: [`docs/FORECAST_DOSSIER.md`](FORECAST_DOSSIER.md)
- ETD spec: [`docs/ETD_SPEC.md`](ETD_SPEC.md)
- Benchmark deliverable contract: [`benchmark/README.md`](../benchmark/README.md), [`benchmark/RECREATE.md`](../benchmark/RECREATE.md), [`benchmark/DATASET.md`](../benchmark/DATASET.md)
- v2.1 entrypoint: [`scripts/build_benchmark.py`](../scripts/build_benchmark.py)
- v2.2 refactor backlog: [`docs/V2_2_REFACTOR_BACKLOG.md`](V2_2_REFACTOR_BACKLOG.md)

---

## 13. Change log

| Version | Date | Summary |
|---|---|---|
| 2.2-draft | 2026-04-22 | Initial design doc. Documents GDELT DOC bulk pipeline, hybrid retrieval contract, encode-once-score-many cache, optional OpenAI Batch backend, resumability invariants, and refactor backlog. No code changes yet. |
