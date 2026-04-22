# EMR-ACH Benchmark Pipeline — Documentation

**Version**: 2.0 (2026-04-22, "forecast from prior news" framing)
**Entry point**: [`scripts/build_benchmark.py`](../scripts/build_benchmark.py)
**Deliverable layout**: [`benchmark/data/{cutoff}/`](../benchmark/data/)

---

## 1. Overview

The EMR-ACH benchmark pipeline produces a unified, leakage-guarded, self-contained
**Forecast Dossier** (FD) dataset for evaluating LLM-based forecasting. Every FD
fuses three inputs — a question, a candidate hypothesis set, and a pool of news
articles published before the forecast point — into the same schema across three
heterogeneous domains:

| Domain | Benchmark source | Task | Classes |
|---|---|---|---|
| Geopolitics | GDELT-CAMEO Geopolitics (our own build on public GDELT 2.0 KG) | Forecast country-pair **conflict intensity** on a target date from prior news | **Peace / Tension / Violence** (ordinal) |
| Prediction markets | ForecastBench | Answer a binary Yes/No prediction-market question | Yes / No |
| Finance | Earnings (yfinance + Finnhub) | Classify a quarterly EPS report vs analyst consensus | Beat / Meet / Miss |

**Design note — GDELT-CAMEO target (v2.0, 2026-04-22):** the benchmark uses a
3-class **intensity** reduction of the CAMEO event taxonomy (see
[`src/common/cameo_intensity.py`](../src/common/cameo_intensity.py) for the
mapping):

  * **Peace** — CAMEO roots 01–09 (cooperative: statements, appeals, consults,
    diplomatic / material cooperation, aid, yielding, investigations)
  * **Tension** — roots 10–17 (non-violent friction: demands, disapproval,
    rejection, threats, protests, force posture, reduced relations, coercion)
  * **Violence** — roots 18–20 (physical force: assault, fight, unconventional
    mass violence)

The ordering is **ordinal** (Peace < Tension < Violence), enabling
ordinal-error metrics alongside nominal accuracy.

**Design note — stability vs change partition (all three benchmarks):** every FD
is annotated with a domain-appropriate **`prior_state`** and partitioned:

| Benchmark | prior_state source | Stability means | Change means |
|---|---|---|---|
| gdelt-cameo   | Modal Peace/Tension/Violence class over prior 30 days' events for the actor pair | Already-violent pair stays violent; peaceful pair stays peaceful | Onset, cessation, escalation, de-escalation |
| earnings      | Mode of prior **4 quarters'** surprise class (Beat/Meet/Miss) for the same ticker | Chronic beater beats again; chronic misser misses again | Unexpected miss / unexpected beat / regime change |
| forecastbench | `crowd_probability` — market's implied majority (≥0.5 → Yes, else No) | Outcome matches the market's majority call | Upset — crowd consensus contradicted by resolution |

The split exposes **where forecasting skill actually matters**: status-quo FDs
are trivially predicted by a classifier that reads the prior state; change FDs
require genuine evidence reading. **Headline metrics are reported on the Change
subset.** See [`scripts/annotate_prior_state.py`](../scripts/annotate_prior_state.py).

**Design note — evidence channel isolation:** GDELT is used only as the
question/label source. Evidence (for baselines and EMR-ACH) comes from
independent editorial sources — NYT, Guardian, Google News RSS. The GDELT
DOC API is no longer in the default `fetch_gdelt_cameo_news.py --source`
cascade (kept as `--source gdelt` opt-in for legacy ablation). This decouples
the retrieval channel from the label channel and makes the leakage argument
reviewer-proof.

Every FD carries a **strict temporal contract**: all linked articles are
published before the forecast point, and every FD's resolution date is after
the evaluator LLM's training cutoff. The pipeline enforces these at build time
via `scripts/quality_filter.py`.

---

## 2. Quick start

```bash
# Full build (22k articles, 3k FDs, ~30 minutes end-to-end on a warm cache):
python scripts/build_benchmark.py \
    --config configs/default_config.yaml \
    --cutoff 2026-01-01

# What will happen without executing (dry-run):
python scripts/build_benchmark.py --cutoff 2026-01-01 --dry-run

# Restrict to a subset of benchmarks:
python scripts/build_benchmark.py --cutoff 2026-01-01 --benchmarks forecastbench,earnings

# Reuse already-downloaded raw inputs (skip the expensive fetch step):
python scripts/build_benchmark.py --cutoff 2026-01-01 --skip-raw

# Full clean rebuild from scratch (wipes data/unified/):
python scripts/build_benchmark.py --cutoff 2026-01-01 --fresh
```

**Outputs**: `benchmark/data/{cutoff}/` contains the self-contained deliverable
(forecasts.jsonl, articles.jsonl, benchmark.yaml, build_manifest.json).
See §9 for the deliverable contract.

---

## 3. Dependency graph

```
build_benchmark.py (orchestrator)
│
├── configs/default_config.yaml  (all tunable parameters)
│
├── STAGE 1 — RAW BUILD (optional; --skip-raw bypasses)
│   ├── build_gdelt_cameo.py          (downloads GDELT KG exports, generates test queries)
│   ├── build_earnings_benchmark.py   (yfinance earnings scrape, beat/meet/miss classification)
│   └── download_forecastbench.py     (static repo clone, no rebuild needed after initial pull)
│
├── STAGE 2 — FIRST UNIFY
│   ├── unify_articles.py             (merge article pools; dedup by URL)
│   └── unify_forecasts.py            (merge FD records; compute ground truth;
│                                     GDELT-CAMEO uses Peace/Tension/Violence)
│     [snapshot: 01_after_first_unify]
│
├── STAGE 3 — MULTI-SOURCE PER-FD NEWS FETCH (unified 90-day analysis window)
│   ├── fetch_forecastbench_news.py   (NYT + Guardian + Google News + GDELT DOC)
│   ├── fetch_gdelt_cameo_news.py     (NYT + Guardian + Google — default editorial;
│   │                                  NO GDELT DOC — keeps label/evidence separate)
│   └── fetch_earnings_news.py        (Finnhub + yfinance + Google News + GDELT DOC)
│   → re-run unify_articles.py and unify_forecasts.py to include new news
│     [snapshot: 01b_after_multi_source_news]
│
├── STAGE 3b — GDELT PRIOR-STATE ANNOTATION (GDELT-CAMEO only)
│   └── annotate_gdelt_prior_state.py (adds fd_type ∈ {stability, change} +
│                                     prior_state_30d; no new API calls)
│     [snapshot: 01c_after_prior_state_annotation]
│
├── STAGE 4 — RELEVANCE SCORING (SBERT cross-match)
│   └── compute_relevance.py          (per-benchmark filter, top-K articles per FD)
│     [snapshot: 02_after_first_relevance]
│
├── STAGE 5 — GDELT CONTEXT RELINK (GDELT-CAMEO only, legacy oracle fix)
│   └── relink_gdelt_context.py       (replaces same-day oracle Docids with pre-event context)
│   → re-run fetch_gdelt_text.py and fetch_article_text.py for new URLs
│   → re-run unify + relevance if text was added
│     [snapshots: 03_after_first_relink, 04_after_gdelt_text_fetch, 05_after_fb_text_fetch,
│                 06_after_reunify, 07_after_re_relevance_and_relink]
│
├── STAGE 6 — QUALITY FILTER
│   └── quality_filter.py             (min_articles, min_days, min_chars, leakage guard)
│     [snapshot: 08_after_quality_filter]
│
├── STAGE 7 — DIAGNOSTICS
│   ├── diagnostic_report.py          (drop-reason report)
│   └── build_eda_report.py           (HTML EDA artifacts)
│     [snapshot: 09_final_before_publish]
│
└── STAGE 8 — PUBLISH
    └── step_publish()                (promotes filtered output to benchmark/data/{cutoff}/)
```

---

## 4. Configuration reference

All pipeline behavior is controlled by [`configs/default_config.yaml`](../configs/default_config.yaml).
CLI flags in `build_benchmark.py` override config values.

### 4.1 Top-level

| Key | Type | Description |
|---|---|---|
| `model_cutoff` | YYYY-MM-DD | Evaluator LLM's training cutoff. Any FD with `resolution_date <= model_cutoff + cutoff_buffer_days` is dropped as training leakage. |
| `cutoff_buffer_days` | int | Safety margin. `0` means require resolution strictly after cutoff; `30` would require resolution at least a month past cutoff. |
| `output.root` | path | Base directory for the published deliverable. Default: `benchmark/data`. |

### 4.2 `benchmarks.forecastbench`

| Key | Description |
|---|---|
| `enabled` | Toggle inclusion. |
| `prediction_market_sources` | List of permitted sources: `polymarket`, `metaculus`, `manifold`, `infer`. |

### 4.3 `benchmarks.gdelt_cameo`

| Key | Format | Description |
|---|---|---|
| `enabled` | bool | Toggle inclusion. |
| `context_start`, `context_end` | YYYYMM | Pre-event context window (e.g. `202601`, `202602`). |
| `test_month` | YYYYMM | Month whose events are used as ground truth. |
| `all_end` | YYYYMM | End of the full download window (includes test month). |
| `min_daily_mentions` | int (default 50) | Reliability threshold per event row. |
| `max_download_workers` | int (default 16) | Parallel downloads of GDELT zip archives. |

### 4.4 `benchmarks.earnings`

| Key | Description |
|---|---|
| `enabled` | Toggle inclusion. |
| `start`, `end` | YYYY-MM-DD window of earnings calls to include. |
| `tickers` | Optional explicit list (CSV or YAML list). If `null`, uses `data/sp500_tickers.txt` with a hardcoded 50-ticker fallback. |
| `threshold` | Fractional EPS surprise threshold for Beat/Miss classification. Default `0.02` = 2%. |

### 4.5 `quality`

| Key | Default | Description |
|---|--:|---|
| `min_articles` | 3 | Minimum articles per FD after leakage-prune. |
| `min_distinct_days` | 2 | Articles must span at least this many distinct publication dates. |
| `min_chars` | 1500 | Total character count across all linked articles. |
| `min_question_chars` | 20 | Minimum question length (avoids degenerate queries). |
| `min_distinct_days_per_benchmark` | `{}` | Per-benchmark override, e.g. `{gdelt-cameo: 1}` for oracle-clustered data. |

### 4.6 `relevance`

| Key | Description |
|---|---|
| `embedding_model` | SBERT model path, default `sentence-transformers/all-mpnet-base-v2`. |
| `batch_size` | SBERT encode batch size (default 32). |
| `max_text_chars` | Per-article text truncation before embedding (default 500). |
| `sources.<name>.*` | Per-source scoring knobs. See table below. |

**Per-source scoring knobs** (`relevance.sources.<forecastbench|gdelt-kg|yfinance|...>`):

| Knob | Description |
|---|---|
| `embedding_threshold` | Cosine-similarity floor. Articles below this score are dropped. |
| `keyword_overlap_min` | Minimum overlap between FD question tokens and article tokens (integer count). |
| `lookback_days` | Only articles published in `[forecast_point − lookback_days, forecast_point)` are eligible. |
| `top_k` | Maximum articles kept per FD (default 10). |
| `recency_weight` | Blend factor between cosine score and recency score (0 = pure cosine, 1 = pure recency). |
| `actor_match_required` | For GDELT-CAMEO: require article to mention at least one of the FD's actor countries. |

### 4.7 `secrets`

Environment-variable names where credentials are looked up (not the keys themselves):

```yaml
secrets:
  openai_api_key:   OPENAI_API_KEY
  newsapi_key:      NEWSAPI_KEY
  metaculus_token:  METACULUS_TOKEN
  finnhub_api_key:  FINNHUB_API_KEY
  guardian_api_key: GUARDIAN_API_KEY
  nyt_api_key:      NYT_API_KEY
```

The actual keys live in `.env` (gitignored); `build_benchmark.py` loads them
via `src.config.get_config()`.

---

## 5. Stage-by-stage reference

### 5.1 `build_gdelt_cameo.py` — GDELT-CAMEO raw build

Downloads GDELT's 15-minute KG exports for the configured window, filters for
country-coded bilateral events with ≥ `min_daily_mentions` daily mentions,
deduplicates URLs, and emits the quad-class test queries.

**Runs in 5 steps**; use `--steps 1,2,3,4,5` to control which:
1. List master file.
2. Download zip exports (parallel).
3. Clean / filter events.
4. Deduplicate URLs, emit `data_news.csv`.
5. Build `relation_query.csv` (the FD-question template source).

**Outputs**:
- `data/gdelt_cameo/data_kg.csv` — filtered event rows.
- `data/gdelt_cameo/data_news.csv` — unique URL → publication date lookup.
- `data/gdelt_cameo/test/relation_query.csv` — one row per country-pair test FD.

### 5.2 `build_earnings_benchmark.py` — Earnings raw build

Iterates the configured ticker list, queries yfinance for earnings history in
the `[start, end]` window, classifies surprise direction using `threshold`, and
emits FD records.

**Ground truth rule**: `surprise_pct = (actual − estimate) / |estimate|`.
If `surprise_pct ≥ threshold` → `Beat`; if ≤ `−threshold` → `Miss`; else `Meet`.
Zero-estimate cases fall back to absolute-EPS comparison at ±$0.02.

**Output**: `data/earnings/earnings_forecasts.jsonl`.

### 5.3 `unify_articles.py` — Article pool merge

Merges article records from six sources (GDELT-CAMEO oracle, ForecastBench
supplemental, and the four per-FD news fetches) into a single
`data/unified/articles.jsonl`. Deduplicates by URL hash.

**Dedup policy**:
- Key = `sha1(url)[:12]` (first 12 hex chars of URL SHA1).
- On collision: keep the article with the longer `text` body.
- Provenance is unioned (multiple tags per article, e.g. `["forecastbench", "gdelt-cameo"]`).

**Schema** (one article per line):
```json
{
  "id": "art_5a916a6d2275",
  "url": "...",
  "title": "...",
  "text": "...",                    // cleaned full body (may be empty)
  "publish_date": "YYYY-MM-DD",
  "source_domain": "reuters.com",
  "gdelt_themes": [],
  "gdelt_tone": 0.0,
  "actors": ["ISR", "PAL"],         // country codes or tickers
  "cameo_code": "195",              // empty for non-GDELT sources
  "char_count": 1234,
  "provenance": ["forecastbench", "gdelt-cameo"]
}
```

### 5.4 `unify_forecasts.py` — Forecast pool merge

Merges FD records from three sources into `data/unified/forecasts.jsonl`.
Computes ground truth, attaches hypothesis definitions, and sets the
`lookback_days = 90` analysis window (unified across all three benchmarks).

For GDELT-CAMEO FDs, ground truth is computed from the event's CAMEO root
code (via `src/common/cameo_intensity.event_to_intensity`): roots 01-09
→ Peace, 10-17 → Tension, 18-20 → Violence. `EventBaseCode` is emitted by
`build_gdelt_cameo.py` step 5 into `relation_query.csv` for this purpose.

**Schema**:
```json
{
  "id": "gdc_1234",
  "benchmark": "gdelt-cameo",
  "source": "gdelt-kg",
  "question": "Based on news from the preceding months, what is the dominant
               intensity of interaction between Israel and Palestine on or
               around 2026-03-01: peace, tension, or violence?",
  "background": "...",
  "forecast_point": "2026-03-01",
  "resolution_date": "2026-03-01",
  "hypothesis_set": ["Peace", "Tension", "Violence"],
  "hypothesis_definitions": {"Peace": "...", "Tension": "...", "Violence": "..."},
  "ground_truth": "Violence",
  "ground_truth_idx": 2,             // ordinal: Peace=0, Tension=1, Violence=2
  "crowd_probability": null,         // only populated for ForecastBench
  "lookback_days": 90,
  "article_ids": [],                 // populated by compute_relevance.py

  // Added by annotate_gdelt_prior_state.py (GDELT-CAMEO only; omitted elsewhere):
  "prior_state_30d": "Violence",
  "prior_state_stability": 0.73,     // fraction of prior 30d matching mode
  "prior_state_n_events": 84,
  "fd_type": "stability"             // "stability" | "change" | "unknown"
}
```

### 5.5 `fetch_forecastbench_news.py`, `fetch_gdelt_cameo_news.py`, `fetch_earnings_news.py`

All three follow the same pattern: iterate FDs, build a per-FD retrieval
query, hit 3–4 news sources in a fallback cascade, and append unique articles
to a benchmark-specific JSONL file.

**Query construction**:
| Benchmark | Query source | Example |
|---|---|---|
| ForecastBench | Keywords extracted from `question + background` via `extract_keywords()` (proper-noun-preferring, stop-word-filtered) | `"Bitcoin price 100000"` |
| GDELT-CAMEO | Actor country names from FD metadata | `"Israel" "Palestine"` |
| Earnings | Company name + ticker | `"Apple Inc" OR AAPL earnings` |

**Sources**:
- **Finnhub** (earnings only) — Finance-first, 60 req/min free tier, historical archive. Filters: `related` must contain the ticker; `category ∈ {company, earnings}`.
- **NYT Article Search** (FB + GDELT) — 1000 req/day free tier, archive back to 1851.
- **The Guardian Open Platform** (FB + GDELT) — 5000 req/day free tier, 1999+.
- **GDELT DOC API** (all) — Unlimited free, multilingual, global. Rate-limited via client-side throttle + exponential retries.
- **Google News RSS** (all) — Unlimited free, no key, recency-biased (weak for old FDs).
- **yfinance** (earnings only) — Most recent ~10 news items per ticker, no historical.

**Cross-source dedup**:
- By raw URL (exact match).
- By normalized URL (drop query/fragment, lowercase host).
- By `(fd_id, title_prefix[:80].lower())` — same-fact-different-URL detection.

**Spam filter**: 30+ auto-generated SEO-spam domains are blocked at fetch time.
See [`src/common/spam_domains.py`](../src/common/spam_domains.py) for the list.

**Resumability**: `--skip-completed` checks which FDs already have any article
in the output file and skips them. Safe to re-run after new data lands.

**Sharding** (`fetch_gdelt_cameo_news.py` only): `--shard N/K` partitions the
FD space deterministically by `hash(fd_id) % K`. Used for parallel workers.

### 5.6 `compute_relevance.py` — SBERT cross-match

For each FD, embed the question; for each candidate article, embed the
title+text (truncated to `max_text_chars`). Compute cosine similarity,
apply per-source filters, select top-K, write back into
`forecasts.jsonl[article_ids]`.

**Per-source filters applied before top-K**:
- `embedding_threshold`: drops articles with cosine below floor.
- `keyword_overlap_min`: drops articles with too little query-token overlap.
- `lookback_days`: drops articles outside the analysis window.
- `actor_match_required` (GDELT only): drops articles not mentioning a
  relevant actor country.

**Incremental cache**: per-row fingerprint (`MD5(fact.text + article.id)`)
is stored alongside the embeddings; unchanged rows skip re-embed on the next
run. Saves ~5 min per rebuild on a warm cache.

**Outputs**:
- `data/unified/forecasts.jsonl` (in-place: `article_ids` populated).
- `data/unified/article_embeddings.npy`, `data/unified/forecast_embeddings.npy`.
- `data/unified/relevance_meta.json` (per-source match rates, top-K coverage).

### 5.7 `relink_gdelt_context.py` — Legacy oracle fix

**Why it exists**: the original (pre-2026-04-22) oracle-retrieval methodology used same-day Docids,
which causes ground-truth leakage (the articles describe the event itself).
Under the new *"forecast from prior news"* framing (v2.0), this step is
partially redundant with `fetch_gdelt_cameo_news.py`. Kept for backwards
compatibility.

Replaces `article_ids` for each GDELT-CAMEO FD with the 10 most-recent
pre-event articles about the same country pair.

### 5.8 `quality_filter.py` — Acceptance criteria

Drops FDs that fail any of the following (see `forecasts_dropped.jsonl` for
per-FD drop reasons):

| Reason tag | Rule |
|---|---|
| `n_articles<3` | After leakage-prune, fewer than `min_articles` articles remain. |
| `day_spread<2` | Articles span fewer than `min_distinct_days` distinct publication dates. |
| `char_count<1500` | Total character count across articles is below `min_chars`. |
| `question_len<20` | Question shorter than `min_question_chars`. |
| `resolution_date<=cutoff(...)` | FD resolves on or before `model_cutoff + cutoff_buffer_days`. |

**Leakage-prune**: articles with `publish_date >= forecast_point` are removed
from the FD's `article_ids` before the above rules apply (no fact about the
outcome itself sneaks in).

### 5.9 `step_publish()` — Deliverable emission

Copies the filtered forecasts and subset of articles to
`benchmark/data/{cutoff}/` under the self-contained contract. Scrubs any
stale `meta/`, `intermediate/`, `staging/` subdirs left over from earlier
builds. Writes:

- `forecasts.jsonl` — primary data.
- `articles.jsonl` — only articles referenced by the filtered FDs.
- `benchmark.yaml` — full effective build config (fully reproducible).
- `build_manifest.json` — provenance: timestamp, git sha, FD counts per
  benchmark, article count, quality thresholds, layout spec.

Diagnostic material (EDA, drop reasons, quality meta) is routed to
`benchmark/audit/{cutoff}/` — never mixed with the deliverable.

---

## 6. Reproducibility

### 6.1 What's captured at build time

Every published `benchmark/data/{cutoff}/build_manifest.json` contains:

```json
{
  "generated_at": "2026-04-22T17:58:34",
  "model_cutoff": "2026-01-01",
  "cutoff_buffer_days": 0,
  "benchmarks_included": ["forecastbench", "gdelt_cameo", "earnings"],
  "n_fds": 2456,
  "n_fds_by_benchmark": {"forecastbench": 461, "gdelt-cameo": 1938, "earnings": 57},
  "n_articles": 27814,
  "git_sha": "f1114c2d08f1d71d97aa10f65ed47b2b53384795",
  "python": "/c/Python314/python",
  "quality_thresholds": {
    "min_articles": 3, "min_distinct_days": 2, "min_chars": 1500, "min_question_chars": 20
  }
}
```

And `benchmark.yaml` is the **effective** config used — pasting it back into
`configs/` and re-running `build_benchmark.py` reproduces the same deliverable
(modulo external API non-determinism in news fetch).

### 6.2 Non-determinism sources

- **News APIs**: article rankings may shift between fetches. Mitigated by
  `--skip-completed` on every fetcher (once an article is captured, it stays).
- **GDELT DOC API**: transient timeouts cause occasional missing articles.
  Retry logic at fetch time + SBERT top-K selection downstream make this
  a small-variance source.
- **LLM extraction** (if ETD is enabled): gpt-4o-mini at T=0 is deterministic
  modulo OpenAI's internal model versioning.

### 6.3 Versioning

- **Dataset**: the `{cutoff}` directory name encodes the model-cutoff
  date; earlier cutoffs remain available side-by-side.
- **Schema**: the deliverable's JSON records are stable across minor
  pipeline updates; any breaking change bumps `schema_version` in the
  manifest and requires a new cutoff directory.
- **Code**: `git_sha` in the manifest pins the exact code version; paired
  with the saved `benchmark.yaml` it is a complete reproduction recipe.

---

## 7. Performance notes

Observed wall-clock times on a warm cache (no network fetches, no
re-embedding):

| Stage | Time |
|---|---|
| Unify | ~15 s |
| Multi-source news fetch (--skip-completed, cache hit) | ~2 min |
| Relevance (incremental cache hit) | ~1 min |
| Relink GDELT | ~30 s |
| Quality filter | ~5 s |
| Publish | <5 s |
| **Total (warm cache)** | **~4 min** |

Cold rebuild (full GDELT KG download + all news fetches + full embed) takes
**~12 h** mostly due to GDELT DOC API rate limits.

See §8 for parallelism knobs.

---

## 8. Parallelism knobs

| Knob | Where | Default | Effect |
|---|---|--:|---|
| `benchmarks.gdelt_cameo.max_download_workers` | config | 16 | Concurrent GDELT zip downloads. |
| `relevance.batch_size` | config | 32 | SBERT encoder batch size. Increase for GPU (64–128). |
| `fetch_*.py --shard N/K` | CLI | — | Partitions FDs across K parallel workers. |
| `compute_relevance.py --device cuda` | CLI | `cpu` | Enables GPU embedding. |

---

## 9. Deliverable contract

`benchmark/data/{cutoff}/` MUST contain exactly these files:

- `forecasts.jsonl` — one FD per line, schema per §5.4.
- `articles.jsonl` — all articles referenced by `forecasts.jsonl[article_ids]`. No extras.
- `benchmark.yaml` — effective build config used for this cutoff.
- `build_manifest.json` — provenance (§6.1).

No intermediate data, no staging, no audit reports. Those live in
`benchmark/audit/{cutoff}/` per §5.9.

This layout is enforced by `scripts/build_benchmark.py step_publish()`, which
scrubs stale subdirs on every publish. Consumers of the benchmark should rely
only on the four files above.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `n_articles<3` dominates drops on a new benchmark | News fetch didn't run for that benchmark | Run `scripts/fetch_<benchmark>_news.py --source all --lookback 90` manually, then re-unify. |
| `day_spread<2` dominates on GDELT-CAMEO | Oracle relink failed; all articles on event day | Check `data/unified/gdelt_cameo_relink_meta.json`; re-run `relink_gdelt_context.py`. |
| `resolution_date<=cutoff` drops everything | Cutoff is too recent for the benchmark's resolution dates | Lower `model_cutoff` or use an earlier-resolving benchmark slice. |
| Relevance scoring re-embeds every FD despite cache | Prompt text changed for a source | Expected: fingerprint changed. Cache refills on the next run. |
| `publish` leaves old meta/ subdir | Stale from a pre-scrub build | Fixed in current `step_publish`; re-run `build_benchmark.py --skip-raw` to reinvoke scrub. |
| GDELT DOC API hanging / 429s | Rate-limited or network issue | Fetchers have 3-attempt retry with backoff; transient. For persistent issues, drop `--source nyt` or reduce worker count. |

---

## 11. Authors' notes / paper-ready citations

If you use this pipeline in published work, cite:

```bibtex
@misc{emrach_pipeline_2026,
  author       = {{EMR-ACH Authors}},
  title        = {EMR-ACH Benchmark Pipeline, v2.0 ("Forecast from Prior News")},
  year         = {2026},
  url          = {https://github.com/<org>/emr-ach/blob/main/docs/PIPELINE.md}
}
```

Key methodology references:

- **MIRAI** (Ye et al., 2024) — prior-work GDELT-based geopolitical forecasting benchmark; cited as external anchor only, not a shared methodology.
- **ForecastBench** (Karger et al., 2025) — prediction-market benchmark.
- **FactScore** (Min et al., 2023) — atomic claim decomposition (ETD).
- **ISO-TimeML** (ISO 24617-1:2012) — temporal annotation standard (ETD).
- **SBERT** (Reimers & Gurevych, 2019) — cross-match relevance backbone.

---

## 12. Change log

| Version | Date | Summary |
|---|---|---|
| 2.0 | 2026-04-22 | Unified "forecast from prior news" framing. Three benchmarks, 90-day analysis window, multi-source per-FD news fetch (NYT/Guardian/Google News/GDELT DOC/Finnhub/yfinance), stratified bootstrap CIs + MCC + NormBalAcc, ETD fact-extraction pipeline (docs/ETD_SPEC.md), publication-audit hardening. |
| 1.0 | 2026-04-15 | Initial release. Two benchmarks (FB + GDELT-CAMEO oracle), ad-hoc per-source fetch, Brier/ECE metrics. Superseded. |

See git history for per-commit changes.
