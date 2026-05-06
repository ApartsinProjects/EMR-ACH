# EMR-ACH v2.2 End-to-End Audit

**Status**: audit only, drafted 2026-04-23 by a fresh-eyes pass over the
v2.1 deliverable plus the v2.2 design + backlog. No code, config, or
existing v2.2 doc modified by this exercise.

**Companion documents** (read first; this audit refers to them rather than
duplicating their content):

- `docs/V2_2_ARCHITECTURE.md` (v2.2 design proposal, 13 sections)
- `docs/V2_2_REFACTOR_BACKLOG.md` (65 actionable items in 6 categories)
- `docs/PIPELINE.md` (v2.1 stage map)
- `docs/FORECAST_DOSSIER.md` (FD canonical schema)
- `docs/ETD_SPEC.md` (Event Timeline Dossier v1.0)

---

## Section 1. Scope and method

### 1.1 Why this audit exists

The previous v2.2 audit (also today) framed the work as "what should
v2.2 do differently from v2.1." That framing produced a strong design
doc and a usable backlog, but it left two structural blind spots:

1. **Stages that v2.1 has but v2.2 documentation never enumerates** (for
   example the `step_pool_audits` post-relevance audit pair, the
   `step_checksums` SHA256 sidecar, the `step_relink_gdelt` second pass,
   the ETD post-publish orchestrator). These are stages a clean-slate
   reproducer of v2.2 would not know to run because they sit in the
   v2.1 entrypoint and were never lifted into the v2.2 architecture
   diagram in `V2_2_ARCHITECTURE.md` Section 2.
2. **Items that shipped today (2026-04-23 PM) but post-date the design
   doc**. These include the `--strict-quotes` validator on
   `articles_to_facts.py`, the OpenAI embeddings backend at
   `src/common/openai_embeddings.py` plus its CLI driver
   `scripts/embed_pool_openai.py`, the ETD post-publish orchestrator at
   `scripts/etd_post_publish.py`, the Phase A `>=`-to-`>` validator
   recovery, the v3 anchor-date prompt, and the new Category F backlog
   items (B10 hybrid baseline and friends).

This audit walks the FULL pipeline end-to-end as a new contributor
would: download to publish to ETD post-publish to baselines to paper.
The outputs are a stage table (Section 2), gap and inconsistency
findings (Section 3), a reconciliation of newly-shipped pieces against
the v2.2 docs (Section 4), a reproducibility audit (Section 5), a
resumability matrix (Section 6), proposed v2.2 doc patches (Section 7),
proposed backlog additions (Section 8), and open questions for the
human (Section 9).

### 1.2 Method

For each script in `scripts/`, the audit:

1. Reads the file's docstring (if any), CLI surface (`add_argument`),
   and orchestration callers (mostly `scripts/build_benchmark.py:main`
   and `scripts/etd_post_publish.py:main`).
2. Records `inputs`, `outputs`, `config keys consumed`, `dependencies on
   earlier stage outputs`, `idempotency / resume`, and `whether v2.2
   docs cover it`.
3. Cross-references with the v2.2 architecture diagram in
   `V2_2_ARCHITECTURE.md` Section 2 and the backlog item list in
   `V2_2_REFACTOR_BACKLOG.md`.

The audit explicitly does NOT modify in-flight files
(`scripts/build_benchmark.py`, `scripts/compute_relevance.py`,
`scripts/annotate_prior_state.py`, `scripts/articles_to_facts.py`,
`configs/default_config.yaml`, `src/common/openai_embeddings.py`,
`scripts/embed_pool_openai.py`).

### 1.3 Notation

- `[gap]` flags a stage that exists in code but is missing from
  `V2_2_ARCHITECTURE.md` Section 2 or from `PIPELINE.md` Section 3.
- `[plan]` flags a stage that exists in v2.2 docs but is not yet shipped
  (correctly listed as P0/P1 in the backlog).
- `[ship]` flags a stage shipped today that the v2.2 docs predate.
- `[drift]` flags a contract mismatch between two scripts or between a
  script and its documented contract.

---

## Section 2. Stage table

The table walks the canonical clean-slate sequence. "Owner" is the file
authoritative for the stage. "v2.2 doc?" lists where the stage is
described in the v2.2 documentation set; "missing" means a clean-slate
reproducer would have to read source code to know the stage exists.

| # | Stage | Owner | Reads | Writes | Config keys | Resumable | v2.2 doc? |
|---|---|---|---|---|---|---|---|
| 0  | Raw GDELT KG download | `scripts/build_gdelt_cameo.py` | GDELT 15-min KG zips | `data/gdelt_cameo/data_kg.csv`, `data_news.csv`, `test/relation_query.csv` | `benchmarks.gdelt_cameo.{context_start,context_end,test_month,all_end,min_daily_mentions,max_download_workers}` | per-zip | yes (Stage 0 in §2) |
| 0a | Raw earnings build | `scripts/build_earnings_benchmark.py` | yfinance API | `data/earnings/earnings_forecasts.jsonl` | `benchmarks.earnings.{start,end,tickers,threshold}` | re-runs full | yes (Stage 0) |
| 0b | Raw ForecastBench download | `scripts/download_forecastbench.py` | upstream FB repo | `data/forecastbench/*` | env `EMRACH_FB_SUBJECT_FILTER` (default `all`; legacy `geopolitics`) | static clone | partial; the env var is undocumented in v2.2 docs |
| 1  | Unify articles | `scripts/unify_articles.py` | per-source `*_articles.jsonl` | `data/unified/articles.jsonl` | (none directly) | re-runs full | mentioned only in §8.2 (refactor target) |
| 1a | Unify forecasts | `scripts/unify_forecasts.py` | per-benchmark `*_forecasts.jsonl` | `data/unified/forecasts.jsonl` | env `EMRACH_FB_HORIZON_DAYS`, `EMRACH_GDELT_HORIZON_DAYS`, `EMRACH_EARN_HORIZON_DAYS` | re-runs full | mentioned only in §8.2 |
| 2  | Multi-source per-FD news fetch | `scripts/fetch_forecastbench_news.py`, `scripts/fetch_gdelt_cameo_news.py`, `scripts/fetch_earnings_news.py` | unified FDs | `data/{benchmark}/{benchmark}_articles.jsonl` | `secrets.*`, `pipeline.http_timeout_sec` | per-FD via `--skip-completed` | yes (§4 hybrid retrieval contract) |
| 2a | Re-unify after fetch | `step_unify` | per-source articles | `data/unified/articles.jsonl` | (none) | re-runs full | not explicit |
| 3  | Annotate prior state | `scripts/annotate_prior_state.py` | unified FDs + per-benchmark metadata | unified FDs (in-place) | `--benchmarks` CLI | re-runs full; in-place mutation | not in v2.2 design (carried forward implicitly) |
| 4  | Compute relevance | `scripts/compute_relevance.py` | unified FDs + articles | unified FDs (in-place `article_ids`), `data/unified/article_embeddings.npy`, `forecast_embeddings.npy`, `relevance_meta.json` | `relevance.{embedding_model,batch_size,max_text_chars,sources.*}` | per-row fingerprint cache | yes (§5 encode-once-score-many; planned refactor) |
| 5  | GDELT context relink | `scripts/relink_gdelt_context.py` | unified FDs + GDELT KG news lookup | unified FDs (in-place) | (none) | re-runs full | not in v2.2 architecture diagram |
| 5a | Fetch GDELT body text | `scripts/fetch_gdelt_text.py` | relinked URLs | augments unified articles with bodies | `pipeline.gdelt_text_fetch_workers`, `pipeline.http_timeout_sec` | per-URL skip on body present | called out only as a refactor target (§3.3 trafilatura) |
| 5b | Fetch ForecastBench body text | `scripts/fetch_article_text.py` | unified articles missing bodies | augments unified articles | `pipeline.text_fetch_workers` | per-URL skip on body present | A5 (parallelize) |
| 5c | Re-unify + re-relevance + second relink | `step_unify`, `step_relevance_parallel`, `step_relink_gdelt` | augmented articles | unified FDs (in-place) | (same as 4) | re-runs full | not in v2.2 architecture diagram |
| 6  | Quality filter | `scripts/quality_filter.py` | unified FDs | `data/unified/forecasts_filtered_*.jsonl`, drop reports | `quality.*`, `model_cutoff`, `cutoff_buffer_days` | re-runs full | yes (Stage 6+ in §2) |
| 7  | Diagnostic + EDA + pool audits | `scripts/diagnostic_report.py`, `scripts/build_eda_report.py`, `scripts/articles_audit.py`, `scripts/fd_audit.py` | unified FDs + articles | `data/unified/audit/*.md`, EDA HTML | (none) | re-runs full | not in v2.2 docs |
| 8  | Publish | `step_publish` in `scripts/build_benchmark.py` | filtered unified FDs + articles | `benchmark/data/{cutoff}/{forecasts,articles}.jsonl`, `benchmark.yaml`, `build_manifest.json`, `meta/` | `output.root` | re-runs full; scrubs stale dirs | yes (Stage 6+) |
| 8a | Checksums sidecar | `step_checksums` in same file | published deliverable | `benchmark/data/{cutoff}/checksums.sha256` | (none) | re-runs full | not in v2.2 docs |
| 9  | ETD Stage 1 (extract facts) | `scripts/articles_to_facts.py` | published `articles.jsonl` (or unified pool) | `data/etd/facts.v1.jsonl`, `facts.errors.jsonl`, `extract_runs.jsonl` | `secrets.openai_api_key`, prompt at `docs/prompts/etd_extraction_v3.txt`; CLI flags `--strict-dates`, `--strict-quotes`, `--only-articles`, `--prompt`, `--chunk-size`, `--run-id` | per-article via `(article_id, extract_run)` skip | partial (§7.4 mentions ETD Stages 2+3 explicitly out of scope; Stage 1 is in scope but its v3 prompt + `--strict-quotes` flag is not yet in v2.2 docs) |
| 10 | ETD Stage 2 (dedup) | `scripts/etd_dedup.py` | `facts.v1.jsonl` | `facts.v1_canonical.jsonl`, `dedup_meta.json` | `--threshold`, `--window-days`, `--batch-size` CLI | re-runs full; idempotent | NOT in v2.2 (intentionally OOS per §11) |
| 11 | ETD Stage 3 (FD link) | `scripts/etd_link.py` | `facts.v1_canonical.jsonl` (or `.v1.jsonl`) + published FDs | `facts.v1_linked.jsonl`, `link_meta.json` | `--cutoff`, `--use-stage1`, `--secondary-articles` CLI | rebuilds from scratch | NOT in v2.2 (OOS) |
| 12 | ETD production filter | `scripts/etd_filter.py` | `facts.v1_linked.jsonl` (or `.v1.jsonl`) | `facts.v1_production_{cutoff}.jsonl`, `filter_meta.json` | CLI: `--source-blocklist`, `--min-confidence`, `--polarity`, `--no-future`, `--require-linked-fd`, `--require-entities`, `--min-cluster-size`, `--benchmark`, `--max-date-skew-days` | re-runs full | NOT in v2.2 |
| 13 | ETD audit | `scripts/etd_audit.py` | filtered facts | `data/etd/audit/*.md` | (CLI in/out paths) | re-runs full | NOT in v2.2 |
| 14 | ETD facts-vs-articles compare | `scripts/etd_compare_facts_vs_articles.py` | facts + per-bench FDs | per-bench markdown + diffs | `--cutoff`, `--n`, `--bench` | re-runs full | NOT in v2.2 |
| 15 | ETD post-publish orchestrator | `scripts/etd_post_publish.py` | published bundle + Stage-1 facts | delta JSONL, then drives Stages 9-14 | `--cutoff`, `--skip-*` CLI | per-step `--skip-` re-entry | NOT in v2.2 docs (shipped today) |
| 16 | OpenAI embeddings (alt to SBERT) | `scripts/embed_pool_openai.py` + `src/common/openai_embeddings.py` | `data/unified/{articles,forecasts}.jsonl` | `data/unified/{article,forecast}_embeddings_openai.npy` + `.fp.txt`, `embeddings_openai_meta.json` | CLI: `--mode {sync,batch}`, `--model`, `--max-text-chars`, `--rebuild` | per-row fingerprint cache; batch resumes via `state.json` in `data/etd_openai_batches/{model}/` | partial (§6 design only; the shipped module diverges from §6.2 design, see Section 4.2 below) |
| 17 | Baselines run | `benchmark/evaluation/baselines/runner.py` + per-method modules `b1..b9`, `b3b_rag_claims` | published `forecasts.jsonl` + `articles.jsonl` (and ETD facts for `b3b`) | `benchmark/results/{cutoff}/{baseline}/...` | `benchmark/configs/baselines.yaml`, env API keys | resumable per-FD via cached raw files | NOT in v2.2 docs; new B10/B10b/F3/F4 items only in backlog Category F |
| 18 | Paper tables | `paper/index.html` (manual) | baseline metrics JSONs | `paper/index.html` | (none) | manual | partial (F4 backlog) |

20 stages total when 0a/0b/1a/2a/5a/5b/5c/8a are counted; 17 distinct
"stage IDs" if you collapse the parallel sub-stages. The v2.2 design doc
diagram in `V2_2_ARCHITECTURE.md` Section 2 shows 7 boxes (Stages 0
through 6+); a clean-slate reproducer reading only the v2.2 design will
miss Stages 0a, 0b, 1a, 2a, 5, 5a, 5b, 5c, 7, 8a, 9-15, 17, 18.

---

## Section 3. Cross-reference findings

This section enumerates concrete gaps, inconsistencies, and broken
sequences detected by walking the table above against the v2.2 docs.

### 3.1 Stages in code but missing from `V2_2_ARCHITECTURE.md` Section 2

[gap] **Stage 3 (`annotate_prior_state.py`)**. The v2.2 architecture
diagram jumps directly from "Stage 5: encode-once-score-many" to "Stage
6+: annotate_prior_state, quality, diagnostics, publish (unchanged)".
That phrasing is ambiguous about ordering: in v2.1 the annotator runs
**before** relevance, not after, because `annotate_prior_state.py`
mutates `forecasts.jsonl` in place to set `fd_type`/`prior_state_*` and
the relevance step does not depend on those fields. A reproducer reading
only the v2.2 doc could place the annotator after relevance and the
build would still appear to succeed. Action: see Section 7 patch to
§2 caption.

[gap] **Stage 5 (`relink_gdelt_context.py`) and Stage 5c (re-relevance
plus second relink)**. The v2.1 build runs the GDELT relink twice (once
after the first relevance pass, again after body fetch and re-unify),
because the second relink can pull in new pre-event articles that only
exist after the trafilatura body fetch. The v2.2 architecture diagram
shows zero relink steps. If a reproducer skips relink, GDELT-CAMEO
ends up with same-day oracle URLs and the `day_spread<2` quality drop
balloons (see `docs/PIPELINE.md` §10 troubleshooting row).

[gap] **Stages 5a and 5b body fetch (`fetch_gdelt_text.py`,
`fetch_article_text.py`)**. The v2.2 design discusses parallelizing the
body fetch (A5 in backlog) but does not name the two existing scripts
as separate stages. They emit no JSONL of their own; they augment
existing articles by filling the `text` field. A reproducer who
"replaces v2.1 cascade with the GDELT DOC index" per §3 needs to know
that the body-fetch step is still required for non-GDELT-DOC sources
and for the survivors of the index lookup.

[gap] **Stage 7 (audits and EDA)**. `articles_audit.py`, `fd_audit.py`,
`diagnostic_report.py`, `build_eda_report.py` all run between quality
filter and publish. They emit reports, not data, but they are guard
rails: a v2.1 build that fails an audit (e.g. zero `change` FDs in
earnings, see commit `8076a54`) catches the regression here, not after
publish. None of these are in the v2.2 design.

[gap] **Stage 8a checksums sidecar**. `step_checksums` writes
`checksums.sha256` next to the deliverable. v2.2 docs do not mention
it. If a future v2.2 publishes the deliverable through a refactored
publisher and forgets the checksum, downstream consumers lose the byte-
identical-deliverable guarantee that `C2` (v2.2/v2.1 parity test) is
supposed to verify.

[gap] **Stages 9-15 ETD post-publish**. `V2_2_ARCHITECTURE.md` §11
explicitly lists "ETD Stages 2 plus 3 redesign" as out of scope, and
that is the right call. But the post-publish orchestrator
(`scripts/etd_post_publish.py`, shipped today) sequences Stage 1 delta
extract, Stage 2 dedup, Stage 3 link, production filter, audit, and
facts-vs-articles compare in one CLI. A v2.2 reproducer reading only
the v2.2 design doc will not know this orchestrator exists; reading
the backlog item E13 reveals it but not its CLI surface or input
contract. This is the single biggest "stage exists in v2.1 today but
v2.2 docs forgot" item.

[gap] **Stages 17-18 baselines and paper tables**. The v2.2 design is
explicit (Section 1.4) that this is a build-system refactor, not a
benchmark redesign, so baselines are correctly out of scope for the
architecture doc. But Category F was added to the backlog and
introduces four new baselines (B10 hybrid, B10b facts-only, F3 EMR-ACH
facts-as-rows, F4 paper Table 3). These depend on the ETD post-publish
artefact at `data/etd/facts.v1_production_{cutoff}.jsonl`, which is
itself only described in the etd_post_publish source code. The chain
`Stage 9 -> Stage 11 -> Stage 12 -> Stage 17 (B10)` is implicit in
backlog F1 but not stated as a sequence anywhere.

### 3.2 Stages in v2.2 docs but not in code (correctly P0/P1 backlog)

[plan] **Stage 1 GDELT DOC archive download (§3.1)**. Backlog A1, P0.
Not yet shipped; correctly tracked.

[plan] **Stage 2 GDELT DOC index build (§3.2)**. Backlog A2, P0. Not
yet shipped.

[plan] **Stage 3 GDELT DOC index lookup (§3.3)**. Backlog A3, P0. Not
yet shipped.

[plan] **Encode-once-score-many in `compute_relevance.py` (§5)**.
Backlog A4, P0. Not yet shipped (the in-flight `compute_relevance.py`
still uses sequential `--benchmark-filter` calls).

[plan] **`src/common/embeddings_backend.py` unified encoder API (§6)**.
Backlog A6, P0. Not shipped under that name. **Today's ship** at
`src/common/openai_embeddings.py` is a distinct module with a different
API surface (see Section 4.2). The backlog A6 still needs to land as
the unifying API; A7 then becomes "wrap the existing
`openai_embeddings.py` behind A6's API."

[plan] **Hybrid retrieval contract module `src/retrieval/contract.py`
(§4)**. Backlog B1, P0. Not yet shipped.

[plan] **`src/common/gdelt_aggregator_domains.py`**. Backlog B8, P0.
Not yet shipped.

[plan] **Per-stage config slices and config-hash (§7.1)**. Backlog B5,
P1. Not yet shipped.

[plan] **`src/common/paths.py`, `src/unify/`, `src/common/news_fetcher.py`,
`src/common/layout.py`, `src/etd/date_validators.py`**. All correctly
P1 in backlog category B/E.

### 3.3 Path / contract mismatches between scripts

[drift] **`compute_relevance.py` legacy config fallback**. The script
at `scripts/compute_relevance.py:43-47` reads from `CONFIG_LEGACY =
configs/relevance.yaml` if the file exists. That file does not exist in
the current tree. The fallback is dead code today; backlog B11 / E3
already flag this. No data hazard, just confusion for a clean-slate
reader.

[drift] **`unify_articles.py` art_id prefix versus per-fetcher
prefixes**. The unifier emits ids of form `art_<sha1>` (see
`docs/PIPELINE.md` §5.3 example), while fetchers emit `fbn_`, `gdc_`,
`earn_` for their per-source articles. The two ID spaces co-exist in
`data/unified/articles.jsonl` (the unifier preserves the per-source IDs
as the canonical id when the article already has an id). A reproducer
who naively assumes all article ids start with `art_` will write broken
joins. Backlog E5 notes this.

[drift] **ETD Stage 1 reads from `data/unified/articles.jsonl`, not
from the published `benchmark/data/{cutoff}/articles.jsonl`**.
`scripts/articles_to_facts.py:61` hardcodes `ARTICLES_FILE = DATA /
"unified" / "articles.jsonl"`. The post-publish orchestrator
(`scripts/etd_post_publish.py:_step_compute_delta`) instead reads from
the published bundle. So a delta extraction (Stage 9 driven by
post-publish) joins published articles (~120k) against Stage-1 already-
covered (~17k of 216k unified). Articles that the unify pool contains
but quality filter dropped are NOT in the delta and never get extracted,
which is correct for production but surprising for a debugger. The
contract is implicit; should be documented.

[drift] **`etd_post_publish.py:--prompt` defaults to v3 but
`articles_to_facts.py:--prompt` defaults to v1**.
`scripts/etd_post_publish.py:145` hard-codes `default=
"docs/prompts/etd_extraction_v3.txt"`. `scripts/articles_to_facts.py:63`
hard-codes `PROMPT_PATH = PROMPT_DIR / "etd_extraction_v1.txt"`. A
reproducer who runs `articles_to_facts.py` directly (without going
through the orchestrator) gets the v1 prompt and the 15.5%-unsupported
hallucination floor instead of the v3 production 11.8%. This is a real
production hazard, not just a documentation gap.

[drift] **Stage 3 link script reads `STAGE2 = facts.v1_canonical.jsonl`
but the post-publish orchestrator does NOT pass any flag to switch to
Stage 1**. `scripts/etd_link.py:79` falls back to `STAGE1` only if
`STAGE2` is missing. If `etd_dedup.py` has ever produced a
`facts.v1_canonical.jsonl` and then someone deletes `facts.v1.jsonl`
or appends to it without re-deduping, the linker silently uses the
stale canonical file. Atomicity of the canonical file is not enforced
beyond `_atomic_write`; there is no `(stage1_sha, stage2_sha)` pin.

[drift] **Filter in/out path branching in
`scripts/etd_post_publish.py`**. Lines 204-210 build the filter input as
`facts.v1_linked.jsonl` if it exists, else `facts.v1.jsonl`. This means
a build that runs `--skip-link` will silently filter the unlinked Stage-
1 facts and tag them as "production". `--require-linked-fd` is
correctly NOT added when `--skip-link` is set (line 212), but the file
name `facts.v1_production_{cutoff}.jsonl` is identical regardless of
whether linkage was applied. A downstream consumer cannot tell the two
artefacts apart from the filename alone.

[drift] **ETD post-publish does not re-run Stage 2 dedup automatically
on a delta extract**. After `articles_to_facts.py --only-articles
delta_{cutoff}.jsonl`, new facts are appended to `facts.v1.jsonl` but
NOT yet clustered against the existing canonical. The orchestrator does
call Stage 2 next (line 185), which reads the WHOLE updated
`facts.v1.jsonl` and re-derives canonical clusters. That is correct,
but Stage 2 dedup re-encodes every fact (~80-90k) on every post-publish
run; there is no incremental path. ETD spec §6.6 says Stage 2 is
idempotent but does not say it is incremental.

### 3.4 Newly-shipped items not yet in v2.2 docs

These are covered in detail in Section 4 below.

### 3.5 Hidden v2.1 mutations (in-place writes)

The pipeline has multiple in-place mutating stages. v2.2 §7 explicitly
calls for snapshot semantics; the current state is:

- **`unify_forecasts.py`** writes `data/unified/forecasts.jsonl` from
  scratch. NOT in-place. Safe.
- **`annotate_prior_state.py`** mutates `data/unified/forecasts.jsonl`
  IN PLACE, adding `prior_state_*`, `fd_type`, and binary-promoting
  `ground_truth` to Comply/Surprise. Resumability: re-running is safe
  (idempotent on already-annotated FDs) but a partial crash mid-write
  can leave some FDs annotated and others not. No `.tmp + rename`
  guarantee inspected in source.
- **`compute_relevance.py`** mutates `forecasts.jsonl` in place to set
  `article_ids`. Same partial-crash hazard.
- **`relink_gdelt_context.py`** mutates `forecasts.jsonl` in place to
  replace `article_ids` with pre-event context articles for GDELT-CAMEO
  FDs.
- **`fetch_gdelt_text.py`, `fetch_article_text.py`** mutate
  `articles.jsonl` in place by adding `text` bodies. This is the most
  expensive in-place mutation; a partial crash means some articles
  have bodies and some do not, and the only signal is the `text` field
  being empty.

This bundle of in-place mutations is what makes the pipeline so hard to
"start from a known-good intermediate". V2.2 backlog B6 (atomic stage-
meta writes) is necessary but not sufficient; the data files themselves
need atomic writes, not just the meta files. Recommend new backlog item
in Section 8.

---

## Section 4. Newly-shipped items not yet in v2.2 docs

The four-paragraph summaries below describe each shipped item and
propose where it should land in the v2.2 docs.

### 4.1 `--strict-quotes` validator on `scripts/articles_to_facts.py`
(commit `9cbe9a1`)

[ship] Strategy 3 evidence-quote validator: when `--strict-quotes` is
set, every fact must carry an `evidence_quote` field whose contents are
a verbatim substring of the article body. Implementation at
`scripts/articles_to_facts.py:344-362` and CLI at line 446. Prompt
support is in `docs/prompts/etd_extraction_v3.txt:54,81` (the v3 prompt
solicits `evidence_quote` per fact). The flag is NOT yet wired into
`scripts/etd_post_publish.py`, so the orchestrator currently runs delta
extracts without strict-quotes verification.

**Proposed home in v2.2 docs**:
- `V2_2_ARCHITECTURE.md` §11 ("Out of scope") add a sentence noting
  ETD Stage 1 prompt v3 plus `--strict-quotes` is now in v2.1 and the
  v2.2 build does not change it.
- `V2_2_REFACTOR_BACKLOG.md` Category C: add C7b "Wire `--strict-quotes`
  into `etd_post_publish.py` delta extract by default; expose as a
  `--no-strict-quotes` opt-out." P1.

### 4.2 v2.2 OpenAI embeddings backend (commit `9a27816`)

[ship] `src/common/openai_embeddings.py` (299 lines) plus
`scripts/embed_pool_openai.py` (149 lines). Implements sync and batch
embedding paths against `text-embedding-3-small` with an L2-normalized
float32 cache schema-compatible with the SBERT cache (different file
suffix `*_openai.npy`).

**Divergence from `V2_2_ARCHITECTURE.md` §6 design**: the design calls
for a unified `src/common/embeddings_backend.py` with signature
`encode(texts, model, backend, dim=768)`, normalizing to 768 by
truncate-then-renormalize. The shipped module is named
`openai_embeddings.py`, has no SBERT path, and uses the model's native
dimension (1536 for `-small`, 3072 for `-large`) without truncation to
768. The cache files therefore have a different shape than the SBERT
cache, so a `compute_relevance.py` consumer cannot transparently
substitute one for the other.

**Implication for backlog A6/A7**: A6 (the unified backend) is still
not landed; A7 (OpenAI Batch backend) has effectively been shipped
under a different module name. The backlog should be updated to
reflect that A7 partially landed and A6 remains the wrapping work.

**Implication for cache-invalidation matrix (§7.2)**: the matrix says
"backend changed -> cache invalidated." The shipped module emits
distinct file names (`*_openai.npy`), so backend swaps do NOT
invalidate the SBERT cache; instead, two parallel caches co-exist.
That is arguably a better design than the doc, but it is undocumented.

**Proposed home in v2.2 docs**:
- `V2_2_ARCHITECTURE.md` §6.2 add a paragraph noting that the shipped
  module uses native dim (1536) and writes to a parallel cache file
  rather than a unified API. Update §7.2 cache row accordingly.
- `V2_2_REFACTOR_BACKLOG.md` A7 mark as "partially landed (commit
  `9a27816`); remaining work is to wrap behind A6 unified API and
  decide on the dim-projection question."

### 4.3 Category F baselines (added to backlog today, commit `9cbe9a1`)

[ship] Backlog Category F (items F1-F4) was added today. F1 (B10
hybrid baseline) and F4 (paper Table 3 update) are P1; F2 (B10b facts-
only) and F3 (EMR-ACH facts-as-rows) are P2.

**Architectural implication**: B10/B10b consume the production-filtered
ETD facts at `data/etd/facts.v1_production_{cutoff}.jsonl`. That file
is only produced if `etd_post_publish.py` has been run. The chain is:
v2.1 build -> publish -> ETD post-publish -> baseline run. The v2.2
architecture diagram in §2 ends at "publish"; the baseline arc is not
shown. Even though baselines are explicitly out of scope per §1.4, the
NEW Category F means the v2.2 deliverable contract effectively now
includes the production facts file, which is currently produced by a
script (`etd_post_publish.py`) that v2.2 does not document.

**Proposed home in v2.2 docs**:
- `V2_2_ARCHITECTURE.md` §1.4 ("What v2.2 explicitly does not change"):
  add note that the deliverable contract now optionally includes
  `data/etd/facts.v1_production_{cutoff}.jsonl` as a sibling artefact
  consumed by B10/B10b.
- `V2_2_ARCHITECTURE.md` §11 ("Out of scope"): clarify that Category F
  baselines are a downstream consumer of the v2.1+ETD pipeline, not a
  v2.2 build-system change.

### 4.4 ETD post-publish orchestrator `scripts/etd_post_publish.py`
(commit `e22395c`)

[ship] 254-line orchestrator that sequences delta-compute -> Phase D
extract -> Stage 2 dedup -> Stage 3 link -> production filter -> audit
-> facts-vs-articles compare per benchmark. Each step independently
`--skip-`able. Output: `data/etd/facts.v1_production_{cutoff}.jsonl`
plus per-bench audits in `data/etd/audit/`.

**Why it matters for v2.2**: this orchestrator is the bridge between
the v2.1 published bundle and the Category F baselines. It is
referenced once in `V2_2_REFACTOR_BACKLOG.md` E13 ("Listed for
completeness. Not a v2.2 work item; v2.1 closed it.") but its CLI
contract, default flags, and file outputs are not documented anywhere
except the source file's docstring.

**Proposed home in v2.2 docs**: a new §3.5 in `docs/PIPELINE.md` titled
"ETD post-publish (after the deliverable is published)" describing
the orchestrator's stage sequence, default flags, file outputs, and
how to re-enter via `--skip-` flags after a partial failure.

### 4.5 ETD Phase A validator fix (commit `7237553`)

[ship] `scripts/articles_to_facts.py:230` (now `:341`) was changed
from `>=` to `>`, recovering 8,647 same-day facts that had been
wrongly rejected as leakage. The same-day facts are legitimate news
of the day; real leakage protection lives at experiment time in
`apply_experiment_horizon()` in the baselines runner.

**Status**: documented in `project_emrach.md` "Late-cycle bug fixes"
and in `V2_2_REFACTOR_BACKLOG.md` B7/E4. The fix is shipped; the
backlog item B7 (`src/etd/date_validators.py` extraction with unit
tests) is correctly P1 to prevent future regressions.

### 4.6 ETD Phase C v3 prompt + production filter recipe (commits
`231ef99`, `28f56fd`)

[ship] v3 prompt `docs/prompts/etd_extraction_v3.txt` adds anchor-date
table, verbatim grounding, and `evidence_quote` field. Production
filter recipe: `--source-blocklist news.fjsen.com,world.people.com.cn
--min-confidence high --polarity asserted --no-future
--require-linked-fd`. Hardcoded into `etd_post_publish.py:51` as
`DEFAULT_BLOCKLIST` and lines 204-213.

**Documentation status**: documented in `project_emrach.md` "Phase C
ETD prompt revisions" and in backlog C10 ("Production filter recipe
in `build_manifest.json`"). C10 is P1 and not yet implemented; the
recipe lives only in the orchestrator source.

**Proposed home**: backlog C10 should be elevated to P0 because every
v2.2-published deliverable risks being non-reproducible without the
recipe pinned in the manifest.

---

## Section 5. Reproducibility audit

### 5.1 Random seeds

| Stage | Random seed pinned? | Where |
|---|---|---|
| 0 GDELT KG download | not applicable | network only |
| 1 unify | not applicable | deterministic merge |
| 2 multi-source fetch | NOT pinned | per-source RSS / API non-determinism |
| 4 compute_relevance | not applicable (deterministic given embeddings) | SBERT encode is deterministic at fp32; FP16 introduces tiny noise |
| 9 ETD Stage 1 | OpenAI temperature=0 (`articles_to_facts.py:268`) | model-version pinned |
| 10 ETD Stage 2 | not applicable (clustering deterministic given embeddings) | |
| 17 baselines | per-baseline (some methods sample; e.g. b4 self-consistency) | `benchmark/configs/baselines.yaml` defaults `temperature: 0.0` |

**Gap**: no global `--seed` plumbed through. For Category F baselines
the sampling baselines (b4, b9, etc.) need explicit seed control to
reproduce a published run.

### 5.2 Model SHAs

| Component | Pinned? | How |
|---|---|---|
| SBERT encoder | model name only (`sentence-transformers/all-mpnet-base-v2`) | `configs/default_config.yaml:91` |
| OpenAI Stage-1 extractor | model id with date suffix (`gpt-4o-mini-2024-07-18`) | `articles_to_facts.py:424` default |
| OpenAI embeddings | model name only (`text-embedding-3-small`) | `src/common/openai_embeddings.py:43` |
| Baselines model | model id with date suffix | `benchmark/configs/baselines.yaml:21` |

**Gap**: SBERT model revision (HF commit hash) is NOT pinned; a
`sentence-transformers` package upgrade could change the underlying
model bytes silently. v2.2 §3.2 calls for `model_revision` in the
GDELT DOC index manifest; the same should apply to
`relevance_meta.json` and the build manifest.

**Gap**: OpenAI embeddings module pins model NAME only, not the
`embedding_dim` parameter; if OpenAI changes the default native dim,
cached vectors become incompatible.

### 5.3 Prompt SHAs

`scripts/articles_to_facts.py` writes `extract_runs.jsonl` per
`docs/ETD_SPEC.md` §3.3 with a `prompt_sha1` field. **Need to verify
this field is actually written** by the current implementation (the
spec calls for it; grep was not exhaustive). Baselines: prompt strings
in `benchmark/evaluation/baselines/prompts.py` are not SHA'd into
results.

**Gap**: baseline prompt SHA is not pinned per result file. A change
to `prompts.py:_BASE_USER` between runs would silently produce
incomparable results.

### 5.4 Config snapshots

`step_publish` writes `benchmark.yaml` (effective config) and
`build_manifest.json` (provenance). Per `V2_2_REFACTOR_BACKLOG.md`
C3, the manifest needs to gain `embedding_backend`,
`embedding_model`, `embedding_model_revision`. Per C10, the manifest
needs the production filter recipe. Neither is shipped.

**Gap**: `benchmark.yaml` does not snapshot the env vars
(`EMRACH_FB_SUBJECT_FILTER`, `EMRACH_FB_HORIZON_DAYS`,
`EMRACH_GDELT_HORIZON_DAYS`, `EMRACH_EARN_HORIZON_DAYS`) that drive
unify behavior. A reproducer who replays the same `benchmark.yaml`
without setting these env vars gets a different deliverable. Recommend
backlog item: capture env-var snapshot into the manifest.

### 5.5 Output checksums

`step_checksums` writes `forecasts.jsonl` and `articles.jsonl` SHA256
sums. **Gap**: `benchmark.yaml`, `build_manifest.json`, and
`forecasts_change.jsonl`/`forecasts_stability.jsonl` are NOT included
in the checksum sidecar. A consumer cannot verify the secondary
artefacts against the canonical sums.

---

## Section 6. Stage-by-stage resumability matrix

| Stage | Safe to kill mid-run? | Resume mechanism | Hazards |
|---|---|---|---|
| 0 GDELT KG download | yes | per-zip; `--steps` CLI bypasses already-done steps | network restart |
| 0a Earnings build | mostly | reruns full window; cheap | yfinance rate limits |
| 1 Unify articles | NO (full overwrite) | none; re-runs full | mid-write crash leaves partial JSONL |
| 1a Unify forecasts | NO | none; re-runs full | same |
| 2 Multi-source fetch | yes | per-FD via `--skip-completed`; per-source independent | partial fetch leaves FD with subset of sources; next `--skip-completed` skips it as "done" though it has fewer than full coverage |
| 3 Annotate prior state | NO (in-place mutation) | re-runs full; idempotent on already-annotated | mid-write leaves mixed state |
| 4 Compute relevance | partial (per-row fingerprint cache survives crashes; in-place forecast mutation does not) | embedding cache + per-row fingerprint | forecasts.jsonl mid-write |
| 5 Relink GDELT | NO (in-place mutation) | re-runs full | mid-write leaves mixed state |
| 5a Body fetch GDELT | yes | per-URL skip if `text` populated | partial means some articles have body, some not; downstream filter drops the unbodied ones |
| 5b Body fetch FB | yes | same | same |
| 6 Quality filter | NO (full overwrite) | re-runs full | none significant |
| 7 Audits | yes | re-runs full; cheap | none |
| 8 Publish | NO (atomic copy + scrub) | re-runs full | scrub removes stale subdirs |
| 8a Checksums | yes | re-runs full | none |
| 9 ETD Stage 1 | yes | `(article_id, extract_run)` skip; OpenAI Batch resumable via `BatchClient` | partial batch leaves some articles uncovered; next run picks them up |
| 10 ETD Stage 2 | NO (full overwrite of canonical) | re-runs full; idempotent | none significant |
| 11 ETD Stage 3 | NO (full overwrite of linked) | re-runs full | none significant |
| 12 ETD filter | NO | re-runs full | overwrites production file silently |
| 13 ETD audit | yes | re-runs full | none |
| 14 ETD compare | yes | per-bench independent; OpenAI calls cached by `BatchClient` | partial leaves missing per-bench artefact |
| 15 ETD post-publish (orchestrator) | yes | per-step `--skip-` re-entry | dedup re-encodes full pool every run (~10 min on RTX 2060) |
| 16 OpenAI embeddings | yes | batch resumable via `state.json`; per-row fingerprint cache | none significant |
| 17 Baselines | yes | per-FD raw response cache | none significant |

**Key observation**: 7 of 21 stages are NOT safe to kill mid-run because
they do in-place writes without `.tmp + rename`. Backlog B6 covers
meta files but not data files. Recommend extending B6.

---

## Section 7. Recommendations to update `V2_2_ARCHITECTURE.md`

These are proposed diffs. They are described, not applied.

### 7.1 §2 (Pipeline shape)

**Diff**: extend the diagram to show Stage 5 (relink GDELT) as a
separate box between Stage 4 (relevance) and Stage 6 (annotate prior
state), with an arrow indicating it runs twice (once after first
relevance, once after re-relevance). Add a parenthetical to the Stage
6+ box: "(annotate_prior_state runs BEFORE relevance for FB and
earnings; AFTER relevance for GDELT-CAMEO; sequenced by
build_benchmark.py main())". This corrects the ambiguity flagged in
Section 3.1.

### 7.2 §3.3 (`query_gdelt_doc_index.py`)

**Diff**: under "Lookup flow" step 5, add a sub-bullet noting that
body fetch is delegated to `fetch_article_text.py` plus
`fetch_gdelt_text.py` (existing scripts), running with
`pipeline.text_fetch_workers` and `pipeline.gdelt_text_fetch_workers`
from config (currently 24 each). The current §3.3 text says "internal
ThreadPoolExecutor over trafilatura"; that overlaps with the v2.1
scripts and risks two parallel implementations.

### 7.3 §6.2 (OpenAI Batch schema-compatible drop-in)

**Diff**: add a paragraph noting that `src/common/openai_embeddings.py`
shipped today as a standalone module that uses the model's native dim
(1536 for `-small`, 3072 for `-large`) and writes to parallel cache
files (`*_openai.npy`) rather than truncating to 768 and unifying. The
A6 unifying backend remains future work.

### 7.4 §7.1 (Snapshot semantics)

**Diff**: extend snapshot semantics to cover DATA files, not just meta
files. Specifically: every script that mutates `forecasts.jsonl` or
`articles.jsonl` in place must write to `.tmp` then atomic-rename. This
is a strengthening of the current "X.meta.json is written last and
atomically" rule.

### 7.5 §7.2 (Cache invalidation matrix)

**Diff**: add a row: "OpenAI embeddings cache (`*_openai.npy` plus
`.fp.txt`) - invalidates when fingerprint mismatch (text changed),
embedding model changed, native-dim changed (model upgrade)." Note
parallel coexistence with SBERT cache.

### 7.6 §11 (Out of scope)

**Diff**: add three explicit "is in v2.1 today, not changed by v2.2"
items: (a) ETD Stage 1 v3 prompt and `--strict-quotes` validator,
(b) ETD post-publish orchestrator at `scripts/etd_post_publish.py`,
(c) Category F baselines (B10/B10b/F3/F4) which are downstream
consumers of the v2.1+ETD pipeline.

### 7.7 New §14 (post-publish addenda)

**Diff**: new section titled "Post-publish addenda". Two paragraphs.
First paragraph: ETD post-publish orchestrator drives the chain
delta-compute -> Phase D extract -> Stage 2 dedup -> Stage 3 link ->
production filter -> audit -> facts-vs-articles compare. Reference
`scripts/etd_post_publish.py:--cutoff` and the per-step `--skip-`
flags. Second paragraph: Category F baselines (B10/B10b/F3/F4)
consume `data/etd/facts.v1_production_{cutoff}.jsonl` and run via
`benchmark/evaluation/baselines/runner.py` per
`benchmark/configs/baselines.yaml`. Reference backlog F1-F4.

---

## Section 8. Recommendations to add items to `V2_2_REFACTOR_BACKLOG.md`

Each item below is a new backlog entry, in the existing category /
priority format.

### 8.1 Category B additions (architecture)

**B15. Atomic data-file writes for in-place mutators.**
Files: `scripts/annotate_prior_state.py`, `scripts/compute_relevance.py`,
`scripts/relink_gdelt_context.py`, `scripts/fetch_gdelt_text.py`,
`scripts/fetch_article_text.py`. Effort: M. Priority: P0. Deps: none.
Every script that mutates `forecasts.jsonl` or `articles.jsonl` in
place writes to `.tmp` then atomic-rename. Strengthens B6 (which
currently covers meta files only).

**B16. Snapshot env vars into `build_manifest.json`.**
Files: `scripts/build_benchmark.py:write_run_manifest`. Effort: S.
Priority: P0. Deps: none. Capture `EMRACH_FB_SUBJECT_FILTER`,
`EMRACH_FB_HORIZON_DAYS`, `EMRACH_GDELT_HORIZON_DAYS`,
`EMRACH_EARN_HORIZON_DAYS` (and any future EMRACH_*) into the
manifest. Today's `benchmark.yaml` is config-only and a reproducer
who omits the env vars gets a different deliverable.

**B17. SBERT model revision pinning.**
Files: `scripts/compute_relevance.py`, `scripts/etd_dedup.py`,
`scripts/embed_pool_openai.py`. Effort: S. Priority: P1. Deps: none.
Read the underlying HF model commit SHA and write into
`relevance_meta.json` / `dedup_meta.json` / `embeddings_openai_meta.json`.
Mirrors the GDELT DOC index manifest pattern in §3.2.

**B18. ETD Stage 2 incremental dedup.**
Files: `scripts/etd_dedup.py`. Effort: M. Priority: P2. Deps: B7.
Per-fact fingerprint cache so re-running Stage 2 after a delta extract
only encodes the new facts. Currently re-encodes all 80-90k facts
every run (~10 min on RTX 2060), which dominates orchestrator wall
clock.

### 8.2 Category C additions (quality and rigor)

**C7b. Wire `--strict-quotes` into `etd_post_publish.py`.**
Files: `scripts/etd_post_publish.py:_run` for the extract step. Effort:
S. Priority: P1. Deps: none. Add `--strict-quotes` to the default
delta-extract command; expose `--no-strict-quotes` as opt-out. The
v3 prompt solicits the field; the validator should be on by default.

**C11. Extend `step_checksums` to all deliverable files.**
Files: `scripts/build_benchmark.py:step_checksums`. Effort: S.
Priority: P1. Deps: none. Add `benchmark.yaml`, `build_manifest.json`,
`forecasts_change.jsonl`, `forecasts_stability.jsonl` to the
`checksums.sha256` sidecar.

**C12. Baseline prompt SHA pinning.**
Files: `benchmark/evaluation/baselines/runner.py`,
`benchmark/evaluation/baselines/prompts.py`. Effort: S. Priority: P2.
Deps: none. Compute `prompts.py:SHA1` at run time and write into each
baseline's `metrics_*.json`. Detects silent prompt drift between runs.

**C13. Global `--seed` plumbed through baselines.**
Files: `benchmark/evaluation/baselines/runner.py` plus per-method
modules that sample. Effort: S. Priority: P2. Deps: none.

### 8.3 Category D additions (documentation)

**D8. Document `etd_post_publish.py` in `docs/PIPELINE.md` §3.5.**
Effort: S. Priority: P0. Deps: none. The orchestrator is the only
documented bridge between v2.1 publish and Category F baselines; its
CLI surface and stage sequence belong in PIPELINE.md.

**D9. Document the prompt-default mismatch between
`articles_to_facts.py` (defaults to v1) and `etd_post_publish.py`
(defaults to v3).** Effort: S. Priority: P1. Deps: none. Either
document the rationale or change one default; current state risks
production runs accidentally using v1.

**D10. Document the `EMRACH_*` env-var surface.** Effort: S. Priority:
P1. Deps: none. Single table in `docs/PIPELINE.md` listing
`EMRACH_FB_SUBJECT_FILTER`, `EMRACH_FB_HORIZON_DAYS`,
`EMRACH_GDELT_HORIZON_DAYS`, `EMRACH_EARN_HORIZON_DAYS` with default,
where read, what it controls.

### 8.4 Category E additions (drift cleanup)

**E19. `etd_link.py` Stage-2 file freshness check.** Effort: S.
Priority: P1. Deps: none. Currently the linker silently uses
`facts.v1_canonical.jsonl` if it exists, even if `facts.v1.jsonl` is
newer. Add a `(stage1_mtime, stage2_mtime)` pin or a fingerprint
check that fails fast if Stage 2 is stale.

**E20. ETD post-publish production-filename collision.** Effort: S.
Priority: P1. Deps: none. The output file
`facts.v1_production_{cutoff}.jsonl` is identical regardless of
whether `--skip-link` was set. Either suffix the filename
(`_unlinked.jsonl`) or refuse to write production output without
linkage.

### 8.5 Category F additions (baselines)

**F5. Pin production filter recipe into `build_manifest.json`.**
Files: `scripts/build_benchmark.py:step_publish` plus
`scripts/etd_post_publish.py`. Effort: S. Priority: P0. Deps: none.
Mirrors backlog C10 but elevates priority because B10/B10b
reproducibility depends on the recipe.

---

## Section 9. Open questions for the human

1. **ETD post-publish: should it be folded into `build_benchmark.py`,
   or stay separate?** Today the build pipeline ends at publish; the
   ETD post-publish chain is a separate CLI. Folding it in would let
   one entrypoint produce the full Category F-ready deliverable;
   keeping it separate preserves the v2.2 §1.4 scope boundary
   ("v2.2 is a build-system refactor, not a benchmark redesign").
   Recommendation: keep separate, but document the chain as the
   canonical "after the build, run the ETD post-publish" step.

2. **OpenAI embeddings dim policy (768 vs native).** The v2.2 design
   §6.2 calls for truncate-to-768 to keep cache compatibility with
   SBERT. The shipped module uses native (1536/3072) and writes a
   parallel cache. Should A6 (the unified backend) try to harmonize
   the two, or should the parallel-cache approach be ratified?

3. **`articles_to_facts.py --prompt` default.** Currently v1
   (production was reverted from v3b to v3 today). Should the default
   move to v3? Recommendation: yes, with a `--prompt v1` opt-in for
   replication of older runs.

4. **Backlog C10 priority.** Currently P1. Given that B10/B10b
   reproducibility depends on the production filter recipe being
   pinned into the manifest, should this be elevated to P0?

5. **Stage 5 (relink GDELT) future.** Once the GDELT DOC index path
   (§3.3) lands and provides editorial pre-event news, is the
   `relink_gdelt_context.py` script obsolete? The v2.2 design implies
   yes ("replaces ~70% of per-FD HTTP scraping") but does not state
   it explicitly. A clean-slate v2.2 reproducer could leave the
   relinker out and the build would silently skip the second-pass
   pre-event context replacement.

6. **Atomic in-place writes (B15 above).** This is a significant
   change to multiple long-running scripts. Is the scope worth the
   resume-safety win, or is the lower-effort "rebuild from snapshot"
   approach via the existing `data/staged/` snapshots sufficient?
   Recommendation: yes, atomic writes are worth it; the staged
   snapshots are coarse-grained (one per stage) and a partial-stage
   crash today still leaves the live `forecasts.jsonl` in a mixed
   state.

7. **Baseline reproducibility (C12, C13).** Are these required for the
   paper, or only for the long-term archive? Recommendation: required
   for paper, because Category F adds new baselines (B10/B10b) whose
   results need to be cite-stable.

8. **SEC EDGAR is documented in PIPELINE.md but not in the v2.2
   hybrid retrieval contract table (§4).** The v2.2 §4 table lists
   "SEC EDGAR (filings)" as primary for earnings, which matches v2.1.
   But `scripts/fetch_earnings_news.py:fetch_sec_edgar` was wired in
   only on 2026-04-23 (commit `6540ccb`). Are EDGAR filings already
   in the published 2026-01-01 deliverable, or only in subsequent
   builds? A footnote in §4 would help reproducers.

---

**End of audit.**
