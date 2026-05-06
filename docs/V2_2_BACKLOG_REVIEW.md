# EMR-ACH v2.2 Refactor Backlog Review

**Status**: review only, drafted 2026-04-23. No code, config, or backlog
modified by this exercise. Reviews [`docs/V2_2_REFACTOR_BACKLOG.md`](V2_2_REFACTOR_BACKLOG.md)
(75 items across categories A through G, last updated commit `57abb8d`).

**Companion documents** (read in this order):
- [`docs/V2_2_ARCHITECTURE.md`](V2_2_ARCHITECTURE.md) (the design doc the backlog supports)
- [`docs/V2_2_END_TO_END_AUDIT.md`](V2_2_END_TO_END_AUDIT.md) (the gaps the backlog is meant to close)

---

## Section 1. Method

### 1.1 What I checked

For each of the 75 items I evaluated:

1. **Validity**: read the file the item references (when listed) and
   verified the described problem actually exists in the current code.
2. **Spec quality**: confirmed presence of file paths, an implied
   acceptance criterion, and dependency declarations.
3. **Priority sanity**: cross-referenced against §1.4 of
   `V2_2_ARCHITECTURE.md` (what v2.2 explicitly does or does not change)
   and the migration plan in §9.
4. **Dependencies**: verified listed deps exist as backlog IDs and
   checked for hidden dependencies between items.
5. **Duplication**: compared item titles, file lists, and effort tags
   for overlap.
6. **Effort estimates**: sanity-checked S/M/L tags against the file
   sizes and complexity called out by the item.
7. **Already done**: cross-referenced commits since `2026-04-22` against
   each item.

### 1.2 What I sampled in code

Direct reads of:
- [`scripts/compute_relevance.py`](../scripts/compute_relevance.py)
  lines 39, 47, 187-252, 382-400 (sys.path bootstrap, legacy config
  fallback, embedder wiring, atomic write).
- [`scripts/articles_to_facts.py`](../scripts/articles_to_facts.py)
  lines 63, 292-362, 441-450 (PROMPT_PATH default, strict-quotes
  validator, CLI flag).
- [`scripts/etd_post_publish.py`](../scripts/etd_post_publish.py) line
  145 (`--prompt` default).
- [`scripts/annotate_prior_state.py`](../scripts/annotate_prior_state.py)
  lines 26, 345-397 (atomic_write_jsonl, FC_FILE atomic write).
- [`scripts/build_benchmark.py`](../scripts/build_benchmark.py) lines
  268-369 (step_relevance, step_relevance_parallel, step_checksums,
  step_publish dispatch sites).
- [`src/common/openai_embeddings.py`](../src/common/openai_embeddings.py)
  lines 7, 12, 44, 228, 242 (native dim, parallel cache, signature).
- Directory checks: `src/common/` (no `paths.py`, no
  `news_fetcher.py`, no `gdelt_aggregator_domains.py`, no
  `embeddings_backend.py`, no `config_slices.py`, no `layout.py`, no
  `sources.py`); `src/retrieval/` does not exist; `src/etd/` does not
  exist as a package; `src/unify/` does not exist; `scripts/archive/`
  does not exist; `configs/relevance.yaml` does not exist.

### 1.3 What I skipped

- Did not exhaustively read the three news fetchers
  (`fetch_forecastbench_news.py`, `fetch_gdelt_cameo_news.py`,
  `fetch_earnings_news.py`); the audit document and existing
  cross-references give enough signal.
- Did not verify the ETD Stage 2 / Stage 3 dedup-and-link internals;
  out of scope per `V2_2_ARCHITECTURE.md` §11.
- Did not run the build pipeline.
- Did not validate the empirical baseline numbers in C7 (Phase A/C
  recovery percentages); accepted them as stated by the project memory.

---

## Section 2. Item-by-item table

Legend for "Validity": V = valid (problem exists in current code),
P = partially valid (claim correct but partially shipped), D = done
(already addressed in a recent commit), Q = needs human verification
(could not confirm from code sample alone).

Legend for "Spec quality": G = good, F = fair (some gap), W = weak.

| ID  | Title                                                | Validity | Spec | Priority OK? | Effort OK? | Deps OK? | Notes |
|-----|------------------------------------------------------|:--------:|:----:|:------------:|:----------:|:--------:|-------|
| A1  | fetch_gdelt_doc_archive.py                           | V        | G    | yes          | yes        | yes      | Largest single P0 win                       |
| A2  | build_gdelt_doc_index.py                             | V        | G    | yes          | yes        | yes (A6) | GPU-bound; sequential with other GPU work   |
| A3  | query_gdelt_doc_index.py                             | V        | G    | yes          | yes        | yes      | Replaces per-FD HTTP cascade                |
| A4  | encode-once score-many                                | V        | G    | yes          | yes        | yes      | Partly superseded by A13 for earnings       |
| A5  | parallelize trafilatura body fetch                   | V        | G    | maybe P0     | yes        | yes      | A5 is half the wall-clock win; consider P0  |
| A6  | embeddings_backend.py                                | V        | F    | yes          | yes        | yes      | Now retroactively must wrap shipped openai_embeddings.py |
| A7  | OpenAI Batch backend                                 | P (shipped) | F | demote to P2 | yes      | yes      | Shipped under different name; needs RESPEC  |
| A8  | FAISS shard-prune by date                            | V        | G    | yes          | yes        | yes      |                                              |
| A9  | per-FD query-embedding cache                         | V        | G    | yes          | yes        | yes      |                                              |
| A10 | SBERT batch_size + FP16 plumbed via config           | V        | G    | yes          | yes        | yes      |                                              |
| A11 | "do not touch" placeholder                           | V        | G    | yes          | yes        | n/a      | Document-only; no work                       |
| A12 | bulk-fetch bodies for survivors only                  | V        | G    | yes          | yes        | yes      | Subset of A3; consider MERGE                |
| A13 | per-benchmark retrieval router                        | V        | G    | yes          | yes        | yes      | Earnings part shipped (`ac0b031`)           |
| B1  | hybrid retrieval contract module                     | V        | G    | yes          | yes        | yes      |                                              |
| B2  | NewsFetcher base class                                | V        | G    | yes          | yes        | yes      | Subsumes E1 + E6                            |
| B3  | src/unify/ package                                    | V        | G    | yes          | yes        | yes      | Subsumes E2                                 |
| B4  | src/common/paths.py single owner                      | V        | F    | yes          | maybe L    | yes      | "Incremental" is open-ended                 |
| B4a | bootstrap_sys_path()                                  | V        | G    | yes          | yes        | yes      | 17 affected scripts confirmed               |
| B5  | per-stage config slices                               | V        | G    | yes          | yes        | yes      |                                              |
| B6  | atomic stage-meta writes                              | V        | G    | yes          | yes        | yes      | Partial overlap with B15; retain both       |
| B7  | src/etd/date_validators.py                            | V        | G    | yes          | yes        | yes      | Subsumes E4                                 |
| B8  | gdelt_aggregator_domains.py                           | V        | G    | yes          | yes        | yes      |                                              |
| B9  | Layout dataclass for cutoff outputs                   | V        | F    | maybe P1     | yes        | yes      | Touches every audit script; large surface   |
| B10 | retire benchmark/build.py shim                        | V        | G    | yes          | yes        | yes      |                                              |
| B11 | retire configs/relevance.yaml fallback                | V        | G    | yes          | yes        | yes      | Verified: file does not exist; fallback is dead |
| B12 | Source enum                                           | V        | G    | yes          | yes        | yes      |                                              |
| B13 | pull GDELT BigQuery fetcher into cascade or delete    | V        | G    | yes          | yes        | yes      |                                              |
| B14 | move pre-v2.0 artifacts to scripts/archive/           | V        | F    | yes          | yes        | yes      | scripts/archive/ does not exist; create     |
| B15 | atomic writes for in-place data mutators              | P        | G    | yes          | yes (M)    | yes      | Already done in compute_relevance and annotate_prior_state |
| C1  | two-layer leakage enforcement                         | V        | G    | yes          | yes        | yes      |                                              |
| C2  | side-by-side parity test                              | V        | G    | yes          | yes        | yes      |                                              |
| C3  | embedding-backend identity in manifest                | V        | G    | yes          | yes        | yes      |                                              |
| C4  | resume protocol invariants test                       | V        | G    | yes          | yes        | yes      |                                              |
| C5  | config-hash regression test                           | V        | G    | yes          | yes        | yes      |                                              |
| C6  | domain blocklist round-trip test                      | V        | G    | yes          | yes        | yes      |                                              |
| C7  | ETD Stage-1 hallucination floor monitoring            | V        | G    | yes          | yes        | yes      | Empirical baseline well-specified           |
| C8  | per-row fingerprint collision test                    | V        | G    | yes          | yes        | yes      |                                              |
| C9  | schema-version bump policy                            | V        | G    | yes          | yes        | n/a      | Doc-only no-op                              |
| C10 | production filter recipe in manifest                  | V        | G    | yes (P1)     | yes        | yes      | Now duplicated with F5 (P0); see RESOLUTION |
| D1  | v2.2 migration guide                                  | V        | G    | yes          | yes        | yes      |                                              |
| D2  | update PIPELINE.md                                    | V        | G    | yes          | yes        | yes      |                                              |
| D3  | new CACHE_INVARIANTS.md                               | V        | G    | yes          | yes        | yes      |                                              |
| D4  | update DATASET.md provenance tags                     | V        | G    | yes          | yes        | yes      |                                              |
| D5  | update README quickstart                              | V        | G    | yes          | yes        | yes      |                                              |
| D6  | archive SUMMARY.md / CONTRIBUTION_ANALYSIS.md         | V        | G    | yes          | yes        | yes      |                                              |
| D7  | v2.2 changelog entry                                  | V        | G    | yes          | yes        | yes      |                                              |
| E1  | duplicate sys.path.insert in fetchers                 | V        | G    | yes          | yes        | yes      | Subsumed by B4a; can DROP                   |
| E2  | _with_raised_csv_field_limit decorator                | V        | G    | yes          | yes        | yes      | Subsumed by B3                              |
| E3  | compute_relevance legacy config fallback              | V        | G    | yes          | yes        | yes      | Exact dup of B11; MERGE                     |
| E4  | articles_to_facts validator surface                   | V        | G    | yes          | yes        | yes      | Exact dup of B7; MERGE                      |
| E5  | art_id prefix replication                             | V        | G    | yes          | yes        | yes      | Subsumed by B2 / B3                         |
| E6  | per-fetcher HEADERS / TIMEOUT                         | V        | G    | yes          | yes        | yes      | Subsumed by B2                              |
| E7  | _fast_jsonl.py is script-local                        | V        | G    | yes          | yes        | yes      |                                              |
| E8  | optional_imports.py is sound                          | V        | G    | yes          | n/a (0)    | n/a      | No-op record                                |
| E9  | build orchestrator step ordering implicit             | V        | F    | yes          | yes (M)    | yes      | "Defer to v2.3" call-out is correct         |
| E10 | spam blocklist only filters fetch-time                | V        | G    | yes          | yes        | yes      |                                              |
| E11 | earnings annotator metadata key (DONE in 8076a54)     | D        | G    | n/a          | n/a (0)    | n/a      | Mark DONE; keep audit-trail item            |
| E12 | audit field-name mismatches (DONE in e22395c)         | D        | G    | n/a          | n/a (0)    | n/a      | Mark DONE                                   |
| E13 | ETD post-publish orchestrator (DONE in e22395c)       | D        | G    | n/a          | n/a (0)    | n/a      | Mark DONE; but documentation gap remains    |
| E14 | parallel scripts/build_gdelt_cameo trees              | Q        | G    | yes          | yes        | yes      | Need to verify benchmark/scripts/gdelt_cameo state |
| E15 | three text-fetch utilities                            | V        | G    | yes          | yes        | yes      |                                              |
| E16 | gdelt_retry_orphans.py purpose unclear                | V        | G    | yes          | yes        | yes      |                                              |
| E17 | debug_flows.py is dev utility                         | V        | G    | yes          | yes        | yes      |                                              |
| E18 | per-script ROOT pattern                               | V        | G    | yes          | yes        | yes      | Subsumed by B4 / B4a; DROP                  |
| F1  | B10 hybrid baseline                                   | V        | G    | yes          | yes        | yes      |                                              |
| F2  | B10b facts-only RAG                                   | V        | G    | yes          | yes        | yes      |                                              |
| F3  | EMR-ACH analysis matrix accepts facts                 | V        | G    | yes          | maybe L    | yes      | Touches the analysis-matrix scoring path    |
| F4  | paper Table 3 with hybrid results                     | V        | G    | yes          | yes        | yes      |                                              |
| F5  | pin production filter recipe                          | V        | G    | yes          | yes        | yes      | Now duplicates C10; promote F5, demote C10 |
| G1  | reuse-contract table implementation                   | V        | G    | yes          | yes        | yes      |                                              |
| G2  | per-benchmark article files as first-class artefacts  | V        | G    | yes          | yes        | yes      | Closes the deleted-earnings-articles bug    |
| G3  | fix step_publish silent overwrite                     | V        | G    | promote P0   | yes        | yes      | This is a concrete production bug, not P1   |
| G4  | fix quality_filter silent no-op                       | V        | G    | promote P0   | yes        | yes      | Concrete production bug found by audit      |
| G5  | integrity check at publish time                       | V        | G    | yes          | yes        | yes      |                                              |
| G6  | reuse_check.py CLI                                    | V        | G    | yes          | yes        | yes      |                                              |

Total: 75 items reviewed.

---

## Section 3. Detailed findings per category

### Category A: Performance

**A1, A2, A3, A12, A13**: KEEP. The four-script GDELT DOC chain
(A1 -> A2 -> A3 + A12 + A13) is the architectural core of v2.2.
A12 is technically a sub-bullet of A3, but the explicit call-out is
worth keeping because the "filter first, fetch survivors" inversion is
the single largest wall-clock win and easy to miss when implementing
A3. Recommendation: KEEP A12 separate, add a dependency note "A12 must
land in the same PR as A3".

**A4 vs A13**: RESCOPE. A4 reads as "encode once across all three
benchmarks". A13 introduces a per-benchmark router that explicitly
says A4's encode-once-score-many produces zero benefit for the
earnings slice (because earnings will use a ticker-date join, not
SBERT cosine). The two items are not duplicates, but A4's
acceptance criterion ("same per-benchmark relevance_meta.json
outputs") needs to be qualified: the earnings benchmark is no longer
in scope for A4 once A13 lands. Recommendation: RESPEC A4 to read
"encode-once-score-many for forecastbench + gdelt-cameo (earnings
goes through A13's relational join)".

**A5**: RETAG-PRIORITY. The audit document explicitly states the
single-threaded body fetch is "the second hottest path after Google
News" and is the bottleneck along with the per-FD HTTP cascade.
Keeping A5 at P1 while A12 is P0 underweights the contribution.
Recommendation: promote A5 to P0 or, alternatively, document that A5
is implicit in A3 + A12 (the GDELT DOC index path replaces most body
fetches).

**A6 vs shipped `src/common/openai_embeddings.py`**: RESPEC. The
shipped module is named differently and uses native dim 1536 instead
of truncating to 768. A6 is now "wrap the existing
openai_embeddings.py module behind a unified `embeddings_backend.py`
API; preserve the parallel-cache approach (do not project to 768)".
The dim-projection question raised in the audit Section 9.2 should be
resolved before A6 starts; recommendation is to ratify the parallel-
cache approach because it preserves SBERT cache validity on backend
swap.

**A7**: DROP or RESPEC as DONE. The OpenAI Batch backend has shipped
under a different module name (commit `9a27816` + `a373e89`). The
remaining work is just A6's wrapping. Recommendation: mark A7 DONE
with a pointer to the shipped module; absorb any residual work into
A6.

**A11**: KEEP as documentation marker. Zero-effort placeholder is
appropriate.

### Category B: Architecture

**B1, B5, B7, B8**: KEEP. P0 / P1 ratings are correct.

**B2**: KEEP. The 1,600-line aggregate of three fetchers is the
single largest refactor surface. The "-400 lines" estimate seems
plausible.

**B3**: KEEP. Same pattern as `benchmark/build.py` deprecation in
commit `d60a4ba`; the pattern is proven.

**B4 vs B4a vs E18**: MERGE. B4a is a narrowly-scoped subset of B4
(the path-bootstrap helper). E18 is the "per-script ROOT" pattern.
Recommendation: keep B4 as the umbrella, keep B4a as a P1 quick-win
(17 scripts to migrate, confirmed by grep), DROP E18 as a duplicate
of B4. The B4a empirical evidence (three production crashes today
listed in the table) is excellent justification.

**B6 vs B15**: KEEP BOTH but rescope. B6 covers stage-meta files
(`X.meta.json`). B15 covers data files (`forecasts.jsonl`,
`articles.jsonl`). They are complementary, not redundant.
**Validity finding**: B15 is partially DONE.
[`scripts/compute_relevance.py:382-400`](../scripts/compute_relevance.py)
already does atomic write via `os.replace`, and
[`scripts/annotate_prior_state.py:345-397`](../scripts/annotate_prior_state.py)
has `atomic_write_jsonl()`. The remaining scripts in B15's file list
(`relink_gdelt_context.py`, `fetch_gdelt_text.py`,
`fetch_article_text.py`) are the actual work. Recommendation:
RESCOPE B15 to "the three remaining mutators".

**B9**: KEEP at P2. Touches every audit script; large surface for
modest payoff. P2 is correct.

**B10**: KEEP at P2.

**B11 vs E3**: MERGE. E3 says "Same as B11". DROP E3.

**B12**: KEEP at P2.

**B13**: KEEP at P2.

**B14**: KEEP at P2. Verified `scripts/archive/` does not exist;
item must include "create the directory".

### Category C: Quality and rigor

**C1 through C7**: KEEP. C7's empirical-baseline table (v1, v2, v3,
v3 + blocklist) is the single best-specified item in the backlog;
serves as a model.

**C8**: KEEP at P2.

**C9**: KEEP. Doc-only no-op; effort 0.

**C10 vs F5**: MERGE / RESOLVE PRIORITY CONFLICT. F5 explicitly says
"Elevates backlog C10 from P1 to P0". C10 is still tagged P1 in the
backlog table, but the priority summary at the bottom of the backlog
file lists F5 (not C10) as P0. The two items have nearly identical
file lists (both touch `step_publish`) and identical acceptance
criteria. Recommendation: MERGE C10 into F5 with priority P0; drop
C10 from the C category.

### Category D: Documentation

**D1 through D7**: KEEP. All correctly low-effort and gated on the
relevant code items landing.

D7's "P0 (at v2.2 release)" framing is unusual but correct: this is a
release-day item, not a prerequisite-of-development item.

### Category E: Drift cleanup

**E1, E6**: DROP. Both subsumed by B2.

**E2**: DROP. Subsumed by B3.

**E3**: DROP. Exact duplicate of B11.

**E4**: DROP. Exact duplicate of B7.

**E5**: KEEP as a sub-task within B2 / B3, but consider DROP because
it is fully covered by either of those parent items.

**E7**: KEEP. Specific enough to be actionable on its own.

**E8**: KEEP. Zero-effort placeholder, useful audit trail.

**E9**: KEEP at P2 with the "defer to v2.3" qualifier intact.

**E10**: KEEP. Real inconsistency in spam-filter scope.

**E11, E12, E13**: Already marked as DONE in the backlog; KEEP as
audit-trail items.

**E14**: KEEP. Needs human verification (could not confirm from this
review whether `benchmark/scripts/gdelt_cameo/` is empty).

**E15, E16, E17**: KEEP at P2.

**E18**: DROP. Subsumed by B4 + B4a.

### Category F: Baselines

**F1, F2, F3, F4**: KEEP. Well-specified.

**F5**: KEEP at P0 (resolves the C10 priority conflict; see Category C).

### Category G: Reuse and reproducibility

**G1**: KEEP at P0. Largest single architectural item in this
category; depends on B5 (config slices).

**G2**: KEEP at P0. Closes the deleted-earnings-articles failure
mode from today's build.

**G3**: PROMOTE TO P0. The current P1 tag understates the severity:
this is the bug that caused today's stale-Apr 21 forecasts.jsonl
shipping incident. The acceptance criterion (`mtime + line-count
assertion`) is sharp and the work is small. Promote.

**G4**: PROMOTE TO P0. The "1,555 zero-article FDs survived the
quality filter" failure is concrete production damage. Cannot ship
v2.2 with this latent bug.

**G5**: KEEP at P1. Belongs after G2 lands.

**G6**: KEEP at P2.

---

## Section 4. Critical-path analysis

### 4.1 Minimum coherent v2.2

The smallest set of items that produces a working, reproducible
v2.2 deliverable:

**Tier 1 (must land first; foundation)**:
- B4a (sys.path bootstrap) - prevents the production crashes that
  blocked today's build cycle.
- B5 (config slices) - prerequisite for G1.
- B6 (atomic stage-meta writes) - small, prerequisite for resume
  invariants.
- B15 (atomic data-file writes; remaining 3 scripts only) - same.

**Tier 2 (the v2.2 wall-clock thesis)**:
- A6 (embeddings_backend wrapper around shipped openai_embeddings.py).
- A1 -> A2 -> A3 + A12 (GDELT DOC chain).
- B1 + B8 (hybrid retrieval contract + aggregator domains; gate A3).
- A4 (RESPEC: encode-once-score-many for FB + GDELT-CAMEO only).
- A13 (per-benchmark router; earnings half is already shipped).

**Tier 3 (correctness gates)**:
- C1 (two-layer leakage enforcement).
- C2 (side-by-side parity test).
- C3 (embedding-backend identity in manifest).
- G1 (reuse-contract implementation).
- G2 (per-benchmark article files as first-class artefacts).
- G3 (fix step_publish silent overwrite; PROMOTE TO P0).
- G4 (fix quality_filter silent no-op; PROMOTE TO P0).
- G5 (publish-time integrity check).
- C10/F5 (production filter recipe in manifest; MERGED, P0).

**Tier 4 (release-day)**:
- D1, D7 (migration guide, changelog entry).

**Critical path total**: roughly 19 items, of which 6 are S, 9 are M,
and 4 are mixed. Estimated ~3 to 4 weeks of focused work assuming the
GDELT DOC archive download (A1) does not stall on rate limits.

### 4.2 Tail items (do not gate v2.2 release)

Everything in Category B P2 (B9, B10, B11, B12, B13, B14), most of
Category E (E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E14, E15, E16,
E17, E18), Category F items F2 + F3 (the ablation triangle and
EMR-ACH facts-as-rows), Category C P2 (C8, C9), Category G P2 (G6),
and most of Category D P1 (D2, D3, D4, D5).

### 4.3 Deferred to v2.3

E9 (declarative DAG orchestration) is explicitly deferred. B18
(incremental ETD Stage 2 dedup, proposed in audit §8.1) should also
be v2.3 because it changes the ETD spec.

---

## Section 5. Already-done items

The backlog already marks E11, E12, E13 as DONE. These additional
items have either fully or partially shipped since drafting:

| ID    | Status            | Commit        | Notes |
|-------|-------------------|---------------|-------|
| A7    | shipped (different module) | `9a27816` + `a373e89` | OpenAI Batch shipped under `src/common/openai_embeddings.py`, not `embeddings_backend.py`. Mark partially done; remaining work folds into A6. |
| A13 (earnings half) | shipped | `ac0b031` | `scripts/link_earnings_articles.py` + `_earnings_meta` bypass. The GDELT-CAMEO router half remains. |
| B15 (compute_relevance) | shipped | (current tree) | `compute_relevance.py:382-400` already does `os.replace` atomic write. Confirmed by grep. |
| B15 (annotate_prior_state) | shipped | (current tree) | `annotate_prior_state.py` has `atomic_write_jsonl()` at line 345, used at line 397. Confirmed. |
| E11 | DONE              | `8076a54`     | Already marked DONE in backlog. |
| E12 | DONE              | `e22395c`     | Already marked DONE in backlog. |
| E13 | DONE              | `e22395c`     | Already marked DONE; but documentation gap remains (see §6). |

Recommendation: update B15's file list to remove `compute_relevance.py`
and `annotate_prior_state.py`; they are done. Remaining scripts:
`relink_gdelt_context.py`, `fetch_gdelt_text.py`,
`fetch_article_text.py`.

---

## Section 6. Missing items

Cross-checked against `V2_2_END_TO_END_AUDIT.md` Section 8 (which
proposes B15, B16, B17, B18, C7b, C11, C12, C13, D8, D9, D10, E19,
E20, F5). Most have been folded into the backlog at commit `57abb8d`,
but the following are still missing or under-specified:

### 6.1 Items proposed in audit §8 but not in backlog

- **B16: snapshot env vars into build_manifest.json**. Not in backlog.
  Critical because `EMRACH_FB_SUBJECT_FILTER`, `EMRACH_FB_HORIZON_DAYS`,
  etc. drive unify behavior; replaying `benchmark.yaml` without them
  produces a different deliverable. RECOMMEND ADD as P1.
- **B17: SBERT model revision pinning** (HF commit SHA). Not in
  backlog. Mirrors the GDELT DOC index manifest model_revision pattern
  (§3.2). RECOMMEND ADD as P1.
- **B18: ETD Stage 2 incremental dedup**. Not in backlog. P2 / v2.3.
- **C7b: wire `--strict-quotes` into etd_post_publish.py default**.
  Not in backlog. Verified mismatch: `articles_to_facts.py` exposes
  the flag (line 446) but `etd_post_publish.py:171` does not pass it
  through. RECOMMEND ADD as P1.
- **C11: extend step_checksums to all deliverable files**
  (`benchmark.yaml`, `build_manifest.json`,
  `forecasts_change.jsonl`, `forecasts_stability.jsonl`). Not in
  backlog. RECOMMEND ADD as P1.
- **C12: baseline prompt SHA pinning**. Not in backlog. Important for
  paper reproducibility once Category F lands. RECOMMEND ADD as P2.
- **C13: global `--seed` for sampling baselines**. Not in backlog.
  RECOMMEND ADD as P2.
- **D8: document etd_post_publish.py in PIPELINE.md §3.5**. Not in
  backlog. Critical because the orchestrator is the bridge between
  v2.1 publish and Category F baselines. RECOMMEND ADD as P0.
- **D9: document the prompt-default mismatch**. Verified concrete bug
  (articles_to_facts.py line 63 defaults to v1; etd_post_publish.py
  line 145 defaults to v3). A user invoking articles_to_facts.py
  directly gets the 15.5% unsupported floor. Either rename one default
  or document the rationale. RECOMMEND ADD as P1.
- **D10: document the EMRACH_* env-var surface**. Not in backlog.
  Pairs with B16. RECOMMEND ADD as P1.
- **E19: etd_link.py Stage-2 file freshness check**. Not in backlog.
  RECOMMEND ADD as P1.
- **E20: ETD post-publish production-filename collision** (the
  `--skip-link` path produces a file with the same name as the
  linked-and-filtered path). Not in backlog. RECOMMEND ADD as P1.

### 6.2 Items neither in audit §8 nor in backlog

- **A14 (proposed): GDELT DOC archive disk-quota guard**. The
  estimated 6-8 GB on disk per cutoff window assumes English filter.
  No item caps the disk usage or warns if free space is insufficient.
  Low-effort safeguard. Suggest ADD as P2.
- **C14 (proposed): two-cutoff leakage probe regression test**. The
  v2.1 design includes the leakage-probe protocol
  (`configs/leakage_probe_config.yaml`); v2.2's encode-once-score-many
  refactor could silently regress the probe. Recommend ADD as P1
  acceptance test alongside C2.
- **D11 (proposed): document the parallel-cache approach for
  OpenAI embeddings** (separate from D3 which is the matrix). The
  shipped behavior diverges from the §6.2 design and the cache-
  invalidation matrix is now stale. Folded into D3 if D3 is rewritten,
  or added as a separate item.
- **G7 (proposed): pre-flight check that `data/{bench}/` per-benchmark
  files exist before unify**. The audit's earnings-articles deletion
  failure mode could fire again at unify time, not just publish time;
  G2 covers publish, but unify is the silent-failure surface. Recommend
  ADD as P1.

### 6.3 Items under-specified

- **B4**: "M (incremental)" effort tag is open-ended. Recommend
  RESPEC: scope B4 to the initial module + 5 high-traffic scripts;
  defer the long-tail migration to a B4b "complete the path migration"
  item.
- **F3**: "M" effort underestimates the work. Touching the analysis-
  matrix scoring step in `experiments/02_emrach/run_emrach.py`
  (existence not verified) plus adding three modes
  (`{article, fact, both}`) is closer to L. Recommend re-tag to L or
  RESCOPE.

---

## Section 7. Recommendations

In rough priority order:

**REC-01: Promote G3 and G4 to P0; demote nothing in exchange.**
Affected items: G3, G4. Rationale: both are concrete production bugs
verified by today's audit. G3 caused the stale-Apr 21 forecasts ship;
G4 let 1,555 zero-article FDs through. Cannot launch v2.2 with these
latent. Adds two items to the P0 list (now 22).

**REC-02: Resolve the C10 vs F5 priority conflict.**
Affected items: C10, F5. Rationale: F5 explicitly elevates C10 from
P1 to P0 but C10 is still tagged P1 in the backlog body. Recommended
action: MERGE C10 into F5; mark F5 as the canonical entry; remove
C10 or downgrade to a pointer.

**REC-03: Mark A7 as partially DONE; absorb residual into A6.**
Affected items: A6, A7. Rationale: the OpenAI Batch backend shipped
under `src/common/openai_embeddings.py` (not `embeddings_backend.py`).
A7's remaining work is the API wrapping, which is precisely A6.
Recommended action: mark A7 with a "partially landed in `9a27816` +
`a373e89`; remaining work folded into A6" note; remove A7 from P1.

**REC-04: Add the 11 items proposed in audit §8 that are still missing.**
Affected items: B16, B17, C7b, C11, C12, C13, D8, D9, D10, E19, E20.
Rationale: the audit doc identified these explicitly; B16 + D10
together close the env-var reproducibility gap; D8 + D9 close the
post-publish documentation gap that blocks Category F adoption; E19 +
E20 close the silent-fail modes in the ETD orchestrator. Most are S
effort.

**REC-05: Drop the four duplicate Category E items; trust the parent
Category B items.**
Affected items: E1 (->B2), E2 (->B3), E3 (->B11), E4 (->B7),
E5 (->B2/B3), E6 (->B2), E18 (->B4/B4a). Rationale: each duplicate
is acknowledged as "subsumed by" its parent. Carrying them forward
inflates the item count and confuses the priority summary. Audit-
trail value is preserved by the cross-reference in the parent items.

**REC-06: RESCOPE B15 to remove already-done files.**
Affected items: B15. Rationale: verified that
`compute_relevance.py` and `annotate_prior_state.py` already do
atomic data-file writes. The remaining work is `relink_gdelt_context.py`,
`fetch_gdelt_text.py`, `fetch_article_text.py`. Update file list and
re-estimate effort from M to S.

**REC-07: Promote A5 to P0 or fold its scope into A3 + A12.**
Affected items: A5, A3, A12. Rationale: the audit says single-threaded
trafilatura is "the second hottest path after Google News". Currently
P1, while A12 (also a body-fetch optimization) is P0. Two paths:
(a) promote A5 to P0; (b) document that A3 + A12 obviate most of A5's
work because GDELT DOC bodies are fetched in bulk. Either is acceptable.
Recommend (b) with a note in A5.

**REC-08: RESPEC A4 to exclude earnings.**
Affected items: A4, A13. Rationale: A13 (per-benchmark router) means
A4's encode-once-score-many produces zero benefit on the earnings
slice. A4 should explicitly say "for forecastbench + gdelt-cameo
only" to avoid wasted work during implementation.

**REC-09: Mark B4a as the authoritative `sys.path` solution; DROP E18.**
Affected items: B4a, E18. Rationale: B4a contains the empirical
evidence (3 production crashes today) and the canonical pattern.
E18's "subsumed by B4" remark is correct; carrying both confuses
the work item count.

**REC-10: Add documentation items D8, D9, D10 (audit §8).**
Affected items: D8, D9, D10. Rationale: these are the
post-publish-orchestrator documentation gaps that the audit
identified. Each is S effort.

**REC-11: Add F5 explicitly into the priority summary block at the
bottom of the backlog.**
Affected items: priority summary line. Rationale: F5 is mentioned in
the body but the summary line still reads "F5" once it now needs to
displace C10. Sync the summary with the resolved priorities.

**REC-12: Add G7 (pre-flight check at unify) as a P1.**
Affected items: G7 (new). Rationale: G2 closes the publish-time
silent-fail; the parallel risk at unify-time is not covered.

**REC-13: RESCOPE F3 to L effort (was M).**
Affected items: F3. Rationale: touching `experiments/02_emrach/run_emrach.py`
plus adding three evidence-unit modes is bigger than M.

**REC-14: Spawn a separate v2.3 backlog file for items E9 + B18 +
analogous deferred items.**
Affected items: E9, B18 (proposed). Rationale: keep v2.2 backlog
focused; deferred items stay visible without inflating the v2.2 P-
counts.

**REC-15: Cross-link Section 4b of `V2_2_ARCHITECTURE.md` (reuse
contract) into G1's spec body.** Affected items: G1. Rationale: the
table in §4b is the canonical spec for G1's implementation; the
backlog item references it but does not summarize the contract.
Make the table the acceptance criterion for G1.

---

**End of review.**
