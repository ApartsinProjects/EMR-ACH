# Gold-Subset Evaluation Compatibility Audit

Static (code-only) audit: can every B1-B9 baseline, the B10 / B3b variants, and
the main EMR-ACH method run on the v2.1 gold subset
(`benchmark/data/{cutoff}-gold/`) and produce the paper metrics without source
modification?

Scope: read-only inspection of runner, base, prompts, method files, baselines
config, `scripts/build_gold_subset.py`, and the first records of the published
parent cutoff. No execution.

Files inspected (11):

- `benchmark/evaluation/baselines/runner.py`
- `benchmark/evaluation/baselines/base.py`
- `benchmark/evaluation/baselines/prompts.py`
- `benchmark/evaluation/baselines/methods/b1_direct.py`
- `benchmark/evaluation/baselines/methods/b2_cot.py`
- `benchmark/evaluation/baselines/methods/b3_rag.py`
- `benchmark/evaluation/baselines/methods/b3b_rag_claims.py`
- `benchmark/evaluation/baselines/methods/b4_self_consistency.py`
- `benchmark/evaluation/baselines/methods/b5_multi_agent_debate.py`
- `benchmark/evaluation/baselines/methods/b6_tree_of_thoughts.py`
- `benchmark/evaluation/baselines/methods/b7_reflexion.py`
- `benchmark/evaluation/baselines/methods/b8_verbalized_confidence.py`
- `benchmark/evaluation/baselines/methods/b9_llm_ensemble.py`
- `benchmark/evaluation/baselines/methods/b10_hybrid_facts_articles.py`
- `benchmark/evaluation/baselines/methods/b10b_facts_only.py`
- `benchmark/configs/baselines.yaml`
- `scripts/build_gold_subset.py`
- `experiments/02_emrach/run_emrach.py`
- `src/data/mirai.py` (for EMR-ACH data contract)
- Parent first-line sample of `benchmark/data/2026-01-01/{forecasts,articles}.jsonl`

Parent FD keys observed on line 1: `article_ids, background, benchmark,
crowd_probability, default_horizon_days, fd_type, forecast_point, ground_truth,
ground_truth_idx, hypothesis_definitions, hypothesis_set, id, lookback_days,
metadata, prior_state_30d, prior_state_n_events, prior_state_stability,
question, resolution_date, source, x_multiclass_ground_truth,
x_multiclass_ground_truth_idx, x_multiclass_hypothesis_definitions,
x_multiclass_hypothesis_set`.

Parent article keys observed on line 1: `actors, cameo_code, char_count,
gdelt_themes, gdelt_tone, id, provenance, publish_date, source_domain, text,
title, title_text, url`.

Gold subset (per `build_gold_subset.py`) emits `forecasts.jsonl` and
`articles.jsonl` as a strict subset of the parent (same schemas,
`additionalProperties: True`), plus `facts.jsonl`, `build_manifest.json`
(with `kind: gold_subset`), `benchmark.yaml`, `checksums.sha256`, `schema/`,
`examples/`, `meta/`, `README.md`, `LICENSE`, `CITATION.cff`,
`selection_criteria.md`.

---

## 1. Data contract compatibility (FD fields)

| Reader | file:line | FD field read | Gold provides? | Notes |
|---|---|---|---|---|
| runner | runner.py:109 | `default_horizon_days` | yes (optional) | used if `--horizon` omitted, else 14. |
| runner | runner.py:111,116,121,133 | `resolution_date`, `article_ids`, `forecast_point`, `experiment_horizon_days` | yes | `experiment_horizon_days` is WRITTEN back to fd, not read. |
| runner | runner.py:488,620,638,649 | `id`, `hypothesis_set`, `ground_truth` | yes (required in schema) |  |
| runner | runner.py:762-767 | `x_multiclass_hypothesis_set`, `x_multiclass_hypothesis_definitions`, `x_multiclass_ground_truth`, `x_multiclass_ground_truth_idx` | yes (optional in gold schema) | Only touched when `--multiclass` flag is passed. |
| runner `_metrics_single_group` | 275-349 | `predicted_class`, `ground_truth`, `parse_failed` | n/a (prediction rows) | Fields are synthesized by `prediction_row`. |
| runner `compute_metrics` | 370,376 | `benchmark`, `hypothesis_set`, `fd_type` | yes (required) | Every gold FD has `fd_type` in `{stability, change}` (or `unknown` only with `--keep-unknown`). |
| base `prediction_row` | 266-281 | `id`, `hypothesis_set`, `ground_truth`, `benchmark`, `fd_type` | yes | |
| prompts `render_user` | 130-141 | `hypothesis_set`, `id`, `hypothesis_definitions`, `question`, `background`, `forecast_point`, `resolution_date` | yes (all required in schema, except `background` optional) | |
| prompts `_prior_expectation_block` | 78-103 | `prior_state_30d`, `prior_state_stability`, `prior_state_n_events`, `benchmark`, `lookback_days`, `hypothesis_set` | yes (all optional in gold schema; renderer emits empty string when missing) | GDELT branch references `lookback_days` with default 90; earnings / forecastbench branches do not. No crash on missing keys. |
| base `articles_block` | 69 | `article_ids` | yes | |
| b1_direct, b2_cot, b3_rag, b8 | `fd['id']`, `fd['hypothesis_set']` | yes | |
| b3b_rag_claims | b3b:31-36,68 | `question`, `hypothesis_set`, `article_ids`, `id` | yes | |
| b4, b5, b6, b7, b9 | similar | `id`, `hypothesis_set` | yes | |
| b10 hybrid | b10:119-121 | `article_ids`, `facts` | `facts` NOT IN gold FD schema — **warning** | Degrades gracefully to articles-only per docstring (`fd['facts']` absent is handled). |
| b10b facts-only | b10b:55 | `facts` | `facts` NOT IN gold FD schema — **warning** | Degrades to "no atomic facts available" stub per `build_facts_block`. |

Explicit gaps:

- **`fd["facts"]`** is populated by `scripts/etd_post_publish.py` on the
  in-memory FD objects; it is never written to the published `forecasts.jsonl`
  (gold or parent). B10 and B10b both check and degrade to a no-facts prompt.
  **B10b becomes degenerate (no evidence at all) without a pre-step to inject
  per-FD facts from `facts.jsonl`.** See Section 9 (shim recommendation).

## 2. Article contract compatibility

| Reader | file:line | Article field read | Gold provides? | Notes |
|---|---|---|---|---|
| runner `apply_experiment_horizon` | 121 | `publish_date` | yes (required) | ISO date. |
| prompts `build_articles_block` | 219,228,233-236 | `title`, `text`, `publish_date`, `source_domain`, `id` (via dict key) | yes | |
| b3b extraction | b3b:33-36 | `title`, `publish_date`, `source_domain`, `text` | yes | |
| b10 hybrid | reuses `build_articles_block` | as above | yes | |

No article-field gaps.

## 3. Cutoff-folder assumptions (regex / suffix sensitivity)

Grep of `re.match|re.search|re.fullmatch` across `benchmark/`:

- `benchmark/evaluation/baselines/runner.py:590` `infer_cutoff`:
  `re.search(r"(\d{4}-\d{2}-\d{2})", str(fds_path))`. **Non-anchored**, so a
  path containing `2026-01-01-gold` returns `"2026-01-01"`. **Effect**: all
  gold runs will write under `benchmark/results/2026-01-01/{method}/...`, i.e.
  they will share the results namespace with the parent-cutoff runs. Verdict:
  does not crash, but results directories will collide if both parent and gold
  evaluations are run for the same cutoff. See Section 9 fix.
- `benchmark/evaluation/baselines/base.py:237` is a hypothesis-probability
  regex; unrelated to cutoff folders.

No code path performs `re.match(r"^\d{4}-\d{2}-\d{2}$", cutoff)` anywhere under
`benchmark/` — the published schemas pin that pattern on FD
`forecast_point`/`resolution_date` fields, not on folder names. Safe.

## 4. `kind: gold_subset` manifest

Grep for `kind.*gold`, `kind.*subset`, and `build_manifest` across
`benchmark/evaluation/` returned no matches. The baselines runner never reads
`build_manifest.json`. No guard path is triggered by the gold-subset kind.
Verdict: evaluator is manifest-agnostic.

## 5. `facts.jsonl` side-effect risk

The runner does not glob the cutoff folder. It takes explicit `--fds` and
`--articles` paths (runner.py:668-669). It does not open `facts.jsonl`.
`facts.jsonl` in the gold folder is inert with respect to the baselines
battery. B10 / B10b consume `fd["facts"]`, which is an in-memory field
(normally injected by `scripts/etd_post_publish.py`) and is absent from the
published JSONL — they degrade gracefully.

Grep of `glob(`, `rglob(`, `iterdir(`, `listdir` in `benchmark/evaluation/`:
zero hits. No accidental folder scan occurs.

## 6. Metrics on smaller N

`compute_metrics` → `_with_slices` → `_metrics_single_group` + `_bootstrap_cis`
(runner.py:220-401):

- No hardcoded minimum-N check. `_metrics_single_group` returns `{"n": 0}` on
  empty input but otherwise works at any N.
- `_bootstrap_cis` uses `B=1000, seed=42`, stratified by default. For each
  bootstrap draw it resamples within each ground-truth class. Per-class pool
  must be non-empty; empty classes are silently skipped (`if not pool: continue`).
  This is safe for gold's ~300-500 FDs and the balanced `{Comply, Surprise}`
  target.
- Edge case: if a gold draw has a stratum with support 0, the class is simply
  dropped from balanced-accuracy and MCC calculations — no crash, but the
  reported balanced-accuracy denominator shrinks. The default gold quotas
  (`DEFAULT_TARGETS` in `build_gold_subset.py`: ~60-100 per stratum) make this
  near-impossible.
- Stratified bootstrap has **no per-stratum minimum**. A stratum with 1 FD
  will be resampled with replacement and will produce a constant — valid but
  noisy. Not a crash risk.
- `by_fd_type` breakdown (runner.py:376-383): requires any row has
  `fd_type` set and the set is not exactly `{"unknown"}`. Gold's default
  `fd_type` filter drops `unknown`, so all rows will be `stability` or
  `change` — `by_fd_type` will be populated and the paper headline
  (`change` slice) is computable.

Verdict: metrics pipeline is **N-agnostic** and **ready for gold**.

## 7. EMR-ACH entrypoint

Located: `experiments/02_emrach/run_emrach.py`. Entrypoint function: `main()`.

**Critical finding: EMR-ACH currently runs on the MIRAI dataset, not on Forecast
Dossiers.** Evidence:

- Imports `src.data.mirai.MiraiDataset`, `HYPOTHESES = ["VC", "MC", "VK", "MK"]`
  (run_emrach.py:29).
- `ds = MiraiDataset(cfg); queries = ds.queries(n)` (run_emrach.py:67-68).
- Uses `src.pipeline.{indicators, influence, retrieval, presence, aggregation,
  calibration, deep_analysis, multi_agent}`; all of these operate on
  `MiraiQuery` objects, not FDs.
- Results written under `cfg.results_dir / "processed"`, not under
  `benchmark/results/{cutoff}/emrach/`.

Per-stage data dependencies (all in-memory; fed from MIRAI, not FD JSONL):

1. Stage 1 indicators: per-query `(subject, relation, object, timestamp)`.
2. Stage 2 influence: per-query indicators × 4 MIRAI hypotheses.
3. Stage 3 retrieval: `retriever.retrieve_batch(queries, n)` — either mock or
   `get_retriever(cfg)`. Retriever contract expects `MiraiQuery.query_text`,
   not FD article_ids.
4. Stage 4 / 4b: presence + background — per-article × per-indicator.
5. Multi-agent + Deep Analysis: consume retrieved articles + indicators.

Hardcoded paths: the script does NOT reach into `benchmark/data/{cutoff}/`.
It reads MIRAI files via `cfg.data_root` (see `src/data/mirai.py`).

**Consequence: EMR-ACH cannot run on `benchmark/data/{cutoff}-gold/`
out-of-the-box. It was never wired to the FD schema.** An adapter is required
(see Section 9).

No `-gold` suffix issues; no cutoff-folder regex in the EMR-ACH entrypoint at all.

## 8. Per-method verdicts

| Method | Verdict | Rationale |
|---|---|---|
| B1 direct | GREEN | Only reads `id`, `hypothesis_set`; all in gold schema. |
| B2 CoT | GREEN | Same as B1. |
| B3 RAG | GREEN | Reads `article_ids` + article fields that gold guarantees. |
| B3b RAG-Claims | GREEN | Reads `question`, `hypothesis_set`, `article_ids`, per-article `title/publish_date/source_domain/text`. All present. Not in `baselines.yaml` — must be invoked via `--method b3b_rag_claims` only if config entry is added, otherwise the method is loadable but unregistered. **YELLOW** on config registration. |
| B4 Self-Consistency | GREEN | Uses shared renderer; no novel field. |
| B5 Multi-Agent Debate | GREEN | Uses shared renderer; multi-round logic depends only on custom_ids. |
| B6 Tree-of-Thoughts | GREEN | Same. |
| B7 Reflexion | GREEN | Same. |
| B8 Verbalized Confidence (deprecated shim) | GREEN | Identical to B3. |
| B9 LLM Ensemble | GREEN | Config deviations allowed (`allow_model_override=True`). |
| B10 hybrid (facts + articles) | YELLOW | Runs, but `fd["facts"]` is absent from gold JSONL; degrades to articles-only unless a pre-step injects facts from `facts.jsonl`. Also not registered in `baselines.yaml`. |
| B10b facts-only | YELLOW → RED (for meaningful eval) | Without `fd["facts"]`, the evidence block is `(no atomic facts available)` for every FD — baseline becomes a no-evidence prompt and the ablation is meaningless. Code runs; experiment is degenerate. Also not registered in `baselines.yaml`. |
| EMR-ACH (`experiments/02_emrach/run_emrach.py`) | RED | Wired to MIRAI, not to the FD benchmark. No FD-loader, no `article_ids → articles.jsonl` retrieval path, no `fd_type` / `prior_state_30d` consumption, no Comply/Surprise hypothesis mapping. Needs an adapter entrypoint before gold eval is possible. |

Summary: 9 GREEN, 3 YELLOW, 1 RED.

## 9. Recommended NEW files (no changes applied yet)

1. **`scripts/eval/emrach_on_gold.py`** — adapter that wraps
   `experiments/02_emrach/run_emrach.py`-style stages but:
   - Loads FDs from `benchmark/data/{cutoff}-gold/forecasts.jsonl` and
     articles from `articles.jsonl`.
   - Maps each FD to an "EMR-ACH query" carrying the FD's `question`,
     `hypothesis_set` (`["Comply", "Surprise"]`), `hypothesis_definitions`,
     `article_ids`, `fd_type`, `prior_state_30d`.
   - Skips retrieval (use `fd.article_ids` as the fixed evidence pool).
   - Produces prediction rows in the same schema as the baselines
     (`predicted_class`, `ground_truth`, `parse_failed`, `fd_type`, etc.) so
     `runner.compute_metrics` can score them unchanged.
   - Writes to `benchmark/results/{cutoff}-gold/emrach/{run_id}/` once the
     cutoff disambiguation below is applied.

2. **`scripts/eval/inject_facts_into_fds.py`** — pre-step for B10/B10b: reads
   `benchmark/data/{cutoff}-gold/facts.jsonl`, groups facts by
   `primary_article_id`, joins onto each FD via `fd["article_ids"]`, and
   writes a decorated `forecasts.with_facts.jsonl` that adds a `facts` field
   per FD. The baselines runner then takes
   `--fds forecasts.with_facts.jsonl --articles articles.jsonl` unmodified.

3. **Config registration**: propose adding `b3b_rag_claims`, `b10_hybrid`, and
   `b10b_facts_only` entries to `benchmark/configs/baselines.yaml`
   (they are code-complete but not listed; recommendation only, no code change
   in this audit).

4. **Cutoff-folder disambiguation**: propose a patch to
   `runner.infer_cutoff` to preserve the `-gold` suffix:
   change the regex from
   `r"(\d{4}-\d{2}-\d{2})"` to
   `r"(\d{4}-\d{2}-\d{2}(?:-gold)?)"` (or equivalent path-segment inspection),
   so gold results go to `benchmark/results/2026-01-01-gold/{method}/...` and
   do not collide with parent-cutoff results. Flag as recommendation; do not
   apply in-flight.

5. **Tests to add once gold materializes** (under `tests/eval/`):
   - `test_gold_schema_contract.py`: asserts every FD field read by the
     baselines is present on the gold subset's first N records.
   - `test_metrics_small_N.py`: runs `compute_metrics` on a synthetic 20-FD
     gold sample to verify bootstrap CIs and `by_fd_type` slices render.
   - `test_cutoff_suffix.py`: asserts `infer_cutoff` (post-fix) returns
     `2026-01-01-gold` for `.../benchmark/data/2026-01-01-gold/forecasts.jsonl`.

## 10. Open questions for the human

1. **Scope of EMR-ACH for the gold headline**: does the paper's main result
   use the MIRAI-side EMR-ACH pipeline at all, or only EMR-ACH adapted to the
   FD benchmark? If the former, gold is irrelevant for EMR-ACH. If the latter,
   the adapter in Section 9 item 1 is a prerequisite; confirm whether an FD
   adapter already exists elsewhere (the grep hit 38 files containing
   `emrach` — mostly docs and paper — but only the MIRAI runner contains the
   orchestration).

2. **B10 / B10b scope**: are these meant to ship as headline methods (paper
   Table 1) or as v2.2 "facts channel" ablations only? If the latter, the
   absent `fd["facts"]` is expected and these only run after
   `scripts/etd_post_publish.py` has decorated the FD snapshot; fine to keep
   them out of the gold headline. Confirm.

3. **Results-dir collision**: is it acceptable for gold and parent-cutoff
   results to share `benchmark/results/2026-01-01/{method}/`? The per-run
   `run_id` prevents file overwrite (manifest captures `cutoff`), but they
   would be visually indistinguishable in the `latest.txt` pointer and in
   any per-cutoff aggregation. The Section 9 item 4 fix is cheap.

4. **`by_fd_type` slices on `unknown`**: gold drops `unknown` by default
   (`--keep-unknown` opt-in). Paper headline is the `change` subset. Confirm
   the evaluation will not pass `--keep-unknown`; otherwise bootstrap bins
   may end up with a tiny `unknown` stratum that confuses the reader.

5. **Multiclass ablation on gold**: gold preserves `x_multiclass_*` fields
   (see FD schema). The runner's `--multiclass` swap path at runner.py:757-769
   works unchanged on gold. Is this ablation expected to be part of the gold
   evaluation run, or parent-only? Affects quota sanity-checking against
   per-class support.
