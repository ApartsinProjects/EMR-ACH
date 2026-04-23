# EMR-ACH Implementation Audit (Phase 1, pre-run)

Date: 2026-04-23
Scope: Verify that EMR-ACH is runnable end-to-end against the v2.1 gold subset
(`benchmark/data/2026-01-01-gold/`) before firing an OpenAI Batch run for
`paper/index.html` Table 1 + Table 4 cells.

Verdict: **BLOCKED — do not proceed to Phase 2.**

The single most critical component (the FD-aware EMR-ACH entry point,
`scripts/eval/emrach_on_gold.py`) is a documented DRY-RUN STUB. The module
itself says so and refuses to execute in live mode. Firing the Batch API for
B1-B9 alone would spend budget without producing the EMR-ACH row the paper
needs, and would leave Table 1 partially filled.

---

## 1. EMR-ACH entry point on the gold subset

**File:** `scripts/eval/emrach_on_gold.py` (232 lines, shipped in `cd166e7`).

**Verdict: STUB.**

Hard evidence from the source:

* Docstring, lines 22-25:
  > "When F3's full EMR-ACH wiring lands, the call to `run_emrach.run(...)`
  > inside `run_emrach_on_bundle` can be flipped on; until then the adapter
  > is shape-only and is exercised by the test suite against the existing
  > parent-cutoff bundle."
* `run_emrach_on_bundle`, lines 156-167:
  ```python
  if dry_run:
      rows = [
          emrach_result_to_prediction_row(
              fd, pick=None,
              metadata={"adapter_status": "dry-run", ...})
          for fd, q in zip(fds, queries)
      ]
  else:  # pragma: no cover - real EMR-ACH path is in-flight per F3 scope
      raise NotImplementedError(
          "Live EMR-ACH invocation is gated on F3's pipeline wiring; "
          "run with --dry-run until that lands."
      )
  ```
* `main()`, lines 214-220: refuses to run without `--dry-run` and exits 2
  with "live mode is gated on F3's EMR-ACH wiring".

What the adapter DOES implement (cleanly):
* `fd_to_emrach_query`: pure data transformation FD -> MIRAI-ish query dict.
* `build_emrach_inputs`: bulk transform across a bundle.
* `emrach_result_to_prediction_row`: translation back to runner row schema.
* `write_predictions_jsonl`: IO plumbing.

What the adapter does NOT do: invoke contrastive indicator generation,
diagnostic weighting, multi-agent debate, or hybrid retrieval. The live path
literally raises `NotImplementedError`.

---

## 2. Underlying EMR-ACH pipeline components (`src/pipeline/`)

The pipeline modules exist as real code (not placeholders), but they are
**MIRAI-locked** and therefore cannot be plugged into FD-shaped v2.1 inputs
without additional FD-aware adapters that do not exist yet. Per-component:

### 2a. Contrastive indicators — `src/pipeline/indicators.py` (128 lines)
**Verdict: PARTIAL (MIRAI-only; not FD-wired).**

* Real `build_indicator_requests` + `parse_indicator_responses` pair that
  produces `m` indicators per query, each tagged `primarily_supports` in
  `{VC, MC, VK, MK}` and a `distinguishes` field.
* Accepts a `contrastive: bool = True` flag on the builder — so contrastive
  vs non-contrastive is present as a knob.
* **Hard-coded to the MIRAI CAMEO 4-hypothesis schema** via the
  `valid_supports = {"VC", "MC", "VK", "MK"}` set and `MiraiQuery` type.
  There is no code path that consumes an FD's `hypothesis_set` (which is
  arbitrary string labels per FD) and maps indicators onto those.
* Callable only from `MiraiQuery` objects, not from FD dicts.

### 2b. Diagnostic weighting
**Verdict: STUB / MISSING.**

Searched for diagnosticity/ACH-style weighting across the repo (case-insensitive
grep for `diagnostic`, `weighting`, etc). The `indicators.py` output carries
only `primarily_supports` (a label) and `distinguishes` (free-text). There is
no code that computes per-indicator diagnosticity scores (i.e. how much an
indicator discriminates among hypotheses) and applies them as weights in an
analysis-matrix A computation. `src/emrach/facts_rows.py` carries an
`evidence_strength_prior` that maps `extraction_confidence` categorical
(high/medium/low) to `{0.9, 0.6, 0.3}` — that is a confidence prior, not
ACH diagnosticity. The paper's Section 4.3 "analysis matrix A" is not
materialised in code anywhere I could locate.

### 2c. Multi-agent debate — `src/pipeline/multi_agent.py` (205 lines)
**Verdict: PARTIAL (real code, but single-round judge + MIRAI-locked).**

* Real advocate/judge structure: one advocate request per hypothesis plus one
  judge request per query, over OpenAI Batch.
* **Single round only**: advocate -> judge. No inter-agent rebuttal loop, no
  multi-round argumentation. The paper claims "multi-round inter-agent
  argumentation before final pick"; the code is a one-shot
  advocate-then-judge flow.
* Hardcoded to MIRAI's 4 hypotheses and their canonical descriptions
  (`HYPOTHESIS_NAMES`, `HYPOTHESIS_DESCRIPTIONS` constants). Cannot accept
  FD-provided `hypothesis_definitions` without code change.

### 2d. Hybrid retrieval — `src/pipeline/retrieval.py` (254 lines)
**Verdict: STUB / MISSING hybrid features.**

* Provides `ManualRetriever` (oracle via doc IDs), `MockRetriever`,
  and (by structure) a Weaviate dense retriever.
* **No MMR re-ranking**, **no RRF fusion**, **no temporal decay**. All three
  are claimed by the paper; none of them are in the file. Retrieval is
  oracle-or-dense-vector, nothing hybrid.
* Operates on `MiraiQuery`/`MiraiArticle`/`MiraiDataset` types only.

### 2e. FD adapter `src/emrach/facts_rows.py`
**Verdict: SHIPPED but narrow.** Pure data projection from ETD atomic facts
to an indicator-row dict; does not itself constitute an EMR-ACH pipeline.

---

## 3. Baselines runner readiness

**File:** `benchmark/evaluation/baselines/runner.py` (920 lines), plus per-method
modules `benchmark/evaluation/baselines/methods/b1_direct.py` through
`b10b_facts_only.py`.

**Verdict: SHIPPED for B1-B9 batch dispatch (functional to the extent of
reading line 1-100; not exhaustively traced end-to-end).**

* Runner supports `--dry-run`, `--smoke N --sync`, and Batch API flow per
  module docstring (lines 12-24).
* Methods exist for b1 through b10 including b3b and b10b. The "B10" slot
  exists (hybrid facts+articles) which is not in the original B1-B9 target
  list but can be skipped.

### Per-baseline cost estimate (rough, for 81 FDs on the gold subset)

These are order-of-magnitude figures using published OpenAI Batch pricing for
`gpt-4o-mini` / `gpt-4o` at 50% batch discount; actual costs depend on which
models `benchmark/configs/baselines.yaml` pins for each baseline and on the
evidence window size. Ranges assume gpt-4o-mini for picks and gpt-4o where
debate is explicit.

| Method | API calls per FD | Est. cost / FD | Est. total (81 FDs) |
|---|---|---|---|
| B1 direct | 1 | ~$0.01 | ~$0.80 |
| B2 CoT | 1 | ~$0.02 | ~$1.60 |
| B3 RAG | 1 | ~$0.02 | ~$1.60 |
| B3b RAG+claims | 1-2 | ~$0.03 | ~$2.40 |
| B4 self-consistency (k=5) | 5 | ~$0.05 | ~$4.00 |
| B5 multi-agent debate | 4+1 | ~$0.08 | ~$6.50 |
| B6 tree-of-thoughts | 3-5 | ~$0.06 | ~$5.00 |
| B7 reflexion | 2-3 | ~$0.04 | ~$3.20 |
| B8 verbalized confidence | 1 | ~$0.02 | ~$1.60 |
| B9 LLM ensemble (k=3-4) | 3-4 | ~$0.06 | ~$5.00 |
| EMR-ACH (blocked) | n/a | n/a | n/a |
| **B1-B9 total** |  |  | **~$31** |

Note: the user-supplied budget was $5-15 total. The rough estimate above is
~$31 for just B1-B9 at 81 FDs; the user may want to halt even before F3 lands
to re-check model pinnings in `benchmark/configs/baselines.yaml` and reduce
the context window (per-FD article counts) before firing a real batch.

Wall-clock: OpenAI Batch API advertises up to 24 h; typical completion for
small volumes is 1-4 h. Across 10 sequential methods that could be 10-40 h
unless submitted in parallel.

---

## 4. Blockers requiring human intervention

1. **F3 scope (live EMR-ACH wiring) is not landed.** The adapter explicitly
   notes it is gated on F3. No FD-aware `run_emrach.run(...)` entry point
   exists.
2. **MIRAI-only pipeline.** Even if the adapter were un-gated, the underlying
   `src/pipeline/*` modules are hardcoded to the MIRAI 4-hypothesis CAMEO
   schema. An FD with e.g. 3 custom hypothesis labels would fail or be
   silently bucketed into VC/MC/VK/MK.
3. **Missing hybrid retrieval.** MMR + RRF + temporal-decay components claimed
   by the paper are not implemented in `src/pipeline/retrieval.py`.
4. **Missing diagnostic weighting.** The ACH analysis-matrix-A computation
   is not implemented; `primarily_supports` labels are present but no
   diagnosticity scores or weighted aggregation.
5. **Multi-agent debate is single-round.** Paper claims multi-round
   inter-agent argumentation; code is one-shot advocate-then-judge.
6. **Budget mismatch.** Rough B1-B9 estimate at 81 FDs is ~$31, above the
   stated $5-15 budget. Re-check `benchmark/configs/baselines.yaml` model
   pinnings before authorising a real batch.

---

## 5. Recommendation

Do **not** fire Phase 2. Table 1's EMR-ACH column (and Table 4 ablation rows,
which assume a full EMR-ACH to ablate from) cannot be populated until:
* F3 ships the live `run_emrach.run(...)` call path behind
  `emrach_on_gold.py --dry-run=false`.
* The MIRAI-locked pipeline is generalised to arbitrary FD hypothesis sets,
  OR a dedicated FD-native EMR-ACH implementation is written.
* Hybrid retrieval (MMR + RRF + temporal decay) is added to
  `src/pipeline/retrieval.py`.
* Diagnostic weighting is added to `src/pipeline/indicators.py` /
  `aggregation.py`.
* Multi-agent debate is extended beyond the current one-shot advocate-judge.

Running B1-B9 alone would burn budget, produce partial Table 1 data, and
still leave the EMR-ACH headline cell empty. Better to hold until EMR-ACH is
real, then run everything in one coherent pass.

Files of interest for the next engineer:
* `E:/Projects/ACH/scripts/eval/emrach_on_gold.py` (stub)
* `E:/Projects/ACH/src/pipeline/indicators.py` (contrastive: partial, MIRAI-only)
* `E:/Projects/ACH/src/pipeline/multi_agent.py` (debate: single-round, MIRAI-only)
* `E:/Projects/ACH/src/pipeline/retrieval.py` (no hybrid features)
* `E:/Projects/ACH/src/pipeline/aggregation.py` (not audited in depth; likely the natural home for diagnosticity weighting)
* `E:/Projects/ACH/experiments/02_emrach/run_emrach.py` (entry point the adapter is meant to call)
* `E:/Projects/ACH/benchmark/evaluation/baselines/runner.py` (ready)
* `E:/Projects/ACH/benchmark/evaluation/baselines/methods/*.py` (ready)
