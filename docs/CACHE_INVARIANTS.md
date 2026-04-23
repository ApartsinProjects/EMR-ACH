# Cache Invariants (v2.2 [D3])

This doc is the invalidation matrix for every cache surface introduced
or formalized in v2.2. It elevates the table from
[`V2_2_ARCHITECTURE.md`](V2_2_ARCHITECTURE.md) Section 7.2 into a
standalone reference with worked examples.

## Cache surfaces

| Cache | Key | Storage | Invalidates on |
|---|---|---|---|
| Stage meta (G1) | `stage_cache_key(config, stage)` (see B5) | `data/stage_meta/{cutoff}/{stage}.json` | any slice key listed in `STAGE_SLICES[stage]`; any recorded output missing |
| SBERT pool embedding | `MD5(texts || model || batch_size)` | `data/cache/embeddings/sbert/{hash}.npy` | text corpus change; SBERT model revision change |
| OpenAI pool embedding | `MD5(texts || model)` | `data/cache/embeddings/openai/{hash}.npy` | text corpus change; OpenAI model name change |
| Per-FD query embedding (A9) | `MD5(question || background || backend || model)` | `data/cache/query_embeddings/{backend}/{model}/{key}.npy` | question or background edit; backend swap; model swap |
| GDELT DOC FAISS index (A2) | per-shard `manifest.json` with model + revision | `data/gdelt_doc/index/{YYYY-MM}/` | backend swap; model revision change; upstream raw shard replacement |
| Per-benchmark article checksum (G2) | SHA256 of `{bench}_articles.jsonl` | `data/{bench}/{bench}_articles.checksums.json` | article file mtime or line-count change |

## Invalidation triggers

1. **Config slice mutation.** The slice defined in
   [`src/common/config_slices.py`](../src/common/config_slices.py) for a
   stage lists exactly the dotted config keys that affect that stage's
   output. Mutating any key in the slice flips the stage's
   `cache_key` and triggers a rerun. Mutating a key outside the slice
   is a no-op for that stage; see
   [`tests/test_config_slices.py`](../tests/test_config_slices.py) for
   the regression guarantee.
2. **Output deletion.** `StageCache.is_valid` (G1) returns `False` if
   any output path recorded at `record()` time is missing. This catches
   the 2026-04-22 earnings-articles deletion failure mode on resume.
3. **Backend swap.** Every embedding cache key includes the backend
   name. Swapping from SBERT to OpenAI does NOT invalidate the SBERT
   cache (parallel-cache approach per review REC-03); the OpenAI path
   populates a sibling directory.
4. **Model revision bump.** The GDELT DOC index manifest records
   `model_revision`; a revision change forces a rebuild. SBERT
   revisions are best-effort via `huggingface_hub`.

## Worked examples

### Editing `relevance.top_k_per_fd`

* Slice: `compute_relevance`.
* Effect: invalidates the relevance cache for every cutoff, forces
  rerun of `compute_relevance`. Downstream stages (`annotate_prior_state`,
  `articles_to_facts`) invalidate because their inputs change.
* Unrelated stages (e.g. `fetch_forecastbench`) remain valid.

### Editing `unrelated_key.foo`

* Slice: none.
* Effect: no stage's `cache_key` changes. No reruns triggered.

### Swapping backend from SBERT to OpenAI

* Slice: `compute_relevance.relevance.encoder.backend`.
* Effect: invalidates `compute_relevance` stage. SBERT cache remains on
  disk untouched. OpenAI cache populates a parallel directory.
* The published `build_manifest.json` records `embedding_backend =
  openai` (C3); the cutoff directory is treated as immutable with
  respect to backend (refuse to mix backends within one `{cutoff}/`).

### Deleting `data/earnings/earnings_articles.jsonl` mid-build

* Stage: `unify_articles` would pick up the missing file at load time.
* With G2 preflight_publish: the assertion fails and the build halts.
* Without G2: `unify_articles` silently produces a short-pool output
  and the stage completes; this is the exact failure that motivated
  G2 + G5.

## Migration note for v2.1 -> v2.2

The v2.1 resume logic rehashed the entire config and invalidated
everything on any edit. Under v2.2, stages that are unaffected by an
edit now correctly reuse their cache. This is a strict relaxation of
the v2.1 behaviour, so the first rebuild after upgrading MAY reuse
stages that previously would have rerun; review
`python scripts/reuse_check.py --cutoff <cutoff>` before launching
the first v2.2 build if that is a concern.
