# v2.1 -> v2.2 Migration Guide (D1)

Step-by-step migration for sites upgrading from a v2.1 tree. Designed
to be safe to read mid-cycle: every section says whether the step is
blocking for a build or safe to defer.

## 1. Pre-flight

Before upgrading:

1. Let any in-flight v2.1 build finish. v2.2's cache invariants
   (see [`CACHE_INVARIANTS.md`](CACHE_INVARIANTS.md)) are a strict
   relaxation of v2.1's; upgrading mid-build is safe but not tested.
2. Run `python scripts/reuse_check.py --cutoff <cutoff>` to preview
   which stages would reuse vs rerun. No side effects.

## 2. Switch to the v2.2 `sys.path` helper (B4a)

Every script under `scripts/` that imports `from src.common.* import ...`
should replace its local `sys.path.insert` block with:

```python
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from src.common.paths import bootstrap_sys_path
bootstrap_sys_path()
```

The first three lines are the minimal prelude (self-bootstrap so
`src.common.paths` is importable); the helper then takes over and
provides idempotency + fail-fast assertions. This pattern prevents
the `ModuleNotFoundError: No module named 'src'` crashes that hit
three scripts on 2026-04-23.

## 3. Adopt the preflight + checksum gates (G2 + G3 + G5)

Between your v2.1 fetcher completion and your publish step, add:

```bash
python scripts/preflight_publish.py --cutoff $CUTOFF
python scripts/check_quality_filter.py --cutoff $CUTOFF \
    --before  $STAGING/forecasts.jsonl \
    --after   $STAGING/forecasts_filtered.jsonl \
    --quality-meta $STAGING/quality_meta.json
```

Non-zero exit halts the publish. Both scripts are read-only audits;
they do not edit the in-flight build orchestrator.

## 4. Switch to the embeddings-backend API (A6)

Call sites that currently import `sentence_transformers` directly
can migrate to:

```python
from src.common.embeddings_backend import encode

res = encode(texts, backend="sbert")      # or backend="openai_batch"
vectors = res.vectors
```

Backward-compat: the existing `compute_relevance.py:embed` remains and
continues to work; the new API is additive. The `openai_batch` backend
is the Batch API path with 50% discount.

## 5. Adopt the B10 / B10b baselines (F1 + F2) for evaluation

Add to your `baselines.yaml`:

```yaml
b10_hybrid_facts_articles:
  class: benchmark.evaluation.baselines.methods.b10_hybrid_facts_articles.B10HybridFactsArticles
  max_facts: 20
  max_articles: 10

b10b_facts_only:
  class: benchmark.evaluation.baselines.methods.b10b_facts_only.B10bFactsOnly
  max_facts: 20
```

Requires the production fact set to be linked into the FDs via
`fd["facts"]`; the existing `scripts/etd_post_publish.py` already
populates this when `--require-linked-fd` is set.

## 6. Rollback procedure

Every v2.2 ship in this pass is additive (new files only); rolling
back is as simple as reverting the commits. No data files move, no
schemas change. See the commit log with `git log --grep 'v2.2 \['`
for the per-item audit trail.
