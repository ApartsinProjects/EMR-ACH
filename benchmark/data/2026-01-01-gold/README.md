# EMR-ACH Gold Subset, cutoff `2026-01-01`

A small, balanced, high-quality forecasting benchmark curated from the
EMR-ACH v2.1 release. Self-contained: nothing in this folder references
files outside the folder.

## TL;DR

- **81 Forecast Dossiers** across 3 benchmarks (geopolitics, finance, prediction markets), unified primary target **Comply vs Surprise**.
- **690 retrieved news articles**, full body text inline.
- **ETD atomic facts** (`facts.jsonl`) linked to those articles, providing a structured-evidence channel alongside the raw text. May be empty if Stage 1 ETD has not been run for the parent cutoff.
- **2-week forecast horizon** strictly enforced: every article is dated at least 14 days before the resolution date.
- **fd_type stratified** ({stability, change}) so the headline metric on the change subset measures real forecasting skill, not status-quo persistence.

## Files

- `forecasts.jsonl` (81 records). Each line is one FD.
- `articles.jsonl` (690 records). Each line is one article.
- `facts.jsonl`. ETD atomic facts whose `primary_article_id` is in `articles.jsonl`. One line per fact.
- `schema/`: full JSON Schema (Draft 2020-12) for all three record types (FD + article + ETD fact).
- `examples/load.py`: 50-line stdlib-only loader (loads FDs + articles + facts).
- `examples/validate.py`: SHA + schema validator (covers all three).
- `examples/eval_template.py`: pick-only evaluation skeleton.
- `selection_criteria.md`: exact filter cascade.
- `build_manifest.json`: counts + git_sha + thresholds + parent SHA.
- `meta/`: selection audit + distribution stats.

## Quickstart

```python
import json
from pathlib import Path

fds = [json.loads(l) for l in open("forecasts.jsonl", encoding="utf-8")]
arts = {a["id"]: a for a in (json.loads(l) for l in open("articles.jsonl", encoding="utf-8"))}

for fd in fds[:3]:
    print(fd["id"], fd["benchmark"], fd["hypothesis_set"], "ground_truth:", fd["ground_truth"])
    for aid in fd["article_ids"][:2]:
        a = arts[aid]
        print(f"   - {a['publish_date']} {a['url']}")
```

See `examples/usage.md` for richer queries and `examples/eval_template.py` for a baseline-runner skeleton.

## Data card

| Benchmark | n FDs | stability | change |
|---|---:|---:|---:|
| `earnings` | 1 | 0 | 1 |
| `forecastbench` | 77 | 60 | 17 |
| `gdelt-cameo` | 3 | 2 | 1 |

## Provenance

- Parent: `benchmark/data/2026-01-01/forecasts.jsonl` (SHA256 in `meta/parent_manifest_sha256.txt`).
- Parent SHA256: `39a2ebf87f541331...`
- Build: see `build_manifest.json` for git_sha + timestamps.

## License

MIT. See `LICENSE`. Citation: `CITATION.cff`.

## Schema versioning

Schema version `2.1-gold` (see `schema/FIELD_REFERENCE.md`). Backwards-compatible additions only; field removals would bump to `3.0`.
