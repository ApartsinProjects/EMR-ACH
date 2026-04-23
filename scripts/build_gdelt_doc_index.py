#!/usr/bin/env python
"""scripts/build_gdelt_doc_index.py: SBERT + FAISS per-shard index
(v2.2 [A2]).

For each monthly shard produced by A1 (``data/gdelt_doc/raw/{YYYY-MM}/
shard.jsonl.zst``), encode article titles + leads with SBERT (via
:mod:`src.common.embeddings_backend`) and write:

* ``data/gdelt_doc/index/{YYYY-MM}/faiss.flatip`` -- L2-normalized IP
  index (cosine-equivalent).
* ``data/gdelt_doc/index/{YYYY-MM}/meta.parquet`` -- row-aligned
  metadata (article_id, url, domain, publish_date, language).
* ``data/gdelt_doc/index/manifest.json`` -- model name, model revision,
  dim, backend, shard list.

Per the v2.2 review's REC-03, the backend is A6's unified ``encode()``
API so the SBERT vs OpenAI swap is a config knob.

This pass ships the orchestrator + CLI surface with ``--dry-run``
default so the heavy GPU/FAISS work does not run in an agent context.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from src.common.paths import DATA_DIR, bootstrap_sys_path  # noqa: E402

bootstrap_sys_path()

from src.common.embeddings_backend import backend_identity  # noqa: E402

DEFAULT_RAW_ROOT = DATA_DIR / "gdelt_doc" / "raw"
DEFAULT_INDEX_ROOT = DATA_DIR / "gdelt_doc" / "index"


@dataclass(frozen=True)
class ShardIndexSpec:
    month: str
    raw_path: Path
    index_dir: Path

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "faiss.flatip"

    @property
    def meta_path(self) -> Path:
        return self.index_dir / "meta.parquet"


def _discover_shards(raw_root: Path, index_root: Path) -> list[ShardIndexSpec]:
    if not raw_root.exists():
        return []
    out: list[ShardIndexSpec] = []
    for child in sorted(raw_root.iterdir()):
        if not child.is_dir():
            continue
        raw = child / "shard.jsonl.zst"
        if not raw.exists():
            continue
        out.append(ShardIndexSpec(
            month=child.name, raw_path=raw,
            index_dir=index_root / child.name,
        ))
    return out


def build_shard(spec: ShardIndexSpec, *, backend: str, model: str | None,
                dry_run: bool) -> dict:
    if dry_run:
        return {
            "shard": spec.month,
            "status": "dry-run (no encode)",
            "planned_faiss": str(spec.faiss_path),
            "planned_meta": str(spec.meta_path),
        }
    raise NotImplementedError(
        "Real index build is not enabled in this build pass. "
        "Implement: (1) stream decompressed JSONL, (2) encode via "
        "src.common.embeddings_backend.encode(..., backend=backend, "
        "model=model), (3) write FAISS flat-IP index + parquet "
        "metadata row-aligned with the embedding matrix."
    )


def write_manifest(index_root: Path, shards: list[ShardIndexSpec],
                   *, backend: str, model: str | None) -> Path:
    backend_name, model_name, model_revision = backend_identity(
        backend,  # type: ignore[arg-type]
        model,
    )
    manifest = {
        "backend": backend_name,
        "model": model_name,
        "model_revision": model_revision,
        "shards": [s.month for s in shards],
    }
    index_root.mkdir(parents=True, exist_ok=True)
    path = index_root / "manifest.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(path)
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    p.add_argument("--backend", default="sbert",
                   choices=["sbert", "openai", "openai_batch"])
    p.add_argument("--model", default=None)
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    shards = _discover_shards(args.raw_root, args.index_root)
    results = []
    for s in shards:
        try:
            results.append(build_shard(
                s, backend=args.backend, model=args.model,
                dry_run=args.dry_run,
            ))
        except NotImplementedError as exc:
            sys.stderr.write(f"{exc}\n")
            return 3
    manifest_path = write_manifest(
        args.index_root, shards, backend=args.backend, model=args.model,
    )
    sys.stdout.write(json.dumps({
        "raw_root": str(args.raw_root),
        "index_root": str(args.index_root),
        "manifest": str(manifest_path),
        "shards": [s.month for s in shards],
        "results": results,
        "dry_run": args.dry_run,
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
