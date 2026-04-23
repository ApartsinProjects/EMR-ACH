"""FD-aware adapter for the main EMR-ACH method on the gold subset.

Per the recommendation in ``docs/GOLD_EVALUATION_AUDIT.md``, this script
shims between the v2.1 benchmark deliverable
(``benchmark/data/{cutoff}[-gold]/forecasts.jsonl`` + ``articles.jsonl``)
and the input shape that ``experiments/02_emrach/run_emrach.py``
historically expects (MIRAI format), then translates EMR-ACH's
predictions back into the baselines runner schema (one JSONL line per FD,
matching ``benchmark/evaluation/baselines/methods/b1_direct.py``).

It deliberately does **not** import ``run_emrach.py`` directly --
that module is in the v2.1 in-flight scope. Instead, this adapter
exposes:

* :func:`fd_to_emrach_query` -- pure data transformation (FD -> MIRAI-shape).
* :func:`emrach_result_to_prediction_row` -- prediction translation back.
* :func:`build_emrach_inputs` -- bulk-transform a forecasts.jsonl bundle.
* :func:`write_predictions_jsonl` -- emits the baselines-runner schema.
* a CLI entry point that wires these together with ``--dry-run`` and
  ``--help``.

When F3's full EMR-ACH wiring lands, the call to ``run_emrach.run(...)``
inside :func:`run_emrach_on_bundle` can be flipped on; until then the
adapter is shape-only and is exercised by the test suite against the
existing parent-cutoff bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def fd_to_emrach_query(fd: dict, articles_by_id: dict[str, dict]) -> dict:
    """Project one FD into the MIRAI-shape query dict EMR-ACH expects.

    The MIRAI shape (per ``src/data/mirai.py``'s ``MiraiDataset``) carries
    a ``query`` (question text), a ``hypotheses`` list, a ``label``
    string, plus a list of ``evidence`` items each with ``id``, ``date``,
    and ``text``. We map the FD primary fields onto that shape and pull
    the article subset referenced by ``fd['article_ids']`` from the
    bundle's articles.jsonl.

    Returns a fresh dict; does not mutate inputs.
    """
    evidence: list[dict] = []
    for aid in fd.get("article_ids", []):
        art = articles_by_id.get(aid)
        if art is None:
            continue
        evidence.append({
            "id": art.get("id", aid),
            "date": art.get("publish_date"),
            "title": art.get("title", ""),
            "text": art.get("text", "") or "",
            "source_domain": art.get("source_domain"),
        })
    return {
        "fd_id": fd.get("id"),
        "benchmark": fd.get("benchmark"),
        "query": fd.get("question", ""),
        "background": fd.get("background"),
        "forecast_point": fd.get("forecast_point"),
        "resolution_date": fd.get("resolution_date"),
        "hypotheses": list(fd.get("hypothesis_set", [])),
        "hypothesis_definitions": dict(fd.get("hypothesis_definitions", {})),
        "label": fd.get("ground_truth"),
        "label_idx": fd.get("ground_truth_idx"),
        "prior_state": fd.get("prior_state"),
        "fd_type": fd.get("fd_type"),
        "default_horizon_days": fd.get("default_horizon_days"),
        "evidence": evidence,
    }


def build_emrach_inputs(
    fds: Iterable[dict],
    articles_by_id: dict[str, dict],
) -> list[dict]:
    """Bulk-transform an FD iterable into the EMR-ACH input list."""
    return [fd_to_emrach_query(fd, articles_by_id) for fd in fds]


def emrach_result_to_prediction_row(
    fd: dict,
    pick: str | None,
    metadata: dict | None = None,
) -> dict:
    """Translate one EMR-ACH per-FD result back into the baselines-runner row.

    The schema mirrors ``benchmark/evaluation/baselines/methods/b1_direct.py``:
    ``{fd_id, prediction, metadata}`` (we add ``benchmark`` for sanity).
    """
    return {
        "fd_id": fd.get("id"),
        "benchmark": fd.get("benchmark"),
        "prediction": pick,
        "metadata": dict(metadata or {}),
    }


def write_predictions_jsonl(rows: Iterable[dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def run_emrach_on_bundle(
    bundle_dir: Path,
    out_path: Path,
    dry_run: bool = True,
    limit: int | None = None,
) -> dict:
    """Top-level entry: load the bundle, transform, optionally predict, write rows.

    In ``--dry-run`` mode (the default until F3 lands) we emit one
    placeholder prediction per FD with ``prediction=None`` and
    ``metadata={"adapter_status": "dry-run"}``, so downstream consumers
    can shape-check end-to-end.
    """
    fds_path = bundle_dir / "forecasts.jsonl"
    arts_path = bundle_dir / "articles.jsonl"
    if not fds_path.exists():
        raise FileNotFoundError(f"forecasts.jsonl not found at {fds_path}")
    if not arts_path.exists():
        raise FileNotFoundError(f"articles.jsonl not found at {arts_path}")

    fds = _read_jsonl(fds_path)
    if limit is not None:
        fds = fds[:limit]
    articles = _read_jsonl(arts_path)
    articles_by_id = {a["id"]: a for a in articles if "id" in a}

    queries = build_emrach_inputs(fds, articles_by_id)

    if dry_run:
        rows = [
            emrach_result_to_prediction_row(
                fd, pick=None, metadata={"adapter_status": "dry-run", "evidence_n": len(q["evidence"])}
            )
            for fd, q in zip(fds, queries)
        ]
    else:  # pragma: no cover - real EMR-ACH path is in-flight per F3 scope
        raise NotImplementedError(
            "Live EMR-ACH invocation is gated on F3's pipeline wiring; "
            "run with --dry-run until that lands."
        )

    n_written = write_predictions_jsonl(rows, out_path)
    return {
        "fds_in": len(fds),
        "queries_built": len(queries),
        "rows_written": n_written,
        "out_path": str(out_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="emrach_on_gold",
        description=(
            "FD-aware adapter that runs EMR-ACH against a v2.1 benchmark bundle "
            "(parent or gold) and emits baselines-runner-compatible predictions."
        ),
    )
    p.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to a benchmark bundle dir containing forecasts.jsonl + articles.jsonl.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output predictions JSONL path.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Shape-check only; emit placeholder predictions without invoking EMR-ACH.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of FDs processed (debug).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run:
        print(
            "[emrach-on-gold] live mode is gated on F3's EMR-ACH wiring; "
            "use --dry-run for now.",
            file=sys.stderr,
        )
        return 2
    summary = run_emrach_on_bundle(
        bundle_dir=args.bundle,
        out_path=args.out,
        dry_run=True,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
