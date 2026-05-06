"""Watchdog that drives the v2.1 lock-in sequence to completion.

Polls every 10 min:
  1. OpenAI chunk_00 status (chunk_01 already completed + downloaded).
  2. Whether the resume chain (etd_post_publish) has produced its expected
     downstream artefacts: facts.v1_canonical.jsonl, facts.v1_linked.jsonl,
     facts.v1_production.jsonl.
  3. Whether the gold subset has been built (benchmark/data/{cutoff}-gold/).

When each gate passes, it auto-fires the next pending step:
  - chunk_00 complete + chain produces artefacts -> commit ETD outputs
  - ETD committed -> run build_gold_subset.py
  - Gold built -> commit gold folder
  - All done -> exit 0 with "READY FOR PUSH+TAG" log line

Does NOT push or tag (requires explicit user approval).

Each loop iteration logs to stdout. Designed to be run in a background bash
task so the user can poll status by tailing the log.

Usage:
  python scripts/watchdog_etd_to_gold.py --cutoff 2026-01-01
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ETD = ROOT / "data" / "etd"
PY = sys.executable

CHUNK_00_BATCH_ID = "batch_69e9fe9cb8b0819097eb2486ec6b6025"
CHUNK_01_BATCH_ID = "batch_69ea01ab210c819084ee003e65a349d5"

POLL_SECONDS = 600  # 10 min


def _log(msg: str) -> None:
    print(f"[{datetime.utcnow().isoformat(timespec='seconds')}Z watchdog] {msg}", flush=True)


def _load_openai_key() -> None:
    """Force the proj key from .env (overriding any stale shell env)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
            return


def _batch_status(batch_id: str) -> tuple[str, int, int]:
    """Returns (status, completed, total)."""
    from openai import OpenAI
    b = OpenAI().batches.retrieve(batch_id)
    rc = b.request_counts
    return (b.status, rc.completed, rc.total)


def _run(cmd: list[str], cwd: Path = ROOT, check: bool = True) -> int:
    _log(f"$ {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd))
    if check and p.returncode != 0:
        _log(f"FAILED (exit {p.returncode}): {' '.join(cmd)}")
    return p.returncode


def _git(args: list[str]) -> tuple[int, str]:
    p = subprocess.run(["git"] + args, cwd=str(ROOT),
                       capture_output=True, text=True)
    return (p.returncode, p.stdout.strip() + p.stderr.strip())


def _ensure_etd_committed() -> bool:
    """If facts.v1_canonical/linked/production exist and aren't committed yet,
    stage and commit them. Returns True if commit was made (or files already
    committed); False if files don't exist yet."""
    targets = [
        DATA_ETD / "facts.v1_canonical.jsonl",
        DATA_ETD / "facts.v1_linked.jsonl",
        DATA_ETD / "facts.v1_production.jsonl",
        DATA_ETD / "dedup_meta.json",
        DATA_ETD / "filter_meta.json",
    ]
    present = [p for p in targets if p.exists()]
    if not (DATA_ETD / "facts.v1_canonical.jsonl").exists():
        return False

    # Anything to commit?
    rc, out = _git(["status", "--porcelain"] + [str(p.relative_to(ROOT)) for p in present])
    dirty = [line for line in out.splitlines() if line.strip()]
    if not dirty:
        _log("ETD outputs already in tree (no diff to commit)")
        return True

    # Stage + commit
    rel = [str(p.relative_to(ROOT)).replace("\\", "/") for p in present]
    audit_dir = DATA_ETD / "audit"
    if audit_dir.exists():
        rel.append(str(audit_dir.relative_to(ROOT)).replace("\\", "/"))
    _git(["add"] + rel)
    msg = (
        "v2.2 ETD post-publish: facts.v1_canonical + linked + production\n\n"
        "Auto-committed by scripts/watchdog_etd_to_gold.py after the\n"
        "etd_post_publish chain (resume chain bvo7owfye) completed.\n\n"
        "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
    )
    rc, out = _git(["commit", "-m", msg])
    _log(f"git commit -> rc={rc} {out[:200]}")
    return rc == 0


def _build_gold(cutoff: str) -> bool:
    gold_dir = ROOT / "benchmark" / "data" / f"{cutoff}-gold"
    if gold_dir.exists() and (gold_dir / "forecasts.jsonl").exists():
        _log(f"gold subset already exists at {gold_dir}")
        return True
    rc = _run([PY, "scripts/build_gold_subset.py", "--cutoff", cutoff], check=False)
    return rc == 0 and (gold_dir / "forecasts.jsonl").exists()


def _commit_gold(cutoff: str) -> bool:
    rel = f"benchmark/data/{cutoff}-gold/"
    if not (ROOT / rel).exists():
        return False
    rc, out = _git(["status", "--porcelain", rel])
    if not out.strip():
        _log("gold subset already committed")
        return True
    _git(["add", rel])
    msg = (
        f"v2.2 gold subset: build benchmark/data/{cutoff}-gold/\n\n"
        f"Auto-committed by scripts/watchdog_etd_to_gold.py after\n"
        f"scripts/build_gold_subset.py --cutoff {cutoff} produced the\n"
        f"self-contained subset folder (forecasts + articles + facts +\n"
        f"schema + examples + README).\n\n"
        "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
    )
    rc, out = _git(["commit", "-m", msg])
    _log(f"git commit gold -> rc={rc} {out[:200]}")
    return rc == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-01-01")
    ap.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    ap.add_argument("--max-iters", type=int, default=144,  # 24 hr at 10-min
                    help="Safety cap on poll iterations.")
    args = ap.parse_args()

    _load_openai_key()
    key_pref = (os.environ.get("OPENAI_API_KEY") or "")[:14]
    _log(f"started; cutoff={args.cutoff}, poll={args.poll_seconds}s, key={key_pref}")

    iters = 0
    while iters < args.max_iters:
        iters += 1
        try:
            status, done, total = _batch_status(CHUNK_00_BATCH_ID)
            _log(f"iter {iters}: chunk_00 {status} {done}/{total}")
        except Exception as e:
            _log(f"iter {iters}: poll error: {e!r}")
            time.sleep(args.poll_seconds)
            continue

        if status not in ("completed", "failed", "expired", "cancelled"):
            time.sleep(args.poll_seconds)
            continue

        if status != "completed":
            _log(f"chunk_00 ended in {status}; aborting watchdog")
            return 2

        # chunk_00 done. Now wait for the resume chain to write its outputs.
        canonical = DATA_ETD / "facts.v1_canonical.jsonl"
        linked    = DATA_ETD / "facts.v1_linked.jsonl"
        prod      = DATA_ETD / "facts.v1_production.jsonl"

        if not canonical.exists():
            _log("chunk_00 done; waiting for facts.v1_canonical.jsonl from resume chain")
            time.sleep(args.poll_seconds)
            continue

        if not linked.exists():
            _log("canonical present; waiting for facts.v1_linked.jsonl (Stage 3)")
            time.sleep(args.poll_seconds)
            continue

        if not prod.exists():
            _log("linked present; waiting for facts.v1_production.jsonl (filter step)")
            time.sleep(args.poll_seconds)
            continue

        _log("all ETD artefacts present -> committing")
        if not _ensure_etd_committed():
            _log("ETD commit failed; will retry next iter")
            time.sleep(args.poll_seconds)
            continue

        _log(f"building gold subset for cutoff {args.cutoff}")
        if not _build_gold(args.cutoff):
            _log("gold build failed; will retry next iter")
            time.sleep(args.poll_seconds)
            continue

        if not _commit_gold(args.cutoff):
            _log("gold commit failed; will retry next iter")
            time.sleep(args.poll_seconds)
            continue

        _log("===== READY FOR PUSH+TAG =====")
        _log("Outstanding manual steps:")
        _log("  1. review uncommitted benchmark/evaluation/* + data/* diffs")
        _log("  2. tidy commit (paper SHAs already in tree)")
        _log("  3. git push origin master")
        _log("  4. git tag v2.1-data-ready")
        _log("  5. git push origin v2.1-data-ready")
        return 0

    _log(f"hit --max-iters={args.max_iters} without completion; exiting 1")
    return 1


if __name__ == "__main__":
    sys.exit(main())
