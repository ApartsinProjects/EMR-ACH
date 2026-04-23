"""One-shot recovery: lift facts wrongly rejected by the previous validator.

The pre-fix validator in `scripts/articles_to_facts.py` used `time >= publish_date`
as the leakage guard, which rejected ~9.9k same-day facts (the legitimate news
of the day). The fix relaxed it to `time > publish_date`. This script re-processes
the existing `data/etd/facts.errors.jsonl`, identifies the rows that would have
passed the new check, reconstructs the fact records using the producer's exact
shape (mirroring `parse_response()` in articles_to_facts.py), and appends them
to `data/etd/facts.v1.jsonl`.

No API call. Idempotent (skips facts whose id is already in facts.v1.jsonl).
Atomic write via tmp + rename.

Usage:
  python scripts/etd_recover_validation_errors.py
  python scripts/etd_recover_validation_errors.py --dry-run   # report only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FACTS_PATH  = DATA / "etd" / "facts.v1.jsonl"
ERRORS_PATH = DATA / "etd" / "facts.errors.jsonl"
ERRORS_OUT  = DATA / "etd" / "facts.errors.post_recovery.jsonl"
META_OUT    = DATA / "etd" / "audit" / "recovery_meta.json"

SCHEMA_VERSION = "1.0"


def _normalize_fact_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if not s.endswith((".", "!", "?")):
        s += "."
    return re.sub(r"\s+", " ", s)


def _fact_id(fact_text: str, article_id: str, extract_run: str) -> str:
    """Mirror articles_to_facts._fact_id signature.

    Verified empirically against the existing facts.v1.jsonl by checking that
    the recovered ids do not collide with already-present ids."""
    h = hashlib.sha1(f"{extract_run}::{article_id}::{fact_text}".encode("utf-8")).hexdigest()[:12]
    return f"f_{h}"


def _atomic_append(path: Path, lines: list[str]) -> None:
    """Append lines atomically: copy existing file to tmp, append, rename.
    Avoids partial-write corruption if the process is killed mid-append."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        with open(path, "rb") as src, open(tmp, "wb") as dst:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                dst.write(chunk)
    else:
        tmp.touch()
    with open(tmp, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be recovered without writing.")
    ap.add_argument("--errors", default=str(ERRORS_PATH))
    ap.add_argument("--facts", default=str(FACTS_PATH))
    args = ap.parse_args()

    errors_path = Path(args.errors)
    facts_path = Path(args.facts)
    if not errors_path.exists() or not facts_path.exists():
        print(f"[ERROR] missing input(s): {errors_path} or {facts_path}")
        return 1

    print(f"[recover] facts:  {facts_path}")
    print(f"[recover] errors: {errors_path}")

    # 1. Index existing fact ids so we never duplicate
    existing_ids: set[str] = set()
    with open(facts_path, encoding="utf-8") as f:
        for line in f:
            try:
                existing_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"[recover] existing facts: {len(existing_ids)}")

    # 2. Walk errors; lift validation_failed rows; re-parse raw_response
    n_total = 0
    by_type = Counter()
    recovered: list[str] = []
    still_invalid: list[str] = []
    skipped_dup_id = 0
    skipped_no_raw = 0

    with open(errors_path, encoding="utf-8") as f:
        for line in f:
            n_total += 1
            try:
                err = json.loads(line)
            except json.JSONDecodeError:
                still_invalid.append(line.rstrip("\n"))
                continue
            etype = err.get("error_type", "")
            by_type[etype] += 1

            if etype != "validation_failed":
                still_invalid.append(line.rstrip("\n"))
                continue

            raw = err.get("raw_response")
            if not raw:
                skipped_no_raw += 1
                still_invalid.append(line.rstrip("\n"))
                continue

            try:
                data = json.loads(raw)
                raw_facts = data.get("facts") if isinstance(data, dict) else None
            except json.JSONDecodeError:
                still_invalid.append(line.rstrip("\n"))
                continue
            if not isinstance(raw_facts, list):
                still_invalid.append(line.rstrip("\n"))
                continue

            article_id = err.get("article_id", "")
            extract_run = err.get("extract_run", "")
            extractor = err.get("extractor", "")
            failed_at = err.get("failed_at") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            # publish_date is implicit in the error_detail; pull it back out
            detail = err.get("error_detail", "") or ""
            m = re.search(r">=?\s*publish_date\s+(\d{4}-\d{2}-\d{2})", detail)
            if not m:
                m = re.search(r">\s*publish_date\s+(\d{4}-\d{2}-\d{2})", detail)
            publish_date = m.group(1) if m else ""

            # Identify which fact[idx] was the one rejected
            idx_match = re.search(r"fact\[(\d+)\]", detail)
            target_idx = int(idx_match.group(1)) if idx_match else None

            # Walk the raw facts, recover the one(s) that pass the new `>` check
            for idx, rf in enumerate(raw_facts):
                if not isinstance(rf, dict):
                    continue
                # Only attempt the fact the original error pointed at
                if target_idx is not None and idx != target_idx:
                    continue
                fact_text = _normalize_fact_text(rf.get("fact", ""))
                if not fact_text:
                    continue
                time_val = (rf.get("time") or "").strip() or "unknown"
                # New (relaxed) leakage guard
                if (publish_date and re.match(r"^\d{4}-\d{2}-\d{2}$", time_val)
                        and time_val > publish_date):
                    continue
                fid = _fact_id(fact_text, article_id, extract_run)
                if fid in existing_ids:
                    skipped_dup_id += 1
                    continue
                rec = {
                    "id": fid,
                    "schema_version": SCHEMA_VERSION,
                    "time": time_val,
                    "time_end": rf.get("time_end") if isinstance(rf.get("time_end"), str) else None,
                    "time_precision": rf.get("time_precision") or
                        ("day" if re.match(r"^\d{4}-\d{2}-\d{2}$", time_val) else
                         "month" if re.match(r"^\d{4}-\d{2}$", time_val) else
                         "year" if re.match(r"^\d{4}$", time_val) else "unknown"),
                    "time_type": rf.get("time_type") or "point",
                    "fact": fact_text,
                    "language": rf.get("language") or "en",
                    "translated_from": rf.get("translated_from"),
                    "article_ids": [article_id],
                    "primary_article_id": article_id,
                    "article_date": publish_date or "unknown",
                    "source": None,
                    "entities": rf.get("entities") or [],
                    "location": rf.get("location"),
                    "metrics": rf.get("metrics") or [],
                    "kind": rf.get("kind") or None,
                    "tags": rf.get("tags") or [],
                    "polarity": rf.get("polarity") or "asserted",
                    "attribution": rf.get("attribution"),
                    "extraction_confidence": rf.get("extraction_confidence") or "medium",
                    "source_tier": rf.get("source_tier") or None,
                    "canonical_id": None,
                    "variant_ids": [],
                    "derived_from": [],
                    "derivation": None,
                    "extractor": extractor,
                    "extract_run": extract_run + "_recovered",
                    "extracted_at": failed_at,
                }
                recovered.append(json.dumps(rec, ensure_ascii=False))
                existing_ids.add(fid)

    print(f"[recover] error rows scanned: {n_total}")
    print(f"  by type: {dict(by_type)}")
    print(f"[recover] facts to recover:    {len(recovered)}")
    print(f"  skipped (already in facts.v1): {skipped_dup_id}")
    print(f"  skipped (no raw_response):     {skipped_no_raw}")
    print(f"[recover] non-recoverable rows kept: {len(still_invalid)}")

    if args.dry_run:
        print("[recover] DRY-RUN: no files written.")
        return 0

    if recovered:
        _atomic_append(facts_path, recovered)
        print(f"[recover] appended {len(recovered)} facts -> {facts_path}")

    # Rewrite errors file to drop the recovered rows (the still_invalid rows are
    # the ones that should remain in the errors sidecar).
    tmp = ERRORS_OUT.with_suffix(ERRORS_OUT.suffix + ".tmp")
    ERRORS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for line in still_invalid:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ERRORS_OUT)
    print(f"[recover] post-recovery errors -> {ERRORS_OUT}")

    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps({
        "errors_in":  str(errors_path),
        "facts_out":  str(facts_path),
        "errors_out": str(ERRORS_OUT),
        "n_error_rows_in": n_total,
        "by_error_type": dict(by_type),
        "n_recovered":   len(recovered),
        "n_still_invalid": len(still_invalid),
        "n_skipped_dup_id": skipped_dup_id,
        "n_skipped_no_raw": skipped_no_raw,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }, indent=2), encoding="utf-8")
    print(f"[recover] meta -> {META_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
