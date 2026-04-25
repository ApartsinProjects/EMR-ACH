#!/usr/bin/env python
"""Standalone validator: SHA256 + minimal schema check.

No external dependencies. Run from the gold-folder directory.
Exits non-zero on any mismatch; prints a short report.
"""
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    expected = {}
    with open("checksums.sha256", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                expected[parts[1]] = parts[0]

    failed = 0
    for name, want in expected.items():
        got = sha256(name)
        ok = got == want
        print(f"{'OK ' if ok else 'BAD'}  {name}  (sha256 {got[:16]}{'==' if ok else ' != '}{want[:16]})")
        if not ok:
            failed += 1

    # Minimal schema check: every FD has the required v2.1-gold fields.
    REQUIRED_FD = {"id", "benchmark", "source", "hypothesis_set", "hypothesis_definitions",
                   "question", "forecast_point", "resolution_date", "ground_truth",
                   "ground_truth_idx", "article_ids", "fd_type"}
    n = 0; bad_fd = 0
    with open("forecasts.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            n += 1
            if REQUIRED_FD - set(d.keys()):
                bad_fd += 1
    print(f"\nFDs: {n}; missing-required: {bad_fd}")
    if bad_fd:
        failed += 1

    # Article schema spot-check
    REQUIRED_ART = {"id", "url", "title", "publish_date"}
    n = 0; bad_art = 0
    with open("articles.jsonl", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            n += 1
            if REQUIRED_ART - set(a.keys()):
                bad_art += 1
    print(f"Articles: {n}; missing-required: {bad_art}")
    if bad_art:
        failed += 1

    # ETD fact schema spot-check (file may be empty)
    if Path("facts.jsonl").exists():
        REQUIRED_FACT = {"id", "fact", "time", "primary_article_id",
                         "polarity", "extraction_confidence"}
        n = 0; bad_f = 0
        with open("facts.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                n += 1
                if REQUIRED_FACT - set(d.keys()):
                    bad_f += 1
        print(f"ETD facts: {n}; missing-required: {bad_f}")
        if bad_f:
            failed += 1

    if failed:
        print(f"\nFAIL: {failed} integrity issue(s).")
        return 1
    print("\nOK: integrity verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
