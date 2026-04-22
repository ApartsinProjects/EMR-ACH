"""Poll OpenAI Batch API status for all in-flight batches and print a summary.

Reads batch IDs from src/batch_client's cache (results/raw/batch_jobs.json or
similar) plus listing recent batches via the OpenAI API directly.

Usage:
  python scripts/poll_openai_batches.py              # one-shot status snapshot
  python scripts/poll_openai_batches.py --watch 300  # loop every 300s (5min)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load_env():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.open(encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip())


def snapshot() -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("[poll] openai SDK not installed; pip install openai")
        sys.exit(1)
    client = OpenAI()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== OpenAI batch snapshot @ {ts} ===")
    try:
        # Most recent 20 batches
        resp = client.batches.list(limit=20)
    except Exception as e:
        print(f"[poll] list failed: {e}")
        return
    in_flight = 0
    for b in resp.data:
        created = datetime.utcfromtimestamp(b.created_at).strftime("%Y-%m-%d %H:%M:%S")
        rc = getattr(b, "request_counts", None)
        counts = ""
        if rc:
            counts = f"{rc.completed}/{rc.total} completed, {rc.failed} failed"
        flag = "[IN-FLIGHT]" if b.status in ("in_progress", "finalizing", "validating") else ""
        if flag:
            in_flight += 1
        print(f"  {b.id[:24]}...  status={b.status:<14} created={created}  {counts}  {flag}")
    print(f"[poll] in-flight: {in_flight}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0,
                    help="Watch mode: poll every N seconds. 0 = one-shot.")
    args = ap.parse_args()
    load_env()
    while True:
        snapshot()
        if args.watch <= 0:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
