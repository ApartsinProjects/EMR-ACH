"""Bypass a stuck OpenAI Batch chunk by sync-encoding its inputs locally.

Reads a chunk's `requests.jsonl` (the same JSONL used to submit the batch),
calls the sync embeddings endpoint, and writes a synthetic `output.jsonl`
in the same shape OpenAI's batch produces. Once that file exists,
`encode_batch`'s LOCAL-COMPLETED short-circuit (Phase 2) treats the chunk
as done and Phase 3 reassembles from disk without ever calling
files.content().

Use case: a batch stalls in `in_progress` at 99%+ for tens of minutes with
no upstream forward motion. Sync runs at OpenAI's full RPM/TPM and
typically finishes 50k inputs in a few minutes for ~$0.10 (no batch
discount; trivial).

Usage:
  python scripts/offline_sync_chunk.py \
      --chunk-dir data/etd_openai_batches/text-embedding-3-small/chunk_00 \
      --model text-embedding-3-small
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True,
                    help="Path to chunk directory containing requests.jsonl.")
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--chunk-size", type=int, default=200,
                    help="Inputs per sync request.")
    ap.add_argument("--max-attempts", type=int, default=5,
                    help="Retry attempts per chunk on transient errors.")
    args = ap.parse_args()

    _load_dotenv()
    chunk_dir = Path(args.chunk_dir)
    req_path = chunk_dir / "requests.jsonl"
    out_path = chunk_dir / "output.jsonl"

    if not req_path.exists():
        print(f"[ERROR] {req_path} not found")
        return 1

    if out_path.exists() and out_path.stat().st_size > 0:
        # Count rows; bail if already complete
        with out_path.open(encoding="utf-8") as fh:
            existing = sum(1 for _ in fh)
        with req_path.open(encoding="utf-8") as fh:
            requested = sum(1 for _ in fh)
        if existing >= requested:
            print(f"[skip] {out_path} already has {existing} rows >= {requested}")
            return 0
        print(f"[warn] {out_path} has {existing}/{requested} rows; "
              f"overwriting with full sync run")

    # Read all requests, preserving order; capture (custom_id, text)
    items: list[tuple[str, str]] = []
    with req_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            cid = r["custom_id"]
            text = r["body"]["input"]
            if isinstance(text, list):
                text = text[0] if text else " "
            items.append((cid, text))
    n = len(items)
    print(f"[offline_sync] loaded {n} requests from {req_path}")

    from openai import OpenAI
    client = OpenAI()

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    t0 = time.time()
    written = 0
    with tmp_path.open("w", encoding="utf-8") as out_fh:
        for i in range(0, n, args.chunk_size):
            batch = items[i:i + args.chunk_size]
            batch_texts = [t if t and t.strip() else " " for _, t in batch]
            for attempt in range(args.max_attempts):
                try:
                    resp = client.embeddings.create(model=args.model, input=batch_texts)
                    break
                except Exception as e:
                    if attempt == args.max_attempts - 1:
                        raise
                    wait = 2 ** attempt
                    print(f"  [retry {attempt+1}/{args.max_attempts}] sleep {wait}s: {e}")
                    time.sleep(wait)
            for (cid, _), data in zip(batch, resp.data):
                row = {
                    "id": f"sync_{cid}",
                    "custom_id": cid,
                    "response": {
                        "status_code": 200,
                        "request_id": f"sync_{cid}",
                        "body": {
                            "object": "list",
                            "data": [
                                {
                                    "object": "embedding",
                                    "index": 0,
                                    "embedding": list(data.embedding),
                                }
                            ],
                            "model": args.model,
                            "usage": {"prompt_tokens": 0, "total_tokens": 0},
                        },
                    },
                    "error": None,
                }
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            if (i // args.chunk_size) % 10 == 0 or i + len(batch) == n:
                rate = written / max(0.001, time.time() - t0)
                eta = (n - written) / max(1.0, rate)
                print(f"  [{int(time.time()-t0)}s] {written}/{n} "
                      f"({rate:.0f}/s; ETA {int(eta)}s)")
        out_fh.flush()
        os.fsync(out_fh.fileno())
    os.replace(tmp_path, out_path)
    print(f"[offline_sync] wrote {written} rows -> {out_path} "
          f"in {int(time.time()-t0)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
