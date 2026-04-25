"""Download CC-News WARC shards to local disk WITHOUT extraction.

Decouples the bandwidth-bound download phase from the CPU-bound
extraction phase. Once all WARCs are on local disk, you can run any
number of extraction passes (different whitelists, different
trafilatura settings, different date windows) without re-fetching.

Sibling: ``scripts/fetch_cc_news_archive.py`` does the inline
download + filter + extract + discard. Use that for a one-shot
extracted output. Use this script when you want to keep the raw
WARCs around for repeated re-extraction.

Output layout::

    {out}/{YYYY-MM}/CC-NEWS-{timestamp}-{NNN}.warc.gz
    {out}/{YYYY-MM}/CC-NEWS-{timestamp}-{NNN}.warc.gz.done

Resume: any shard whose ``.done`` exists is skipped.

Usage::

    python scripts/download_cc_news_warcs.py \\
        --start 2025-12 --end 2025-12 \\
        --workers 8 \\
        --out data/cc_news_warcs/

Disk: ~500 MB-1 GB per shard, ~616 shards/month -> ~300-600 GB/month.
Time at 8 workers: ~3-6 hours per month, bandwidth-bound (Common Crawl
mirror caps per IP at ~30-50 MB/s observed).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.fetch_cc_news_archive import (  # noqa: E402
    CC_BASE,
    DEFAULT_MAX_SHARD_BYTES,
    RETRY_ATTEMPTS,
    RETRY_BASE_SLEEP,
    THROTTLE_SLEEP_RANGE,
    _TransientHTTPError,
    _download_shard,
    _month_range,
    list_shards_for_month,
)


def _default_workers() -> int:
    cpu = os.cpu_count() or 2
    return min(8, max(2, cpu - 1))


def _download_one(shard_path: str, out_dir: Path, max_bytes: int) -> dict:
    """Download a single WARC to ``out_dir/<basename>``. Atomic + resumable."""
    name = Path(shard_path).name
    final = out_dir / name
    done = final.with_suffix(final.suffix + ".done")
    if done.exists() and final.exists():
        return {"shard": name, "status": "skip", "bytes": final.stat().st_size,
                "wall_s": 0.0, "attempts": 0}
    tmp = final.with_suffix(final.suffix + ".dl")
    url = f"{CC_BASE}/{shard_path}"
    t0 = time.time()
    last_err: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            n = _download_shard(url, tmp, max_bytes)
            os.replace(tmp, final)
            done.write_text(f"{n}\n", encoding="utf-8")
            return {"shard": name, "status": "ok", "bytes": n,
                    "wall_s": time.time() - t0, "attempts": attempt}
        except _TransientHTTPError as exc:
            last_err = exc
            if attempt >= RETRY_ATTEMPTS:
                break
            time.sleep(random.uniform(*THROTTLE_SLEEP_RANGE))
        except Exception as exc:
            last_err = exc
            if attempt >= RETRY_ATTEMPTS:
                break
            time.sleep(RETRY_BASE_SLEEP * (2 ** (attempt - 1)))
    return {"shard": name, "status": "fail", "bytes": 0,
            "wall_s": time.time() - t0, "attempts": RETRY_ATTEMPTS,
            "error": repr(last_err)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="Start month YYYY-MM")
    p.add_argument("--end", required=True, help="End month YYYY-MM (inclusive)")
    p.add_argument("--out", default="data/cc_news_warcs/", help="Output root")
    p.add_argument("--workers", type=int, default=_default_workers(),
                   help=f"Parallel downloads (default: {_default_workers()})")
    p.add_argument("--max-shards", type=int, default=0,
                   help="Cap shards across all months (0 = all)")
    p.add_argument("--max-shard-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES,
                   help="Abort any shard exceeding this size")
    p.add_argument("--dry-run", action="store_true",
                   help="List shards without downloading")
    args = p.parse_args(argv)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    months = _month_range(args.start, args.end)
    print(f"[dl-warcs] window={args.start}..{args.end}, workers={args.workers}, out={out_root}")

    jobs: list[tuple[str, Path]] = []
    for (y, m) in months:
        shards = list_shards_for_month(y, m)
        month_dir = out_root / f"{y:04d}-{m:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)
        print(f"[dl-warcs] {y}-{m:02d}: {len(shards)} shards in manifest")
        for s in shards:
            name = Path(s).name
            if (month_dir / (name + ".done")).exists():
                continue
            jobs.append((s, month_dir))
            if args.max_shards and len(jobs) >= args.max_shards:
                break
        if args.max_shards and len(jobs) >= args.max_shards:
            break

    print(f"[dl-warcs] {len(jobs)} shards to download")
    if args.dry_run:
        for j, (s, _) in enumerate(jobs[:20]):
            print(f"[dl-warcs]   would: {s}")
        if len(jobs) > 20:
            print(f"[dl-warcs]   ... and {len(jobs) - 20} more")
        return 0
    if not jobs:
        print("[dl-warcs] nothing to do (all shards already downloaded)")
        return 0

    n_ok = n_skip = n_fail = 0
    bytes_total = 0
    t_total = time.time()
    print(f"[dl-warcs] dispatching {len(jobs)} shards across {args.workers} workers")
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_download_one, s, d, args.max_shard_bytes): (s, d)
                for s, d in jobs}
        done_n = 0
        for fut in cf.as_completed(futs):
            res = fut.result()
            done_n += 1
            if res["status"] == "ok":
                n_ok += 1
                bytes_total += res["bytes"]
            elif res["status"] == "skip":
                n_skip += 1
            else:
                n_fail += 1
                print(f"[dl-warcs]   FAIL {res['shard']}: {res.get('error', '?')}")
            if done_n % 10 == 0 or done_n == len(jobs):
                gb = bytes_total / (1024 ** 3)
                rate = bytes_total / max(0.001, time.time() - t_total) / (1024 ** 2)
                print(f"[dl-warcs]   {done_n}/{len(jobs)}: ok={n_ok} skip={n_skip} "
                      f"fail={n_fail} | {gb:.1f} GB | {rate:.1f} MB/s")

    print(f"[dl-warcs] done: ok={n_ok} skip={n_skip} fail={n_fail} | "
          f"{bytes_total/(1024**3):.1f} GB total | "
          f"{(time.time()-t_total)/60:.1f} min wall-clock")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
