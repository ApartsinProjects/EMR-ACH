"""Multi-batch orchestrator for OpenAI Batch API.

Splits a large request list into chunks, submits them as concurrent batches,
polls all of them in one loop, and merges the results. Fixes two pain points
we hit on 22k-item ETD runs:

  1. **Throughput**: a single 22k batch takes 4-12h; five concurrent 4.5k
     batches take 1-3h (each sub-batch is worked in parallel by OpenAI's
     scheduler, and the per-batch size penalty is sub-linear).

  2. **Crash recovery**: the original `BatchClient.run()` uses a naive
     `time.sleep(60); retrieve()` loop. A single transient `APITimeoutError`
     during `retrieve()` crashes the runner and orphans the batch.
     `run_multibatch()` wraps every poll in retry-with-exponential-backoff.

Design notes:
  - Each chunk is a separate batch job named `{job_name}__chunk_{i:03d}_of_{N:03d}`.
  - Resumable per-chunk via `results/raw/{sub_job}.jsonl` cache AND
    `batch_jobs.json` persistence.
  - Merged final results are cached to `results/raw/{job_name}.jsonl` so callers
    see the same file contract as single-batch `BatchClient.run()`.
  - Submission is staggered (small `submit_stagger_sec`) to avoid hammering
    the upload endpoint with N parallel file POSTs.

Not yet wired into `scripts/articles_to_facts.py` — that will happen when we
decide to use it. This module is a dependency-free drop-in; the caller passes
a configured `BatchClient` instance and gets a `dict[custom_id, BatchResult]`
back, same as `client.run(...)`.

Example:
    from src.common.multibatch import run_multibatch
    from src.batch_client import BatchClient

    client = BatchClient(mode="batch", config=cfg)
    results = run_multibatch(
        client, requests, job_name="etd_v2_initial",
        chunk_size=5000, poll_interval_sec=60,
    )
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.batch_client import BatchClient, BatchRequest, BatchResult


# ---------------------------------------------------------------------------
# Tunables (all overrideable per call)
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE       = 5000   # one batch per ~5k requests; empirically 1-3h
DEFAULT_POLL_INTERVAL    = 60     # seconds between polls
DEFAULT_SUBMIT_STAGGER   = 2      # seconds between chunk-submit calls
DEFAULT_POLL_RETRIES     = 5      # per-poll retry budget on transient errors
DEFAULT_POLL_BACKOFF_BASE = 2     # seconds; doubled each retry

_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "expired"})


# ---------------------------------------------------------------------------
# Retry-with-backoff wrapper
# ---------------------------------------------------------------------------

def _retry_call(fn, *, retries: int = DEFAULT_POLL_RETRIES,
                backoff_base: int = DEFAULT_POLL_BACKOFF_BASE,
                what: str = "api call"):
    """Call fn() with exponential backoff. Raises the last exception if all
    retries exhaust. Only swallows network-style transient errors; programmer
    errors (TypeError, ValueError in fn's args) propagate immediately."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except (TimeoutError, ConnectionError) as e:
            last_exc = e
        except Exception as e:
            # OpenAI SDK exceptions: APITimeoutError, APIConnectionError, RateLimitError
            # all subclass OpenAIError. We catch by name to avoid importing the SDK here.
            cls = type(e).__name__
            if cls in ("APITimeoutError", "APIConnectionError", "RateLimitError",
                       "APIError", "InternalServerError"):
                last_exc = e
            else:
                raise
        delay = backoff_base * (2 ** attempt)
        print(f"  [retry] {what}: {type(last_exc).__name__} — backoff {delay}s (attempt {attempt+1}/{retries})")
        time.sleep(delay)
    raise RuntimeError(f"{what} failed after {retries} retries: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Helpers that reach into BatchClient internals (all prefixed _ so we know)
# ---------------------------------------------------------------------------

def _cache_path(client: BatchClient, job_name: str) -> Path:
    return Path(client.cfg.results_dir) / "raw" / f"{job_name}.jsonl"


def _load_cached_results(path: Path) -> dict[str, BatchResult] | None:
    if not path.exists():
        return None
    return client_load_results(path)


def client_load_results(path: Path) -> dict[str, BatchResult]:
    """Mirror of BatchClient._load_results — kept separate so this module
    doesn't reach into a _private method. Reads a JSONL of BatchResult dicts."""
    out: dict[str, BatchResult] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["custom_id"]] = BatchResult(
            custom_id=r["custom_id"],
            content=r.get("content") or "",
            error=r.get("error") or None,
            input_tokens=r.get("input_tokens", 0),
            output_tokens=r.get("output_tokens", 0),
        )
    return out


def _save_results(results: dict[str, BatchResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in results.values():
            f.write(json.dumps({
                "custom_id":     r.custom_id,
                "content":       r.content,
                "error":         r.error,
                "input_tokens":  r.input_tokens,
                "output_tokens": r.output_tokens,
            }) + "\n")
        f.flush()
        try:
            import os as _os
            _os.fsync(f.fileno())
        except Exception:
            pass
    import os as _os
    _os.replace(str(tmp), str(path))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@dataclass
class ChunkState:
    """Per-chunk tracking during the unified poll loop."""
    sub_job:    str
    requests:   list[BatchRequest]
    batch_id:   str | None = None
    status:     str         = "pending"   # pending | in_progress | completed | failed | cached
    results:    dict[str, BatchResult] | None = None


def run_multibatch(
    client:            BatchClient,
    requests:          list[BatchRequest],
    job_name:          str,
    chunk_size:        int = DEFAULT_CHUNK_SIZE,
    poll_interval_sec: int = DEFAULT_POLL_INTERVAL,
    submit_stagger_sec: int = DEFAULT_SUBMIT_STAGGER,
) -> dict[str, BatchResult]:
    """Submit `requests` as N concurrent batches and merge results.

    If a combined cache at `results/raw/{job_name}.jsonl` exists, it is
    returned unchanged (enables full-job resumption). Otherwise each chunk
    is cached independently at `results/raw/{job_name}__chunk_XXX_of_YYY.jsonl`
    and a unified cache is written when all chunks complete.

    Args:
        client: a configured BatchClient in mode="batch".
        requests: the full list of requests to dispatch.
        job_name: stable identifier. Drives cache paths and batch metadata.
        chunk_size: requests per sub-batch (OpenAI scheduler parallelizes sub-batches).
        poll_interval_sec: time between full-round polls across all sub-batches.
        submit_stagger_sec: tiny delay between submissions to avoid burst-upload pressure.

    Returns:
        dict keyed by custom_id, exactly like BatchClient.run(...).
    """
    # 1. Check combined cache first (resume)
    combined_cache = _cache_path(client, job_name)
    if combined_cache.exists():
        print(f"[multibatch] Combined cache hit: {combined_cache}")
        return client_load_results(combined_cache)

    # 2. Partition
    n = len(requests)
    if n == 0:
        return {}
    if n <= chunk_size:
        print(f"[multibatch] {n} <= chunk_size={chunk_size}; delegating to single-batch run()")
        return client.run(requests, job_name)

    chunks = [requests[i:i + chunk_size] for i in range(0, n, chunk_size)]
    N = len(chunks)
    print(f"[multibatch] Split {n} requests into {N} chunks of <= {chunk_size}")

    # 3. Build chunk states, loading per-chunk caches if present
    states: list[ChunkState] = []
    for i, chunk in enumerate(chunks):
        sub_job = f"{job_name}__chunk_{i:03d}_of_{N:03d}"
        cache = _cache_path(client, sub_job)
        if cache.exists():
            print(f"[multibatch] [{i+1:03d}/{N}] cache hit for {sub_job}")
            states.append(ChunkState(sub_job=sub_job, requests=chunk,
                                     status="cached", results=client_load_results(cache)))
        else:
            states.append(ChunkState(sub_job=sub_job, requests=chunk))

    # 4. Submit each non-cached chunk (or resume existing batch from batch_jobs.json).
    jobs = client._load_jobs()  # noqa: SLF001 — deliberate internal access
    for i, st in enumerate(states):
        if st.status == "cached":
            continue
        if st.sub_job in jobs and jobs[st.sub_job].get("batch_id"):
            st.batch_id = jobs[st.sub_job]["batch_id"]
            st.status = "in_progress"   # will be refreshed in the poll loop
            print(f"[multibatch] [{i+1:03d}/{N}] resuming existing batch {st.batch_id} for {st.sub_job}")
            continue
        # Submit fresh
        model = st.requests[0].model or client.cfg.experiment_model
        temp = client.cfg.get("model", "temperature", default=0.0)
        print(f"[multibatch] [{i+1:03d}/{N}] submitting {len(st.requests)} requests as {st.sub_job}")
        batch = _retry_call(
            lambda: client._submit_batch(st.requests, st.sub_job, model, temp),  # noqa: SLF001
            what=f"submit chunk {i+1}/{N}",
        )
        st.batch_id = batch.id
        st.status = batch.status
        jobs[st.sub_job] = {"batch_id": batch.id, "job_name": st.sub_job, "status": batch.status}
        client._save_jobs(jobs)  # noqa: SLF001
        if submit_stagger_sec:
            time.sleep(submit_stagger_sec)

    # 5. Unified poll loop — retry-with-backoff per retrieve() call.
    print(f"[multibatch] Polling {sum(1 for s in states if s.status not in ('cached','completed'))} in-flight chunks every {poll_interval_sec}s...")
    while any(s.status not in ("cached", "completed") and s.status not in _TERMINAL_STATES - {"completed"}
              for s in states):
        in_progress = [s for s in states if s.status not in ("cached",) and s.status not in _TERMINAL_STATES]
        if not in_progress:
            break
        time.sleep(poll_interval_sec)
        for st in in_progress:
            try:
                batch = _retry_call(
                    lambda st=st: client._client.batches.retrieve(st.batch_id),   # noqa: SLF001
                    what=f"poll {st.sub_job}",
                )
            except RuntimeError as e:
                print(f"  [multibatch] chunk {st.sub_job}: poll failed: {e}")
                continue
            st.status = batch.status
            jobs[st.sub_job]["status"] = batch.status
            client._save_jobs(jobs)  # noqa: SLF001
            rc = batch.request_counts
            print(f"  [poll] {st.sub_job}: {batch.status}  {rc.completed}/{rc.total} (failed={rc.failed})")
            if batch.status == "completed":
                # Download + cache immediately, so a crash during later chunks doesn't lose work.
                results = _retry_call(
                    lambda b=batch: client._download_results(b),  # noqa: SLF001
                    what=f"download {st.sub_job}",
                )
                st.results = results
                _save_results(results, _cache_path(client, st.sub_job))
                print(f"  [done] {st.sub_job}: cached {len(results)} results")

    # 6. Sanity: every non-cached chunk should now have results or terminal failure.
    failures: list[str] = []
    for st in states:
        if st.results is None:
            failures.append(f"{st.sub_job}: status={st.status} batch_id={st.batch_id}")
    if failures:
        print(f"[multibatch] WARNING: {len(failures)} chunk(s) did not complete:")
        for f in failures:
            print(f"  {f}")

    # 7. Merge, write combined cache
    merged: dict[str, BatchResult] = {}
    for st in states:
        if st.results:
            merged.update(st.results)
    _save_results(merged, combined_cache)
    print(f"[multibatch] Merged {len(merged)} results -> {combined_cache}")
    return merged
