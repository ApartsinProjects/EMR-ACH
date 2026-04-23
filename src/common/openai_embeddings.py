"""OpenAI-API embeddings as a drop-in alternative to local SBERT for relevance.

v2.2 motivation: local SBERT (mpnet-base-v2) on RTX 2060 takes ~2-3h per
benchmark relevance pass on the 218k-article pool, even with batch=256 +
fp16. OpenAI text-embedding-3-small via the Batch API does the same work
in ~30 minutes wall-clock for ~$0.30, removes the local-GPU dependency,
and produces 1536-dim embeddings (vs mpnet's 768-dim).

Compatibility contract: this module emits L2-normalized float32 numpy
arrays in the same shape as `compute_relevance.embed()`, so the cosine-
similarity scoring downstream is unchanged. The .npy cache files are
written under separate names (`*_openai.npy` / `*_openai.fp.txt`) so they
do not collide with the SBERT cache.

Two execution paths:
  - sync   : straight client.embeddings.create() per chunk; rate-limited
             at OpenAI's tier-1 RPM/TPM. Good for <50k items.
  - batch  : submit a JSONL via client.batches.create(); 50% discount;
             ~30-60 min wall-clock; resumable via batch_id; recommended
             for production-scale runs.

Cost (April 2026 pricing, may need update):
  text-embedding-3-small: $0.020 / 1M input tokens (sync) / $0.010 (batch)
  text-embedding-3-large: $0.130 / 1M input tokens (sync) / $0.065 (batch)

Volume math for our pool:
  218k articles x ~120 tokens/article = ~26M input tokens
  -> small/batch: ~$0.26
  -> large/batch: ~$1.69
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIM = 1536          # text-embedding-3-small native; -large is 3072
DEFAULT_CHUNK = 200         # items per sync request (well under 8192-input limit)
DEFAULT_MAX_TOKENS_PER_BATCH_FILE = 4_000_000   # OpenAI batch file size cap
DEFAULT_BATCH_REQUESTS_CAP = 50_000              # OpenAI per-batch maximum_requests cap


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize rows for cosine-similarity downstream."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return (v / n).astype(np.float32, copy=False)


def _truncate(text: str, max_chars: int) -> str:
    return (text or "")[:max_chars]


def encode_sync(texts: list[str], model: str = DEFAULT_MODEL,
                chunk: int = DEFAULT_CHUNK, client=None,
                progress_every: int = 50) -> np.ndarray:
    """Synchronous embedding via client.embeddings.create() in chunks of `chunk`.
    Returns L2-normalized float32 ndarray of shape (N, D)."""
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    all_vecs: list[np.ndarray] = []
    n = len(texts)
    t0 = time.time()
    for i in range(0, n, chunk):
        batch_texts = texts[i:i + chunk]
        # OpenAI rejects empty strings; substitute a single space.
        batch_texts = [t if t.strip() else " " for t in batch_texts]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=model, input=batch_texts)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                print(f"  [encode_sync] retry {attempt+1}/5 after {wait}s: {e}")
                time.sleep(wait)
        vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
        all_vecs.append(vecs)
        done = i + len(batch_texts)
        if (done % (progress_every * chunk)) == 0 or done == n:
            rate = done / max(0.001, time.time() - t0)
            print(f"  [encode_sync] {done}/{n}  ({rate:.0f} items/s)")
    return _normalize(np.vstack(all_vecs))


def _build_batch_jsonl(texts: list[str], model: str, out_path: Path) -> int:
    """Write the JSONL of /v1/embeddings batch requests. Returns count.
    OpenAI Batch API expects one request per line:
      {"custom_id":"...", "method":"POST", "url":"/v1/embeddings",
       "body":{"model":"text-embedding-3-small", "input":["text"]}}
    Each request takes one input string (we do not pack multiple inputs
    per request because the batch parser then has to map them back, and
    the cost is identical either way at the input-token level)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            line = {
                "custom_id": f"emb_{i:07d}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": model, "input": text if text.strip() else " "},
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)
    return n


def encode_batch(texts: list[str], model: str = DEFAULT_MODEL,
                 work_dir: Path | None = None, client=None,
                 poll_interval_sec: int = 30,
                 timeout_sec: int = 24 * 3600) -> np.ndarray:
    """Submit OpenAI Batch job(s) for all `texts` and block until completion.
    Returns L2-normalized float32 ndarray of shape (N, D).

    Auto-chunking: OpenAI caps each batch at DEFAULT_BATCH_REQUESTS_CAP
    (50,000) requests. When `len(texts)` exceeds the cap, the input is
    split into sequential sub-batches; each gets its own work_dir
    suffix + state.json so any individual chunk can resume independently.
    Results are concatenated in input order.

    Resume semantics: writes the request file + batch_id to `work_dir`
    (or `work_dir / chunk_NN/` for chunked runs). A subsequent call with
    the same `work_dir` will re-poll the existing batch_id rather than
    create a new one. Delete the work_dir to force a fresh submission.
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    work_dir = work_dir or Path("data/etd_openai_batches") / model
    work_dir.mkdir(parents=True, exist_ok=True)

    # Auto-chunk above the OpenAI per-batch cap. Submit all chunks up front
    # (in parallel on OpenAI's side), then poll each to completion. This halves
    # wall-clock for 2-chunk runs vs the earlier sequential implementation; the
    # cost is unchanged because it is token-based, not request-based.
    if len(texts) > DEFAULT_BATCH_REQUESTS_CAP:
        cap = DEFAULT_BATCH_REQUESTS_CAP
        n_chunks = (len(texts) + cap - 1) // cap
        print(f"[encode_batch] N={len(texts)} > {cap}; splitting into {n_chunks} PARALLEL sub-batches")

        # Phase 1: submit all (or resume-attach to) chunks; return immediately
        # after each submission. Polling comes in phase 2.
        chunks = []
        for i in range(n_chunks):
            chunk_texts = texts[i * cap : (i + 1) * cap]
            chunk_dir = work_dir / f"chunk_{i:02d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            state_path = chunk_dir / "state.json"
            request_path = chunk_dir / "requests.jsonl"
            state: dict = {}
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    state = {}
            if "batch_id" not in state:
                n_req = _build_batch_jsonl(chunk_texts, model, request_path)
                with open(request_path, "rb") as fp:
                    file_obj = client.files.create(file=fp, purpose="batch")
                batch = client.batches.create(
                    input_file_id=file_obj.id,
                    endpoint="/v1/embeddings",
                    completion_window="24h",
                    metadata={"workflow": "emr-ach-embeddings", "model": model,
                              "chunk": str(i), "n_inputs": str(n_req)},
                )
                state.update({
                    "batch_id":      batch.id,
                    "input_file_id": file_obj.id,
                    "model":         model,
                    "n_inputs":      n_req,
                    "submitted_at":  datetime.utcnow().isoformat() + "Z",
                })
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                print(f"[encode_batch]   chunk {i+1}/{n_chunks} submitted: {batch.id}")
            else:
                print(f"[encode_batch]   chunk {i+1}/{n_chunks} resume: {state['batch_id']}")
            chunks.append((i, chunk_dir, chunk_texts, state["batch_id"]))

        # Phase 2: poll all chunks in a round-robin loop until each is terminal.
        t0 = time.time()
        done_batches: dict[int, any] = {}
        while len(done_batches) < n_chunks and time.time() - t0 < timeout_sec:
            for i, chunk_dir, chunk_texts, bid in chunks:
                if i in done_batches:
                    continue
                b = client.batches.retrieve(bid)
                if b.status in ("completed", "failed", "expired", "cancelled"):
                    done_batches[i] = b
                    rc = b.request_counts
                    print(f"[encode_batch]   chunk {i+1}/{n_chunks} {b.status}: "
                          f"total={rc.total} done={rc.completed} fail={rc.failed}")
            if len(done_batches) < n_chunks:
                time.sleep(poll_interval_sec)

        # Phase 3: validate + download + reassemble
        parts = []
        for i, chunk_dir, chunk_texts, bid in chunks:
            b = done_batches.get(i)
            if b is None:
                raise TimeoutError(f"chunk {i} batch {bid} did not complete in {timeout_sec}s")
            if b.status != "completed":
                raise RuntimeError(f"chunk {i} batch {bid} ended in status {b.status}")
            out_path = chunk_dir / "output.jsonl"
            out_bytes = client.files.content(b.output_file_id).read()
            out_path.write_bytes(out_bytes)
            n = len(chunk_texts)
            vecs = np.zeros((n, _expected_dim(model)), dtype=np.float32)
            for line in out_path.open(encoding="utf-8"):
                r = json.loads(line)
                cid = r.get("custom_id", "")
                if not cid.startswith("emb_"):
                    continue
                idx = int(cid[len("emb_"):])
                data = (r.get("response", {}).get("body", {}) or {}).get("data") or []
                if not data:
                    continue
                v = np.asarray(data[0]["embedding"], dtype=np.float32)
                if v.shape[0] != vecs.shape[1]:
                    vecs = np.zeros((n, v.shape[0]), dtype=np.float32)
                vecs[idx] = v
            parts.append(_normalize(vecs))

        return np.vstack(parts)

    request_path = work_dir / "requests.jsonl"
    state_path = work_dir / "state.json"

    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}

    if "batch_id" not in state:
        n_req = _build_batch_jsonl(texts, model, request_path)
        print(f"[encode_batch] wrote {n_req} requests to {request_path}")
        with open(request_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        print(f"[encode_batch] uploaded file_id={file_obj.id}")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={"workflow": "emr-ach-embeddings", "model": model,
                      "n_inputs": str(n_req)},
        )
        state.update({
            "batch_id": batch.id,
            "input_file_id": file_obj.id,
            "model": model,
            "n_inputs": n_req,
            "submitted_at": datetime.utcnow().isoformat() + "Z",
        })
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[encode_batch] submitted batch_id={batch.id}")
    else:
        print(f"[encode_batch] resuming batch_id={state['batch_id']}")

    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        b = client.batches.retrieve(state["batch_id"])
        rc = b.request_counts
        elapsed = int(time.time() - t0)
        print(f"  [{elapsed}s] {b.status}  total={rc.total} done={rc.completed} fail={rc.failed}")
        if b.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(poll_interval_sec)
    else:
        raise TimeoutError(f"batch {state['batch_id']} did not complete in {timeout_sec}s")

    if b.status != "completed":
        raise RuntimeError(f"batch {state['batch_id']} ended in status {b.status}")

    # Download the output file
    output_path = work_dir / "output.jsonl"
    out_bytes = client.files.content(b.output_file_id).read()
    output_path.write_bytes(out_bytes)
    print(f"[encode_batch] downloaded {len(out_bytes)//1024}KB -> {output_path}")

    # Parse + reorder by custom_id index
    n = len(texts)
    vecs = np.zeros((n, _expected_dim(model)), dtype=np.float32)
    seen = 0
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cid = r.get("custom_id", "")
            if not cid.startswith("emb_"):
                continue
            idx = int(cid[len("emb_"):])
            body = r.get("response", {}).get("body", {})
            data = body.get("data") or []
            if not data:
                continue
            v = np.asarray(data[0]["embedding"], dtype=np.float32)
            if v.shape[0] != vecs.shape[1]:
                # Reshape on first encounter if model returned different dim
                vecs = np.zeros((n, v.shape[0]), dtype=np.float32)
            vecs[idx] = v
            seen += 1
    if seen != n:
        print(f"[warn] only {seen}/{n} embeddings recovered from batch output")
    return _normalize(vecs)


def _expected_dim(model: str) -> int:
    if "large" in model:
        return 3072
    return 1536


# --- Incremental cache (mirrors compute_relevance.load_or_embed signature) ---

def _per_row_fingerprints(items: list[dict], text_fn) -> list[str]:
    out = []
    for r in items:
        h = hashlib.md5()
        h.update(text_fn(r).encode("utf-8", errors="replace"))
        out.append(h.hexdigest())
    return out


def load_or_embed_openai(items: list[dict], text_fn, cache_path: Path,
                         model: str = DEFAULT_MODEL, mode: str = "batch",
                         rebuild: bool = False) -> np.ndarray:
    """Incremental embedding cache compatible with compute_relevance's
    `load_or_embed`. Embeds only the rows whose fingerprint changed since
    the last run; reuses cached vectors for the rest.

    cache_path -> {cache}.npy + {cache}.fp.txt (fingerprints, one per line).
    """
    fp_path = cache_path.with_suffix(".fp.txt")
    current_fps = _per_row_fingerprints(items, text_fn)

    cached_emb = None
    cached_fps: list[str] = []
    if not rebuild and cache_path.exists() and fp_path.exists():
        cached_emb = np.load(cache_path)
        cached_fps = fp_path.read_text(encoding="utf-8").splitlines()
        if len(cached_fps) != cached_emb.shape[0]:
            cached_emb = None
            cached_fps = []

    fp_to_idx: dict[str, int] = {fp: i for i, fp in enumerate(cached_fps)} if cached_emb is not None else {}

    n = len(items)
    new_indices: list[int] = []
    new_texts: list[str] = []
    final = None
    if cached_emb is not None:
        final = np.zeros((n, cached_emb.shape[1]), dtype=np.float32)
    for i, fp in enumerate(current_fps):
        if cached_emb is not None and fp in fp_to_idx:
            final[i] = cached_emb[fp_to_idx[fp]]
        else:
            new_indices.append(i)
            new_texts.append(text_fn(items[i]))

    print(f"  [openai cache] reused {n - len(new_indices)}/{n} "
          f"({100*(n - len(new_indices))/max(1,n):.1f}%), "
          f"embedding {len(new_indices)} new/changed rows via {mode}")

    if new_texts:
        if mode == "batch":
            new_vecs = encode_batch(new_texts, model=model)
        else:
            new_vecs = encode_sync(new_texts, model=model)
        if final is None:
            final = np.zeros((n, new_vecs.shape[1]), dtype=np.float32)
            for i in range(n):
                if i not in set(new_indices):
                    raise RuntimeError("inconsistent cache state")
        for k, idx in enumerate(new_indices):
            final[idx] = new_vecs[k]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, final.astype(np.float32))
    fp_path.write_text("\n".join(current_fps), encoding="utf-8")
    return final
