"""ETD Stage 2: cluster near-duplicate facts and assign canonical IDs.

The Stage-1 LLM extractor often produces multiple paraphrased versions of the
same atomic fact (different articles describing the same press conference,
the same casualty count from a single Reuters wire that hit five outlets).
This stage groups paraphrases by semantic similarity, picks one canonical
fact per cluster, and tags every other fact with `canonical_id` pointing at
the chosen representative.

Algorithm
---------
1. Encode every fact's `fact` string (OpenAI text-embedding-3-small by
   default; SBERT mpnet-base-v2 as fallback; both L2-normalized).
2. Build candidate-pair shortlist via kNN (top-k similar facts above
   cosine threshold tau).
     - `--knn-mode bucket` (default, v2.2): group facts by fact-date, then
       for each day query only against the ±window-days bucket; uses FAISS
       `IndexFlatIP` per bucket when available, numpy matmul otherwise.
       Complexity O(N x B) vs brute's O(N^2) where B = mean bucket size;
       recall is exact within the window (which is also the downstream
       clustering constraint, so no semantic loss).
     - `--knn-mode brute` (legacy): single sliced matmul over the full
       embedding matrix.
3. Filter pairs: keep only pairs with fact `time` within --window-days
   (redundant under `--knn-mode bucket` but retained for the brute path).
4. Union-find clustering on filtered pairs.
5. Per cluster: canonical = highest extraction_confidence, ties broken by
   earliest article_date. All others get `canonical_id` = canonical.id.
6. Canonical's `variant_ids` = list of cluster members' ids (excluding self).

Idempotent: re-running with the same inputs re-derives clusters from scratch
and overwrites the `canonical_id` / `variant_ids` fields. All other fact
fields are preserved verbatim.

Reads:
  data/etd/facts.v1.jsonl                  Stage-1 output

Writes (atomic):
  data/etd/facts.v1_canonical.jsonl        all facts with canonical_id /
                                            variant_ids filled
  data/etd/dedup_meta.json                  cluster-size histogram + audit

Usage:
  python scripts/etd_dedup.py
  python scripts/etd_dedup.py --threshold 0.92 --window-days 3
  python scripts/etd_dedup.py --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "etd"
INPUT = DATA / "facts.v1.jsonl"
OUTPUT = DATA / "facts.v1_canonical.jsonl"
META = DATA / "dedup_meta.json"

# v2.2 default: OpenAI embeddings (no GPU; matches compute_relevance.py).
# Local SBERT path is preserved as a fallback via --embedder sbert.
DEFAULT_EMBEDDER = "openai"
EMBED_MODEL_SBERT = "sentence-transformers/all-mpnet-base-v2"
EMBED_MODEL_OPENAI = "text-embedding-3-small"

# Bootstrap sys.path so `from src.common.openai_embeddings import ...`
# resolves when invoked directly. (Backlog item B4a generalizes this.)
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


def _atomic_write(path: Path, render) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        render(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        return None


# Union-find ---------------------------------------------------------------

class UF:
    __slots__ = ("p",)

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


# kNN search strategies ----------------------------------------------------

def _knn_brute(embs, threshold: float, top_k: int, chunk: int = 1024):
    """Legacy brute-force: single sliced (chunk x N) matmul over the full
    embedding matrix. O(N^2) time + memory per chunk. Preserved as
    `--knn-mode brute` for parity with v2.1 behavior."""
    import numpy as np
    n = embs.shape[0]
    candidates: list[tuple[int, int, float]] = []
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        sims = embs[i0:i1] @ embs.T
        for r, gi in enumerate(range(i0, i1)):
            sims[r, gi] = -1.0
        top_idx = np.argpartition(-sims, top_k, axis=1)[:, :top_k]
        for r, gi in enumerate(range(i0, i1)):
            for j in top_idx[r]:
                s = float(sims[r, j])
                if s >= threshold and gi < j:
                    candidates.append((gi, int(j), s))
        if (i1 % 8192) == 0 or i1 == n:
            print(f"  [brute] scanned {i1}/{n}; pairs so far: {len(candidates)}")
    return candidates


def _knn_bucketed(embs, fact_dates, window_days: int, threshold: float,
                  top_k: int):
    """v2.2 default: bucket facts by date, for each day query only against
    the +/- window-days neighborhood. Exact recall within the window (which
    is the downstream clustering constraint, so no semantic loss). Uses
    FAISS IndexFlatIP per bucket when available; numpy matmul otherwise.

    Complexity: O(N x B) where B = mean bucket size. For 78k facts spread
    over 365 days with window=3, B ~= 1500 vs brute's N=78k -> ~50x fewer
    FLOPs, plus MKL/SIMD from FAISS on top.

    Returns a deduplicated list of (i, j, sim) pairs with i < j."""
    import numpy as np
    try:
        import faiss
        use_faiss = True
    except ImportError:
        faiss = None
        use_faiss = False

    n = embs.shape[0]
    # Group indices by ISO date string; skip facts with no parseable date
    by_date: dict[str, list[int]] = defaultdict(list)
    nodate: list[int] = []
    for i, d in enumerate(fact_dates):
        if d is None:
            nodate.append(i)
        else:
            by_date[d.date().isoformat()].append(i)

    sorted_keys = sorted(by_date.keys())
    key_to_ord = {k: i for i, k in enumerate(sorted_keys)}
    key_dates = [datetime.strptime(k, "%Y-%m-%d").date() for k in sorted_keys]

    print(f"[etd_dedup] bucketed kNN: {len(sorted_keys)} distinct dates, "
          f"{len(nodate)} no-date facts skipped, FAISS={use_faiss}")

    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int, float]] = []

    # Two-pointer sliding window over sorted_keys
    lo = 0
    hi = 0
    for center_ord, center_date in enumerate(key_dates):
        while lo < center_ord and (center_date - key_dates[lo]).days > window_days:
            lo += 1
        while hi + 1 < len(key_dates) and (key_dates[hi + 1] - center_date).days <= window_days:
            hi += 1

        bucket_idxs: list[int] = []
        for k in range(lo, hi + 1):
            bucket_idxs.extend(by_date[sorted_keys[k]])
        day_idxs = by_date[sorted_keys[center_ord]]
        if len(bucket_idxs) < 2 or not day_idxs:
            continue

        bucket_arr = np.asarray(bucket_idxs, dtype=np.int64)
        day_arr = np.asarray(day_idxs, dtype=np.int64)

        q = embs[day_arr]
        b = embs[bucket_arr]
        k_eff = min(top_k + 1, len(bucket_idxs))  # +1 slot for self-hit

        if use_faiss:
            # L2-normalized embeddings => inner product == cosine
            index = faiss.IndexFlatIP(b.shape[1])
            index.add(np.ascontiguousarray(b))
            D, I = index.search(np.ascontiguousarray(q), k_eff)
            for r in range(len(day_arr)):
                qi = int(day_arr[r])
                for c in range(k_eff):
                    s = float(D[r, c])
                    if s < threshold:
                        continue
                    j = int(bucket_arr[I[r, c]])
                    if j == qi:
                        continue
                    a, b_ = (qi, j) if qi < j else (j, qi)
                    if (a, b_) in seen:
                        continue
                    seen.add((a, b_))
                    candidates.append((a, b_, s))
        else:
            sims = q @ b.T  # (day_n, bucket_n)
            # Mask self-similarity: each day_arr[r] might appear in bucket_arr
            # Build a map only for the query row (rare hit)
            bucket_pos = {int(bucket_arr[c]): c for c in range(len(bucket_arr))}
            for r in range(len(day_arr)):
                qi = int(day_arr[r])
                if qi in bucket_pos:
                    sims[r, bucket_pos[qi]] = -1.0
            k_part = min(top_k, sims.shape[1] - 1)
            top_idx = np.argpartition(-sims, k_part, axis=1)[:, :k_part]
            for r in range(len(day_arr)):
                qi = int(day_arr[r])
                for col in top_idx[r]:
                    s = float(sims[r, col])
                    if s < threshold:
                        continue
                    j = int(bucket_arr[col])
                    if j == qi:
                        continue
                    a, b_ = (qi, j) if qi < j else (j, qi)
                    if (a, b_) in seen:
                        continue
                    seen.add((a, b_))
                    candidates.append((a, b_, s))

        if (center_ord + 1) % 50 == 0 or center_ord == len(key_dates) - 1:
            print(f"  [bucket] day {center_ord+1}/{len(key_dates)}; "
                  f"pairs so far: {len(candidates)}")

    return candidates


# Main ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.92,
                    help="Cosine-sim floor for treating two facts as paraphrases.")
    ap.add_argument("--window-days", type=int, default=3,
                    help="Max difference in fact `time` for a pair to cluster.")
    ap.add_argument("--top-k", type=int, default=10,
                    help="Per-fact neighbor candidates to inspect.")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Encoding batch size (used by SBERT path; ignored by OpenAI).")
    ap.add_argument("--embedder", choices=["openai", "sbert"], default=DEFAULT_EMBEDDER,
                    help="Embedding backend. 'openai' uses Batch API (~$0.04 for 78k facts, "
                         "~10-15 min wall-clock, no local GPU); 'sbert' uses local "
                         "mpnet-base-v2 on CUDA if available (~5-10 min on RTX 2060).")
    ap.add_argument("--openai-model", default=EMBED_MODEL_OPENAI,
                    help="OpenAI embedding model when --embedder=openai.")
    ap.add_argument("--openai-mode", choices=["sync", "batch"], default="batch",
                    help="OpenAI execution mode when --embedder=openai. Batch = 50%% cheaper.")
    ap.add_argument("--knn-mode", choices=["bucket", "brute"], default="bucket",
                    help="kNN search strategy. 'bucket' (default, v2.2): group facts by "
                         "event date and search only within the +/- window-days bucket; "
                         "O(N x B) time, uses FAISS IndexFlatIP per bucket when available. "
                         "'brute': legacy single sliced matmul over the full matrix.")
    ap.add_argument("--limit", type=int, default=None,
                    help="(Smoke) only process first N facts.")
    args = ap.parse_args()

    if not INPUT.exists():
        print(f"[ERROR] {INPUT} missing. Run scripts/articles_to_facts.py first.")
        return 1

    print(f"[etd_dedup] reading {INPUT}")
    facts: list[dict] = []
    with open(INPUT, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            facts.append(json.loads(line))
    n = len(facts)
    print(f"[etd_dedup] loaded {n} facts")

    import numpy as np

    texts = [f.get("fact", "") for f in facts]
    t0 = time.time()
    if args.embedder == "openai":
        # OpenAI Batch API path: no GPU; ~$0.04 for 78k facts; ~10-15 min wall-clock.
        # Reuses the same caching layer as compute_relevance.py via load_or_embed_openai.
        from src.common.openai_embeddings import encode_batch, encode_sync
        print(f"[etd_dedup] encoding via OpenAI {args.openai_model} (mode={args.openai_mode})")
        encoder = encode_batch if args.openai_mode == "batch" else encode_sync
        embs = encoder(texts, model=args.openai_model)
        embs = np.asarray(embs, dtype=np.float32)
    else:
        # Legacy SBERT path: kept as a fallback (set --embedder sbert). Uses GPU
        # via sentence-transformers if CUDA is available; FP16 for speed.
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[etd_dedup] encoding via SBERT {EMBED_MODEL_SBERT} on {device}")
        model = SentenceTransformer(EMBED_MODEL_SBERT, device=device)
        if device == "cuda":
            try:
                model = model.half()
            except Exception:
                pass
        embs = model.encode(texts, batch_size=args.batch_size,
                            normalize_embeddings=True, show_progress_bar=True)
        embs = np.asarray(embs, dtype=np.float32)
    print(f"[etd_dedup] encoded {len(texts)} facts in {time.time() - t0:.1f}s "
          f"-> shape {embs.shape}")

    fact_dates = [_parse_date(f.get("time") or f.get("article_date")) for f in facts]

    t1 = time.time()
    if args.knn_mode == "bucket":
        print(f"[etd_dedup] kNN mode=bucket (window={args.window_days}d, top-{args.top_k})")
        candidates = _knn_bucketed(
            embs, fact_dates,
            window_days=args.window_days,
            threshold=args.threshold,
            top_k=args.top_k,
        )
        # Bucket search already enforces the +/- window constraint, so pairs
        # are already date-window-filtered. No extra filter pass needed.
        kept = len(candidates)
    else:
        print(f"[etd_dedup] kNN mode=brute (top-{args.top_k})")
        candidates = _knn_brute(embs, args.threshold, args.top_k)
        # Date filter: only cluster pairs whose fact `time` is within window
        window = timedelta(days=args.window_days)
        kept = 0
        filtered: list[tuple[int, int, float]] = []
        for i, j, s in candidates:
            di, dj = fact_dates[i], fact_dates[j]
            if di is None or dj is None:
                continue
            if abs(di - dj) <= window:
                filtered.append((i, j, s))
                kept += 1
        candidates = filtered

    print(f"[etd_dedup] kNN took {time.time() - t1:.1f}s; "
          f"candidate pairs above {args.threshold}: {kept}")

    uf = UF(n)
    for i, j, _s in candidates:
        uf.union(i, j)

    # Group by cluster root
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    confidence_rank = {"high": 3, "medium": 2, "low": 1}

    def _rank_key(idx: int) -> tuple:
        f = facts[idx]
        c = confidence_rank.get((f.get("extraction_confidence") or "").lower(), 0)
        d = fact_dates[idx] or datetime(9999, 12, 31)
        return (-c, d)

    # Reset canonical / variant fields, then refill
    for f in facts:
        f["canonical_id"] = None
        f["variant_ids"] = []

    cluster_sizes = Counter()
    for members in groups.values():
        cluster_sizes[len(members)] += 1
        if len(members) == 1:
            continue
        members.sort(key=_rank_key)
        canon_idx = members[0]
        canon_id = facts[canon_idx]["id"]
        variants = [facts[m]["id"] for m in members[1:]]
        facts[canon_idx]["variant_ids"] = variants
        for m in members[1:]:
            facts[m]["canonical_id"] = canon_id

    deduped_n = sum(1 for f in facts if f["canonical_id"])
    print(f"[etd_dedup] clusters: {len(groups)} | "
          f"deduped facts: {deduped_n} | unique canonicals: {n - deduped_n}")
    print(f"[etd_dedup] cluster-size histogram (top 10): "
          f"{dict(sorted(cluster_sizes.items())[:10])}")

    def _w(f):
        for fact in facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")

    print(f"[etd_dedup] writing {OUTPUT}")
    _atomic_write(OUTPUT, _w)

    meta = {
        "input": str(INPUT),
        "output": str(OUTPUT),
        "n_facts": n,
        "n_clusters": len(groups),
        "n_deduped": deduped_n,
        "n_unique_canonicals": n - deduped_n,
        "threshold": args.threshold,
        "window_days": args.window_days,
        "top_k": args.top_k,
        "embedder": args.embedder,
        "embed_model": (args.openai_model if args.embedder == "openai" else EMBED_MODEL_SBERT),
        "knn_mode": args.knn_mode,
        "cluster_size_histogram": dict(sorted(cluster_sizes.items())),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    def _wm(f):
        json.dump(meta, f, indent=2)

    _atomic_write(META, _wm)
    print(f"[etd_dedup] meta -> {META}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
