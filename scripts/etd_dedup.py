"""ETD Stage 2: cluster near-duplicate facts and assign canonical IDs.

The Stage-1 LLM extractor often produces multiple paraphrased versions of the
same atomic fact (different articles describing the same press conference,
the same casualty count from a single Reuters wire that hit five outlets).
This stage groups paraphrases by semantic similarity, picks one canonical
fact per cluster, and tags every other fact with `canonical_id` pointing at
the chosen representative.

Algorithm
---------
1. SBERT-encode every fact's `fact` string (mpnet-base-v2, normalized).
2. Build candidate-pair shortlist via FAISS / brute-force kNN (top-k similar
   facts above cosine threshold tau).
3. Filter pairs: keep only same-day (or same-week) pairs, since unrelated
   events can phrase identically.
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

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


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
                    help="SBERT encoding batch size.")
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

    # Lazy GPU import; keeps non-Stage-2 use of this module cheap.
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[etd_dedup] encoding via {EMBED_MODEL} on {device}")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL, device=device)
    texts = [f.get("fact", "") for f in facts]
    embs = model.encode(texts, batch_size=args.batch_size,
                        normalize_embeddings=True, show_progress_bar=True)
    embs = np.asarray(embs, dtype=np.float32)
    print(f"[etd_dedup] encoded in {time.time() - t0:.1f}s")

    # Brute-force kNN via single big matmul, sliced (n^2 memory at 69k * 768
    # floats = ~14 GB if dense; we slice in row-chunks). For 70k facts this
    # is fine; FAISS would only matter beyond ~500k.
    print(f"[etd_dedup] computing pairwise sims (top-{args.top_k}) ...")
    candidates: list[tuple[int, int, float]] = []
    chunk = 1024
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        sims = embs[i0:i1] @ embs.T  # (chunk, n)
        # Mask self-similarity
        for r, gi in enumerate(range(i0, i1)):
            sims[r, gi] = -1.0
        # Top-k indices per row
        top_idx = np.argpartition(-sims, args.top_k, axis=1)[:, :args.top_k]
        for r, gi in enumerate(range(i0, i1)):
            for j in top_idx[r]:
                s = float(sims[r, j])
                if s >= args.threshold and gi < j:  # dedup pair (i,j) i<j
                    candidates.append((gi, int(j), s))
        if (i1 % 8192) == 0 or i1 == n:
            print(f"  scanned {i1}/{n}; pairs so far: {len(candidates)}")
    print(f"[etd_dedup] candidate pairs above {args.threshold}: {len(candidates)}")

    # Date filter: only cluster pairs whose fact `time` is within window
    fact_dates = [_parse_date(f.get("time") or f.get("article_date")) for f in facts]
    window = timedelta(days=args.window_days)
    kept = 0
    uf = UF(n)
    for i, j, _s in candidates:
        di, dj = fact_dates[i], fact_dates[j]
        if di is None or dj is None:
            continue
        if abs(di - dj) <= window:
            uf.union(i, j)
            kept += 1
    print(f"[etd_dedup] pairs after date-window filter: {kept}")

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
        "embed_model": EMBED_MODEL,
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
