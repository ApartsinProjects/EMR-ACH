"""Per-FD top-K retrieval against the CC-News FAISS index.

Reads FDs from ``--fds data/unified/forecasts.jsonl``, encodes each FD's
query text with the same backend used for index build (recorded in the
index manifest), and returns top-K candidate articles per FD after a
date-window prefilter and optional host filter.

Output: one JSONL row per FD with its top-K matches:

    {"fd_id": "...", "matches": [
        {"url": "...", "host": "...", "publish_date": "...",
         "title": "...", "score": 0.73}, ...]}

Sibling: ``scripts/query_gdelt_doc_index.py`` (see v2.2 Section 3.3).

Usage
-----
    python scripts/query_cc_news_index.py \\
        --index data/cc_news/index/ \\
        --fds data/unified/forecasts.jsonl \\
        --top-k 10 \\
        --out data/cc_news/enrichments.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).parent.parent))


def _parse_date(val) -> datetime | None:
    if not val:
        return None
    try:
        if isinstance(val, datetime):
            return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _fd_query_text(fd: dict) -> str:
    """Extract retrievable text from an FD. Falls back across schemas."""
    for k in ("question", "query", "title"):
        v = fd.get(k)
        if v:
            return str(v).strip()
    meta = fd.get("_meta") or fd.get("_earnings_meta") or {}
    bits = [str(meta.get("company_name", "")), str(meta.get("ticker", ""))]
    return " ".join(b for b in bits if b) or str(fd.get("fd_id", ""))


def _fd_forecast_point(fd: dict) -> datetime | None:
    for k in ("forecast_point", "announce_date", "event_date", "resolution_date"):
        dt = _parse_date(fd.get(k))
        if dt is not None:
            return dt
    meta = fd.get("_meta") or fd.get("_earnings_meta") or {}
    for k in ("announce_date", "event_date"):
        dt = _parse_date(meta.get(k))
        if dt is not None:
            return dt
    return None


def load_manifest(index_dir: Path) -> dict:
    p = index_dir / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}: build the index first via build_cc_news_index.py")
    return json.loads(p.read_text(encoding="utf-8"))


def encode_queries(texts: list[str], manifest: dict) -> "np.ndarray":  # type: ignore
    """Encode FD queries using the same backend as the index build."""
    import numpy as np

    backend = manifest.get("embedder", "openai")
    model = manifest.get("model")
    if backend == "openai":
        from src.common.openai_embeddings import encode_sync  # type: ignore

        emb = encode_sync(texts, model=model)
    else:
        from sentence_transformers import SentenceTransformer  # type: ignore

        m = SentenceTransformer(model, device="cpu")
        emb = m.encode(texts, batch_size=64, convert_to_numpy=True,
                       normalize_embeddings=True)
    return emb.astype(np.float32)


def _iter_fds(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def query_month(index_dir: Path, month: str, query_emb, top_k: int):
    import faiss  # type: ignore
    import numpy as np

    faiss_path = index_dir / f"{month}.faiss"
    meta_path = index_dir / f"{month}.parquet"
    if not faiss_path.exists():
        return None, None
    index = faiss.read_index(str(faiss_path))
    scores, idx = index.search(query_emb, top_k)

    # Load metadata (parquet preferred, fallback JSONL).
    if meta_path.exists():
        try:
            import pandas as pd  # type: ignore

            meta = pd.read_parquet(meta_path).to_dict(orient="records")
        except ImportError:
            meta = None
    else:
        jsonl_path = meta_path.with_suffix(".jsonl")
        meta = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return (scores, idx), meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", default="data/cc_news/index/", help="Index directory")
    p.add_argument("--fds", required=True, help="Path to forecasts.jsonl")
    p.add_argument("--out", required=True, help="Output enrichments JSONL")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--date-window-days", type=int, default=90,
                   help="Keep only matches whose publish_date is within this many days prior to the FD's forecast point")
    p.add_argument("--host-filter", default=None,
                   help="Optional comma-separated host suffix allowlist (e.g. 'reuters.com,bloomberg.com')")
    p.add_argument("--limit", type=int, default=0, help="Process at most this many FDs (0 = all)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index_dir = Path(args.index)
    manifest = load_manifest(index_dir)
    months = list(manifest.get("months", {}).keys())
    if not months:
        print("[cc-news-query] manifest has no months; nothing to query")
        return 1

    host_allow = None
    if args.host_filter:
        host_allow = {h.strip().lower() for h in args.host_filter.split(",") if h.strip()}

    fds = list(_iter_fds(Path(args.fds)))
    if args.limit:
        fds = fds[: args.limit]
    print(f"[cc-news-query] {len(fds)} FD(s), {len(months)} month shard(s), top-k={args.top_k}")

    queries = [_fd_query_text(fd) for fd in fds]
    q_emb = encode_queries(queries, manifest)

    # Pre-load each month's FAISS + metadata once.
    month_cache: dict[str, tuple] = {}
    for m in months:
        faiss_path = index_dir / f"{m}.faiss"
        meta_path = index_dir / f"{m}.parquet"
        if not faiss_path.exists():
            continue
        import faiss  # type: ignore

        idx = faiss.read_index(str(faiss_path))
        if meta_path.exists():
            try:
                import pandas as pd  # type: ignore

                meta = pd.read_parquet(meta_path).to_dict(orient="records")
            except ImportError:
                meta = []
        else:
            jpath = meta_path.with_suffix(".jsonl")
            meta = [json.loads(l) for l in jpath.read_text(encoding="utf-8").splitlines() if l.strip()] if jpath.exists() else []
        month_cache[m] = (idx, meta)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, fd in enumerate(fds):
            fp = _fd_forecast_point(fd)
            lo = (fp - timedelta(days=args.date_window_days)) if fp else None
            hi = fp if fp else None

            pooled: list[tuple[float, dict]] = []
            for month, (idx, meta) in month_cache.items():
                if not meta:
                    continue
                scores, ids = idx.search(q_emb[i:i + 1], args.top_k * 3)
                for rank, row_idx in enumerate(ids[0]):
                    if row_idx < 0 or row_idx >= len(meta):
                        continue
                    row = meta[row_idx]
                    dt = _parse_date(row.get("publish_date"))
                    if lo is not None and (dt is None or dt < lo or dt > hi):
                        continue
                    if host_allow is not None:
                        host = str(row.get("host") or "").lower().replace("www.", "")
                        if host not in host_allow:
                            # allow parent-domain match
                            parts = host.split(".")
                            if not any(".".join(parts[k:]) in host_allow for k in range(len(parts) - 1)):
                                continue
                    pooled.append((float(scores[0][rank]), row))
            pooled.sort(key=lambda t: -t[0])
            top = pooled[: args.top_k]
            out_row = {
                "fd_id": fd.get("fd_id") or fd.get("id"),
                "query": queries[i],
                "matches": [
                    {
                        "url": r.get("url"),
                        "host": r.get("host"),
                        "publish_date": r.get("publish_date"),
                        "title": r.get("title"),
                        "score": round(s, 6),
                    }
                    for s, r in top
                ],
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[cc-news-query] wrote {n_written} enrichment rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
