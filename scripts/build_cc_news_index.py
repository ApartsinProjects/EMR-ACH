"""Build a FAISS index plus parquet metadata from CC-News shards.

Reads zstd-compressed JSONL shards produced by
``scripts/fetch_cc_news_archive.py``, encodes ``title + first-N-chars(text)``
with either OpenAI Batch (default, per v2.2 §6) or local SBERT, and
writes per-month FAISS flat-IP indices plus aligned parquet metadata.

Outputs (per month, under ``--out``):
- ``{YYYY-MM}.faiss``: FAISS flat-IP index (cosine via L2-normalized vectors).
- ``{YYYY-MM}.parquet``: metadata aligned by row index (url, host,
  publish_date, title, shard).
- ``manifest.json``: model name, embedding dim, row counts, backend,
  build timestamp.

Usage
-----
    python scripts/build_cc_news_index.py \\
        --in data/cc_news/ --out data/cc_news/index/ \\
        --embedder openai

See ``docs/V2_2_ARCHITECTURE.md`` Section 3.2 for the sibling GDELT DOC
index design; this script follows the same shape.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent))


def _iter_zst_jsonl(path: Path) -> Iterator[dict]:
    import zstandard as zstd  # type: ignore

    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        buf = b""
        while True:
            chunk = reader.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    yield json.loads(line.decode("utf-8"))
        if buf.strip():
            yield json.loads(buf.decode("utf-8"))


def load_month_shards(in_dir: Path, month: str) -> list[dict]:
    """Load all rows from every shard in ``in_dir/{month}/``.

    Skips shards that have no matching ``.done`` sidecar (meaning the
    fetcher crashed mid-shard and the output is incomplete).
    """
    month_dir = in_dir / month
    if not month_dir.exists():
        return []
    rows: list[dict] = []
    for shard in sorted(month_dir.glob("shard_*.jsonl.zst")):
        done = shard.with_suffix(shard.suffix + ".done")
        if not done.exists():
            print(f"[cc-news-index]   skipping incomplete shard: {shard.name}")
            continue
        for rec in _iter_zst_jsonl(shard):
            rows.append(rec)
    return rows


def _text_for_embedding(row: dict, max_chars: int = 1200) -> str:
    title = (row.get("title") or "").strip()
    body = (row.get("text") or "").strip()
    if title:
        snippet = f"{title}\n\n{body[:max_chars]}"
    else:
        snippet = body[:max_chars]
    return snippet or "(empty)"


def _encode_openai(rows: list[dict], cache_path: Path, model: str) -> "np.ndarray":  # type: ignore
    from src.common.openai_embeddings import load_or_embed_openai  # type: ignore

    return load_or_embed_openai(
        items=rows,
        text_fn=_text_for_embedding,
        cache_path=cache_path,
        model=model,
        mode="batch",
    )


def _encode_sbert(rows: list[dict], model_name: str, device: str = "cpu") -> "np.ndarray":  # type: ignore
    import numpy as np

    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name, device=device)
    texts = [_text_for_embedding(r) for r in rows]
    emb = model.encode(texts, batch_size=128, convert_to_numpy=True,
                       show_progress_bar=True, normalize_embeddings=True)
    return emb.astype(np.float32)


def _build_faiss(embeddings, out_path: Path) -> int:
    import faiss  # type: ignore

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(out_path))
    return dim


def _write_parquet(rows: list[dict], out_path: Path) -> None:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        print("[cc-news-index]   WARNING: pandas not available, writing JSONL fallback")
        out_jsonl = out_path.with_suffix(".jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({
                    "url": r.get("url"),
                    "host": r.get("host"),
                    "publish_date": r.get("publish_date"),
                    "title": r.get("title"),
                    "shard": r.get("shard"),
                }, ensure_ascii=False) + "\n")
        return
    df = pd.DataFrame([{
        "url": r.get("url"),
        "host": r.get("host"),
        "publish_date": r.get("publish_date"),
        "title": r.get("title"),
        "shard": r.get("shard"),
    } for r in rows])
    df.to_parquet(out_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_dir", default="data/cc_news/", help="CC-News shard root")
    p.add_argument("--out", default="data/cc_news/index/", help="Index output directory")
    p.add_argument("--embedder", choices=["openai", "sbert"], default="openai")
    p.add_argument("--openai-model", default="text-embedding-3-small")
    p.add_argument("--sbert-model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--device", default="cpu", help="SBERT device (cpu or cuda)")
    p.add_argument("--month", default=None,
                   help="Restrict to a single month (YYYY-MM); default: all subdirs of --in")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.month:
        months = [args.month]
    else:
        months = sorted(p.name for p in in_dir.iterdir() if p.is_dir() and p.name != "index")

    manifest = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "embedder": args.embedder,
        "model": args.openai_model if args.embedder == "openai" else args.sbert_model,
        "months": {},
    }

    for month in months:
        t0 = time.time()
        print(f"[cc-news-index] {month}: loading shards...")
        rows = load_month_shards(in_dir, month)
        if not rows:
            print(f"[cc-news-index]   no rows, skipping")
            continue
        print(f"[cc-news-index]   loaded {len(rows):,} rows")

        if args.embedder == "openai":
            cache = out_dir / f"{month}.openai_cache.npy"
            emb = _encode_openai(rows, cache, args.openai_model)
        else:
            emb = _encode_sbert(rows, args.sbert_model, device=args.device)

        faiss_path = out_dir / f"{month}.faiss"
        dim = _build_faiss(emb, faiss_path)
        meta_path = out_dir / f"{month}.parquet"
        _write_parquet(rows, meta_path)
        dt = time.time() - t0
        manifest["months"][month] = {
            "rows": len(rows),
            "dim": dim,
            "faiss": faiss_path.name,
            "parquet": meta_path.name,
            "duration_s": round(dt, 2),
        }
        print(f"[cc-news-index]   wrote {faiss_path.name} ({len(rows):,} x {dim}), "
              f"{meta_path.name}, {dt:.1f}s")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[cc-news-index] manifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
