"""v2.2 CLI: encode the unified article + forecast pool via OpenAI embeddings.

Drop-in alternative to the SBERT path in `scripts/compute_relevance.py`.
Produces L2-normalized .npy cache files compatible with the cosine-sim
scoring downstream, but written under separate names so the SBERT and
OpenAI caches do not collide:

  data/unified/article_embeddings_openai.npy   + .fp.txt
  data/unified/forecast_embeddings_openai.npy  + .fp.txt

The relevance scoring step (compute_relevance.py main loop) can then
consume either cache by checking which exists. v2.2 work item: extend
compute_relevance to take a `--embedder {sbert,openai}` flag that
selects the cache; for now this script just produces the OpenAI cache,
and consumers can swap manually.

Cost (April 2026 pricing):
  text-embedding-3-small via Batch API: ~$0.30 for full v2.1 pool
                                        (218k articles + 11k FDs ~= 26M tokens)
  text-embedding-3-large via Batch API: ~$1.90 for the same pool
Wall-clock for batch: ~30-60 min including OpenAI queue time.

Usage:
  # Default: text-embedding-3-small, batch mode, full pool
  python scripts/embed_pool_openai.py

  # Sync mode (no batch wait, ~5x more expensive but instant)
  python scripts/embed_pool_openai.py --mode sync

  # Articles only (skip forecasts)
  python scripts/embed_pool_openai.py --skip-forecasts

  # Force re-encode (ignore cache)
  python scripts/embed_pool_openai.py --rebuild

  # Bigger model (1536-dim small -> 3072-dim large; ~6x cost)
  python scripts/embed_pool_openai.py --model text-embedding-3-large
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.openai_embeddings import load_or_embed_openai, DEFAULT_MODEL  # noqa: E402

DATA = ROOT / "data"
UNI = DATA / "unified"
ARTICLES_PATH = UNI / "articles.jsonl"
FORECASTS_PATH = UNI / "forecasts.jsonl"
ART_CACHE = UNI / "article_embeddings_openai.npy"
FC_CACHE = UNI / "forecast_embeddings_openai.npy"
META_OUT = UNI / "embeddings_openai_meta.json"


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _article_text(a: dict, max_chars: int) -> str:
    return ((a.get("title") or "") + "\n" + (a.get("text") or "")[:max_chars]).strip() or " "


def _forecast_text(f: dict) -> str:
    parts = [f.get("question") or ""]
    bg = (f.get("background") or "").strip()
    if bg:
        parts.append(bg[:1500])
    return "\n".join(parts).strip() or " "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sync", "batch"], default="batch",
                    help="OpenAI execution path. Batch is 50%% cheaper, ~30-60 min "
                         "wall-clock; sync is instant but rate-limited.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="OpenAI embedding model (default: text-embedding-3-small).")
    ap.add_argument("--max-text-chars", type=int, default=500,
                    help="Per-article text truncation. Matches SBERT path's default.")
    ap.add_argument("--skip-articles", action="store_true",
                    help="Skip the article-pool encoding step.")
    ap.add_argument("--skip-forecasts", action="store_true",
                    help="Skip the forecast-pool encoding step.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Ignore existing .npy cache; re-encode everything.")
    args = ap.parse_args()

    if not ARTICLES_PATH.exists() or not FORECASTS_PATH.exists():
        print(f"[ERROR] missing inputs: {ARTICLES_PATH} or {FORECASTS_PATH}")
        print(f"        Run scripts/build_benchmark.py first to populate data/unified/.")
        return 1

    summary: dict = {"model": args.model, "mode": args.mode,
                     "max_text_chars": args.max_text_chars,
                     "rebuild": args.rebuild}

    if not args.skip_articles:
        print(f"[articles] loading {ARTICLES_PATH}")
        arts = _load_jsonl(ARTICLES_PATH)
        print(f"[articles] {len(arts)} records")
        emb = load_or_embed_openai(
            arts,
            text_fn=lambda a: _article_text(a, args.max_text_chars),
            cache_path=ART_CACHE,
            model=args.model,
            mode=args.mode,
            rebuild=args.rebuild,
        )
        print(f"[articles] embeddings shape: {emb.shape} -> {ART_CACHE}")
        summary["articles"] = {"n": len(arts), "shape": list(emb.shape),
                               "cache": str(ART_CACHE)}

    if not args.skip_forecasts:
        print(f"[forecasts] loading {FORECASTS_PATH}")
        fds = _load_jsonl(FORECASTS_PATH)
        print(f"[forecasts] {len(fds)} records")
        emb = load_or_embed_openai(
            fds,
            text_fn=_forecast_text,
            cache_path=FC_CACHE,
            model=args.model,
            mode=args.mode,
            rebuild=args.rebuild,
        )
        print(f"[forecasts] embeddings shape: {emb.shape} -> {FC_CACHE}")
        summary["forecasts"] = {"n": len(fds), "shape": list(emb.shape),
                                "cache": str(FC_CACHE)}

    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] meta -> {META_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
