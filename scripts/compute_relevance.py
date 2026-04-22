"""
Compute per-forecast relevance over the unified article pool using SBERT.

Step A of the coverage-recovery plan: for every forecast, search the ENTIRE
unified article pool (not just the per-question articles originally downloaded
by GDELT) and attach the top-k most relevant articles that pass date + keyword
filters. This recovers forecasts that had no per-question articles but do have
semantically-relevant articles elsewhere in the pool.

Reads:
  data/unified/forecasts.jsonl
  data/unified/articles.jsonl
  configs/default_config.yaml  (`relevance:` section)

Writes:
  data/unified/forecasts.jsonl             - article_ids overwritten
  data/unified/relevance_meta.json         - stats
  data/unified/article_embeddings.npy      - cached SBERT embeddings
  data/unified/forecast_embeddings.npy

Usage:
  python scripts/compute_relevance.py               # CPU default
  python scripts/compute_relevance.py --device cuda # if GPU available
  python scripts/compute_relevance.py --rebuild     # force re-embed
"""
import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer, util
# Fast JSONL I/O (orjson if available, stdlib json fallback) — see _fast_jsonl.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from _fast_jsonl import loads as _j_loads, dumps as _j_dumps

ROOT = Path(__file__).parent.parent
UNI = ROOT / "data" / "unified"
# Relevance config lives under the `relevance:` section of default_config.yaml.
# Falls back to the legacy standalone configs/relevance.yaml if present (deprecated).
CONFIG_MAIN   = ROOT / "configs" / "default_config.yaml"
CONFIG_LEGACY = ROOT / "configs" / "relevance.yaml"
FC_FILE = UNI / "forecasts.jsonl"
ART_FILE = UNI / "articles.jsonl"
META = UNI / "relevance_meta.json"
ART_EMB = UNI / "article_embeddings.npy"
FC_EMB = UNI / "forecast_embeddings.npy"

STOP = set("will the a of to in on for at by an from and or when with which be is are was were have has had that this".split())

_NGRAM_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9\-]+\b")


def key_ngrams(text: str, n: int = 2) -> set[str]:
    words = [w.lower() for w in _NGRAM_RE.findall(text or "") if w.lower() not in STOP and len(w) > 2]
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def load_cfg() -> dict:
    """Read the relevance section from default_config.yaml. If the whole file
    is just the relevance knobs (legacy standalone configs/relevance.yaml),
    return it as-is."""
    path = CONFIG_MAIN if CONFIG_MAIN.exists() else CONFIG_LEGACY
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "relevance" in cfg:
        return cfg["relevance"]
    return cfg


def embed(model, texts: list[str], batch: int, desc: str) -> np.ndarray:
    print(f"  [{desc}] encoding {len(texts)} texts...")
    emb = model.encode(texts, batch_size=batch, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)


import hashlib as _hashlib


def _per_row_fingerprints(items: list[dict], text_fn) -> list[str]:
    """MD5 of each row's text. Used by the incremental cache loader to
    identify exactly which rows changed between builds, so we can re-embed
    only the deltas rather than the entire pool."""
    out = []
    for r in items:
        h = _hashlib.md5()
        h.update(text_fn(r).encode("utf-8", errors="replace"))
        out.append(h.hexdigest())
    return out


def load_or_embed(model, items: list[dict], text_fn, cache_path: Path,
                  batch: int, desc: str, rebuild: bool) -> np.ndarray:
    """Incremental embedding cache.

    State on disk:
      {cache}.npy     numpy array, shape = (N, D)
      {cache}.fp.txt  N lines, each line = per-row MD5 fingerprint (order-aligned)

    Strategy:
      1. Compute current per-row fingerprints.
      2. Load cached fingerprints if present. Build a `fp -> row_idx` map.
      3. For each current row, if its fingerprint is in the cached map,
         copy the old embedding; otherwise mark for (re-)embedding.
      4. Embed only the deltas; assemble the final array; persist new cache.
      5. On save, the previous .npy is preserved under a timestamped backup.
    """
    fp_path = cache_path.with_suffix(".fp.txt")
    current_fps = _per_row_fingerprints(items, text_fn)
    N = len(items)

    cached_emb = None
    cached_fps: list[str] = []
    if not rebuild and cache_path.exists() and fp_path.exists():
        try:
            cached_emb = np.load(cache_path)
            cached_fps = fp_path.read_text(encoding="utf-8").splitlines()
            if cached_emb.shape[0] != len(cached_fps):
                cached_emb = None  # corrupt cache — ignore
                cached_fps = []
        except Exception:
            cached_emb = None
            cached_fps = []

    if cached_emb is None:
        # full cold embed
        print(f"  [{desc}] no usable cache; embedding all {N} rows")
        emb = embed(model, [text_fn(r) for r in items], batch, desc)
    else:
        # incremental path — reuse rows whose fingerprint is in the old cache
        fp_to_idx = {fp: i for i, fp in enumerate(cached_fps)}
        D = cached_emb.shape[1]
        out = np.zeros((N, D), dtype=cached_emb.dtype)
        to_embed_idx = []
        for i, fp in enumerate(current_fps):
            j = fp_to_idx.get(fp, -1)
            if j >= 0:
                out[i] = cached_emb[j]
            else:
                to_embed_idx.append(i)

        n_reused = N - len(to_embed_idx)
        print(f"  [{desc}] incremental cache: reused {n_reused}/{N} ({100*n_reused/N:.1f}%), "
              f"embedding {len(to_embed_idx)} new/changed rows")
        if to_embed_idx:
            texts = [text_fn(items[i]) for i in to_embed_idx]
            new_emb = embed(model, texts, batch, f"{desc}-delta")
            for pos, i in enumerate(to_embed_idx):
                out[i] = new_emb[pos]
        # Normalize: sentence-transformers with normalize_embeddings=True produces
        # L2-normalized vectors; rows copied from cache inherit that. New rows from
        # embed() also have it. No re-normalize needed.
        emb = out

    # Preserve previous cache so nothing is overwritten in place
    if cache_path.exists():
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        backup = cache_path.with_name(f"{cache_path.stem}.prev_{ts}.npy")
        try:
            cache_path.rename(backup)
            if fp_path.exists():
                fp_path.rename(fp_path.with_name(f"{fp_path.stem}.prev_{ts}.txt"))
            print(f"  [{desc}] previous cache preserved as {backup.name}")
        except Exception:
            pass

    np.save(cache_path, emb)
    fp_path.write_text("\n".join(current_fps), encoding="utf-8")
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--benchmark-filter", default="forecastbench",
                    help="Only score forecasts in this benchmark; articles are also filtered to this provenance family")
    args = ap.parse_args()

    cfg = load_cfg()
    model_name = cfg["embedding_model"]
    batch = cfg.get("batch_size", 32)
    max_chars = cfg.get("max_text_chars", 500)
    sources = cfg["sources"]

    all_forecasts = [_j_loads(l) for l in open(FC_FILE, encoding="utf-8")]
    all_articles = [_j_loads(l) for l in open(ART_FILE, encoding="utf-8")]

    # Subset by benchmark/provenance so we don't embed MIRAI's 178K articles
    # when we only need to score ForecastBench cross-matches.
    if args.benchmark_filter:
        bench = args.benchmark_filter
        # Benchmark name normalization: 'gdelt_cameo' (Python identifier style,
        # used in YAML keys) and 'gdelt-cameo' (dash style, stored in FD records)
        # are aliases. Always resolve to the dash form used in data.
        _alias = {'gdelt_cameo': 'gdelt-cameo'}
        bench = _alias.get(bench, bench)
        # Per-benchmark article pool:
        #   - forecastbench → articles with 'forecastbench' in provenance
        #   - gdelt-cameo   → articles with 'gdelt' in provenance
        #   - earnings      → NO provenance filter (cross-search whole pool —
        #                     earnings has no dedicated news source yet)
        if bench == "earnings":
            articles = all_articles
            print(f"Benchmark filter: {bench} -> searching ENTIRE article pool "
                  f"({len(articles)} articles) since earnings has no dedicated news source")
        else:
            prov_tag = bench.split("-")[0]
            articles = [a for a in all_articles
                        if any(prov_tag in p for p in a.get("provenance", []))]
            print(f"Benchmark filter: {bench} -> {len(articles)} articles "
                  f"(provenance contains '{prov_tag}')")
        scored_ids = {f["id"] for f in all_forecasts if f.get("benchmark") == bench}
        print(f"  scoring {len(scored_ids)} forecasts in scope")
    else:
        articles = all_articles
        scored_ids = {f["id"] for f in all_forecasts}
    forecasts = all_forecasts  # keep full list for final write-back
    scored_forecasts = [f for f in forecasts if f["id"] in scored_ids]

    print(f"Loaded {len(forecasts)} total forecasts, {len(articles)} in-scope articles")
    print(f"Loading SBERT model: {model_name} (device={args.device})")
    model = SentenceTransformer(model_name, device=args.device)

    art_emb = load_or_embed(
        model, articles,
        lambda a: (a.get("title", "") + "\n" + a.get("text", "")[:max_chars]).strip() or " ",
        ART_EMB, batch, "articles", args.rebuild,
    )
    fc_emb = load_or_embed(
        model, scored_forecasts,
        lambda q: (q.get("question", "") + "\n" + q.get("background", "")[:max_chars]).strip() or " ",
        FC_EMB, batch, "forecasts", args.rebuild,
    )

    # Parse article publication dates once
    art_dates: list[datetime | None] = []
    for a in articles:
        d = a.get("publish_date") or ""
        try:
            art_dates.append(datetime.strptime(d[:10], "%Y-%m-%d"))
        except Exception:
            art_dates.append(None)

    # Precompute article n-grams for keyword overlap filter
    art_ngrams = [key_ngrams(a.get("title", "") + " " + a.get("text", "")[:max_chars]) for a in articles]

    stats = {
        "before": {"with_arts": 0, "zero": 0},
        "after":  {"with_arts": 0, "zero": 0},
        "by_source": {},
    }
    for fc in forecasts:
        stats["before"]["with_arts" if fc.get("article_ids") else "zero"] += 1

    # main scoring loop — only iterate over scored_forecasts
    print("Scoring relevance...")
    for i, fc in enumerate(scored_forecasts):
        src = fc.get("source", "")
        scfg = sources.get(src) or sources.get("polymarket")  # fall back to generic
        thresh = scfg["embedding_threshold"]
        kw_min = scfg["keyword_overlap_min"]
        lookback = scfg["lookback_days"]
        top_k = scfg["top_k"]
        rec_w = scfg["recency_weight"]
        actor_req = scfg.get("actor_match_required", False)
        fc_actors = set(fc.get("actors", []) or [])

        # time bounds
        try:
            t_star = datetime.strptime(fc["forecast_point"][:10], "%Y-%m-%d")
        except Exception:
            fc["article_ids"] = []
            continue
        lb_cutoff = t_star - timedelta(days=lookback)

        sims = (art_emb @ fc_emb[i])   # cosine sim since both are unit-normalized

        q_ngrams = key_ngrams(fc.get("question", "") + " " + fc.get("background", "")[:max_chars])

        cands = []
        for j, a in enumerate(articles):
            d = art_dates[j]
            if d is None:
                continue
            if d >= t_star or d < lb_cutoff:
                continue
            if sims[j] < thresh:
                continue
            if kw_min > 0 and len(q_ngrams & art_ngrams[j]) < kw_min:
                continue
            if actor_req:
                a_actors = set(a.get("actors", []) or [])
                if not (fc_actors & a_actors):
                    continue
            # blended score
            recency = 1.0 - (t_star - d).days / max(lookback, 1)
            score = rec_w * recency + (1 - rec_w) * float(sims[j])
            cands.append((score, a["id"], float(sims[j]), (t_star - d).days))

        cands.sort(reverse=True)
        # UNION with originally-linked article_ids (preserves GDELT's per-question
        # retrieval), then pad up to top_k from the best remaining SBERT matches.
        original = list(fc.get("article_ids") or [])
        kept = list(original)  # keep original order
        seen = set(kept)
        for _, aid, _sim, _age in cands:
            if aid in seen:
                continue
            if len(kept) >= top_k:
                break
            kept.append(aid)
            seen.add(aid)
        fc["article_ids"] = kept
        fc["_relevance_original_count"] = len(original)
        fc["_relevance_added_count"] = len(kept) - len(original)

    # recompute stats (only for scored forecasts)
    for fc in scored_forecasts:
        k = "with_arts" if fc.get("article_ids") else "zero"
        stats["after"][k] += 1
        src = fc.get("source", "?")
        stats["by_source"].setdefault(src, {"with_arts": 0, "zero": 0, "avg_k": 0.0})
        s = stats["by_source"][src]
        s[k] += 1

    for src, s in stats["by_source"].items():
        total = s["with_arts"] + s["zero"]
        s["pct_with_arts"] = round(100 * s["with_arts"] / total, 1) if total else 0
        lens = [len(fc["article_ids"]) for fc in scored_forecasts if fc.get("source") == src]
        s["avg_k"] = round(sum(lens) / len(lens), 2) if lens else 0

    # write back — atomic: write .tmp then os.replace (crash-safe in-place overwrite)
    import os as _os
    tmp_fc = FC_FILE.with_suffix(FC_FILE.suffix + ".tmp") if hasattr(FC_FILE, "with_suffix") else \
             Path(str(FC_FILE) + ".tmp")
    with open(tmp_fc, "w", encoding="utf-8") as f:
        for fc in forecasts:
            f.write(json.dumps(fc, ensure_ascii=False) + "\n")
        f.flush()
        try: _os.fsync(f.fileno())
        except Exception: pass
    _os.replace(str(tmp_fc), str(FC_FILE))

    tmp_meta = Path(str(META) + ".tmp")
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.flush()
        try: _os.fsync(f.fileno())
        except Exception: pass
    _os.replace(str(tmp_meta), str(META))

    print("\n=== Coverage lift ===")
    print(f"Before:  with_arts={stats['before']['with_arts']}  zero={stats['before']['zero']}")
    print(f"After:   with_arts={stats['after']['with_arts']}   zero={stats['after']['zero']}")
    print("\nPer-source after Step A:")
    for src, s in stats["by_source"].items():
        print(f"  {src:<12} with_arts={s['with_arts']:>3}  zero={s['zero']:>3}  "
              f"pct={s['pct_with_arts']:>5}%  avg_k={s['avg_k']}")


if __name__ == "__main__":
    main()
