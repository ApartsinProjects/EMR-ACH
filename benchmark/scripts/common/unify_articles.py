"""
Unify article records from ForecastBench (GDELT DOC API + trafilatura) and
GDELT-CAMEO (GDELT KG oracle links + trafilatura) into a single article file.

Reads (read-only):
  data/fb_articles_full.jsonl                 - ForecastBench GDELT + full text
  data/gdelt_cameo/data_news_full.csv          - GDELT-CAMEO oracle articles (if exists)
  data/gdelt_cameo/data_news.csv               - fallback if *_full.csv missing

Writes:
  data/unified/articles.jsonl                 - deduplicated by sha1(url)[:12]
  data/unified/articles_meta.json             - summary stats

Article schema:
  { id, url, title, text, title_text, publish_date, source_domain,
    gdelt_themes, gdelt_tone, actors, cameo_code, char_count,
    provenance: ["forecastbench" | "gdelt-cameo", ...] }

Usage:
  python scripts/unify_articles.py
"""
import csv
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

# data_news_full.csv's Text column can exceed the default 128k csv field limit
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
# Fast JSONL I/O (orjson if available, stdlib json fallback) — see _fast_jsonl.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from _fast_jsonl import loads as _j_loads, dumps as _j_dumps

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_ART = OUT_DIR / "articles.jsonl"
OUT_META = OUT_DIR / "articles_meta.json"

FB_FULL = ROOT / "data" / "fb_articles_full.jsonl"
FB_SUPP = ROOT / "data" / "gdelt_articles_supplement.jsonl"   # Step B additions
GDELT_CAMEO_FULL = ROOT / "data" / "gdelt_cameo" / "data_news_full.csv"
GDELT_CAMEO_PLAIN = ROOT / "data" / "gdelt_cameo" / "data_news.csv"


def art_id(url: str) -> str:
    return "art_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def load_jsonl_articles(path: Path, provenance_tag: str) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = _j_loads(line)
            url = r.get("url", "")
            if not url:
                continue
            title = r.get("title", "") or ""
            text = r.get("text", "") or ""
            title_text = (title + "\n" + text).strip()
            out.append({
                "id": art_id(url),
                "url": url,
                "title": title,
                "text": text,
                "title_text": title_text,
                "publish_date": r.get("date", "") or "",
                "source_domain": r.get("source", "") or domain_of(url),
                "gdelt_themes": [],
                "gdelt_tone": float(r.get("tone", 0.0) or 0.0),
                "actors": r.get("country_mentions", []) or [],
                "cameo_code": "",
                "char_count": len(title_text),
                "provenance": [provenance_tag],
            })
    return out


def load_forecastbench() -> list[dict]:
    if not FB_FULL.exists():
        print(f"[WARN] {FB_FULL} not found; skipping ForecastBench articles")
        return []
    return load_jsonl_articles(FB_FULL, "forecastbench")


def load_forecastbench_supplement() -> list[dict]:
    return load_jsonl_articles(FB_SUPP, "forecastbench-stepB")


def load_gdelt_cameo() -> list[dict]:
    src = GDELT_CAMEO_FULL if GDELT_CAMEO_FULL.exists() else (GDELT_CAMEO_PLAIN if GDELT_CAMEO_PLAIN.exists() else None)
    if src is None:
        print(f"[WARN] No MIRAI-2024 article CSV found; skipping (build still running?)")
        return []
    out = []
    with open(src, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            url = row.get("URL", "") or row.get("url", "")
            if not url:
                continue
            title = row.get("Title", "") or ""
            text = row.get("Text", "") or ""
            title_text = (title + "\n" + text).strip()
            out.append({
                "id": art_id(url),
                "url": url,
                "title": title,
                "text": text,
                "title_text": title_text,
                "publish_date": (row.get("Date", "") or "")[:10],
                "source_domain": domain_of(url),
                "gdelt_themes": (row.get("Themes", "") or "").split(";") if row.get("Themes") else [],
                "gdelt_tone": float(row.get("Tone", 0.0) or 0.0) if row.get("Tone") else 0.0,
                "actors": [a for a in (row.get("Actor1CountryCode", ""), row.get("Actor2CountryCode", "")) if a],
                "cameo_code": row.get("EventCode", "") or "",
                "char_count": len(title_text),
                "provenance": ["gdelt-cameo"],
            })
    return out


def dedup_merge(*lists: list[dict]) -> list[dict]:
    """Merge by id; union provenance, prefer record with longer text."""
    merged: dict[str, dict] = {}
    for lst in lists:
        for r in lst:
            rid = r["id"]
            if rid not in merged:
                merged[rid] = r
                continue
            existing = merged[rid]
            if len(r.get("text", "")) > len(existing.get("text", "")):
                existing["text"] = r["text"]
                existing["title_text"] = r["title_text"]
                existing["char_count"] = r["char_count"]
            if not existing.get("title") and r.get("title"):
                existing["title"] = r["title"]
            # union provenance
            prov = set(existing.get("provenance", [])) | set(r.get("provenance", []))
            existing["provenance"] = sorted(prov)
            # fill missing metadata
            for k in ("publish_date", "cameo_code"):
                if not existing.get(k) and r.get(k):
                    existing[k] = r[k]
            existing["actors"] = list({*existing.get("actors", []), *r.get("actors", [])})
            existing["gdelt_themes"] = list({*existing.get("gdelt_themes", []), *r.get("gdelt_themes", [])})
    return list(merged.values())


def main():
    print("Loading ForecastBench articles (primary)...")
    fb = load_forecastbench()
    print(f"  {len(fb)} records")

    print("Loading ForecastBench Step-B supplement (if any)...")
    fb_supp = load_forecastbench_supplement()
    print(f"  {len(fb_supp)} records")

    print("Loading MIRAI-2024 articles...")
    m24 = load_gdelt_cameo()
    print(f"  {len(m24)} records")

    print("Merging + deduplicating by URL hash...")
    merged = dedup_merge(fb, fb_supp, m24)

    with open(OUT_ART, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # stats
    with_text = sum(1 for r in merged if r.get("text"))
    by_prov: dict[tuple, int] = {}
    for r in merged:
        key = tuple(r.get("provenance", []))
        by_prov[key] = by_prov.get(key, 0) + 1

    meta = {
        "total_articles": len(merged),
        "with_full_text": with_text,
        "with_full_text_pct": round(100 * with_text / len(merged), 1) if merged else 0,
        "by_provenance": {"|".join(k): v for k, v in by_prov.items()},
        "input_counts": {"forecastbench": len(fb), "gdelt-cameo": len(m24)},
        "output_path": str(OUT_ART),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Wrote {len(merged)} unique articles to {OUT_ART}")
    print(f"  Full text: {with_text} ({meta['with_full_text_pct']}%)")
    print(f"  By provenance: {meta['by_provenance']}")


if __name__ == "__main__":
    main()
