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

# data_news_full.csv's Text column can exceed the default 128k csv field limit.
# Helper below raises + restores the limit only around the specific reader that
# needs it, so we don't leave a process-wide side effect on import.

def _with_raised_csv_field_limit(fn):
    """Decorator: bump csv.field_size_limit for the duration of fn, then restore."""
    import functools
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        prev = csv.field_size_limit()
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        try:
            return fn(*args, **kwargs)
        finally:
            csv.field_size_limit(prev)
    return inner
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
EARNINGS_NEWS = ROOT / "data" / "earnings" / "earnings_articles.jsonl"  # yfinance + GDELT per-ticker
FB_NEWS       = ROOT / "data" / "forecastbench" / "forecastbench_articles.jsonl"
GDC_NEWS      = ROOT / "data" / "gdelt_cameo" / "gdelt_cameo_news.jsonl"


def art_id(url: str) -> str:
    """Stable 12-hex-char article identifier derived from URL SHA1.

    Collisions are astronomically unlikely at the corpus sizes we deal with
    (50k–200k articles), but if one occurs the downstream dedup_merge() will
    union provenance correctly — the semantics are "same URL → same article".
    """
    return "art_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def domain_of(url: str) -> str:
    """Extract lowercased hostname from a URL, stripping leading `www.`.

    Returns empty string on any parse failure (URLs are permissive by design;
    fetchers call this in hot loops and benefit from a never-raise contract).
    """
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except (AttributeError, ValueError):
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


def _load_per_fd_news(path: Path, id_prefix: str, actors_from: str = "") -> list[dict]:
    """Shared loader for per-FD news files produced by
      - fetch_earnings_news.py         (ticker in `ticker` field)
      - fetch_forecastbench_news.py    (question-keyword retrieval)
      - fetch_gdelt_cameo_news.py      (actor-pair retrieval)
    All three use the same record schema: {id,url,title,text,date,source,provenance,...}
    plus optional domain-specific fields we pass through as `actors`.
    """
    if not path.exists():
        return []
    out = []
    n_bad = 0
    for line in path.open(encoding="utf-8"):
        try:
            r = _j_loads(line)
        except (ValueError, UnicodeDecodeError):
            n_bad += 1
            continue
        url = r.get("url", "")
        if not url:
            continue
        title = r.get("title", "") or ""
        text = r.get("text", "") or ""
        title_text = (title + "\n" + text).strip()
        prov = r.get("provenance") or id_prefix
        actors_val = r.get(actors_from, "") if actors_from else ""
        actors = [actors_val] if isinstance(actors_val, str) and actors_val else (actors_val if isinstance(actors_val, list) else [])
        out.append({
            "id":            art_id(url),
            "url":           url,
            "title":         title,
            "text":          text,
            "title_text":    title_text,
            "publish_date":  r.get("date", "") or "",
            "source_domain": r.get("source", "") or domain_of(url),
            "gdelt_themes":  [],
            "gdelt_tone":    0.0,
            "actors":        actors,
            "cameo_code":    "",
            "char_count":    len(title_text),
            "provenance":    [prov],
        })
    return out


def load_earnings_news() -> list[dict]:
    """Load earnings-specific news (Finnhub + Google News + per-ticker GDELT)."""
    if not EARNINGS_NEWS.exists():
        return []
    out = []
    for line in EARNINGS_NEWS.open(encoding="utf-8"):
        try:
            r = _j_loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        url = r.get("url", "")
        if not url:
            continue
        title = r.get("title", "") or ""
        text = r.get("text", "") or ""
        title_text = (title + "\n" + text).strip()
        prov = r.get("provenance") or "earnings-news"
        ticker = r.get("ticker", "") or ""
        out.append({
            "id":            art_id(url),
            "url":           url,
            "title":         title,
            "text":          text,
            "title_text":    title_text,
            "publish_date":  r.get("date", "") or "",
            "source_domain": r.get("source", "") or domain_of(url),
            "gdelt_themes":  [],
            "gdelt_tone":    0.0,
            "actors":        [ticker] if ticker else [],
            "cameo_code":    "",
            "char_count":    len(title_text),
            "provenance":    [prov],
        })
    return out


@_with_raised_csv_field_limit
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

    print("Loading earnings news (Finnhub + Google News + per-ticker GDELT)...")
    earn = load_earnings_news()
    print(f"  {len(earn)} records")

    print("Loading ForecastBench per-FD news (NYT + Guardian + Google + GDELT)...")
    fb_news = _load_per_fd_news(FB_NEWS, "fb-news")
    print(f"  {len(fb_news)} records")

    print("Loading GDELT-CAMEO pre-event news (replaces same-day oracle)...")
    gdc_news = _load_per_fd_news(GDC_NEWS, "gdelt-cameo-pre")
    print(f"  {len(gdc_news)} records")

    print("Merging + deduplicating by URL hash...")
    merged = dedup_merge(fb, fb_supp, m24, earn, fb_news, gdc_news)

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
        "input_counts": {"forecastbench": len(fb), "gdelt-cameo": len(m24), "earnings-news": len(earn)},
        "output_path": str(OUT_ART),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Wrote {len(merged)} unique articles to {OUT_ART}")
    print(f"  Full text: {with_text} ({meta['with_full_text_pct']}%)")
    print(f"  By provenance: {meta['by_provenance']}")


if __name__ == "__main__":
    main()
