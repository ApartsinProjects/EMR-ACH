"""Fetch pre-event news for each GDELT-CAMEO FD across multiple sources.

Mirrors fetch_forecastbench_news.py but uses the actor country pair as the
retrieval query. Moves GDELT-CAMEO away from same-day oracle Docids toward the
unified "forecast event outcome from prior news" task shared by ForecastBench
and earnings.

For each GDELT-CAMEO FD (question about action of country_s toward country_o
on a specific event_date), pulls news published in the 30 days before the
event, using the two country names as the query.

Sources (same cascade as FB fetcher):
  1. GDELT DOC API (no key, global)
  2. Google News RSS (no key)
  3. NYT Article Search (NYT_API_KEY)
  4. The Guardian (GUARDIAN_API_KEY)

Writes:
  data/gdelt_cameo/gdelt_cameo_news.jsonl

Usage:
  python scripts/fetch_gdelt_cameo_news.py
  python scripts/fetch_gdelt_cameo_news.py --limit 10 --source gdelt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, urlunparse, quote_plus

import requests

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))
from src.common.optional_imports import optional  # noqa: E402

feedparser = optional("feedparser")   # only needed for Google News RSS source

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
OUT_DIR = DATA / "gdelt_cameo"
OUT_FILE = OUT_DIR / "gdelt_cameo_news.jsonl"

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GUARDIAN_API = "https://content.guardianapis.com/search"
NYT_API = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

TIMEOUT = 30

# Shared spam blocklist — see src/common/spam_domains.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))
from src.common.spam_domains import is_spam_url as _is_spam_url_shared  # noqa: E402


def art_id(url: str) -> str:
    return "gdc_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def is_spam_url(url: str) -> bool:
    """Delegate to the shared blocklist in src/common/spam_domains.py."""
    return _is_spam_url_shared(url)


def url_key(url: str) -> str:
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc.lower(), p.path.rstrip("/"), "", "", ""))
    except Exception:
        return url


# ---------------------------------------------------------------------------
# Sources (NYT / Guardian / Google / GDELT)
# Same shape as fetch_forecastbench_news.py — adapted for actor-pair queries.
# ---------------------------------------------------------------------------

def fetch_nyt(query: str, forecast_point: datetime, lookback: int) -> list[dict]:
    key = os.environ.get("NYT_API_KEY", "")
    if not key:
        return []
    start = forecast_point - timedelta(days=lookback)
    end = forecast_point - timedelta(days=1)
    out = []
    for page in range(2):
        try:
            r = requests.get(NYT_API, params={
                "q": query, "begin_date": start.strftime("%Y%m%d"),
                "end_date": end.strftime("%Y%m%d"),
                "api-key": key, "page": page, "sort": "relevance",
            }, timeout=TIMEOUT)
            if r.status_code == 429: time.sleep(6); continue
            if r.status_code != 200: break
            docs = r.json().get("response", {}).get("docs", [])
        except Exception as e:
            print(f"  [WARN] NYT: {e}"); break
        if not docs: break
        for d in docs:
            url = d.get("web_url") or ""
            if not url: continue
            out.append({
                "url": url, "title": d.get("headline", {}).get("main", "") or "",
                "summary": d.get("abstract", "") or d.get("lead_paragraph", "") or "",
                "date": (d.get("pub_date", "") or "")[:10],
                "publisher": "The New York Times", "provenance": "nyt",
            })
        time.sleep(6)
    return out


def fetch_guardian(query: str, forecast_point: datetime, lookback: int,
                   page_size: int = 15) -> list[dict]:
    key = os.environ.get("GUARDIAN_API_KEY", "")
    if not key:
        return []
    start = forecast_point - timedelta(days=lookback)
    end = forecast_point - timedelta(days=1)
    try:
        r = requests.get(GUARDIAN_API, params={
            "q": query, "from-date": start.strftime("%Y-%m-%d"),
            "to-date": end.strftime("%Y-%m-%d"),
            "show-fields": "trailText",
            "page-size": page_size, "order-by": "relevance", "api-key": key,
        }, timeout=TIMEOUT)
        if r.status_code != 200: return []
        results = r.json().get("response", {}).get("results", [])
    except Exception as e:
        print(f"  [WARN] Guardian: {e}"); return []
    out = []
    for a in results:
        url = a.get("webUrl", "")
        if not url: continue
        pub = a.get("webPublicationDate", "")[:10]
        fields = a.get("fields", {}) or {}
        out.append({
            "url": url, "title": a.get("webTitle", "") or "",
            "summary": fields.get("trailText", "") or "",
            "date": pub, "publisher": "The Guardian", "provenance": "guardian",
        })
    return out


def fetch_google_news(query: str, forecast_point: datetime, lookback: int,
                      max_records: int = 25) -> list[dict]:
    if not feedparser: return []  # feedparser not installed; soft-skip Google News
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    start = forecast_point - timedelta(days=lookback)
    end = forecast_point
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        print(f"  [WARN] Google News: {e}"); return []
    out = []
    for entry in getattr(feed, "entries", [])[: max_records * 3]:
        link = getattr(entry, "link", "") or ""
        if not link: continue
        pub = getattr(entry, "published_parsed", None)
        pub_dt = None
        if pub:
            try: pub_dt = datetime(*pub[:6])
            except: pass
        if pub_dt is None or not (start <= pub_dt < end): continue
        out.append({
            "url": link, "title": getattr(entry, "title", "") or "",
            "summary": getattr(entry, "summary", "") or "",
            "date": pub_dt.strftime("%Y-%m-%d"), "provenance": "google-news",
        })
        if len(out) >= max_records: break
    return out


def _gdelt_query(q: str, sd: str, ed: str, max_records: int, fd_id: str) -> list[dict]:
    for attempt in range(3):
        try:
            r = requests.get(GDELT_DOC_API, params={
                "query": q, "mode": "ArtList", "maxrecords": max_records,
                "startdatetime": sd, "enddatetime": ed, "format": "json",
                "sort": "DateDesc",
            }, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json().get("articles", [])
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
        except Exception as e:
            if attempt == 2:
                print(f"  [WARN] GDELT {fd_id} {sd}..{ed}: {e}"); return []
            time.sleep(2 * (attempt + 1))
    return []


def fetch_gdelt(query: str, forecast_point: datetime, lookback: int,
                max_records: int, n_slices: int = 3, fd_id: str = "") -> list[dict]:
    total_end = forecast_point - timedelta(days=1)
    slice_days = max(1, lookback // n_slices)
    per_slice = max(3, max_records // n_slices)
    out = []
    seen: set[str] = set()
    for i in range(n_slices):
        slice_end = total_end - timedelta(days=i * slice_days)
        slice_start = slice_end - timedelta(days=slice_days - 1)
        if slice_start < forecast_point - timedelta(days=lookback):
            slice_start = forecast_point - timedelta(days=lookback)
        sd = slice_start.strftime("%Y%m%d000000")
        ed = slice_end.strftime("%Y%m%d235959")
        articles = _gdelt_query(query, sd, ed, per_slice, fd_id)
        time.sleep(0.7)
        for a in articles:
            url = a.get("url", "")
            if not url or url in seen: continue
            seen.add(url)
            seendate = a.get("seendate", "")
            date_str = ""
            if seendate:
                try: date_str = datetime.strptime(seendate[:8], "%Y%m%d").strftime("%Y-%m-%d")
                except: pass
            out.append({
                "url": url, "title": a.get("title", "") or "",
                "summary": "", "date": date_str, "provenance": "gdelt-cameo-pre",
            })
    return out


# ---------------------------------------------------------------------------
# FD loading — need actor metadata
# ---------------------------------------------------------------------------

def load_gdelt_fds() -> list[dict]:
    """Load GDELT-CAMEO FDs with actor names from metadata."""
    # Try unified forecasts; fall back to staged
    unified = DATA / "unified" / "forecasts.jsonl"
    if unified.exists():
        src = unified
    else:
        staged = sorted((DATA / "staged").glob("*/01_after_first_unify/forecasts.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if not staged:
            raise FileNotFoundError("No FDs found")
        src = staged[0]
    print(f"[load] reading GDELT FDs from {src}")
    out = []
    for line in src.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("benchmark") != "gdelt-cameo":
            continue
        out.append(r)
    return out


def fd_query(fd: dict) -> str:
    """Build retrieval query from FD actor pair + date context.

    Expected FD.metadata structure:
      actors: ["ISR", "PAL"]
      actor_names: ["Israel", "Palestine"]          # preferred if present
    """
    md = fd.get("metadata", {}) or {}
    names = md.get("actor_names") or []
    actors = md.get("actors") or []
    pieces = []
    if len(names) >= 2 and names[0] and names[1]:
        pieces = [f'"{names[0]}"', f'"{names[1]}"']
    elif len(actors) >= 2:
        pieces = actors[:2]
    else:
        # fall back to parsing the question
        return fd.get("question", "")[:120]
    return " ".join(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",
                    choices=["all", "editorial", "gdelt", "google", "guardian", "nyt"],
                    default="editorial",
                    help="Source cascade. Default 'editorial' = Guardian + NYT + Google News "
                         "(no GDELT DOC). Under the 2026-04-22 'forecast from prior news' design, "
                         "the GDELT Event Database supplies the question/ground-truth only; "
                         "evidence MUST come from independent editorial sources to keep the "
                         "retrieval channel information-theoretically separate from the label "
                         "channel. Pass --source all to include GDELT DOC (for legacy "
                         "comparability runs).")
    ap.add_argument("--lookback", type=int, default=90,
                    help="Days before event_date to search (default 90 — unified analysis window)")
    ap.add_argument("--max-gdelt", type=int, default=18)
    ap.add_argument("--max-guardian", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-completed", action="store_true")
    ap.add_argument("--shard", default=None,
                    help="Parallel-worker sharding: 'N/K' processes FDs where "
                         "hash(fd_id) %% K == N. Use 0/4, 1/4, 2/4, 3/4 for 4 workers.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fds = load_gdelt_fds()
    if args.shard:
        try:
            n_str, k_str = args.shard.split("/")
            shard_n, shard_k = int(n_str), int(k_str)
            assert 0 <= shard_n < shard_k
        except Exception:
            print(f"[ERR] bad --shard '{args.shard}'; expected 'N/K' with 0 <= N < K")
            sys.exit(1)
        before = len(fds)
        fds = [fd for fd in fds if hash(fd["id"]) % shard_k == shard_n]
        print(f"[shard] {shard_n}/{shard_k}: kept {len(fds)} of {before} FDs")
    if args.skip_completed and OUT_FILE.exists():
        done: set[str] = set()
        for line in OUT_FILE.open(encoding="utf-8"):
            try: done.add(json.loads(line).get("fd_id", ""))
            except: pass
        before = len(fds)
        fds = [fd for fd in fds if fd["id"] not in done]
        print(f"[filter] --skip-completed dropped {before - len(fds)} FDs")
    if args.limit:
        fds = fds[:args.limit]

    print(f"Processing {len(fds)} GDELT-CAMEO FDs (sources={args.source}, lookback={args.lookback}d)")

    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    if OUT_FILE.exists():
        for line in OUT_FILE.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                seen_urls.add(rec.get("url", ""))
                seen_urls.add(url_key(rec.get("url", "")))
                t = (rec.get("title") or "").strip().lower()[:80]
                if t:
                    seen_titles.add(f"{rec.get('fd_id','')}::{t}")
            except: pass

    written = 0
    per_prov = Counter()
    with OUT_FILE.open("a", encoding="utf-8") as out:
        for i, fd in enumerate(fds):
            fd_id = fd["id"]
            fp_str = fd.get("forecast_point") or fd.get("resolution_date")
            try:
                fp_dt = datetime.strptime(fp_str, "%Y-%m-%d")
            except Exception:
                continue
            query = fd_query(fd)
            if not query.strip():
                continue
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(fds)}] {fd_id}  q={query!r}  written={written}  prov={dict(per_prov)}")

            collected = []
            # Editorial sources (default / --source editorial) — independent from
            # the GDELT label channel, so retrieval doesn't leak into ground truth.
            if args.source in ("all", "editorial", "nyt"):
                collected.extend(fetch_nyt(query, fp_dt, args.lookback))
            if args.source in ("all", "editorial", "guardian"):
                collected.extend(fetch_guardian(query, fp_dt, args.lookback, args.max_guardian))
            if args.source in ("all", "editorial", "google"):
                collected.extend(fetch_google_news(query, fp_dt, args.lookback))
            # GDELT DOC — only when --source all or --source gdelt is explicitly
            # passed. Kept for legacy ablation only; default design avoids it.
            if args.source in ("all", "gdelt"):
                collected.extend(fetch_gdelt(query, fp_dt, args.lookback, args.max_gdelt, fd_id=fd_id))

            for art in collected:
                url = art["url"]
                if not url:
                    continue
                # v2.2 leakage guard: drop anything with publish_date >
                # forecast_point. GDELT DOC + editorial sources have all
                # been observed returning future-dated records occasionally,
                # so the source-side filter is re-asserted here.
                _pd = art.get("date", "") or ""
                if _pd:
                    try:
                        _pd_dt = datetime.strptime(_pd[:10], "%Y-%m-%d")
                        if _pd_dt > fp_dt:
                            per_prov["__dropped_leakage__"] += 1
                            continue
                    except ValueError:
                        pass
                if is_spam_url(url):
                    per_prov["__dropped_spam__"] += 1; continue
                norm = url_key(url)
                if url in seen_urls or norm in seen_urls:
                    per_prov["__dedup_url__"] += 1; continue
                tk = (art.get("title") or "").strip().lower()[:80]
                fdtk = f"{fd_id}::{tk}" if tk else ""
                if fdtk and fdtk in seen_titles:
                    per_prov["__dedup_title__"] += 1; continue
                seen_urls.add(url); seen_urls.add(norm)
                if fdtk: seen_titles.add(fdtk)
                src = art.get("publisher") or domain_of(url)
                rec = {
                    "id":         art_id(url),
                    "fd_id":      fd_id,
                    "query":      query,
                    "url":        url,
                    "title":      art.get("title", ""),
                    "text":       art.get("summary", "") or "",
                    "date":       art.get("date", "") or "",
                    "source":     src,
                    "provenance": art["provenance"],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                per_prov[art["provenance"]] += 1

    print(f"\nDone. Wrote {written} new articles to {OUT_FILE.relative_to(ROOT)}")
    for p, n in sorted(per_prov.items(), key=lambda x: -x[1]):
        print(f"  {p:20s}: {n}")
    _leak = per_prov.get("__dropped_leakage__", 0)
    _total_seen = sum(per_prov.values())
    print(f"[gdelt_cameo] leakage-filtered: {_leak} of {_total_seen} articles")


if __name__ == "__main__":
    main()
