"""
Multi-strategy article text fetcher — the single fetching library used by
`fetch_article_text.py`, `fetch_gdelt_text.py`, and `retry_article_text.py`.

Cascade (stops at first success returning >=MIN_CHARS).
Ordered cheap→expensive so we pay the least work to reach a result:
  1. trafilatura + readability + BeautifulSoup on live URL          (~1-2s)
  2. Wayback Machine snapshot                                        (~1-3s, fixes 404/soft bot-walls)
  3. archive.today / archive.ph snapshot                             (~1-2s, different corpus than Wayback)
  4. Playwright headless Chromium re-render                         (~7-10s, JS-rendered / Cloudflare)
  5. Common Crawl CDX + WARC range request                           (~2-5s, historical deep archive)

Every layer is wrapped in try/except; missing dependencies or endpoint outages
degrade gracefully to the next layer.

Public API:
    fetch_text(url, opts=None) -> str   # returns extracted body text or ""
    FetchOptions(enable_playwright=True, enable_archive_today=True,
                 enable_common_crawl=True, timeout=30, min_chars=200,
                 workers_playwright=3, user_agent=None) -> options dataclass

Optional deps (all free):
    pip install playwright && playwright install chromium
    pip install warcio

Module can be used from shell for ad-hoc test:
    python scripts/fetch_text_multi.py https://example.com/article
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote, urlparse

import requests

# Always-available fallback libs
try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from readability import Document as _ReadabilityDoc
except ImportError:
    _ReadabilityDoc = None

try:
    import chardet
except ImportError:
    chardet = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# Optional dep: Playwright (heavier install)
_PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    sync_playwright = None

# Optional dep: warcio (for Common Crawl range parsing)
_WARCIO_AVAILABLE = False
try:
    from warcio.archiveiterator import ArchiveIterator
    _WARCIO_AVAILABLE = True
except ImportError:
    ArchiveIterator = None


UA_CHROME = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
UA_FIREFOX = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) "
              "Gecko/20100101 Firefox/123.0")

BROWSER_HEADERS = {
    "Accept":               "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language":      "en-US,en;q=0.9",
    "Accept-Encoding":      "gzip, deflate, br",
    "DNT":                  "1",
    "Connection":           "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


@dataclass
class FetchOptions:
    timeout: int = 30
    min_chars: int = 200
    enable_playwright:     bool = True
    enable_wayback:        bool = True
    enable_archive_today:  bool = True
    enable_common_crawl:   bool = True
    playwright_wait_ms:    int = 4000     # extra wait after domcontentloaded
    user_agent:            Optional[str] = None


# ────────────────────────────── extraction-quality validator ──────────────────────────────
# Strings that indicate the extracted text is actually a paywall, bot-wall,
# error page, or JS-shell — NOT a real article.
_BLOCKER_PHRASES = [
    ("cloudflare",                    "cf_challenge"),
    ("please enable javascript",      "js_required"),
    ("enable javascript to continue", "js_required"),
    ("access denied",                 "access_denied"),
    ("403 forbidden",                 "http_403"),
    ("404 not found",                 "http_404"),
    ("this page isn't available",     "http_404"),
    ("page not found",                "http_404"),
    ("verify you are human",          "bot_check"),
    ("checking your browser",         "cf_wait"),
    ("just a moment",                 "cf_wait"),
    ("subscribe to continue reading", "paywall"),
    ("you have reached your article", "paywall"),
    ("sign in to continue reading",   "paywall"),
    ("subscribe now to read",         "paywall"),
    ("this content is available to",  "paywall"),
    ("your browser is unsupported",   "browser_unsupported"),
]


def validate_extraction(text: str,
                        min_chars: int = 400,
                        min_words: int = 50,
                        min_sentences: int = 3,
                        min_avg_sentence_words: float = 4.0,
                        min_unique_word_ratio: float = 0.15,
                        max_nav_line_ratio: float = 0.6) -> tuple[bool, str]:
    """Return (is_valid, reason). Rejects empty/stubby/boilerplate text that
    passed a fetch's length threshold but is actually useless for forecasting."""
    if not text:
        return False, "empty"
    n = len(text)
    if n < min_chars:
        return False, f"too_short({n}<{min_chars})"

    # 1. Blocker-phrase check (paywall/bot-wall/error page marker in first 1k chars)
    head = text[:1000].lower()
    for phrase, tag in _BLOCKER_PHRASES:
        if phrase in head:
            return False, f"blocker:{tag}"

    # 2. Word-count floor
    words = text.split()
    nw = len(words)
    if nw < min_words:
        return False, f"too_few_words({nw}<{min_words})"

    # 3. Unique-word diversity — repetitive menus have ~5% unique
    unique = len({w.lower() for w in words})
    diversity = unique / nw
    if diversity < min_unique_word_ratio:
        return False, f"low_diversity({diversity:.2f}<{min_unique_word_ratio})"

    # 4. Sentence count — real articles have periods, !, ?
    sent_endings = sum(1 for c in text if c in ".!?")
    if sent_endings < min_sentences:
        return False, f"too_few_sentences({sent_endings}<{min_sentences})"

    # 5. Avg sentence length — nav labels are 1-2 words
    avg_sw = nw / max(sent_endings, 1)
    if avg_sw < min_avg_sentence_words:
        return False, f"avg_sentence_too_short({avg_sw:.1f})"

    # 6. Nav-line ratio — if >60% of lines are 1-2 words each, it's a menu dump
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        nav_lines = sum(1 for l in lines if len(l.split()) <= 2)
        nav_ratio = nav_lines / len(lines)
        if nav_ratio > max_nav_line_ratio:
            return False, f"nav_dump({nav_ratio:.2f}>{max_nav_line_ratio})"

    return True, "ok"


# ────────────────────────────── helpers ──────────────────────────────
def _decode_body(resp: requests.Response) -> str:
    raw = resp.content
    enc = (resp.encoding or "").lower() if resp.encoding else ""
    if (not enc or enc == "iso-8859-1") and chardet is not None:
        detected = chardet.detect(raw[:50000])
        enc = detected.get("encoding") or "utf-8"
    try:
        return raw.decode(enc or "utf-8", errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _extract_with_fallbacks(html: str, min_chars: int) -> str:
    """trafilatura → readability → BeautifulSoup, stop at first >= min_chars."""
    if trafilatura is not None:
        try:
            text = trafilatura.extract(html, include_comments=False,
                                       include_tables=False, no_fallback=False) or ""
            if len(text) >= min_chars:
                return text
        except Exception:
            pass
    if _ReadabilityDoc is not None and BeautifulSoup is not None:
        try:
            doc = _ReadabilityDoc(html)
            soup = BeautifulSoup(doc.summary(), "html.parser")
            text2 = soup.get_text("\n", strip=True)
            if len(text2) >= min_chars:
                return text2
        except Exception:
            pass
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer",
                             "aside", "form", "noscript"]):
                tag.decompose()
            target = soup.find("article") or soup.find("main") or soup.body or soup
            text3 = target.get_text("\n", strip=True)
            if len(text3) >= min_chars * 1.5:
                return text3
        except Exception:
            pass
    return ""


# ────────────────────────────── layer 1: plain HTTP ──────────────────────────────
def _fetch_plain(url: str, opts: FetchOptions) -> str:
    session = requests.Session()
    uas = [opts.user_agent] if opts.user_agent else [UA_CHROME, UA_FIREFOX]
    random.shuffle(uas)
    for ua in uas:
        try:
            resp = session.get(url, headers={**BROWSER_HEADERS, "User-Agent": ua},
                               timeout=opts.timeout, allow_redirects=True)
            ct = (resp.headers.get("content-type", "") or "").lower()
            if ct.startswith(("image/", "video/", "audio/", "application/zip",
                              "application/octet")):
                return ""
            if resp.status_code in (429, 503):
                time.sleep(5); continue
            if resp.status_code >= 400:
                continue
            html = _decode_body(resp)
            text = _extract_with_fallbacks(html, opts.min_chars)
            if text:
                return text
        except Exception:
            continue
    return ""


# ────────────────────────────── layer 2: Playwright ──────────────────────────────
_PLAYWRIGHT_BROWSER = None  # reused across calls within one process


def _get_playwright_browser():
    global _PLAYWRIGHT_BROWSER
    if _PLAYWRIGHT_BROWSER is None and _PLAYWRIGHT_AVAILABLE:
        try:
            pw = sync_playwright().start()
            _PLAYWRIGHT_BROWSER = pw.chromium.launch(headless=True, args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ])
        except Exception:
            _PLAYWRIGHT_BROWSER = None
    return _PLAYWRIGHT_BROWSER


def _fetch_playwright(url: str, opts: FetchOptions) -> str:
    if not (opts.enable_playwright and _PLAYWRIGHT_AVAILABLE):
        return ""
    browser = _get_playwright_browser()
    if browser is None:
        return ""
    try:
        context = browser.new_context(user_agent=opts.user_agent or UA_CHROME,
                                      viewport={"width": 1280, "height": 900},
                                      locale="en-US")
        page = context.new_page()
        page.set_default_timeout(opts.timeout * 1000)
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(opts.playwright_wait_ms)
        html = page.content()
        context.close()
        return _extract_with_fallbacks(html, opts.min_chars)
    except Exception:
        try: context.close()
        except Exception: pass
        return ""


# ────────────────────────────── layer 3: Wayback ──────────────────────────────
def _fetch_wayback(url: str, opts: FetchOptions) -> str:
    if not opts.enable_wayback:
        return ""
    try:
        api = f"http://archive.org/wayback/available?url={quote(url, safe='')}"
        meta = requests.get(api, timeout=15).json()
        snap = (meta.get("archived_snapshots", {}) or {}).get("closest", {})
        if not snap.get("available"):
            return ""
        archived_url = snap["url"]
        # Request the raw-content variant (no Wayback toolbar)
        if "/web/" in archived_url and "id_" not in archived_url:
            parts = archived_url.split("/web/", 1)
            ts, real = parts[1].split("/", 1)
            archived_url = f"{parts[0]}/web/{ts}id_/{real}"
        resp = requests.get(archived_url,
                            headers={"User-Agent": opts.user_agent or UA_CHROME},
                            timeout=opts.timeout, allow_redirects=True)
        if resp.status_code != 200:
            return ""
        return _extract_with_fallbacks(_decode_body(resp), opts.min_chars)
    except Exception:
        return ""


# ────────────────────────────── layer 4: archive.today ──────────────────────────────
_ARCHIVE_TODAY_RATE_LIMITER: dict[str, float] = {"last": 0.0}


def _fetch_archive_today(url: str, opts: FetchOptions) -> str:
    if not opts.enable_archive_today:
        return ""
    try:
        # Rate-limit: 1 req/sec globally per process
        now = time.time()
        delta = now - _ARCHIVE_TODAY_RATE_LIMITER["last"]
        if delta < 1.0:
            time.sleep(1.0 - delta)
        _ARCHIVE_TODAY_RATE_LIMITER["last"] = time.time()

        # Look up latest snapshot. archive.ph redirects /newest/{url} -> snapshot.
        lookup = f"https://archive.ph/newest/{url}"
        resp = requests.get(lookup,
                            headers={"User-Agent": opts.user_agent or UA_CHROME,
                                     **BROWSER_HEADERS},
                            timeout=opts.timeout, allow_redirects=True)
        if resp.status_code != 200:
            return ""
        # archive.ph wraps the page in their template; still extract the body.
        return _extract_with_fallbacks(_decode_body(resp), opts.min_chars)
    except Exception:
        return ""


# ────────────────────────────── layer 5: Common Crawl ──────────────────────────────
# CC index is published ~monthly; querying all is slow. We hit the 3 most
# recent indices (covers past ~90 days), fetched dynamically from the CC
# collinfo.json manifest so we don't go stale as new crawls land.
_CC_INDEX_CACHE = None


def _get_cc_indices() -> list[str]:
    global _CC_INDEX_CACHE
    if _CC_INDEX_CACHE is not None:
        return _CC_INDEX_CACHE
    try:
        info = requests.get("https://index.commoncrawl.org/collinfo.json", timeout=10).json()
        # info is a list of dicts: [{"id": "CC-MAIN-2026-13", "name":..., ...}, ...]
        _CC_INDEX_CACHE = [x["id"] for x in info[:3]]  # 3 most recent
    except Exception:
        _CC_INDEX_CACHE = []
    return _CC_INDEX_CACHE


_CC_CDX_URL = "https://index.commoncrawl.org/{index}-index"


def _fetch_common_crawl(url: str, opts: FetchOptions) -> str:
    if not (opts.enable_common_crawl and _WARCIO_AVAILABLE):
        return ""
    import io
    for index in _get_cc_indices():
        try:
            cdx_url = _CC_CDX_URL.format(index=index)
            r = requests.get(cdx_url, params={
                "url": url, "output": "json", "limit": 1,
            }, headers={"User-Agent": opts.user_agent or UA_CHROME}, timeout=15)
            if r.status_code != 200 or not r.text.strip():
                continue
            line = r.text.strip().split("\n")[0]
            meta = json.loads(line)
            filename = meta.get("filename"); offset = int(meta.get("offset", 0))
            length = int(meta.get("length", 0))
            if not (filename and length):
                continue
            # Range-request the WARC segment
            warc_url = f"https://data.commoncrawl.org/{filename}"
            byte_range = f"bytes={offset}-{offset+length-1}"
            wr = requests.get(warc_url, headers={
                "Range": byte_range,
                "User-Agent": opts.user_agent or UA_CHROME,
            }, timeout=opts.timeout)
            if wr.status_code not in (200, 206):
                continue
            for record in ArchiveIterator(io.BytesIO(wr.content)):
                if record.rec_type == "response":
                    raw_html = record.content_stream().read().decode(
                        "utf-8", errors="replace")
                    text = _extract_with_fallbacks(raw_html, opts.min_chars)
                    if text:
                        return text
        except Exception:
            continue
    return ""


# ────────────────────────────── public API ──────────────────────────────
def fetch_text(url: str, opts: Optional[FetchOptions] = None) -> str:
    """Run the full fallback cascade. Returns extracted body text or ""."""
    opts = opts or FetchOptions()
    if not url or not url.startswith(("http://", "https://")):
        return ""

    for strategy in (
        _fetch_plain,
        _fetch_wayback,
        _fetch_archive_today,
        _fetch_playwright,
        _fetch_common_crawl,
    ):
        try:
            text = strategy(url, opts)
            if text and validate_extraction(text, min_chars=opts.min_chars)[0]:
                return text
        except Exception:
            continue
    return ""


def fetch_text_with_provenance(url: str,
                               opts: Optional[FetchOptions] = None) -> tuple[str, str]:
    """Same as fetch_text but also returns which strategy succeeded."""
    opts = opts or FetchOptions()
    if not url or not url.startswith(("http://", "https://")):
        return "", "invalid_url"
    for name, strategy in (
        ("plain",         _fetch_plain),
        ("wayback",       _fetch_wayback),
        ("archive_today", _fetch_archive_today),
        ("playwright",    _fetch_playwright),
        ("common_crawl",  _fetch_common_crawl),
    ):
        try:
            text = strategy(url, opts)
            if text and validate_extraction(text, min_chars=opts.min_chars)[0]:
                return text, name
        except Exception:
            continue
    return "", "all_failed"


def capabilities() -> dict:
    """Report which optional deps are installed."""
    return {
        "trafilatura":  trafilatura is not None,
        "readability":  _ReadabilityDoc is not None,
        "beautifulsoup": BeautifulSoup is not None,
        "chardet":      chardet is not None,
        "playwright":   _PLAYWRIGHT_AVAILABLE,
        "warcio":       _WARCIO_AVAILABLE,
    }


# ────────────────────────────── CLI for ad-hoc test ──────────────────────────────
if __name__ == "__main__":
    caps = capabilities()
    print("Capabilities:", caps, file=sys.stderr)
    if len(sys.argv) < 2:
        print("Usage: python scripts/fetch_text_multi.py <url>", file=sys.stderr)
        sys.exit(1)
    url = sys.argv[1]
    text, provenance = fetch_text_with_provenance(url)
    print(f"[provenance] {provenance}", file=sys.stderr)
    print(f"[length] {len(text)} chars", file=sys.stderr)
    print(text)
