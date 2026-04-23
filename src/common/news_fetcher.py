"""src.common.news_fetcher: shared fetcher primitives (v2.2 [B2], absorbs [E1, E5, E6]).

Base class plus helpers for the three per-benchmark fetchers
(fetch_forecastbench_news, fetch_gdelt_cameo_news, fetch_earnings_news).
Today each fetcher replicates ~400 lines of shared plumbing:

* ``_sys.path.insert`` bootstrap (cleaned up by B4a)
* ``art_id`` construction (``"fbn_"``, ``"gdc_"``, ``"earn_"`` prefixes)
* ``domain_of(url)`` URL parsing
* spam blocklist check via :mod:`src.common.spam_domains`
* HEADERS / TIMEOUT constants
* append-then-dedup loop over existing JSONL
* JSONL atomic write

This module ships the shared surface. The existing fetcher scripts
remain in-flight (one is running right now per the tasking note) so
this pass does NOT rewrite them; the migration lands in a follow-up
where each fetcher becomes an ~80-line subclass.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlparse

__all__ = [
    "DEFAULT_HEADERS",
    "DEFAULT_TIMEOUT_SECONDS",
    "art_id_for",
    "domain_of",
    "NewsFetcher",
    "FetchedArticle",
]

DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
DEFAULT_TIMEOUT_SECONDS: int = 20


# Canonical prefixes per benchmark. Promoting these to a single owner
# addresses E5 (the unify layer hardcoded "art_" while fetchers used
# the benchmark-specific prefixes).
_PREFIXES = {
    "forecastbench": "fbn_",
    "gdelt_cameo": "gdc_",
    "earnings": "earn_",
}


def art_id_for(benchmark: str, url: str, publish_date: str | None = None) -> str:
    """Deterministic article id: ``{prefix}{first-10-of-md5}``.
    Matches the v2.1 format used by the three fetchers. Unknown
    benchmarks fall back to the legacy ``art_`` prefix so callers
    outside the known set are not broken.
    """
    prefix = _PREFIXES.get(benchmark, "art_")
    key = f"{url}|{publish_date or ''}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}{h}"


def domain_of(url: str) -> str:
    """Lowercased hostname with any leading ``www.`` stripped. Empty
    string on malformed URLs (fail open so callers do not crash on
    dirty data).
    """
    if not url:
        return ""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


@dataclass
class FetchedArticle:
    """Record shape written to ``data/{benchmark}/{benchmark}_articles.jsonl``."""

    article_id: str
    url: str
    title: str = ""
    text: str = ""
    publish_date: str = ""
    source_domain: str = ""
    provenance: str = ""
    linked_fd_ids: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        out = {
            "article_id": self.article_id,
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "publish_date": self.publish_date,
            "source_domain": self.source_domain,
            "provenance": self.provenance,
            "linked_fd_ids": list(self.linked_fd_ids),
        }
        if self.extras:
            out["extras"] = dict(self.extras)
        return out


def _load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line).get("article_id")))
            except json.JSONDecodeError:
                continue
    return ids


def _append_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Append JSONL records to ``path`` with atomic rename of the tail
    segment. Returns the number of records written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    # For simple append semantics we use a plain open('a'). True
    # atomicity requires a write-all-and-swap pattern; the base class
    # uses that when full rewrite is needed (see NewsFetcher.write_all).
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n


class NewsFetcher(ABC):
    """Base class for per-benchmark news fetchers.

    Subclasses override :meth:`build_queries`, :meth:`eligible_sources`,
    :attr:`benchmark`, and :meth:`fetch_for_query`. The shared plumbing
    (dedup, spam filter, JSONL write, HEADERS) lives here.
    """

    benchmark: str = "unknown"

    def __init__(
        self,
        out_path: Path,
        *,
        spam_blocklist: set[str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.out_path = out_path
        self.headers = headers or DEFAULT_HEADERS
        self.timeout = timeout
        self._spam = {d.lower() for d in (spam_blocklist or set())}
        self._seen_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Subclass surface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_queries(self, fd: dict) -> list[str]:
        """Return one or more search strings for a single FD."""

    @abstractmethod
    def eligible_sources(self) -> list[str]:
        """Return the provenance tags this fetcher emits (e.g.
        ``['google-news']`` or ``['gdelt-doc']``)."""

    @abstractmethod
    def fetch_for_query(self, query: str, fd: dict) -> Iterable[dict]:
        """Yield raw article records for a single query. Subclasses do
        the HTTP work; dedup, spam filter, and ID construction happen
        in :meth:`process_fd`.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _is_spam(self, url: str) -> bool:
        d = domain_of(url)
        if not d:
            return False
        if d in self._spam:
            return True
        # Suffix match: sub.spam.com matches spam.com.
        for bad in self._spam:
            if d.endswith("." + bad):
                return True
        return False

    def process_fd(self, fd: dict) -> list[FetchedArticle]:
        """Run the fetcher's queries for one FD and return new, deduped
        :class:`FetchedArticle` records. Already-seen article_ids (from
        the existing out_path or earlier calls this session) are
        dropped.
        """
        if not self._seen_ids:
            self._seen_ids = _load_existing_ids(self.out_path)
        queries = self.build_queries(fd)
        out: list[FetchedArticle] = []
        for q in queries:
            for raw in self.fetch_for_query(q, fd):
                url = str(raw.get("url") or "")
                if not url or self._is_spam(url):
                    continue
                art_id = art_id_for(
                    self.benchmark, url,
                    raw.get("publish_date") or "",
                )
                if art_id in self._seen_ids:
                    continue
                self._seen_ids.add(art_id)
                fa = FetchedArticle(
                    article_id=art_id,
                    url=url,
                    title=str(raw.get("title") or ""),
                    text=str(raw.get("text") or ""),
                    publish_date=str(raw.get("publish_date") or ""),
                    source_domain=domain_of(url),
                    provenance=str(raw.get("provenance") or self.eligible_sources()[0]),
                    linked_fd_ids=list(raw.get("linked_fd_ids") or [fd.get("id")]),
                    extras=dict(raw.get("extras") or {}),
                )
                out.append(fa)
        return out

    def append(self, articles: Iterable[FetchedArticle]) -> int:
        """Append articles to the out_path JSONL; returns count."""
        return _append_jsonl(
            self.out_path, (a.to_json() for a in articles)
        )

    def write_all(self, articles: Iterable[FetchedArticle]) -> int:
        """Atomically overwrite the out_path with the supplied articles.
        Writes to a tmp sibling and renames. Returns count.
        """
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path.with_suffix(self.out_path.suffix + ".tmp")
        n = 0
        with tmp.open("w", encoding="utf-8") as f:
            for a in articles:
                f.write(json.dumps(a.to_json()) + "\n")
                n += 1
        os.replace(tmp, self.out_path)
        return n
