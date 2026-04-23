"""src.common.gdelt_aggregator_domains: blocklist of GDELT-affiliated and
news-aggregator domains for the GDELT-CAMEO editorial-only filter
(v2.2 [B8] + [C6]).

Used by the editorial-only filter on the GDELT-CAMEO bulk-retrieval
path (see B1 contract: ``editorial_filter=True`` for that benchmark).
The blocklist is conservative: included domains are those the audit
flagged as either (a) self-syndicating GDELT mirrors that re-host the
same wire copy, (b) topic-agnostic aggregators that dilute the
editorial signal, or (c) state-syndication outlets known to inflate
counts on geopolitics queries.

This is a pure-data module. Callers use :func:`is_aggregator_domain`
or :func:`filter_articles` and never touch the underlying set
directly. The blocklist revision string is exported so it can be
recorded in stage_meta for the cache-key contract (B5 / G1).

Add a domain by appending to ``_AGGREGATOR_DOMAINS`` and bumping
``BLOCKLIST_REVISION``. Do not remove domains without an audit-trail
entry.
"""

from __future__ import annotations

from typing import Iterable
from urllib.parse import urlparse

__all__ = [
    "BLOCKLIST_REVISION",
    "is_aggregator_domain",
    "domain_of",
    "filter_articles",
    "all_blocked_domains",
]

# Bump this string whenever the set below is mutated. Stage_meta uses it
# as a cache invalidation trigger for the GDELT-CAMEO fetch slice.
BLOCKLIST_REVISION = "r1.2026-04-23"

# Conservative; the audit on 2026-04-22 flagged these as syndication or
# pure-aggregator surfaces. Kept lower-case for case-insensitive match.
_AGGREGATOR_DOMAINS: frozenset[str] = frozenset(
    {
        # GDELT and Internet Archive mirrors
        "gdeltproject.org",
        "data.gdeltproject.org",
        "archive.org",
        "web.archive.org",
        # Generic news aggregators
        "news.google.com",
        "news.yahoo.com",
        "news.bing.com",
        "feeds.feedburner.com",
        "rss.cnn.com",
        "smartnews.com",
        "flipboard.com",
        # State-syndication outlets called out in the v3 ETD source
        # blocklist (commit 28f56fd) as inflating geopolitics counts.
        "news.fjsen.com",
        "world.people.com.cn",
        "english.cri.cn",
        "xinhuanet.com",
        # Press-release wires (carry editorial weight at zero diversity)
        "prnewswire.com",
        "businesswire.com",
        "globenewswire.com",
        "newswire.ca",
        "presswire.com",
    }
)


def all_blocked_domains() -> frozenset[str]:
    """Return the immutable set of blocked domains."""
    return _AGGREGATOR_DOMAINS


def domain_of(url: str) -> str:
    """Return the lowercased ``hostname`` portion of ``url`` with a
    leading ``www.`` stripped. Empty string if the URL is malformed.
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


def is_aggregator_domain(url_or_domain: str) -> bool:
    """``True`` if the URL (or bare domain) belongs to a blocked
    aggregator. Match is exact on the registered domain or any subdomain
    suffix (e.g. ``foo.archive.org`` blocks because ``archive.org`` does).
    """
    if not url_or_domain:
        return False
    candidate = url_or_domain.lower()
    if "://" in candidate:
        candidate = domain_of(candidate)
    if candidate.startswith("www."):
        candidate = candidate[4:]
    if candidate in _AGGREGATOR_DOMAINS:
        return True
    for blocked in _AGGREGATOR_DOMAINS:
        # Subdomain suffix match (e.g. data.gdeltproject.org -> gdeltproject.org).
        if candidate.endswith("." + blocked):
            return True
    return False


def filter_articles(
    articles: Iterable[dict],
    *,
    url_field: str = "url",
    domain_field: str | None = "domain",
) -> list[dict]:
    """Return the subset of ``articles`` that are NOT aggregator-domain
    hits. Checks ``domain_field`` first if present (cheaper); falls
    through to parsing ``url_field``.
    """
    out: list[dict] = []
    for art in articles:
        if domain_field and art.get(domain_field):
            if is_aggregator_domain(art[domain_field]):
                continue
        elif art.get(url_field) and is_aggregator_domain(art[url_field]):
            continue
        out.append(art)
    return out
