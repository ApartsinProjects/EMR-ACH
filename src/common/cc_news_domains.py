"""Whitelist of editorial news domains for the CC-News bulk pipeline.

This list is used by ``scripts/fetch_cc_news_archive.py`` to drop articles
whose host is NOT in the whitelist during WARC streaming. The goal is to
trade coverage for tractability: CC-News includes millions of records per
monthly shard set, but less than 1% come from outlets we care about for
S&P 500 corporate news or geopolitics editorial coverage.

The list is intentionally conservative; add domains via PR and re-run the
fetcher with ``--force`` if an outlet is missing. See
``docs/CC_NEWS_PIPELINE.md`` for the rationale and sibling GDELT DOC
pipeline at ``docs/V2_2_ARCHITECTURE.md`` Section 3.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


# Baseline whitelist: roughly 50 outlets covering US, UK, and global
# English-language editorial news. Grouped for readability; order does
# not matter.
_BASELINE: tuple[str, ...] = (
    # US general / geopolitics
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "latimes.com",
    "bostonglobe.com",
    "chicagotribune.com",
    "usatoday.com",
    "npr.org",
    "pbs.org",
    "time.com",
    "theatlantic.com",
    "newyorker.com",
    "politico.com",
    "axios.com",
    "thehill.com",
    # US business and markets
    "bloomberg.com",
    "cnbc.com",
    "marketwatch.com",
    "fortune.com",
    "businessinsider.com",
    "forbes.com",
    "barrons.com",
    "seekingalpha.com",
    "investors.com",
    "thestreet.com",
    "fool.com",
    # Wires
    "reuters.com",
    "apnews.com",
    "afp.com",
    "bloombergquint.com",
    # UK / Commonwealth
    "bbc.com",
    "bbc.co.uk",
    "theguardian.com",
    "ft.com",
    "economist.com",
    "telegraph.co.uk",
    "independent.co.uk",
    "thetimes.co.uk",
    "cbc.ca",
    "abc.net.au",
    # Pan-regional / global
    "aljazeera.com",
    "dw.com",
    "france24.com",
    "rfi.fr",
    "scmp.com",
    "japantimes.co.jp",
    "straitstimes.com",
    "hindustantimes.com",
    "timesofindia.indiatimes.com",
    "thehindu.com",
    # Tech / business adjacencies (light coverage)
    "techcrunch.com",
    "theverge.com",
    "arstechnica.com",
    "wired.com",
)


def load_whitelist(extra_file: Path | None = None) -> frozenset[str]:
    """Return the effective editorial whitelist.

    Parameters
    ----------
    extra_file:
        Optional path to a text file with one additional domain per line.
        Blank lines and ``#`` comments are ignored. Domains are lowercased
        and ``www.`` stripped.
    """
    domains: set[str] = {d.lower() for d in _BASELINE}
    if extra_file is not None and Path(extra_file).exists():
        for raw in Path(extra_file).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            domains.add(_normalize_host(line))
    return frozenset(domains)


def _normalize_host(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def host_in_whitelist(host: str, whitelist: Iterable[str]) -> bool:
    """Return True if ``host`` or any of its parent domains is in the set.

    Example: ``edition.cnn.com`` is accepted if ``cnn.com`` is whitelisted.
    """
    if not host:
        return False
    host = _normalize_host(host)
    wl = set(whitelist)
    if host in wl:
        return True
    parts = host.split(".")
    for i in range(1, len(parts) - 1):
        if ".".join(parts[i:]) in wl:
            return True
    return False


__all__ = ["load_whitelist", "host_in_whitelist"]
