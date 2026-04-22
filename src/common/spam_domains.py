"""Shared SEO-spam domain blocklist used by every news fetcher.

These domains auto-generate boilerplate stock / political summary content from
scraped ticker data + templating. They are blocked at fetch time across all
sources (GDELT, Google News, NYT, Guardian, Finnhub — though Finnhub's editorial
gatekeeping keeps them out anyway).

If you add a domain here, add it in lowercase (without `www.`) and include a
one-line comment explaining why. The list grows over time as we find new
offenders in build audits.

Update cadence: manual, on observation. No programmatic source.

Callers should use `is_spam_url(url)` rather than importing the set directly,
since the helper handles subdomain matching (e.g. `finance.themarketsdaily.com`
should resolve to spam via `themarketsdaily.com`).
"""
from __future__ import annotations

from urllib.parse import urlparse


# Alphabetized, lowercase, no `www.`. Comment each entry.
SPAM_DOMAINS: frozenset[str] = frozenset({
    # Auto-generated stock-summary SEO networks
    "americanbankingnews.com",     # ticker-templating spam farm
    "baseballdailydigest.com",     # non-baseball content auto-templated from tickers
    "communityfinancialnews.com",  # ticker-templating spam farm
    "dailypolitical.com",          # ticker-templating spam farm (political framing)
    "defenseworld.net",            # ticker-templating spam farm
    "etfdailynews.com",            # ticker-templating spam farm
    "financialcontent.com",        # hosted-press-release aggregator (low-signal)
    "markets.financialcontent.com",
    "insidermonkey.com",           # ticker-templating spam farm
    "marketbeat.com",              # heavy SEO spam on tickers
    "marketsdaily.com",            # variant of themarketsdaily.com
    "modernreaders.com",           # ticker-templating spam farm
    "nasdaqchronicle.com",         # spam farm
    "nyse-post.com",               # spam farm
    "pressoracle.com",             # press-release aggregator
    "rivertonroll.com",            # ticker-templating spam farm
    "scoopsquare24.com",           # spam farm
    "sportsperspectives.com",      # sports-framed ticker spam
    "stocknewscrier.com",          # spam farm
    "stocknewsgazette.com",        # spam farm
    "stocknewstimes.com",          # spam farm
    "streetregister.com",          # spam farm
    "thecerbatgem.com",            # spam farm
    "thelincolnianonline.com",     # spam farm
    "themarketsdaily.com",         # spam farm
    "tickerreport.com",            # ticker-templating spam farm (heavy)
    "watchlistnews.com",           # spam farm
    "wkrb13.com",                  # spam farm
    "zerkshire.com",               # spam farm
    "zolmax.com",                  # spam farm
})


def domain_of(url: str) -> str:
    """Extract hostname from a URL, lowercased and with `www.` stripped.

    Returns empty string on parse failure rather than raising — fetchers call
    this in hot loops and benefit from a non-throwing contract.
    """
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def is_spam_url(url: str) -> bool:
    """True if the URL belongs to any blocklisted domain or a subdomain of one.

    Example:
        >>> is_spam_url("https://themarketsdaily.com/2026/...")
        True
        >>> is_spam_url("https://finance.themarketsdaily.com/...")
        True
        >>> is_spam_url("https://reuters.com/...")
        False
    """
    d = domain_of(url)
    if not d:
        return False
    if d in SPAM_DOMAINS:
        return True
    return any(d.endswith("." + root) for root in SPAM_DOMAINS)
