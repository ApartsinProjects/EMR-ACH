"""src.retrieval.contract: per-benchmark source cascade contract (v2.2 [B1]).

Single source of truth for "which source feeds which benchmark, in what
order, with what filters". Until now this knowledge was scattered across
three fetcher scripts (fetch_forecastbench_news.py,
fetch_gdelt_cameo_news.py, fetch_earnings_news.py) and re-derived inside
compute_relevance.py and unify_articles.py. The contract module is the
single import every retrieval-side caller goes through.

Per docs/V2_2_ARCHITECTURE.md Section 4 ("Hybrid retrieval table"):

| Benchmark      | Primary       | Secondary      | Editorial filter | Notes |
|----------------|---------------|----------------|------------------|-------|
| forecastbench  | google-news   | gdelt-doc      | no               | full pool, SBERT cosine |
| gdelt_cameo    | gdelt-doc     | google-news    | yes              | aggregator blocklist applied |
| earnings       | sec-edgar     | google-news    | no               | ticker x date relational join (no SBERT) |

Pure data; no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

__all__ = [
    "Benchmark",
    "Source",
    "RetrievalMode",
    "BenchmarkContract",
    "CONTRACT",
    "contract_for",
    "all_sources_for",
    "primary_source_for",
    "requires_editorial_filter",
    "retrieval_mode_for",
]


class Benchmark(str, Enum):
    """The three first-class benchmarks shipped in v2.1 / v2.2."""

    FORECASTBENCH = "forecastbench"
    GDELT_CAMEO = "gdelt_cameo"
    EARNINGS = "earnings"


class Source(str, Enum):
    """Article-source identifiers used as the ``provenance`` tag on each
    article and as keys in fetch + cache layouts. Values are the wire
    strings used in benchmark/DATASET.md and in build_manifest.json.
    """

    GOOGLE_NEWS = "google-news"
    GDELT_DOC = "gdelt-doc"
    GDELT_DOC_EDITORIAL = "gdelt-doc-editorial"
    SEC_EDGAR = "sec-edgar"
    NYT = "nyt"
    GUARDIAN = "guardian"
    FINNHUB = "finnhub"
    YFINANCE = "yfinance"


class RetrievalMode(str, Enum):
    """How the per-benchmark router selects evidence (see A13)."""

    SBERT_COSINE = "sbert_cosine"
    TICKER_DATE_JOIN = "ticker_date_join"
    ACTOR_PAIR_PREFILTER_THEN_SBERT = "actor_pair_prefilter_then_sbert"


@dataclass(frozen=True)
class BenchmarkContract:
    """Declarative contract for a single benchmark."""

    benchmark: Benchmark
    primary: Source
    secondaries: tuple[Source, ...]
    editorial_filter: bool
    retrieval_mode: RetrievalMode
    notes: str = ""

    def all_sources(self) -> tuple[Source, ...]:
        """Primary first, then secondaries, in cascade order."""
        return (self.primary, *self.secondaries)


CONTRACT: Mapping[Benchmark, BenchmarkContract] = {
    Benchmark.FORECASTBENCH: BenchmarkContract(
        benchmark=Benchmark.FORECASTBENCH,
        primary=Source.GOOGLE_NEWS,
        secondaries=(Source.GDELT_DOC,),
        editorial_filter=False,
        retrieval_mode=RetrievalMode.SBERT_COSINE,
        notes="Full-pool SBERT cosine; no aggregator filter.",
    ),
    Benchmark.GDELT_CAMEO: BenchmarkContract(
        benchmark=Benchmark.GDELT_CAMEO,
        primary=Source.GDELT_DOC,
        secondaries=(Source.GOOGLE_NEWS,),
        editorial_filter=True,
        retrieval_mode=RetrievalMode.ACTOR_PAIR_PREFILTER_THEN_SBERT,
        notes="Editorial-only filter via gdelt_aggregator_domains.py (B8).",
    ),
    Benchmark.EARNINGS: BenchmarkContract(
        benchmark=Benchmark.EARNINGS,
        primary=Source.SEC_EDGAR,
        secondaries=(Source.GOOGLE_NEWS,),
        editorial_filter=False,
        retrieval_mode=RetrievalMode.TICKER_DATE_JOIN,
        notes="Relational join via link_earnings_articles.py (ac0b031); no SBERT.",
    ),
}


def _coerce(b: Benchmark | str) -> Benchmark:
    if isinstance(b, Benchmark):
        return b
    return Benchmark(b)


def contract_for(benchmark: Benchmark | str) -> BenchmarkContract:
    """Look up the contract for a benchmark; accepts string or enum."""
    return CONTRACT[_coerce(benchmark)]


def all_sources_for(benchmark: Benchmark | str) -> tuple[Source, ...]:
    """Cascade-ordered tuple of sources for a benchmark."""
    return contract_for(benchmark).all_sources()


def primary_source_for(benchmark: Benchmark | str) -> Source:
    """The first-priority source for a benchmark."""
    return contract_for(benchmark).primary


def requires_editorial_filter(benchmark: Benchmark | str) -> bool:
    """Whether the editorial-only domain filter applies."""
    return contract_for(benchmark).editorial_filter


def retrieval_mode_for(benchmark: Benchmark | str) -> RetrievalMode:
    """Which retrieval algorithm the per-benchmark router should use (A13)."""
    return contract_for(benchmark).retrieval_mode
