"""src.common.sources: canonical Source enum (v2.2 [B12]).

Re-exports the :class:`src.retrieval.contract.Source` enum at a more
discoverable location for callers (fetchers, unifiers, audit scripts)
that only need the source identifiers and not the full hybrid-retrieval
contract. String literals like ``"gdelt-doc"``, ``"google-news"``,
``"finnhub"``, ``"yfinance"``, ``"sec-edgar"`` should migrate to this
enum incrementally; the enum's ``str`` values match the wire strings
used in benchmark/DATASET.md and build_manifest.json so downstream
JSON serialization is unchanged.
"""

from __future__ import annotations

from src.retrieval.contract import Source

__all__ = ["Source"]
