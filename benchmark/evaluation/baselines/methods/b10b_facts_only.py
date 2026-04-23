"""B10b: facts-only RAG baseline (sanity-check ablation, v2.2 [F2]).

Same evidence schema as B10 (F1) minus the article snippets. Provides
the bottom of the ablation triangle:

  * B3   articles only (production today)
  * B10  facts + articles (hybrid; F1)
  * B10b facts only (this file; F2)

Lets the paper decompose the contribution of (a) raw text and (b)
structured atoms separately. Degrades gracefully to a "no evidence
available" prompt when ``fd['facts']`` is absent so the baseline runs
on legacy snapshots that predate ETD post-publish.
"""

from __future__ import annotations

from .._shim import BatchResult
from ..base import Baseline
from ..prompts import INSTRUCTIONS_RAG, render_user
from .b10_hybrid_facts_articles import (
    DEFAULT_MAX_FACTS,
    DEFAULT_MAX_FACT_CHARS,
    build_facts_block,
)


def build_facts_only_evidence_block(
    facts: list[dict] | None,
    *,
    max_facts: int = DEFAULT_MAX_FACTS,
    max_fact_chars: int = DEFAULT_MAX_FACT_CHARS,
) -> str:
    return (
        "Evidence (atomic facts, dated):\n"
        f"{build_facts_block(facts or [], max_facts=max_facts, max_fact_chars=max_fact_chars)}"
    )


class B10bFactsOnly(Baseline):
    """Facts-only pick-only baseline."""

    name = "b10b_facts_only"

    @property
    def max_facts(self) -> int:
        return int(self.cfg.get("max_facts", DEFAULT_MAX_FACTS))

    @property
    def max_fact_chars(self) -> int:
        return int(self.cfg.get("max_fact_chars", DEFAULT_MAX_FACT_CHARS))

    def evidence_block(self, fd: dict) -> str:
        return build_facts_only_evidence_block(
            fd.get("facts"),
            max_facts=self.max_facts,
            max_fact_chars=self.max_fact_chars,
        )

    def build_requests(self, fds, articles):
        reqs = []
        for fd in fds:
            evidence = self.evidence_block(fd)
            user = render_user(fd, evidence, instructions=INSTRUCTIONS_RAG)
            reqs.append(self.make_request(
                custom_id=f"{fd['id']}::{self.name}", user=user
            ))
        return reqs

    def parse_responses(self, results, fds):
        preds = []
        for fd in fds:
            res = results.get(f"{fd['id']}::{self.name}")
            content = res.content if isinstance(res, BatchResult) else ""
            pick = self.parse_pick(content, fd["hypothesis_set"])
            preds.append(self.prediction_row(fd, pick))
        return preds
