"""B10: hybrid baseline that combines ETD facts + article snippets (v2.2 [F1]).

Pick-only baseline that consumes BOTH the production-filtered ETD facts
and the original article snippets in a single prompt. Sits next to B3
(articles-only, current production) and the future B10b (facts-only)
in the ablation triangle described in docs/V2_2_REFACTOR_BACKLOG.md
Section F.

Evidence block contract::

    Evidence (atomic facts, dated):
    [F1] 2026-03-01  (high) Pakistan FM met Afghan ambassador. [actors: Pakistan, Afghanistan]
    [F2] 2026-03-02  (high) Casualty count rose to 47 per Reuters wire. [actors: Pakistan, Afghanistan]
    ... up to top-K facts (default K=20) sorted by date

    Source articles (truncated):
    [A1] 2026-03-01 nytimes.com -- "Diplomatic meeting between..." [first 400 chars]
    [A2] 2026-03-02 reuters.com -- "Casualty count rises..." [first 400 chars]
    ... up to top-N=10 articles

Facts must be supplied per FD via ``fd["facts"]`` (a list of dicts with
``date``, ``confidence``, ``text``, optional ``actors``); the post-publish
orchestrator (``scripts/etd_post_publish.py``) populates this field from
``data/etd/facts.v1_production_{cutoff}.jsonl`` filtered by
``linked_fd_ids``. When ``fd["facts"]`` is absent or empty the baseline
degrades gracefully to articles-only behavior so it can run on legacy
benchmark snapshots that predate ETD post-publish.
"""

from __future__ import annotations

from .._shim import BatchResult
from ..base import Baseline
from ..prompts import INSTRUCTIONS_RAG, build_articles_block, render_user

DEFAULT_MAX_FACTS = 20
DEFAULT_MAX_FACT_CHARS = 220


def _format_fact(idx: int, fact: dict, max_chars: int) -> str:
    date = (fact.get("date") or "").strip()[:10] or "unknown"
    confidence = (fact.get("confidence") or "").strip().lower() or "med"
    text = (fact.get("text") or "").strip().replace("\n", " ")
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "..."
    actors = fact.get("actors") or []
    actors_str = ""
    if isinstance(actors, list) and actors:
        actors_str = f" [actors: {', '.join(str(a) for a in actors)}]"
    return f"[F{idx+1}] {date}  ({confidence}) {text}{actors_str}"


def build_facts_block(
    facts: list[dict],
    *,
    max_facts: int = DEFAULT_MAX_FACTS,
    max_fact_chars: int = DEFAULT_MAX_FACT_CHARS,
) -> str:
    if not facts:
        return "(no atomic facts available)"
    sortable = [f for f in facts if isinstance(f, dict)]
    sortable.sort(key=lambda f: (f.get("date") or ""))
    sortable = sortable[-max_facts:]  # keep the most-recent K
    lines = [_format_fact(i, f, max_fact_chars) for i, f in enumerate(sortable)]
    return "\n".join(lines)


def build_hybrid_evidence_block(
    article_ids: list[str],
    articles: dict[str, dict],
    facts: list[dict] | None,
    *,
    max_articles: int = 10,
    max_chars_each: int = 400,
    max_facts: int = DEFAULT_MAX_FACTS,
    max_fact_chars: int = DEFAULT_MAX_FACT_CHARS,
) -> str:
    """Render the B10 evidence block: facts first, then truncated articles.

    Article snippets are intentionally shorter than B3's default (400 vs
    900 chars) because the facts block carries the load-bearing causal
    cues; articles supply contextual subtext.
    """
    facts_section = build_facts_block(
        facts or [], max_facts=max_facts, max_fact_chars=max_fact_chars
    )
    articles_section = build_articles_block(
        article_ids, articles,
        max_articles=max_articles, max_chars_each=max_chars_each,
    )
    return (
        "Evidence (atomic facts, dated):\n"
        f"{facts_section}\n\n"
        "Source articles (truncated):\n"
        f"{articles_section}"
    )


class B10HybridFactsArticles(Baseline):
    """Hybrid pick-only baseline: facts + article snippets."""

    name = "b10_hybrid_facts_articles"

    @property
    def max_facts(self) -> int:
        return int(self.cfg.get("max_facts", DEFAULT_MAX_FACTS))

    @property
    def max_fact_chars(self) -> int:
        return int(self.cfg.get("max_fact_chars", DEFAULT_MAX_FACT_CHARS))

    @property
    def article_chars(self) -> int:
        # Tighter default than B3: facts carry the causal load.
        return int(self.cfg.get("max_chars_each", 400))

    def evidence_block(self, fd: dict, articles: dict[str, dict]) -> str:
        return build_hybrid_evidence_block(
            fd.get("article_ids", []),
            articles,
            fd.get("facts"),
            max_articles=self.max_articles,
            max_chars_each=self.article_chars,
            max_facts=self.max_facts,
            max_fact_chars=self.max_fact_chars,
        )

    def build_requests(self, fds, articles):
        reqs = []
        for fd in fds:
            evidence = self.evidence_block(fd, articles)
            user = render_user(
                fd, evidence, instructions=INSTRUCTIONS_RAG
            )
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
