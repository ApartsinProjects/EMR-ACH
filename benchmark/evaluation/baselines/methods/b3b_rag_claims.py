"""B3b: RAG with per-article claim extraction pre-pass (wave-3 experiment).

Two-round pipeline:
  Round 0: for each (FD, article) pair, a cheap extraction call that asks the
           LLM to pull 2-3 short, forecast-relevant claims from the article.
           Articles deemed irrelevant return an empty claim list.
  Round 1: standard B3 prompt, but the articles block contains the extracted
           claims instead of raw title+truncated-text.

Hypothesis: feeding the forecaster distilled facts rather than raw article
noise gives more signal per token and improves selection accuracy on classes
that depend on subtle evidence (e.g., Material Conflict on GDELT-CAMEO).
"""

from __future__ import annotations

from .._shim import BatchResult, parse_json_response

from ..base import Baseline
from ..prompts import INSTRUCTIONS_RAG, render_user


_EXTRACTION_SYS = (
    "You are a news-analysis assistant. Extract forecast-relevant facts only. "
    "Respond with valid JSON only, no code fences."
)


def _build_extraction_prompt(fd: dict, article: dict) -> str:
    return (
        f"Forecasting question: {fd['question']}\n"
        f"Candidate hypotheses: {', '.join(fd['hypothesis_set'])}\n\n"
        f"Article title: {article.get('title','') or '(no title)'}\n"
        f"Article date : {article.get('publish_date','') or '(unknown)'}\n"
        f"Article source: {article.get('source_domain','') or '(unknown)'}\n\n"
        f"Article text:\n{(article.get('text','') or '')[:3000]}\n\n"
        "Extract at most 3 short, forecast-relevant facts from this article. "
        "Each fact should be one sentence, specific (names/dates/numbers), and directly "
        "help distinguish between the candidate hypotheses. If the article is not "
        "relevant to the forecasting question, return an empty list.\n\n"
        'Return JSON only: {"claims": ["fact 1", "fact 2", "fact 3"]}'
    )


def _parse_claims(content: str) -> list[str]:
    if not content:
        return []
    data = parse_json_response(content)
    if not isinstance(data, dict):
        return []
    c = data.get("claims") or data.get("facts") or []
    if not isinstance(c, list):
        return []
    return [str(x).strip() for x in c if isinstance(x, (str, int, float)) and str(x).strip()]


class B3bRAGClaims(Baseline):
    name = "b3b_rag_claims"
    multi_round = True
    n_rounds = 2  # round 0 = extract, round 1 = final pick

    # ------------------------------------------------------------------
    # Round 0 — per-(FD, article) extraction
    # ------------------------------------------------------------------
    def build_requests(self, fds, articles):
        reqs = []
        for fd in fds:
            aids = (fd.get("article_ids") or [])[: self.max_articles]
            for aid in aids:
                art = articles.get(aid)
                if not art:
                    continue
                user = _build_extraction_prompt(fd, art)
                reqs.append(self.make_request(
                    custom_id=f"{fd['id']}::{self.name}::extract::{aid}",
                    user=user,
                    system=_EXTRACTION_SYS,
                    max_tokens=220,  # 3 short bullets
                ))
        return reqs

    # ------------------------------------------------------------------
    # Round 1 — final pick using extracted claims
    # ------------------------------------------------------------------
    def build_requests_round(self, r: int, fds, articles, prior: dict):
        reqs = []
        for fd in fds:
            aids = (fd.get("article_ids") or [])[: self.max_articles]
            chunks: list[str] = []
            for i, aid in enumerate(aids):
                art = articles.get(aid) or {}
                ext_id = f"{fd['id']}::{self.name}::extract::{aid}"
                res = prior.get(ext_id)
                content = res.content if isinstance(res, BatchResult) else ""
                claims = _parse_claims(content)
                if not claims:
                    continue
                title = (art.get("title") or "").strip()
                date = art.get("publish_date", "")
                source = art.get("source_domain", "")
                header = f"[Article {i+1} | {date} | {source}] {title}".rstrip()
                claim_block = "\n".join(f"  - {c}" for c in claims)
                chunks.append(f"{header}\n{claim_block}")
            articles_block = "\n\n".join(chunks) or "(no relevant claims extracted)"
            user = render_user(fd, articles_block=articles_block, instructions=INSTRUCTIONS_RAG)
            reqs.append(self.make_request(
                custom_id=f"{fd['id']}::{self.name}",
                user=user,
            ))
        return reqs

    def parse_responses(self, results, fds):
        preds = []
        for fd in fds:
            res = results.get(f"{fd['id']}::{self.name}")
            content = res.content if isinstance(res, BatchResult) else ""
            pick = self.parse_pick(content, fd["hypothesis_set"])
            # Count how many claims were actually extracted (non-empty) for audit
            aids = (fd.get("article_ids") or [])[: self.max_articles]
            n_with_claims = 0
            for aid in aids:
                ext = results.get(f"{fd['id']}::{self.name}::extract::{aid}")
                if isinstance(ext, BatchResult) and _parse_claims(ext.content):
                    n_with_claims += 1
            preds.append(self.prediction_row(
                fd, pick, extras={"n_articles_with_claims": n_with_claims},
            ))
        return preds
