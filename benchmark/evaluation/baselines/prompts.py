"""
Shared prompt templates for all baselines.

All templates accept the unified FD schema fields (hypothesis_set, hypothesis_definitions,
question, background, articles_block). v2.1 framing (2026-04-22): every FD has the
binary primary target ["Comply", "Surprise"] (the FD's outcome either matches or
breaks the prior-state status-quo expectation). The legacy domain-specific multiclass
target is preserved on the FD as `x_multiclass_*` for ablation but baselines read
the primary target via `fd["hypothesis_set"]`.

Every user prompt injects `_prior_expectation_block(fd)`, a natural-language
status-quo sentence built from `prior_state_30d` + `fd_type`, so the model is
explicitly told what the prior expectation is and asked whether the outcome will
Comply with it or Surprise against it.

Keep all prompt strings here; baseline method files import them.
"""

from __future__ import annotations


SYSTEM_FORECASTER = (
    "You are an expert forecaster. Always respond with valid JSON only. "
    "Do not wrap the JSON in code blocks."
)

SYSTEM_CRITIC = (
    "You are a careful critic of forecasts. Review the provided reasoning "
    "and produce a concise critique. Respond with valid JSON only."
)


def format_hypotheses(hypothesis_set: list[str], hypothesis_definitions: dict[str, str]) -> str:
    lines = []
    for h in hypothesis_set:
        d = hypothesis_definitions.get(h, "").strip()
        lines.append(f"  - {h}: {d}" if d else f"  - {h}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pick-only user template (2026-04-21). Every baseline returns exactly one
# hypothesis label — no probabilities, no calibration.
# ---------------------------------------------------------------------------

_BASE_USER = """Forecasting question: {question}

Background / resolution criteria:
{background}
{prior_expectation_block}
Candidate hypotheses (pick exactly one):
{hypotheses_block}

Relevant articles:
{articles_block}

Forecast point: {forecast_point}
Resolution date: {resolution_date}

{instructions}

Return JSON only (no prose, no code fences):
{{
  "prediction": "<exactly one of: {hypothesis_list}>",
  "reasoning": "<one or two concise sentences citing the articles>"{schema_extra}
}}"""


def _prior_expectation_block(fd: dict) -> str:
    """Render a natural-language 'status quo' sentence from the FD's prior-state
    annotation. Empty string when no annotation is available (keeps the prompt
    unchanged for FDs that weren't annotated).

    Under v2.1 (Comply / Surprise primary target), this block is what tells the
    model what the status-quo expectation IS, so it can decide whether the
    outcome will Comply with it (status quo holds) or Surprise against it
    (status quo breaks)."""
    prior = fd.get("prior_state_30d")
    if not prior:
        return ""
    stability = fd.get("prior_state_stability", 0.0)
    n_events = fd.get("prior_state_n_events", 0)
    bench = fd.get("benchmark", "")
    # Human-readable expectation sentence per benchmark.
    if bench == "gdelt-cameo":
        exp = (f"Over the past {fd.get('lookback_days', 90)} days, the dominant "
               f"interaction class between these actors has been '{prior}' "
               f"(observed in {int(stability*100)}% of {n_events} prior events).")
    elif bench == "earnings":
        exp = (f"In recent prior quarters, this company has mostly reported "
               f"'{prior}' outcomes (matching in {int(stability*100)}% of "
               f"{n_events} prior quarters).")
    elif bench == "forecastbench":
        exp = (f"The prediction-market crowd's implied majority answer at freeze "
               f"was '{prior}' (confidence strength {stability:.2f}).")
    else:
        exp = f"The status-quo expectation for this FD is '{prior}'."
    # Under the binary Comply/Surprise framing, append an explicit framing
    # clause so the model reads the prior as the thing to confirm or contradict.
    hs = fd.get("hypothesis_set") or []
    if "Comply" in hs and "Surprise" in hs:
        exp += (" Predict 'Comply' if the outcome will match this status-quo "
                "expectation, or 'Surprise' if it will break it.")
    return f"\nPrior expectation (status quo): {exp}\n"


def _shuffled_order(hypothesis_set: list[str], fd_id: str, salt: str = "") -> list[str]:
    """Deterministic per-FD shuffle of hypothesis display order. Breaks any
    positional bias the LLM may have toward the first-listed hypothesis.
    The canonical hypothesis_set is unchanged; only the prompt's display
    order is reshuffled. `salt` lets multi-sample baselines (B4/B9) use
    different orders per sample."""
    import hashlib
    import random
    seed_str = f"{fd_id}::{salt}"
    seed = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    out = list(hypothesis_set)
    rng.shuffle(out)
    return out


def render_user(
    fd: dict,
    articles_block: str,
    instructions: str = "Select the single most likely hypothesis given the evidence.",
    schema_extra: str = "",
    shuffle_salt: str = "",
) -> str:
    canonical = fd["hypothesis_set"]
    display = _shuffled_order(canonical, fd.get("id", ""), shuffle_salt)
    return _BASE_USER.format(
        question=fd["question"],
        background=(fd.get("background") or "(none)").strip() or "(none)",
        prior_expectation_block=_prior_expectation_block(fd),
        hypotheses_block=format_hypotheses(display, fd.get("hypothesis_definitions", {})),
        articles_block=articles_block,
        forecast_point=fd.get("forecast_point", "N/A"),
        resolution_date=fd.get("resolution_date", "N/A"),
        instructions=instructions.strip(),
        hypothesis_list=", ".join(display),
        schema_extra=schema_extra,
    )


# ---------------------------------------------------------------------------
# Per-method instruction blocks (pick-only)
# ---------------------------------------------------------------------------

INSTRUCTIONS_DIRECT = (
    "Pick the single most likely hypothesis. Return exactly one label from the list."
)

INSTRUCTIONS_COT = (
    "Think step-by-step using structured hypothesis comparison (ACH-style).\n"
    "For EACH candidate hypothesis, in order:\n"
    "  1. List the strongest evidence SUPPORTING it (articles + general knowledge).\n"
    "  2. List the strongest evidence DISCONFIRMING it.\n"
    "  3. Give a one-sentence verdict: is the evidence more consistent or inconsistent?\n"
    "After reviewing all hypotheses, pick the one whose evidence profile is most "
    "asymmetrically consistent (highest support AND lowest disconfirmation). Do not "
    "default to the most talked-about class; pick based on the evidence profile. "
    "Return exactly one label."
)

INSTRUCTIONS_RAG = (
    "Use ONLY the evidence in the articles above and the background. Weigh conflicting "
    "sources and pick the single most likely hypothesis. Return exactly one label."
)

INSTRUCTIONS_TOT_NODE = (
    "Generate {breadth} distinct candidate reasoning steps toward a single pick. For each, "
    "provide a short thought and a self-score 0-10 for how promising it is."
)

INSTRUCTIONS_REFLEXION_CRITIC = (
    "Critique the previous pick: list at most three concrete weaknesses in the reasoning "
    "or use of evidence. Do not output a hypothesis yourself."
)

INSTRUCTIONS_DEBATE_ROUND1 = INSTRUCTIONS_COT

INSTRUCTIONS_DEBATE_ROUNDN = (
    "Below are the other agents' picks and reasoning from the previous round. Read them, "
    "identify where you agree or disagree, and pick the single most likely hypothesis. "
    "Return exactly one label."
)


# ---------------------------------------------------------------------------
# Article context builder
# ---------------------------------------------------------------------------

def build_articles_block(
    article_ids: list[str],
    articles: dict[str, dict],
    max_articles: int = 10,
    max_chars_each: int = 900,
) -> str:
    """Format retrieved articles into the prompt context block.

    Wave-2 improvements:
      - Chronological sort ASC (oldest first -> newest last; recency is anchored
        at the end of the block where the LLM's attention is strongest).
      - Include source_domain in the header (credibility signal).
      - Dedupe by normalized title prefix (first 80 chars) to drop near-identical
        wire-service copies before the 900-char truncation burns budget on them.
    """
    # Resolve + filter to available articles
    resolved: list[dict] = []
    for aid in list(article_ids)[:max_articles * 2]:  # allow slack for dedupe
        art = articles.get(aid)
        if art:
            resolved.append(art)
    # Dedupe on normalized title prefix
    seen_titles: set[str] = set()
    deduped: list[dict] = []
    for art in resolved:
        key = (art.get("title") or "").strip().lower()[:80]
        if not key:
            deduped.append(art)
            continue
        if key in seen_titles:
            continue
        seen_titles.add(key)
        deduped.append(art)
    # Chronological ascending (unknown dates sorted last as empty string)
    deduped.sort(key=lambda a: a.get("publish_date", "") or "")
    # Cap at max_articles after dedupe+sort
    deduped = deduped[-max_articles:]  # keep the most recent after ASC sort
    chunks: list[str] = []
    for i, art in enumerate(deduped):
        title = (art.get("title") or "").strip()
        body = (art.get("text") or "").strip().replace("\n", " ")
        body = body[:max_chars_each]
        date = art.get("publish_date", "") or "unknown"
        source = (art.get("source_domain") or "").strip() or "unknown"
        header = f"[Article {i+1} | {date} | {source}] {title}"
        chunks.append(f"{header}\n{body}")
    if not chunks:
        return "(no articles available)"
    return "\n\n".join(chunks)
