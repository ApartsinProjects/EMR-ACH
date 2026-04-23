"""ETD hallucination verifier: sample-based LLM check of fact <-> source support.

Picks N random facts from `data/etd/facts.v1.jsonl`, joins each to the text
of its `primary_article_id`, and asks a cheap verifier model:

  "Given the article text, is this fact directly supported, partially
   supported, or unsupported?"

Aggregates verdicts overall and stratified by `extraction_confidence`. A
high unsupported rate at `high` confidence is the canonical signal that the
extractor is hallucinating; the per-confidence breakdown also tells you
what `--min-confidence` threshold to apply in `etd_filter.py`.

Cost: ~$0.01 / 100 facts on gpt-4o-mini sync. Default sample of 500 ~$0.05.

Usage:
  python scripts/etd_verify.py --n 500
  python scripts/etd_verify.py --n 100 --bench gdelt-cameo
  python scripts/etd_verify.py --dry-run    # build prompts without API calls
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FACTS_IN = DATA / "etd" / "facts.v1.jsonl"
ARTS_IN = DATA / "unified" / "articles.jsonl"
OUT_DIR = DATA / "etd" / "audit"


def _atomic_write_text(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


_PROMPT = """You will see an article and an extracted fact claimed to come from that article.
Decide if the fact is directly supported by the article text.

Article (may be truncated):
\"\"\"
{article}
\"\"\"

Extracted fact:
\"{fact}\"

Reply with JSON only, no prose, no code fences:
{{"verdict": "supported" | "partial" | "unsupported",
  "reason": "<one short sentence>"}}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(FACTS_IN))
    ap.add_argument("--articles", default=str(ARTS_IN))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--bench", default=None,
                    help="Optional: restrict sample to facts whose primary "
                         "article carries this benchmark (requires linked-FD "
                         "metadata; skipped if not present).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-article-chars", type=int, default=4000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR / "verifier_report.md"))
    ap.add_argument("--detail-out", default=str(OUT_DIR / "verifier_details.jsonl"))
    args = ap.parse_args()

    facts_path = Path(args.facts)
    arts_path = Path(args.articles)
    if not facts_path.exists() or not arts_path.exists():
        print(f"[ERROR] inputs missing: {facts_path} or {arts_path}")
        return 1

    print(f"[etd_verify] loading {facts_path}")
    facts = []
    with open(facts_path, encoding="utf-8") as f:
        for line in f:
            try:
                facts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"[etd_verify] facts: {len(facts)}")

    print(f"[etd_verify] loading {arts_path}")
    art_idx = {}
    with open(arts_path, encoding="utf-8") as f:
        for line in f:
            try:
                a = json.loads(line)
            except json.JSONDecodeError:
                continue
            if a.get("id"):
                art_idx[a["id"]] = a
    print(f"[etd_verify] articles indexed: {len(art_idx)}")

    pool = [fact for fact in facts
            if fact.get("primary_article_id") in art_idx
            and (art_idx[fact["primary_article_id"]].get("text") or "").strip()]
    if not pool:
        print("[ERROR] no facts with retrievable article text")
        return 1
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))
    print(f"[etd_verify] sampled {len(sample)} of {len(pool)} eligible facts")

    client = None
    if not args.dry_run:
        from openai import OpenAI
        client = OpenAI()

    results = []
    by_verdict = Counter()
    by_conf_verdict: dict[str, Counter] = defaultdict(Counter)

    for i, fact in enumerate(sample, 1):
        a = art_idx[fact["primary_article_id"]]
        article_text = (a.get("text") or "")[:args.max_article_chars]
        prompt = _PROMPT.format(article=article_text, fact=fact.get("fact", ""))

        verdict, reason = None, None
        if not args.dry_run:
            try:
                r = client.chat.completions.create(
                    model=args.model,
                    temperature=0.0,
                    messages=[
                        {"role": "system",
                         "content": "You are a strict fact-verification judge. Reply with valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                content = r.choices[0].message.content
                parsed = json.loads(content)
                verdict = parsed.get("verdict")
                reason = parsed.get("reason")
            except Exception as e:
                verdict = "ERROR"
                reason = str(e)[:200]
            time.sleep(0.05)

        if verdict:
            by_verdict[verdict] += 1
            by_conf_verdict[fact.get("extraction_confidence") or "unknown"][verdict] += 1
        results.append({
            "fact_id": fact.get("id"),
            "primary_article_id": fact.get("primary_article_id"),
            "article_url": a.get("url"),
            "fact_text": fact.get("fact"),
            "extraction_confidence": fact.get("extraction_confidence"),
            "verdict": verdict,
            "reason": reason,
        })

        if i % 50 == 0:
            print(f"  [{i}/{len(sample)}] verdicts so far: {dict(by_verdict)}")

    _atomic_write_jsonl(Path(args.detail_out), results)

    n = len(sample)
    lines = []
    P = lines.append
    P(f"# ETD Hallucination Verifier")
    P(f"")
    P(f"- Facts file: `{facts_path}`")
    P(f"- Articles file: `{arts_path}`")
    P(f"- Generated: {datetime.utcnow().isoformat()}Z")
    P(f"- Sample size: {n} (seed={args.seed}, model=`{args.model}`)")
    P(f"- Mode: {'DRY-RUN (no API)' if args.dry_run else 'live'}")
    P(f"")
    if args.dry_run:
        P(f"Dry-run produced no verdicts. Re-run without --dry-run to call "
          f"the verifier.")
        _atomic_write_text(Path(args.out), "\n".join(lines) + "\n")
        return 0

    total_verdicts = sum(by_verdict.values())
    P(f"## Headline verdict distribution")
    P(f"| verdict | n | share |")
    P(f"|---|---:|---:|")
    for v in ["supported", "partial", "unsupported", "ERROR"]:
        c = by_verdict.get(v, 0)
        share = 100 * c / max(1, total_verdicts)
        P(f"| `{v}` | {c} | {share:.1f}% |")
    P(f"")
    P(f"## Stratified by extraction_confidence")
    P(f"| confidence | n | supported | partial | unsupported | unsupported share |")
    P(f"|---|---:|---:|---:|---:|---:|")
    for conf in ["high", "medium", "low", "unknown"]:
        ctr = by_conf_verdict.get(conf, Counter())
        nc = sum(ctr.values())
        if nc == 0:
            continue
        s = ctr.get("supported", 0)
        p = ctr.get("partial", 0)
        u = ctr.get("unsupported", 0)
        P(f"| `{conf}` | {nc} | {s} | {p} | {u} | {100*u/max(1,nc):.1f}% |")
    P(f"")
    P(f"## Recommended actions")
    P(f"- If `unsupported` share at `high` confidence is >5%, the extractor "
      f"is producing high-confidence hallucinations; revise the Stage-1 "
      f"prompt to demand verbatim grounding.")
    P(f"- If unsupported rate climbs sharply at `low` confidence, add "
      f"`--min-confidence medium` to `etd_filter.py` defaults.")
    P(f"- Per-fact details (with reason) at `{args.detail_out}`. Sort by "
      f"`verdict=unsupported AND extraction_confidence=high` and review the "
      f"top 20 manually before adjusting prompts.")

    _atomic_write_text(Path(args.out), "\n".join(lines) + "\n")
    print(f"[etd_verify] report -> {args.out}")
    print(f"[etd_verify] details -> {args.detail_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
