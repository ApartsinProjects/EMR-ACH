"""Sample-based head-to-head: do ETD facts preserve forecasting ability vs. raw articles?

Picks N FDs from the v2.1 published benchmark, builds TWO evidence prompts
for each (one with the original article texts, one with the ETD facts
linked to those same articles via Stage-3 linkage), runs a direct
forecasting prompt on both, and reports:

  * Per-FD prediction agreement (articles_pred == facts_pred)
  * Headline accuracy parity (articles_acc vs. facts_acc) on the sample
  * Token cost per channel (mean / p95 input tokens)
  * Per-FD diff dump for manual inspection (article excerpt + linked facts +
    both predictions + ground truth)

If facts produce comparable accuracy at <30% of the input-token cost, ETD
is a viable retrieval channel. If facts agree on stability FDs but disagree
on change FDs, the extraction layer is dropping causal evidence (the
exact thing forecasting needs). The per-FD dump lets you see WHICH facts
were missing or distorted.

Cost: ~$0.05 for N=50 against gpt-4o-mini, sync mode (no batch needed for
sample sizes this small). Use --batch only when N >= 200.

Usage:
  python scripts/etd_compare_facts_vs_articles.py --cutoff 2026-01-01 --n 50
  python scripts/etd_compare_facts_vs_articles.py --cutoff 2026-01-01 --n 50 \
      --bench gdelt-cameo --seed 7
  python scripts/etd_compare_facts_vs_articles.py --cutoff 2026-01-01 --n 50 \
      --dry-run                # preview prompts without spending API tokens
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
ETD_LINKED = DATA / "etd" / "facts.v1_linked.jsonl"          # preferred
ETD_RAW    = DATA / "etd" / "facts.v1.jsonl"                  # fallback (no FD linkage)
OUT_DIR    = DATA / "etd" / "audit"


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


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _articles_block(article_ids, art_idx, max_chars=600, max_arts=10):
    """Render top-N articles, truncated to max_chars each."""
    parts = []
    for i, aid in enumerate(article_ids[:max_arts], 1):
        a = art_idx.get(aid)
        if not a:
            continue
        text = (a.get("text") or "").strip()[:max_chars]
        title = (a.get("title") or "").strip()
        date = a.get("date", "")
        src = a.get("source", "")
        parts.append(f"[A{i}] {date} {src} -- {title}\n{text}")
    return "\n\n".join(parts) if parts else "(no articles available)"


def _facts_block(facts, max_facts=20):
    """Render facts as one-line bullets sorted by date."""
    facts = sorted(facts, key=lambda f: f.get("time") or "")
    lines = []
    for i, f in enumerate(facts[:max_facts], 1):
        ents = ", ".join(e.get("name", "") for e in (f.get("entities") or []) if isinstance(e, dict))
        ents = f" [actors: {ents}]" if ents else ""
        lines.append(f"[F{i}] {f.get('time','?')}  ({f.get('extraction_confidence','?')}) "
                     f"{f.get('fact','')}{ents}")
    return "\n".join(lines) if lines else "(no facts linked to this FD)"


_PROMPT = """Forecasting question: {question}

Background:
{background}

Evidence ({channel_name}):
{evidence_block}

Candidate hypotheses (pick exactly one):
{hypotheses}

Forecast point: {forecast_point}
Resolution date: {resolution_date}

Pick the single most likely hypothesis based ONLY on the evidence above.

Return JSON only (no prose, no code fences):
{{"prediction": "<exactly one of: {hypothesis_list}>",
  "reasoning": "<one sentence>"}}"""


def _build_prompt(fd, evidence_block, channel_name):
    hyps = fd.get("hypothesis_set", [])
    hyp_block = "\n".join(f"  - {h}" for h in hyps)
    return _PROMPT.format(
        question=fd.get("question", ""),
        background=(fd.get("background") or "(none)").strip()[:500],
        channel_name=channel_name,
        evidence_block=evidence_block,
        hypotheses=hyp_block,
        forecast_point=fd.get("forecast_point", "?"),
        resolution_date=fd.get("resolution_date", "?"),
        hypothesis_list=", ".join(hyps),
    )


def _call(client, model, system, user):
    """Sync chat completion. Returns (parsed_dict_or_None, input_tokens, output_tokens)."""
    r = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    content = r.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    usage = r.usage
    return parsed, getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True, help="Published benchmark cutoff (YYYY-MM-DD).")
    ap.add_argument("--n", type=int, default=50, help="Sample size of FDs.")
    ap.add_argument("--bench", default=None, choices=[None, "forecastbench", "gdelt-cameo", "earnings"],
                    help="Optional: restrict sample to a single benchmark.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-arts", type=int, default=10, help="Top-N articles per FD.")
    ap.add_argument("--max-facts", type=int, default=20, help="Top-N facts per FD.")
    ap.add_argument("--max-chars-per-article", type=int, default=600)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build prompts and report token counts without calling the API.")
    ap.add_argument("--out", default=str(OUT_DIR / "facts_vs_articles.md"))
    ap.add_argument("--diff-out", default=str(OUT_DIR / "facts_vs_articles_diffs.jsonl"))
    args = ap.parse_args()

    fd_path = ROOT / "benchmark" / "data" / args.cutoff / "forecasts.jsonl"
    art_path = ROOT / "benchmark" / "data" / args.cutoff / "articles.jsonl"
    if not fd_path.exists() or not art_path.exists():
        print(f"[ERROR] missing {fd_path} or {art_path}; build the benchmark first.")
        return 1

    facts_path = ETD_LINKED if ETD_LINKED.exists() else ETD_RAW
    if not facts_path.exists():
        print(f"[ERROR] no ETD facts at {ETD_LINKED} or {ETD_RAW}; run articles_to_facts first.")
        return 1
    print(f"[etd_compare] FDs:      {fd_path}")
    print(f"[etd_compare] articles: {art_path}")
    print(f"[etd_compare] facts:    {facts_path}")

    fds = list(_load_jsonl(fd_path))
    arts = list(_load_jsonl(art_path))
    art_idx = {a["id"]: a for a in arts if "id" in a}
    print(f"[etd_compare] loaded {len(fds)} FDs and {len(arts)} articles")

    # Index facts by primary_article_id (and by linked FD if Stage-3 ran).
    facts_by_aid: dict[str, list[dict]] = {}
    facts_by_fd: dict[str, list[dict]] = {}
    using_linked = "linked" in str(facts_path)
    n_facts = 0
    for f in _load_jsonl(facts_path):
        n_facts += 1
        pid = f.get("primary_article_id")
        if pid:
            facts_by_aid.setdefault(pid, []).append(f)
        if using_linked:
            for fid in f.get("linked_fd_ids", []) or []:
                facts_by_fd.setdefault(fid, []).append(f)
    print(f"[etd_compare] loaded {n_facts} facts; linked-FD index: {using_linked}")

    # Sample FDs
    rng = random.Random(args.seed)
    pool = fds if not args.bench else [fd for fd in fds if fd.get("benchmark") == args.bench]
    pool = [fd for fd in pool if fd.get("article_ids")]
    if not pool:
        print("[ERROR] no eligible FDs (no article_ids); bench filter or build issue?")
        return 1
    sample = rng.sample(pool, min(args.n, len(pool)))
    print(f"[etd_compare] sampled {len(sample)} FDs from pool of {len(pool)}")

    # Build prompts + (optionally) call API
    client = None
    if not args.dry_run:
        from openai import OpenAI
        client = OpenAI()

    diffs = []
    in_tok_arts = []
    in_tok_facts = []
    agree = 0
    correct_arts = 0
    correct_facts = 0
    parsed_arts = 0
    parsed_facts = 0

    for i, fd in enumerate(sample, 1):
        # Pull the right facts: prefer FD-linked (Stage 3), else union by article ids
        if using_linked:
            fd_facts = facts_by_fd.get(fd["id"], [])
        else:
            fd_facts = []
            seen = set()
            for aid in (fd.get("article_ids") or []):
                for f in facts_by_aid.get(aid, []):
                    if f["id"] not in seen:
                        seen.add(f["id"])
                        fd_facts.append(f)

        articles_block = _articles_block(fd.get("article_ids", []), art_idx,
                                         max_chars=args.max_chars_per_article,
                                         max_arts=args.max_arts)
        facts_block = _facts_block(fd_facts, max_facts=args.max_facts)

        prompt_a = _build_prompt(fd, articles_block, "articles")
        prompt_f = _build_prompt(fd, facts_block, "atomic facts")

        # Approx token counting (1 token ~ 4 chars; good enough for cost gist)
        in_tok_arts.append(len(prompt_a) // 4)
        in_tok_facts.append(len(prompt_f) // 4)

        res_a, res_f = None, None
        if not args.dry_run:
            try:
                res_a, ta, _ = _call(client, args.model,
                                     "You are an expert forecaster. Always respond with valid JSON only.",
                                     prompt_a)
                in_tok_arts[-1] = ta
            except Exception as e:
                print(f"  [{i}] articles call failed: {e}")
            time.sleep(0.05)
            try:
                res_f, tf, _ = _call(client, args.model,
                                     "You are an expert forecaster. Always respond with valid JSON only.",
                                     prompt_f)
                in_tok_facts[-1] = tf
            except Exception as e:
                print(f"  [{i}] facts call failed: {e}")
            time.sleep(0.05)

        pred_a = (res_a or {}).get("prediction") if res_a else None
        pred_f = (res_f or {}).get("prediction") if res_f else None
        gt = fd.get("ground_truth")
        if pred_a is not None: parsed_arts += 1
        if pred_f is not None: parsed_facts += 1
        if pred_a == pred_f and pred_a is not None: agree += 1
        if pred_a == gt: correct_arts += 1
        if pred_f == gt: correct_facts += 1

        diffs.append({
            "fd_id": fd.get("id"),
            "benchmark": fd.get("benchmark"),
            "fd_type": fd.get("fd_type"),
            "question": fd.get("question"),
            "ground_truth": gt,
            "prediction_articles": pred_a,
            "prediction_facts": pred_f,
            "n_articles": len(fd.get("article_ids", [])),
            "n_facts_linked": len(fd_facts),
            "tokens_articles": in_tok_arts[-1],
            "tokens_facts": in_tok_facts[-1],
            "reasoning_articles": (res_a or {}).get("reasoning") if res_a else None,
            "reasoning_facts": (res_f or {}).get("reasoning") if res_f else None,
        })

        if i % 10 == 0:
            print(f"  [{i}/{len(sample)}] articles={parsed_arts} facts={parsed_facts} "
                  f"agree={agree}")

    _atomic_write_jsonl(Path(args.diff_out), diffs)

    def _stat(xs):
        if not xs: return (0, 0, 0)
        s = sorted(xs)
        return (sum(xs) / len(xs),
                s[len(s) // 2],
                s[int(len(s) * 0.95)])

    mean_a, med_a, p95_a = _stat(in_tok_arts)
    mean_f, med_f, p95_f = _stat(in_tok_facts)

    # Per-bench / per-fd_type breakdown
    by_type = Counter((d["benchmark"], d["fd_type"]) for d in diffs)
    type_acc_a = Counter()
    type_acc_f = Counter()
    type_total = Counter()
    for d in diffs:
        k = (d["benchmark"], d["fd_type"])
        type_total[k] += 1
        if d["prediction_articles"] == d["ground_truth"]: type_acc_a[k] += 1
        if d["prediction_facts"]    == d["ground_truth"]: type_acc_f[k] += 1

    n = len(sample)
    lines = []
    P = lines.append
    P(f"# Facts vs Articles head-to-head (sample N={n})")
    P(f"")
    P(f"- Cutoff: `{args.cutoff}` | Bench: `{args.bench or 'all'}` | Seed: {args.seed}")
    P(f"- Model: `{args.model}` | Mode: {'DRY-RUN (no API)' if args.dry_run else 'live'}")
    P(f"- Facts source: `{facts_path.name}` (FD-linked: {using_linked})")
    P(f"- Generated: {datetime.utcnow().isoformat()}Z")
    P(f"")
    P(f"## Headline")
    P(f"| metric | articles | facts | delta |")
    P(f"|---|---:|---:|---:|")
    if not args.dry_run:
        acc_a = correct_arts / max(1, n)
        acc_f = correct_facts / max(1, n)
        P(f"| accuracy   | {100*acc_a:.1f}% | {100*acc_f:.1f}% | {100*(acc_f-acc_a):+.1f}pp |")
        P(f"| parsed     | {parsed_arts}/{n} | {parsed_facts}/{n} | |")
        P(f"| agreement  | colspan=3 | {agree}/{n} ({100*agree/max(1,n):.1f}%) | |")
    P(f"| input tokens (mean)   | {mean_a:.0f} | {mean_f:.0f} | "
      f"{(mean_f-mean_a)/max(1,mean_a)*100:+.0f}% |")
    P(f"| input tokens (median) | {med_a} | {med_f} | "
      f"{(med_f-med_a)/max(1,med_a)*100:+.0f}% |")
    P(f"| input tokens (p95)    | {p95_a} | {p95_f} | "
      f"{(p95_f-p95_a)/max(1,p95_a)*100:+.0f}% |")
    P(f"")
    if not args.dry_run:
        P(f"## Per-(benchmark, fd_type) accuracy")
        P(f"| benchmark | fd_type | n | acc(articles) | acc(facts) | delta |")
        P(f"|---|---|---:|---:|---:|---:|")
        for k in sorted(type_total):
            t = type_total[k]
            aa = 100 * type_acc_a[k] / max(1, t)
            af = 100 * type_acc_f[k] / max(1, t)
            P(f"| `{k[0]}` | `{k[1]}` | {t} | {aa:.1f}% | {af:.1f}% | {af-aa:+.1f}pp |")
        P(f"")
    P(f"## Per-FD diffs")
    P(f"Full per-FD records (with reasoning) at `{args.diff_out}`.")
    P(f"")
    P(f"## How to read this")
    P(f"- A negative accuracy delta on the **change** subset is the canonical "
      f"signal that ETD is dropping causal evidence the model needs at the "
      f"hard cases. Inspect those rows in the diff file first.")
    P(f"- If facts beat articles on **stability** but lose on **change**, ETD "
      f"is stripping noise (good) but also stripping the leading indicator "
      f"of regime shift (bad). Tighten Stage-1 to require entity + numeric "
      f"context on flagged-as-change facts.")
    P(f"- A token-count ratio worse than ~30% (facts comparable to articles) "
      f"means Stage-2 dedup or summarization isn't compressing as expected; "
      f"re-check `--max-facts` and Stage-2 cluster-size threshold.")

    _atomic_write_text(Path(args.out), "\n".join(lines) + "\n")
    print(f"[etd_compare] report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
