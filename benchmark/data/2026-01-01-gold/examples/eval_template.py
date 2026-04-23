#!/usr/bin/env python
"""Baseline pick-only evaluation skeleton, gold subset, zero EMR-ACH dependency.

Loads the gold subset, formats a prompt per FD, calls a user-supplied
LLM client, and computes accuracy + per-fd_type breakdown. No reference
to any module outside this folder.

Configure the LLM call by editing `call_llm()`; the rest is inert.
"""
import json
from pathlib import Path
from collections import Counter


def load_gold():
    fds = [json.loads(l) for l in open("forecasts.jsonl", encoding="utf-8")]
    arts = {a["id"]: a for a in (json.loads(l) for l in open("articles.jsonl", encoding="utf-8"))}
    return fds, arts


def render_prompt(fd, arts, max_arts=10, max_chars_per_article=600):
    hyp_block = "\n".join(f"  - {h}: {fd['hypothesis_definitions'].get(h, '')}" for h in fd["hypothesis_set"])
    selected = [arts[a] for a in fd["article_ids"][:max_arts] if a in arts]
    art_block = "\n\n".join(
        f"[A{i+1}] {a['publish_date']} {a.get('source_domain', '?')} -- {a['title']}\n"
        f"{(a.get('text') or '')[:max_chars_per_article]}"
        for i, a in enumerate(selected)
    )
    return (
        f"Forecasting question: {fd['question']}\n\n"
        f"Background: {fd.get('background') or '(none)'}\n\n"
        f"Hypotheses (pick exactly one):\n{hyp_block}\n\n"
        f"Evidence:\n{art_block}\n\n"
        f"Forecast point: {fd['forecast_point']}\n"
        f"Resolution date: {fd['resolution_date']}\n\n"
        f"Return JSON only, no prose:\n"
        f'{{"prediction": "<exactly one of: {", ".join(fd["hypothesis_set"])}>"}}'
    )


def call_llm(prompt: str) -> str:
    """Replace this stub with your provider call. The default returns
    the FD's first hypothesis (a degenerate baseline)."""
    raise NotImplementedError("Edit call_llm() to call your model. The default raises by design.")


def main():
    fds, arts = load_gold()
    print(f"Loaded {len(fds)} FDs and {len(arts)} articles.")

    correct_total = 0
    by_ft = Counter()
    correct_by_ft = Counter()
    skipped = 0

    for i, fd in enumerate(fds, 1):
        try:
            prompt = render_prompt(fd, arts)
            response_text = call_llm(prompt)
            pred = json.loads(response_text).get("prediction")
        except Exception as e:
            skipped += 1
            continue
        ft = fd.get("fd_type", "unknown")
        by_ft[ft] += 1
        if pred == fd["ground_truth"]:
            correct_total += 1
            correct_by_ft[ft] += 1

    n = sum(by_ft.values())
    print(f"\nResults (skipped={skipped}):")
    print(f"  overall: {correct_total}/{n} = {100*correct_total/max(1, n):.1f}%")
    for ft, count in by_ft.items():
        c = correct_by_ft[ft]
        print(f"  fd_type={ft}: {c}/{count} = {100*c/max(1, count):.1f}%")


if __name__ == "__main__":
    main()
