#!/usr/bin/env python
"""Zero-dependency loader for the EMR-ACH Gold Subset.

Uses only Python stdlib. Run from the gold-folder directory.
Loads three record types: FDs, articles, and ETD atomic facts.
"""
import json
from pathlib import Path
from collections import defaultdict


def load_fds(path: str = "forecasts.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_articles(path: str = "articles.jsonl") -> dict[str, dict]:
    """Returns {article_id: article_record}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            out[a["id"]] = a
    return out


def load_facts(path: str = "facts.jsonl") -> list[dict]:
    """Returns the ETD atomic-fact records. May be empty if Stage 1 ETD
    has not been run for the parent cutoff."""
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def index_facts_by_article(facts: list[dict]) -> dict[str, list[dict]]:
    """Group facts by primary_article_id for fast per-article lookup."""
    out = defaultdict(list)
    for f in facts:
        pid = f.get("primary_article_id")
        if pid:
            out[pid].append(f)
    return out


def fd_evidence(fd: dict, articles: dict[str, dict], top_k: int | None = None) -> list[dict]:
    """Return the article records cited in fd.article_ids, in order."""
    aids = fd.get("article_ids") or []
    if top_k is not None:
        aids = aids[:top_k]
    return [articles[a] for a in aids if a in articles]


def fd_facts(fd: dict, facts_by_article: dict[str, list[dict]],
             top_k: int | None = None) -> list[dict]:
    """Return ETD facts whose primary_article_id is in fd.article_ids,
    sorted by fact.time descending."""
    out = []
    for aid in fd.get("article_ids") or []:
        out.extend(facts_by_article.get(aid, []))
    out.sort(key=lambda f: f.get("time") or "", reverse=True)
    return out[:top_k] if top_k else out


if __name__ == "__main__":
    fds = load_fds()
    arts = load_articles()
    facts = load_facts()
    print(f"Loaded {len(fds)} FDs, {len(arts)} articles, {len(facts)} ETD facts.")
    by_bench = {}
    for fd in fds:
        by_bench[fd["benchmark"]] = by_bench.get(fd["benchmark"], 0) + 1
    print(f"By benchmark: {by_bench}")
    sample = fds[0]
    print(f"\nSample FD: {sample['id']} ({sample['benchmark']})")
    print(f"  Q: {sample['question'][:90]}")
    print(f"  Hypotheses: {sample['hypothesis_set']}")
    print(f"  Ground truth: {sample['ground_truth']}  (fd_type={sample['fd_type']})")
    ev = fd_evidence(sample, arts, top_k=3)
    for a in ev:
        print(f"  Article: {a['publish_date']} {a['url'][:70]}")
    if facts:
        idx = index_facts_by_article(facts)
        for fact in fd_facts(sample, idx, top_k=3):
            print(f"  Fact: ({fact.get('time','?')}) {fact['fact'][:90]}")
