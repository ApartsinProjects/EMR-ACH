"""Smoke tests for the B10 hybrid baseline (v2.2 [F1])."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.evaluation.baselines.methods.b10_hybrid_facts_articles import (
    DEFAULT_MAX_FACTS,
    B10HybridFactsArticles,
    build_facts_block,
    build_hybrid_evidence_block,
)


def test_build_facts_block_empty():
    assert "(no atomic facts" in build_facts_block([])


def test_build_facts_block_renders_fields():
    facts = [
        {"date": "2026-03-01", "confidence": "high",
         "text": "Pakistan FM met Afghan ambassador.",
         "actors": ["Pakistan", "Afghanistan"]},
        {"date": "2026-03-02", "confidence": "med",
         "text": "Casualty count rose to 47 per Reuters wire."},
    ]
    block = build_facts_block(facts)
    assert "[F1]" in block
    assert "[F2]" in block
    assert "2026-03-01" in block
    assert "actors: Pakistan, Afghanistan" in block


def test_build_facts_block_truncates_to_max_k():
    facts = [{"date": f"2026-03-{i+1:02d}", "text": f"fact {i}"} for i in range(40)]
    block = build_facts_block(facts, max_facts=5)
    # Only 5 lines (most recent 5 by date).
    assert block.count("\n") == 4
    assert "fact 39" in block
    assert "fact 0" not in block


def test_build_facts_block_truncates_text():
    long = "x" * 1000
    facts = [{"date": "2026-03-01", "text": long}]
    block = build_facts_block(facts, max_fact_chars=50)
    assert len(block) < 200


def test_build_hybrid_evidence_block_includes_both_sections():
    articles = {
        "a1": {"title": "T1", "text": "body 1", "publish_date": "2026-03-01",
               "source_domain": "nytimes.com"},
    }
    facts = [{"date": "2026-03-02", "text": "fact text", "confidence": "high"}]
    block = build_hybrid_evidence_block(["a1"], articles, facts)
    assert "Evidence (atomic facts" in block
    assert "Source articles" in block
    assert "fact text" in block
    assert "T1" in block


def test_b10_baseline_class_uses_hybrid_evidence():
    cfg = {}
    defaults = {
        "model": "gpt-5-nano",
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": None,
    }
    b = B10HybridFactsArticles(cfg, defaults)
    fd = {
        "id": "fd_x",
        "question": "Q?",
        "background": "bg",
        "hypothesis_set": ["Comply", "Surprise"],
        "hypothesis_definitions": {},
        "article_ids": ["a1"],
        "facts": [{"date": "2026-03-01", "text": "atomic fact",
                   "confidence": "high"}],
        "fd_type": "change",
        "benchmark": "gdelt-cameo",
    }
    articles = {
        "a1": {"title": "Title", "text": "body", "publish_date": "2026-03-01",
               "source_domain": "reuters.com"},
    }
    reqs = b.build_requests([fd], articles)
    assert len(reqs) == 1
    user_msg = reqs[0].messages[1]["content"]
    assert "atomic fact" in user_msg
    assert "Title" in user_msg
    assert b.max_facts == DEFAULT_MAX_FACTS


def test_b10_baseline_degrades_to_articles_only_when_no_facts():
    cfg = {}
    defaults = {"model": "gpt-5-nano", "temperature": 0.0,
                "max_tokens": 256, "response_format": None}
    b = B10HybridFactsArticles(cfg, defaults)
    fd = {
        "id": "fd_y", "question": "Q?", "background": "bg",
        "hypothesis_set": ["Comply", "Surprise"],
        "hypothesis_definitions": {},
        "article_ids": ["a1"], "facts": [],
        "fd_type": "change", "benchmark": "forecastbench",
    }
    articles = {"a1": {"title": "T", "text": "b", "publish_date": "2026-03-01"}}
    reqs = b.build_requests([fd], articles)
    user_msg = reqs[0].messages[1]["content"]
    assert "(no atomic facts" in user_msg
    # Articles still present.
    assert "T" in user_msg
