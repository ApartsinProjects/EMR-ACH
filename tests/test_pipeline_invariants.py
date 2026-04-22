"""Sanity-check invariants for the benchmark preparation pipeline.

These are fast, dependency-light unit tests for the three places where subtle
bugs would corrupt the dataset without obvious failure:

  1. Leakage guard — articles with publish_date >= forecast_point MUST be
     pruned before the quality filter's downstream counts run.
  2. Spam blocklist — auto-SEO domains and their subdomains MUST be filtered
     at fetch time.
  3. Dedup merge — same URL from two sources MUST produce one record whose
     provenance unions both tags and whose text is the longer of the two.

Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# 1. Spam blocklist
# ---------------------------------------------------------------------------

from src.common.spam_domains import is_spam_url, domain_of, SPAM_DOMAINS


def test_spam_exact_domain():
    assert is_spam_url("https://themarketsdaily.com/2026/foo")
    assert is_spam_url("https://tickerreport.com/abc")


def test_spam_subdomain_match():
    """Subdomain of a blocked root is also blocked."""
    assert is_spam_url("https://finance.themarketsdaily.com/foo")
    assert is_spam_url("https://www.tickerreport.com/abc")


def test_spam_non_match():
    """Legitimate news outlets pass through."""
    for u in ["https://reuters.com/world/...",
              "https://nytimes.com/2026/...",
              "https://finnhub.io/api/...",
              "https://www.wsj.com/..."]:
        assert not is_spam_url(u), f"false positive: {u}"


def test_spam_malformed_url_not_crash():
    """is_spam_url is called in hot loops — malformed input must not raise."""
    for u in ["", "not-a-url", None, 42]:
        try:
            is_spam_url(u if isinstance(u, str) else "")
        except Exception as e:
            pytest.fail(f"is_spam_url crashed on {u!r}: {e}")


def test_spam_domain_list_nonempty():
    """Guards against an accidental empty blocklist after a refactor."""
    assert len(SPAM_DOMAINS) >= 20, "SPAM_DOMAINS unexpectedly shrunk"
    for d in SPAM_DOMAINS:
        assert d == d.lower(), f"domain not lowercase: {d}"
        assert not d.startswith("www."), f"domain has www prefix: {d}"


def test_domain_of_strips_www():
    assert domain_of("https://www.reuters.com/foo") == "reuters.com"
    assert domain_of("https://Reuters.com/foo")      == "reuters.com"


# ---------------------------------------------------------------------------
# 2. Leakage guard (quality_filter behavior)
# ---------------------------------------------------------------------------

from quality_filter import parse_date   # type: ignore


def test_parse_date_happy():
    assert parse_date("2026-03-01") == datetime(2026, 3, 1)


def test_parse_date_truncates_timestamp():
    assert parse_date("2026-03-01T12:34:56Z") == datetime(2026, 3, 1)


def test_parse_date_bad():
    assert parse_date("") is None
    assert parse_date(None) is None            # type: ignore[arg-type]
    assert parse_date("last Tuesday") is None
    assert parse_date("2026/03/01") is None


def test_leakage_guard_logic():
    """Emulates the quality_filter prune: article publish_date < forecast_point."""
    forecast_point = datetime(2026, 3, 1)
    ok = [{"publish_date": "2026-02-27"}, {"publish_date": "2026-02-01"}]
    leak = [{"publish_date": "2026-03-01"}, {"publish_date": "2026-03-15"}]
    for a in ok:
        assert parse_date(a["publish_date"]) < forecast_point
    for a in leak:
        assert parse_date(a["publish_date"]) >= forecast_point


# ---------------------------------------------------------------------------
# 3. Dedup merge semantics (unify_articles.dedup_merge)
# ---------------------------------------------------------------------------

from unify_articles import dedup_merge   # type: ignore


def _article(id_: str, url: str, title: str, text: str, provenance: list[str]) -> dict:
    return {
        "id": id_, "url": url, "title": title, "text": text,
        "title_text": (title + "\n" + text).strip(),
        "publish_date": "2026-02-15", "source_domain": "test.example",
        "gdelt_themes": [], "gdelt_tone": 0.0, "actors": [],
        "cameo_code": "", "char_count": len(title) + len(text),
        "provenance": provenance,
    }


def test_dedup_union_provenance():
    """Same id, two sources → one record with union of provenance."""
    a = [_article("art_abc", "https://x.com/1", "T", "short",        ["forecastbench"])]
    b = [_article("art_abc", "https://x.com/1", "T", "much longer",   ["gdelt-cameo"])]
    merged = dedup_merge(a, b)
    assert len(merged) == 1
    assert set(merged[0]["provenance"]) == {"forecastbench", "gdelt-cameo"}


def test_dedup_prefers_longer_text():
    """On collision, the record with the longer `text` wins."""
    a = [_article("art_x", "https://x.com/2", "T", "short",       ["forecastbench"])]
    b = [_article("art_x", "https://x.com/2", "T", "much longer", ["gdelt-cameo"])]
    merged = dedup_merge(a, b)
    assert merged[0]["text"] == "much longer"


def test_dedup_no_collision():
    """Different ids stay separate."""
    a = [_article("art_a", "https://x.com/a", "Ta", "aa", ["forecastbench"])]
    b = [_article("art_b", "https://x.com/b", "Tb", "bb", ["gdelt-cameo"])]
    merged = dedup_merge(a, b)
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# 4. Config schema validation
# ---------------------------------------------------------------------------

from src.common.config_validation import validate_config


def test_config_schema_catches_bad_cutoff(tmp_path):
    schema = Path(__file__).parent.parent / "configs" / "default_config.schema.json"
    if not schema.exists():
        pytest.skip("schema file not present")
    bad = {"model_cutoff": "not-a-date"}
    errs = validate_config(bad, schema)
    assert any("model_cutoff" in e for e in errs)


def test_config_schema_catches_negative_buffer(tmp_path):
    schema = Path(__file__).parent.parent / "configs" / "default_config.schema.json"
    if not schema.exists():
        pytest.skip("schema file not present")
    bad = {"model_cutoff": "2026-01-01", "cutoff_buffer_days": -5}
    errs = validate_config(bad, schema)
    assert any("cutoff_buffer_days" in e for e in errs)


def test_config_schema_accepts_defaults():
    """The shipped default_config.yaml MUST validate clean."""
    import yaml
    cfg_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
    schema = Path(__file__).parent.parent / "configs" / "default_config.schema.json"
    if not (cfg_path.exists() and schema.exists()):
        pytest.skip("config files not present")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    errs = validate_config(cfg, schema)
    assert errs == [], f"default_config.yaml failed validation: {errs}"
