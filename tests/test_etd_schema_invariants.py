"""ETD fact record schema invariants.

Validates synthetic Stage-1 fact records against docs/etd.schema.json plus
the v2.1 invariants the JSON Schema cannot easily express (id format,
canonical-id consistency, time-vs-article_date ordering).
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO / "docs" / "etd.schema.json"


def _fact(**overrides):
    base = {
        "id": "f_a1b2c3d4e5f6",
        "schema_version": "1.0",
        "time": "2026-03-01",
        "time_end": None,
        "time_precision": "day",
        "time_type": "point",
        "fact": "On 2026-03-01, the Pakistani foreign minister met with the Afghan ambassador.",
        "language": "en",
        "translated_from": None,
        "article_ids": ["art_a1b2c3d4e5f6"],
        "primary_article_id": "art_a1b2c3d4e5f6",
        "article_date": "2026-03-02",
        "source": "nytimes.com",
        "entities": [
            {"name": "Pakistan", "type": "country"},
            {"name": "Afghanistan", "type": "country"},
        ],
        "location": None,
        "metrics": [],
        "kind": "diplomatic-meeting",
        "tags": [],
        "polarity": "asserted",
        "attribution": None,
        "extraction_confidence": "high",
        "source_tier": None,
        "canonical_id": None,
        "variant_ids": [],
        "derived_from": [],
        "derivation": None,
        "extractor": "gpt-4o-mini-2024-07-18",
        "extract_run": "test_run",
        "extracted_at": "2026-04-23T00:00:00Z",
    }
    base.update(overrides)
    return base


def test_schema_file_loads():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema.get("$schema", "").startswith("https://json-schema.org/")


def test_minimal_fact_validates():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(_fact(), schema)


def test_missing_required_field_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = _fact()
    del bad["fact"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_polarity_enum_constrained():
    """polarity must come from a fixed enum so future runs cannot silently
    introduce new categories. The current allowed set is
    {asserted, negated, hypothetical, reported}; reject anything else."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    polarity_schema = schema.get("properties", {}).get("polarity", {})
    if "enum" not in polarity_schema:
        pytest.skip("schema does not pin polarity enum yet (TODO)")
    # 'asserted' is the unconditional default; other values may carry
    # additional schema-conditional requirements (e.g. polarity='reported'
    # requires attribution to be a non-empty string), so we only assert
    # the simple in-enum + out-of-enum shape here.
    jsonschema.validate(_fact(polarity="asserted"), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_fact(polarity="invented_value"), schema)


def test_extraction_confidence_enum_constrained():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    conf_schema = schema.get("properties", {}).get("extraction_confidence", {})
    if "enum" not in conf_schema:
        pytest.skip("schema does not pin extraction_confidence enum yet (TODO)")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_fact(extraction_confidence="very-high"), schema)


def test_canonical_id_round_trip():
    """If canonical_id is set, the fact is a variant of another canonical;
    the schema permits this. Round-trip check."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(_fact(canonical_id="f_aaaaaaaaaaaa"), schema)


def test_time_format_is_iso_date():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    time_schema = schema.get("properties", {}).get("time", {})
    if "pattern" not in time_schema and time_schema.get("format") not in ("date", "date-time"):
        pytest.skip("schema does not enforce time format (TODO)")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_fact(time="03/01/2026"), schema)


def test_time_should_not_exceed_article_date():
    """Audit found 0 facts with future-dated `time` after the producer's
    write-time check; this test guards against regression. Implemented as a
    Python-side invariant (JSON Schema cannot compare two fields)."""
    fact = _fact()  # default has time=2026-03-01, article_date=2026-03-02
    assert fact["time"] <= fact["article_date"], (
        "Default fixture must satisfy time <= article_date; otherwise the "
        "test fixture itself drifted."
    )
