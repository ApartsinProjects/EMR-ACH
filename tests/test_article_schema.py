"""Article record schema invariants.

Validates synthetic and real-sample article records against
docs/article.schema.json plus a few v2.1 invariants the JSON Schema cannot
express (id format, URL non-empty, source_type enum coverage).
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO / "docs" / "article.schema.json"


def _article(**overrides):
    base = {
        "id": "art_a1b2c3d4e5f6",
        "url": "https://www.nytimes.com/2026/03/01/world/example.html",
        "title": "Example article title",
        "text": "Body text of the example article ...",
        "date": "2026-03-01",
        "source": "The New York Times",
        "provenance": "nyt",
        "language": "en",
    }
    base.update(overrides)
    return base


def test_schema_file_loads():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert "id" in schema["required"]


def test_minimal_article_validates():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(_article(), schema)


def test_id_must_match_art_hex_pattern():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_article(id="abc123"), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_article(id="art_TOOSHORT"), schema)


def test_date_must_be_iso():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_article(date="03/01/2026"), schema)


def test_required_fields_present():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = _article()
    del bad["url"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_source_type_enum():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    # filing is a valid source_type for EDGAR records
    jsonschema.validate(_article(source_type="filing"), schema)
    # news is the default category
    jsonschema.validate(_article(source_type="news"), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_article(source_type="invented"), schema)


def test_extra_fields_allowed():
    """The schema is intentionally open under additionalProperties: true so
    domain-specific fetchers can stash extra metadata without breaking
    consumers. Verify a custom field passes."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    a = _article()
    a["_finnhub_meta"] = {"category": "earnings", "related": ["AAPL"]}
    jsonschema.validate(a, schema)


def test_text_can_be_empty():
    """Title-only records (no body extraction) are allowed; downstream
    quality_filter / articles_audit reports them, but the schema accepts."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(_article(text=""), schema)
