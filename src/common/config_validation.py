"""YAML config schema validator for scripts/build_benchmark.py.

Loads configs/default_config.schema.json and validates a parsed config dict
against it. Fails fast with actionable error messages:

    [config] configs/default_config.yaml:
      model_cutoff: 'not-a-date' does not match '^\\d{4}-\\d{2}-\\d{2}$'
      benchmarks.earnings.threshold: 2.5 is greater than the maximum of 1

Usage:
    from src.common.config_validation import validate_config
    errors = validate_config(cfg, schema_path)
    if errors:
        for e in errors: print(f'  {e}')
        sys.exit(2)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .optional_imports import optional

_jsonschema = optional("jsonschema", install_hint="pip install jsonschema")


def validate_config(cfg: dict[str, Any], schema_path: Path | str) -> list[str]:
    """Validate cfg against the JSON Schema at schema_path.

    Returns a list of human-readable error messages (empty list = valid).
    If jsonschema isn't installed, returns [] (validation skipped with warning).
    """
    if not _jsonschema:
        print("[config] jsonschema not installed; skipping validation. "
              "Install with: pip install jsonschema")
        return []

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator = _jsonschema.Draft202012Validator(schema)
    errs = sorted(validator.iter_errors(cfg), key=lambda e: e.path)
    out: list[str] = []
    for e in errs:
        loc = ".".join(str(p) for p in e.path) or "<root>"
        out.append(f"{loc}: {e.message}")
    return out
