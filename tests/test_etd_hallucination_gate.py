"""Tests for scripts/ci/etd_hallucination_gate.py (v2.2 [C7]).

Covers the dry-run path: parser, threshold pass / fail, missing-report
fail, and insufficient-sample fail.
"""

from __future__ import annotations

import importlib.util as _iu
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE_PATH = REPO_ROOT / "scripts" / "ci" / "etd_hallucination_gate.py"


def _load():
    spec = _iu.spec_from_file_location("_etd_gate", GATE_PATH)
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


GATE = _load()


REPORT_PASS = """# ETD Hallucination Verifier
- Sample size: 200
## Stratified by extraction_confidence
| confidence | n | supported | partial | unsupported | unsupported share |
|---|---:|---:|---:|---:|---:|
| `high` | 200 | 180 | 0 | 20 | 10.0% |
"""

REPORT_FAIL = """# ETD Hallucination Verifier
## Stratified by extraction_confidence
| confidence | n | supported | partial | unsupported | unsupported share |
|---|---:|---:|---:|---:|---:|
| `high` | 200 | 100 | 0 | 100 | 50.0% |
"""

REPORT_SMALL = """| confidence | n | supported | partial | unsupported | unsupported share |
|---|---:|---:|---:|---:|---:|
| `high` | 50 | 49 | 0 | 1 | 2.0% |
"""

REPORT_NO_HIGH = """| confidence | n | supported | partial | unsupported | unsupported share |
|---|---:|---:|---:|---:|---:|
| `medium` | 200 | 150 | 0 | 50 | 25.0% |
"""


def _write_report(tmp_path: Path, body: str) -> Path:
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "verifier_report.md").write_text(body, encoding="utf-8")
    return audit


def test_parse_high_row():
    parsed = GATE._parse_high_unsupported_share(REPORT_PASS)
    assert parsed is not None
    share, n = parsed
    assert n == 200
    assert share == pytest.approx(0.10)


def test_parse_no_high_row_returns_none():
    assert GATE._parse_high_unsupported_share(REPORT_NO_HIGH) is None


def test_dry_run_pass(tmp_path):
    audit = _write_report(tmp_path, REPORT_PASS)
    rc = GATE.run_dry_run(threshold=0.13, min_sample=200, audit_dir=audit)
    assert rc == 0


def test_dry_run_fail_threshold(tmp_path):
    audit = _write_report(tmp_path, REPORT_FAIL)
    rc = GATE.run_dry_run(threshold=0.13, min_sample=200, audit_dir=audit)
    assert rc == 1


def test_dry_run_missing_report(tmp_path):
    empty = tmp_path / "audit"
    empty.mkdir()
    rc = GATE.run_dry_run(threshold=0.13, min_sample=200, audit_dir=empty)
    assert rc == 2


def test_dry_run_insufficient_sample(tmp_path):
    audit = _write_report(tmp_path, REPORT_SMALL)
    rc = GATE.run_dry_run(threshold=0.13, min_sample=200, audit_dir=audit)
    assert rc == 3


def test_parse_args_defaults():
    ns = GATE.parse_args(["--dry-run"])
    assert ns.dry_run is True
    assert ns.threshold == pytest.approx(0.13)
    assert ns.min_sample == 200


def test_main_dry_run_via_cli(tmp_path):
    audit = _write_report(tmp_path, REPORT_PASS)
    rc = GATE.main(["--dry-run", "--audit-dir", str(audit), "--threshold", "0.2", "--min-sample", "100"])
    assert rc == 0
