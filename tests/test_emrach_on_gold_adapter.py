"""Shape tests for scripts/eval/emrach_on_gold.py.

The adapter must NOT import experiments/02_emrach/run_emrach.py (in-flight
scope); these tests only exercise the data transformation + result
translation, against the existing parent-cutoff bundle when available
and synthetic fixtures otherwise.
"""

from __future__ import annotations

import importlib.util as _iu
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = REPO_ROOT / "scripts" / "eval" / "emrach_on_gold.py"
PARENT_BUNDLE = REPO_ROOT / "benchmark" / "data" / "2026-01-01"


def _load():
    spec = _iu.spec_from_file_location("_emrach_on_gold", ADAPTER_PATH)
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


ADAPTER = _load()


def test_does_not_import_run_emrach():
    """Hard contract: the adapter source must not import run_emrach."""
    src = ADAPTER_PATH.read_text(encoding="utf-8")
    assert "from experiments.02_emrach" not in src
    assert "import run_emrach" not in src
    assert "run_emrach" in src  # name is referenced in commentary, that's ok
    # Stricter: no `import run_emrach` line.
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("import run_emrach") or s.startswith("from run_emrach"):
            pytest.fail(f"Forbidden import of run_emrach: {line!r}")


def _synthetic_fd():
    return {
        "id": "fd_test_1",
        "benchmark": "gdelt_cameo",
        "question": "Will Country A escalate against Country B?",
        "background": "Background context.",
        "forecast_point": "2025-12-15",
        "resolution_date": "2025-12-29",
        "hypothesis_set": ["Comply", "Surprise"],
        "hypothesis_definitions": {"Comply": "stays peace", "Surprise": "violence"},
        "ground_truth": "Surprise",
        "ground_truth_idx": 1,
        "prior_state": "Peace",
        "fd_type": "change",
        "default_horizon_days": 14,
        "article_ids": ["art_aaa", "art_bbb", "art_missing"],
    }


def _synthetic_articles():
    return {
        "art_aaa": {
            "id": "art_aaa",
            "url": "https://example.com/a",
            "title": "Headline A",
            "text": "Body A",
            "publish_date": "2025-12-10",
            "source_domain": "example.com",
        },
        "art_bbb": {
            "id": "art_bbb",
            "url": "https://example.com/b",
            "title": "Headline B",
            "text": "",
            "publish_date": "2025-12-11",
            "source_domain": "example.com",
        },
    }


def test_fd_to_emrach_query_shape():
    fd = _synthetic_fd()
    arts = _synthetic_articles()
    q = ADAPTER.fd_to_emrach_query(fd, arts)
    assert q["fd_id"] == "fd_test_1"
    assert q["query"] == fd["question"]
    assert q["hypotheses"] == ["Comply", "Surprise"]
    assert q["label"] == "Surprise"
    # missing article id is skipped, not None-padded
    assert len(q["evidence"]) == 2
    assert {e["id"] for e in q["evidence"]} == {"art_aaa", "art_bbb"}
    assert q["evidence"][0]["text"] == "Body A"
    assert q["evidence"][1]["text"] == ""  # empty body preserved as ""


def test_build_emrach_inputs_bulk():
    fds = [_synthetic_fd(), _synthetic_fd()]
    arts = _synthetic_articles()
    out = ADAPTER.build_emrach_inputs(fds, arts)
    assert len(out) == 2
    assert all("evidence" in q for q in out)


def test_emrach_result_to_prediction_row():
    fd = _synthetic_fd()
    row = ADAPTER.emrach_result_to_prediction_row(
        fd, pick="Surprise", metadata={"score": 0.71}
    )
    assert row == {
        "fd_id": "fd_test_1",
        "benchmark": "gdelt_cameo",
        "prediction": "Surprise",
        "metadata": {"score": 0.71},
    }


def test_dry_run_against_parent_bundle(tmp_path):
    if not (PARENT_BUNDLE / "forecasts.jsonl").exists():
        pytest.skip(f"parent bundle missing at {PARENT_BUNDLE}")
    out = tmp_path / "preds.jsonl"
    summary = ADAPTER.run_emrach_on_bundle(
        bundle_dir=PARENT_BUNDLE,
        out_path=out,
        dry_run=True,
        limit=5,
    )
    assert summary["fds_in"] == 5
    assert summary["rows_written"] == 5
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 5
    for row in rows:
        assert "fd_id" in row
        assert "prediction" in row
        assert row["prediction"] is None  # dry-run placeholder
        assert row["metadata"]["adapter_status"] == "dry-run"


def test_run_emrach_on_bundle_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        ADAPTER.run_emrach_on_bundle(
            bundle_dir=tmp_path, out_path=tmp_path / "x.jsonl", dry_run=True
        )


def test_cli_help_does_not_crash():
    with pytest.raises(SystemExit) as excinfo:
        ADAPTER.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_cli_dry_run_synthetic_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "forecasts.jsonl").write_text(
        json.dumps(_synthetic_fd()) + "\n", encoding="utf-8"
    )
    (bundle / "articles.jsonl").write_text(
        "\n".join(json.dumps(a) for a in _synthetic_articles().values()) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "preds.jsonl"
    rc = ADAPTER.main(
        ["--bundle", str(bundle), "--out", str(out), "--dry-run"]
    )
    assert rc == 0
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["fd_id"] == "fd_test_1"


def test_cli_live_mode_refuses():
    bundle = REPO_ROOT  # path doesn't matter; live mode short-circuits
    rc = ADAPTER.main(["--bundle", str(bundle), "--out", str(bundle / "x.jsonl")])
    # No --dry-run -> rc=2 with the gated-feature notice
    assert rc == 2
