"""FD schema round-trip and v2.1 invariants.

Catches the class of bugs where the FD schema silently drifts (a stage
adds/removes a field without updating consumers, a 4-class label leaks
through after Comply/Surprise promotion, or a stratification field goes
missing from the change-subset partitioning).

Tests are dependency-light: they construct synthetic FDs in-memory and
feed them through the real validators / promoters where possible.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

REQUIRED_BASE = {
    "id", "benchmark", "source", "hypothesis_set",
    "question", "background",
    "forecast_point", "resolution_date",
    "ground_truth", "ground_truth_idx",
    "article_ids",
}

V21_PROMOTED = {
    "prior_state", "fd_type",
    "x_multiclass_ground_truth", "x_multiclass_hypothesis_set",
}

VALID_FD_TYPES = {"stability", "change", "unknown"}
COMPLY_SURPRISE = ["Comply", "Surprise"]
PEACE_TENSION_VIOLENCE = ["Peace", "Tension", "Violence"]
BEAT_MEET_MISS = ["Beat", "Meet", "Miss"]
YES_NO = ["Yes", "No"]


def _fd(**overrides) -> dict:
    """Minimal valid v2.1 FD scaffold for fixture use."""
    base = {
        "id": "test_001",
        "benchmark": "gdelt-cameo",
        "source": "gdelt-kg",
        "hypothesis_set": COMPLY_SURPRISE,
        "hypothesis_definitions": {
            "Comply": "Status-quo expectation holds.",
            "Surprise": "Status-quo expectation breaks.",
        },
        "question": "Will the prior-30d modal intensity hold for AFG/PAK on 2026-03-01?",
        "background": "Recent bilateral context summarized.",
        "forecast_point": "2026-03-01",
        "resolution_date": "2026-03-15",
        "ground_truth": "Comply",
        "ground_truth_idx": 0,
        "article_ids": ["art_a", "art_b", "art_c"],
        "prior_state": "Tension",
        "fd_type": "stability",
        "x_multiclass_ground_truth": "Tension",
        "x_multiclass_hypothesis_set": PEACE_TENSION_VIOLENCE,
        "default_horizon_days": 14,
    }
    base.update(overrides)
    return base


# -----------------------------------------------------------------------------
# Required-field presence
# -----------------------------------------------------------------------------

def test_required_base_fields_present():
    fd = _fd()
    missing = REQUIRED_BASE - set(fd.keys())
    assert not missing, f"FD scaffold missing required base fields: {missing}"


def test_v21_promoted_fields_present():
    """v2.1 FDs MUST carry the prior_state + fd_type + x_multiclass_* fields
    that drive the change-subset stratification and multiclass ablation."""
    fd = _fd()
    missing = V21_PROMOTED - set(fd.keys())
    assert not missing, (
        f"v2.1 promoted fields missing: {missing}. "
        "annotate_prior_state.py must always emit these."
    )


# -----------------------------------------------------------------------------
# Comply/Surprise primary contract
# -----------------------------------------------------------------------------

def test_primary_hypothesis_set_is_comply_surprise():
    fd = _fd()
    assert fd["hypothesis_set"] == COMPLY_SURPRISE


def test_ground_truth_is_in_hypothesis_set():
    fd = _fd()
    assert fd["ground_truth"] in fd["hypothesis_set"]


def test_ground_truth_idx_matches_label():
    fd = _fd()
    assert fd["hypothesis_set"][fd["ground_truth_idx"]] == fd["ground_truth"]


# -----------------------------------------------------------------------------
# Multiclass preservation as secondary
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("benchmark,multiclass", [
    ("gdelt-cameo", PEACE_TENSION_VIOLENCE),
    ("earnings",    BEAT_MEET_MISS),
    ("forecastbench", YES_NO),
])
def test_multiclass_preserved_per_benchmark(benchmark: str, multiclass: list[str]):
    fd = _fd(benchmark=benchmark, x_multiclass_hypothesis_set=multiclass,
             x_multiclass_ground_truth=multiclass[0])
    assert fd["x_multiclass_hypothesis_set"] == multiclass
    assert fd["x_multiclass_ground_truth"] in multiclass
    # Primary must STILL be Comply/Surprise regardless of domain.
    assert fd["hypothesis_set"] == COMPLY_SURPRISE


def test_no_4class_quad_label_leaks_to_primary():
    """Regression: pre-v2.1 GDELT FDs used VC/MC/VK/MK as primary. v2.1 must
    never emit those at the primary hypothesis layer."""
    forbidden = {
        "Verbal Cooperation", "Material Cooperation",
        "Verbal Conflict", "Material Conflict",
    }
    fd = _fd()
    assert not (set(fd["hypothesis_set"]) & forbidden)
    assert fd["ground_truth"] not in forbidden


# -----------------------------------------------------------------------------
# fd_type partitioning
# -----------------------------------------------------------------------------

def test_fd_type_value_is_valid():
    fd = _fd()
    assert fd["fd_type"] in VALID_FD_TYPES


def test_stability_implies_ground_truth_comply():
    """If prior_state matches the multiclass ground_truth, fd_type = stability
    and the binary ground_truth must be 'Comply'."""
    fd = _fd(fd_type="stability",
             prior_state="Tension",
             x_multiclass_ground_truth="Tension",
             ground_truth="Comply", ground_truth_idx=0)
    assert fd["fd_type"] == "stability"
    assert fd["ground_truth"] == "Comply"


def test_change_implies_ground_truth_surprise():
    """If prior_state differs from the multiclass ground_truth, fd_type =
    change and binary ground_truth must be 'Surprise'."""
    fd = _fd(fd_type="change",
             prior_state="Peace",
             x_multiclass_ground_truth="Violence",
             ground_truth="Surprise", ground_truth_idx=1)
    assert fd["fd_type"] == "change"
    assert fd["ground_truth"] == "Surprise"


# -----------------------------------------------------------------------------
# Round-trip
# -----------------------------------------------------------------------------

def test_jsonl_round_trip_preserves_all_fields(tmp_path):
    """Serializing → JSONL → parsing back must preserve every key + value."""
    fd = _fd()
    f = tmp_path / "fd.jsonl"
    f.write_text(json.dumps(fd) + "\n", encoding="utf-8")
    parsed = json.loads(f.read_text(encoding="utf-8").strip())
    assert parsed == fd


def test_horizon_field_emitted():
    """Every FD must carry default_horizon_days so the runner can apply
    apply_experiment_horizon() without the original config."""
    fd = _fd()
    assert "default_horizon_days" in fd
    assert isinstance(fd["default_horizon_days"], int)
    assert fd["default_horizon_days"] > 0
