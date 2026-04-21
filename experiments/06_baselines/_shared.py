"""
Shared helpers for B4-B9 baseline runners on the unified Forecast Dossier (FD) dataset.

Provides:
  - load_fds / load_articles: jsonl loaders
  - concat_articles: builds a bounded article-context block from article_ids
  - BASE_SYSTEM / BASE_USER_YESNO: common Yes/No base prompt
  - extract_prob_yes: parses a probability estimate from a JSON or text response
  - synthetic_response: deterministic placeholder response for --dry-run
  - evaluate_binary: brier + ece + accuracy dict
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics import brier_score  # noqa: E402


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_fds(path: str | Path, n: int | None = None) -> list[dict]:
    path = Path(path)
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if n is not None and len(out) >= n:
                break
    return out


def load_articles(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def concat_articles(
    article_ids: Iterable[str],
    articles: dict[str, dict],
    max_articles: int = 5,
    max_chars_each: int = 700,
) -> str:
    chunks: list[str] = []
    for i, aid in enumerate(list(article_ids)[:max_articles]):
        art = articles.get(aid)
        if not art:
            continue
        title = (art.get("title") or "").strip()
        body = (art.get("text") or "").strip().replace("\n", " ")
        body = body[:max_chars_each]
        chunks.append(f"[Article {i+1}] {title}\n{body}")
    if not chunks:
        return "(no articles available)"
    return "\n\n".join(chunks)


# ------------------------------------------------------------------
# Prompt templates shared across baselines
# ------------------------------------------------------------------

BASE_SYSTEM = (
    "You are an expert forecaster. Always respond with valid JSON only. "
    "Do not wrap in code blocks."
)

BASE_USER_YESNO = """Forecasting question: {question}
Resolution date: {resolution_date}

Relevant articles:
{context}

Assign a calibrated probability to Yes that reflects the specific evidence.
Probabilities must sum to 1.0.

Return JSON only:
{{
  "probabilities": {{"Yes": <float>, "No": <float>}},
  "prediction": "<Yes_or_No>",
  "reasoning": "<one concise sentence>"
}}"""


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------

def extract_prob_yes(text: str, default: float = 0.5) -> float:
    """Pull a Yes probability from a JSON blob or fall back to regex."""
    if not text:
        return default
    # Try full JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            probs = data.get("probabilities", {})
            if isinstance(probs, dict) and "Yes" in probs:
                return float(np.clip(float(probs["Yes"]), 0.0, 1.0))
            if "prob_yes" in data:
                return float(np.clip(float(data["prob_yes"]), 0.0, 1.0))
    except Exception:
        pass
    # Regex fallback: look for "Yes": <number>
    m = re.search(r'"Yes"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if m:
        try:
            return float(np.clip(float(m.group(1)), 0.0, 1.0))
        except Exception:
            pass
    m = re.search(r"\b([01]?\.\d+)\b", text)
    if m:
        try:
            return float(np.clip(float(m.group(1)), 0.0, 1.0))
        except Exception:
            pass
    return default


# ------------------------------------------------------------------
# Dry-run synthetic response
# ------------------------------------------------------------------

def synthetic_response(fd_id: str, variant: str = "", prob: float = 0.5) -> str:
    """Deterministic synthetic JSON response for dry-run plumbing checks."""
    # Slight deterministic jitter so probabilities aren't all identical
    seed = sum(ord(c) for c in f"{fd_id}|{variant}") % 1000
    jitter = (seed / 1000.0 - 0.5) * 0.2  # +/- 0.1
    p = float(np.clip(prob + jitter, 0.05, 0.95))
    return json.dumps({
        "probabilities": {"Yes": round(p, 4), "No": round(1 - p, 4)},
        "prediction": "Yes" if p >= 0.5 else "No",
        "reasoning": f"synthetic dry-run response ({variant})",
    })


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def gt_to_int(gt) -> int:
    """Normalize ground_truth (Yes/No string, 0/1, bool) to 0/1."""
    if isinstance(gt, bool):
        return int(gt)
    if isinstance(gt, (int, float)):
        return int(gt)
    if isinstance(gt, str):
        s = gt.strip().lower()
        if s in ("yes", "y", "true", "1"):
            return 1
        if s in ("no", "n", "false", "0"):
            return 0
    raise ValueError(f"Cannot interpret ground_truth={gt!r}")


def evaluate_binary(preds: list[dict]) -> dict:
    probs = [float(p["prob_yes"]) for p in preds]
    labels = [int(p["gt"]) for p in preds]
    if not probs:
        return {"brier_score": None, "ece": None, "accuracy": None, "n": 0}
    bs = brier_score(probs, labels)
    probs_arr = np.array(probs)
    labels_arr = np.array(labels, dtype=float)
    n = len(probs_arr)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs_arr >= lo) & (probs_arr < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(labels_arr[mask].mean() - probs_arr[mask].mean())
    pred_labels = (probs_arr >= 0.5).astype(int)
    acc = float((pred_labels == labels_arr).mean())
    return {
        "brier_score": round(float(bs), 4),
        "ece": round(float(ece), 4),
        "accuracy": round(acc * 100, 1),
        "n": n,
    }


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def common_argparser(description: str):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--fds", default="data/unified/forecasts_filtered.jsonl")
    p.add_argument("--articles", default="data/unified/articles.jsonl")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n-queries", type=int, default=None)
    return p


def n_for_mode(mode: str, override: int | None) -> int | None:
    if override:
        return override
    if mode == "smoke":
        return 5
    return None
