"""
Abstract Baseline class. Every method in `methods/` subclasses this and implements:
  - build_requests(fds, articles, cfg) -> list[BatchRequest]
  - parse_responses(results, fds, cfg) -> list[dict]  (prediction rows)

All OpenAI calls route through src.batch_client.BatchClient. No direct openai calls.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ._shim import BatchRequest, BatchResult, parse_json_response

from .prompts import (
    SYSTEM_FORECASTER,
    build_articles_block,
    render_user,
)


class Baseline(ABC):
    """Base class for a single baseline method."""

    name: str = "baseline"  # short id, e.g. "b1_direct"

    def __init__(self, cfg: dict[str, Any], defaults: dict[str, Any]):
        """
        cfg: the per-baseline config dict (class/args/overrides) from configs/baselines.yaml
        defaults: the defaults block (model, temperature, max_tokens, batch_api, response_format)
        """
        self.cfg = cfg
        self.defaults = defaults

    # ------------------------------------------------------------------
    # Model/temperature resolution helpers
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self.cfg.get("model", self.defaults["model"])

    @property
    def temperature(self) -> float:
        return float(self.cfg.get("temperature", self.defaults["temperature"]))

    @property
    def max_tokens(self) -> int:
        return int(self.cfg.get("max_tokens", self.defaults["max_tokens"]))

    @property
    def response_format(self) -> dict | None:
        return self.cfg.get("response_format", self.defaults.get("response_format"))

    @property
    def max_articles(self) -> int:
        return int(self.cfg.get("max_articles", 10))

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def articles_block(self, fd: dict, articles: dict[str, dict]) -> str:
        return build_articles_block(
            fd.get("article_ids", []),
            articles,
            max_articles=self.max_articles,
        )

    def make_request(
        self,
        custom_id: str,
        user: str,
        *,
        system: str = SYSTEM_FORECASTER,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        extra: dict | None = None,
    ) -> BatchRequest:
        return BatchRequest(
            custom_id=custom_id,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            response_format=response_format if response_format is not None else self.response_format,
            extra=extra or {},
        )

    def render_user(self, fd: dict, articles: dict[str, dict], **kwargs) -> str:
        return render_user(fd, self.articles_block(fd, articles), **kwargs)

    # ------------------------------------------------------------------
    # Abstract surface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_requests(
        self, fds: list[dict], articles: dict[str, dict]
    ) -> list[BatchRequest]:
        ...

    @abstractmethod
    def parse_responses(
        self,
        results: dict[str, BatchResult],
        fds: list[dict],
    ) -> list[dict]:
        ...

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_pick(content: str, hypothesis_set: list[str]) -> str | None:
        """Parse a single hypothesis label from a JSON `prediction` field.

        Returns None if nothing in the JSON maps cleanly to a hypothesis.
        Never falls back to hypothesis_set[0] or to free-text outside the JSON —
        silent fallbacks would pollute metrics. Callers mark the FD parse_failed
        and count it as wrong.

        Handles:
          - Extra JSON keys (predicted_class, class, choice, decision)
          - List values: {"prediction": ["Yes"]}
          - Nested dict: {"prediction": {"class": "Yes"}}
          - Prefix match: "Verbal Cooperation: because ..." -> "Verbal Cooperation"
          - Substring match with longest-hypothesis priority
          - Case-insensitive + punctuation-stripped
          - Whitespace-stripped ("verbalcooperation" -> "Verbal Cooperation")
        """
        if not content:
            return None
        data = parse_json_response(content)
        candidate_strs: list[str] = []
        if isinstance(data, dict):
            for key in ("prediction", "pick", "answer", "label",
                        "predicted_class", "class", "choice", "decision",
                        "selected_hypothesis", "hypothesis", "final_answer"):
                v = data.get(key)
                if isinstance(v, str):
                    candidate_strs.append(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str):
                            candidate_strs.append(item)
                elif isinstance(v, dict):
                    for subk in ("class", "value", "label", "name", "text"):
                        sv = v.get(subk)
                        if isinstance(sv, str):
                            candidate_strs.append(sv)
                            break
            # Also scan the `reasoning` field as a last-resort *intra-JSON* signal
            # — not free text outside the JSON envelope, so still safe.
            r = data.get("reasoning")
            if isinstance(r, str):
                candidate_strs.append(r)

        if not candidate_strs:
            return None

        # Precompute normalized hypothesis keys for matching
        hs_norm = {h.lower(): h for h in hypothesis_set}
        hs_nospace = {h.lower().replace(" ", ""): h for h in hypothesis_set}

        def _try_match(raw: str) -> str | None:
            s = raw.strip().strip('"\'.,;:!?()[]{}*`')
            if s in hypothesis_set:
                return s
            sl = s.lower()
            if sl in hs_norm:
                return hs_norm[sl]
            if sl in hs_nospace:
                return hs_nospace[sl]
            # Unique prefix match (handles "Verbal Cooperation: the article says...")
            prefix_matches = [h for h in hypothesis_set
                              if sl.startswith(h.lower()) or sl.startswith(h.lower().replace(" ", ""))]
            if len(prefix_matches) == 1:
                return prefix_matches[0]
            # Longest substring match (handles "Most likely: Material Conflict")
            sub_matches = sorted(
                [h for h in hypothesis_set if h.lower() in sl],
                key=lambda h: -len(h),
            )
            if sub_matches:
                # Require the match to be uniquely the longest (avoid "Cooperation" matching both)
                if len(sub_matches) == 1 or len(sub_matches[0]) > len(sub_matches[1]):
                    return sub_matches[0]
            return None

        for raw in candidate_strs:
            m = _try_match(raw)
            if m:
                return m
        return None

    @staticmethod
    def plurality(picks: list[str | None], hypothesis_set: list[str]) -> str | None:
        """Majority vote over a list of picks. None entries (parse failures) are
        ignored. If every pick failed to parse, returns None (marks the FD
        parse_failed). Ties broken by hypothesis_set order."""
        from collections import Counter
        valid = [p for p in picks if p in hypothesis_set]
        if not valid:
            return None
        c = Counter(valid)
        return max(hypothesis_set, key=lambda h: (c[h], -hypothesis_set.index(h)))

    @staticmethod
    def parse_probabilities(content: str, hypothesis_set: list[str]) -> dict[str, float]:
        """Parse the 'probabilities' field from a JSON response. Fallbacks to uniform."""
        data = parse_json_response(content) if content else None
        probs: dict[str, float] = {}
        if isinstance(data, dict):
            p = data.get("probabilities") or data.get("probs") or {}
            if isinstance(p, dict):
                for h in hypothesis_set:
                    if h in p:
                        try:
                            probs[h] = float(p[h])
                        except (TypeError, ValueError):
                            pass
        # Regex fallback per hypothesis
        if len(probs) != len(hypothesis_set) and content:
            for h in hypothesis_set:
                if h in probs:
                    continue
                m = re.search(rf'"{re.escape(h)}"\s*:\s*([0-9]*\.?[0-9]+)', content)
                if m:
                    try:
                        probs[h] = float(m.group(1))
                    except ValueError:
                        pass
        if len(probs) != len(hypothesis_set):
            return {h: 1.0 / len(hypothesis_set) for h in hypothesis_set}
        # Heuristic: if any value > 1.5, the model emitted percentages (e.g.
        # {"Yes": 95, "No": 5}) instead of probabilities. Rescale by /100
        # BEFORE normalizing, so mixed outputs like {"Yes": 95, "No": 0.5}
        # don't get silently corrupted by sum-based normalization.
        if any(v > 1.5 for v in probs.values()):
            probs = {h: v / 100.0 for h, v in probs.items()}
        # Normalize and clip
        total = sum(max(0.0, v) for v in probs.values())
        if total <= 0:
            return {h: 1.0 / len(hypothesis_set) for h in hypothesis_set}
        return {h: float(np.clip(probs[h] / total, 0.0, 1.0)) for h in hypothesis_set}

    @staticmethod
    def argmax_class(prob_distribution: dict[str, float], hypothesis_set: list[str]) -> str:
        return max(hypothesis_set, key=lambda h: prob_distribution.get(h, 0.0))

    @staticmethod
    def prediction_row(fd: dict, predicted_class: str | None, extras: dict | None = None) -> dict:
        """Pick-only prediction row. predicted_class=None indicates a parse
        failure — row is flagged parse_failed and will be counted as wrong by
        the metrics layer (no silent fallback to hypothesis_set[0])."""
        parse_failed = predicted_class is None or predicted_class not in fd["hypothesis_set"]
        row = {
            "id": fd["id"],
            "benchmark": fd.get("benchmark"),
            "hypothesis_set": fd["hypothesis_set"],
            "predicted_class": predicted_class,
            "ground_truth": fd.get("ground_truth"),
            "parse_failed": parse_failed,
        }
        if extras:
            row.update(extras)
        return row
