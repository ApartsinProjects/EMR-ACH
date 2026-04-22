"""B8: DEPRECATED under pick-only framing (2026-04-21).

Verbalized Confidence no longer applies when systems emit a single hypothesis
label (no probabilities to calibrate). Kept as a thin pick-only RAG variant for
backward compatibility with the runner; results should be read as identical in
spirit to B3. The paper replaces B8 with a majority-class reference baseline.
"""

from __future__ import annotations

from .._shim import BatchResult

from ..base import Baseline
from ..prompts import INSTRUCTIONS_RAG


class B8VerbalizedConfidence(Baseline):
    name = "b8_verbalized_confidence"

    def build_requests(self, fds, articles):
        reqs = []
        for fd in fds:
            user = self.render_user(fd, articles, instructions=INSTRUCTIONS_RAG)
            reqs.append(self.make_request(custom_id=f"{fd['id']}::{self.name}", user=user))
        return reqs

    def parse_responses(self, results, fds):
        preds = []
        for fd in fds:
            res = results.get(f"{fd['id']}::{self.name}")
            content = res.content if isinstance(res, BatchResult) else ""
            pick = self.parse_pick(content, fd["hypothesis_set"])
            preds.append(self.prediction_row(fd, pick))
        return preds
