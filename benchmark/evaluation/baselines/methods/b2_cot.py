"""B2: Chain-of-Thought. Same inputs as B1 but explicit CoT instructions."""

from __future__ import annotations

from .._shim import BatchRequest, BatchResult

from ..base import Baseline
from ..prompts import INSTRUCTIONS_COT, render_user


class B2CoT(Baseline):
    name = "b2_cot"

    def build_requests(self, fds, articles):
        reqs: list[BatchRequest] = []
        for fd in fds:
            user = render_user(fd, articles_block="(not provided; use general knowledge)",
                               instructions=INSTRUCTIONS_COT)
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
