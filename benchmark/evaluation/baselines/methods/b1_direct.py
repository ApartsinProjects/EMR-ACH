"""B1: Direct Prompting. No articles, single call, one prediction per FD."""

from __future__ import annotations

from .._shim import BatchRequest, BatchResult

from ..base import Baseline
from ..prompts import INSTRUCTIONS_DIRECT, render_user


class B1Direct(Baseline):
    name = "b1_direct"

    def build_requests(self, fds, articles):
        reqs: list[BatchRequest] = []
        for fd in fds:
            user = render_user(fd, articles_block="(not provided; use general knowledge)",
                               instructions=INSTRUCTIONS_DIRECT)
            reqs.append(self.make_request(custom_id=f"{fd['id']}::{self.name}", user=user))
        return reqs

    def parse_responses(self, results, fds):
        preds: list[dict] = []
        for fd in fds:
            res = results.get(f"{fd['id']}::{self.name}")
            content = res.content if isinstance(res, BatchResult) else ""
            pick = self.parse_pick(content, fd["hypothesis_set"])
            preds.append(self.prediction_row(fd, pick))
        return preds
