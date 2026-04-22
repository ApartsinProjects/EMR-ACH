"""B3: RAG-Only. Concat provided articles + single prediction call."""

from __future__ import annotations

from .._shim import BatchResult

from ..base import Baseline
from ..prompts import INSTRUCTIONS_RAG


class B3RAG(Baseline):
    name = "b3_rag"

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
