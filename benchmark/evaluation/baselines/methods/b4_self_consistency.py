"""B4: Self-Consistency. k CoT samples at T>0, average probabilities."""

from __future__ import annotations

import numpy as np

from .._shim import BatchResult

from ..base import Baseline
from ..prompts import INSTRUCTIONS_COT


class B4SelfConsistency(Baseline):
    name = "b4_self_consistency"

    @property
    def n_samples(self) -> int:
        return int(self.cfg.get("n_samples", 8))

    def build_requests(self, fds, articles):
        reqs = []
        for fd in fds:
            user = self.render_user(fd, articles, instructions=INSTRUCTIONS_COT)
            for k in range(self.n_samples):
                # Per-sample seed enforces genuine diversity at T>0. Without
                # this, the Batch API can dedup identical byte-for-byte
                # requests and return correlated samples — defeating the whole
                # Self-Consistency principle.
                req = self.make_request(
                    custom_id=f"{fd['id']}::{self.name}::s{k}",
                    user=user,
                    temperature=self.temperature,
                )
                # Propagate seed to the OpenAI API via BatchRequest.extra
                # (batch_client forwards extra.* into the request body).
                extra = getattr(req, "extra", None) or {}
                extra["seed"] = k
                try:
                    req.extra = extra
                except Exception:
                    pass
                reqs.append(req)
        return reqs

    def parse_responses(self, results, fds):
        preds = []
        for fd in fds:
            hs = fd["hypothesis_set"]
            picks = []
            for k in range(self.n_samples):
                res = results.get(f"{fd['id']}::{self.name}::s{k}")
                content = res.content if isinstance(res, BatchResult) else ""
                picks.append(self.parse_pick(content, hs))
            final = self.plurality(picks, hs)
            preds.append(self.prediction_row(
                fd, final,
                extras={"n_samples": self.n_samples, "sample_picks": picks},
            ))
        return preds
