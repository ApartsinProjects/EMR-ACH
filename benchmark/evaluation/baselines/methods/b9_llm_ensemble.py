"""B9: Heterogeneous LLM Ensemble. N (model, temperature) configs, mean probabilities.

B9 is the only baseline permitted to use multiple models. The runner fairness check
allowlists b9_llm_ensemble for `model` deviations from defaults.
"""

from __future__ import annotations

import numpy as np

from .._shim import BatchResult

from ..base import Baseline
from ..prompts import INSTRUCTIONS_RAG


class B9LLMEnsemble(Baseline):
    name = "b9_llm_ensemble"
    allow_model_override = True  # runner checks this flag

    @property
    def member_configs(self) -> list[dict]:
        return list(self.cfg.get("configs", []))

    def build_requests(self, fds, articles):
        reqs = []
        for fd in fds:
            user = self.render_user(fd, articles, instructions=INSTRUCTIONS_RAG)
            for i, mc in enumerate(self.member_configs):
                reqs.append(self.make_request(
                    custom_id=f"{fd['id']}::{self.name}::m{i}",
                    user=user,
                    model=mc.get("model", self.model),
                    temperature=float(mc.get("temperature", self.temperature)),
                ))
        return reqs

    def parse_responses(self, results, fds):
        preds = []
        for fd in fds:
            hs = fd["hypothesis_set"]
            picks = []
            for i in range(len(self.member_configs)):
                res = results.get(f"{fd['id']}::{self.name}::m{i}")
                content = res.content if isinstance(res, BatchResult) else ""
                picks.append(self.parse_pick(content, hs))
            final = self.plurality(picks, hs)
            preds.append(self.prediction_row(
                fd, final,
                extras={"n_members": len(self.member_configs), "member_picks": picks},
            ))
        return preds
