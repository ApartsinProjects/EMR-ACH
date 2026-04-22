"""
B7: Reflexion. Self-critique loop of K iterations.

Round 0: initial CoT forecast.
Round r (odd): critique the previous forecast.
Round r (even, >0): revise based on critique.

Rounds alternate; final prediction is taken from the last revision round.
Runner iterates 2*K - 1 rounds; we collapse critique+revise as pairs.

For simplicity here we run K iterations where each iteration is "revise given
previous forecast and self-critique text produced inline". One request per FD
per iteration.
"""

from __future__ import annotations

from .._shim import BatchResult

from ..base import Baseline
from ..prompts import INSTRUCTIONS_COT, INSTRUCTIONS_REFLEXION_CRITIC, SYSTEM_CRITIC


class B7Reflexion(Baseline):
    name = "b7_reflexion"
    multi_round = True

    @property
    def n_iterations(self) -> int:
        return int(self.cfg.get("n_iterations", 3))

    def build_requests(self, fds, articles):
        # Iteration 0: initial forecast
        reqs = []
        for fd in fds:
            user = self.render_user(fd, articles, instructions=INSTRUCTIONS_COT)
            reqs.append(self.make_request(
                custom_id=f"{fd['id']}::{self.name}::i0::forecast",
                user=user,
            ))
        return reqs

    def build_requests_round(self, i: int, fds, articles, prior: dict):
        """Iteration i: critique + revise (submitted as two requests per FD)."""
        reqs = []
        for fd in fds:
            prev_id = f"{fd['id']}::{self.name}::i{i-1}::forecast"
            prev_res = prior.get(prev_id)
            prev_content = prev_res.content if isinstance(prev_res, BatchResult) else ""

            critic_user = (
                f"Forecast to critique:\n{prev_content[:1500]}\n\n"
                f"{INSTRUCTIONS_REFLEXION_CRITIC}"
            )
            reqs.append(self.make_request(
                custom_id=f"{fd['id']}::{self.name}::i{i}::critique",
                user=critic_user,
                system=SYSTEM_CRITIC,
            ))

            revise_user = self.render_user(
                fd, articles,
                instructions=(
                    f"Prior forecast:\n{prev_content[:1000]}\n\n"
                    f"A critic has flagged weaknesses (see 'critique' message history). "
                    f"Produce an improved forecast addressing those weaknesses. "
                    + INSTRUCTIONS_COT
                ),
            )
            reqs.append(self.make_request(
                custom_id=f"{fd['id']}::{self.name}::i{i}::forecast",
                user=revise_user,
            ))
        return reqs

    def parse_responses(self, results, fds):
        preds = []
        final_i = self.n_iterations - 1
        for fd in fds:
            res = results.get(f"{fd['id']}::{self.name}::i{final_i}::forecast")
            content = res.content if isinstance(res, BatchResult) else ""
            pick = self.parse_pick(content, fd["hypothesis_set"])
            preds.append(self.prediction_row(
                fd, pick, extras={"n_iterations": self.n_iterations},
            ))
        return preds
