"""
B5: Multi-Agent Debate. n_agents x n_rounds.

Round 0: each agent produces an independent CoT forecast.
Rounds 1..R-1: each agent sees other agents' previous-round answers and revises.

For batch dispatch, all round-0 requests are submitted first (one batch), then
subsequent rounds depend on prior results. The runner calls `build_requests` for
round 0, dispatches, then calls `build_requests_round(r, ...)` for later rounds.

For simplicity and full Batch API compatibility, we submit ALL rounds' requests
up-front with placeholder context — this is equivalent to independent agents per
round, which matches Du et al. (2023) when prior-round context is included in the
user message. To support true round-dependency via batch, we model it as R
sequential batch submissions driven by `runner.py` (one batch per round).

Here `build_requests` emits only round 0, and `build_requests_round(r, prior)`
emits subsequent rounds given parsed prior-round results.
"""

from __future__ import annotations

import json

import numpy as np

from .._shim import BatchResult

from ..base import Baseline
from ..prompts import INSTRUCTIONS_DEBATE_ROUND1, INSTRUCTIONS_DEBATE_ROUNDN


class B5MultiAgentDebate(Baseline):
    name = "b5_multi_agent_debate"
    multi_round = True  # runner hint

    @property
    def n_agents(self) -> int:
        return int(self.cfg.get("n_agents", 3))

    @property
    def n_rounds(self) -> int:
        return int(self.cfg.get("n_rounds", 2))

    # ----- Round 0 -----
    def build_requests(self, fds, articles):
        reqs = []
        for fd in fds:
            base_user = self.render_user(fd, articles, instructions=INSTRUCTIONS_DEBATE_ROUND1)
            for a in range(self.n_agents):
                reqs.append(self.make_request(
                    custom_id=f"{fd['id']}::{self.name}::r0::a{a}",
                    user=base_user,
                    temperature=max(0.4, self.temperature),
                ))
        return reqs

    # ----- Subsequent rounds (called by runner in a loop) -----
    def build_requests_round(self, r: int, fds, articles, prior: dict):
        """prior: {custom_id: BatchResult} from round r-1."""
        reqs = []
        for fd in fds:
            hs = fd["hypothesis_set"]
            peer_blocks = []
            for a in range(self.n_agents):
                pid = f"{fd['id']}::{self.name}::r{r-1}::a{a}"
                res = prior.get(pid)
                content = res.content if isinstance(res, BatchResult) else ""
                peer_blocks.append(f"[Agent {a} prev round]: {content[:800]}")
            peers = "\n\n".join(peer_blocks)
            user = self.render_user(
                fd, articles,
                instructions=INSTRUCTIONS_DEBATE_ROUNDN + "\n\n" + peers,
            )
            for a in range(self.n_agents):
                reqs.append(self.make_request(
                    custom_id=f"{fd['id']}::{self.name}::r{r}::a{a}",
                    user=user,
                    temperature=max(0.4, self.temperature),
                ))
        return reqs

    def parse_responses(self, results, fds):
        preds = []
        final_round = self.n_rounds - 1
        for fd in fds:
            hs = fd["hypothesis_set"]
            picks = []
            for a in range(self.n_agents):
                pid = f"{fd['id']}::{self.name}::r{final_round}::a{a}"
                res = results.get(pid)
                content = res.content if isinstance(res, BatchResult) else ""
                picks.append(self.parse_pick(content, hs))
            final = self.plurality(picks, hs)
            preds.append(self.prediction_row(
                fd, final,
                extras={"n_agents": self.n_agents, "n_rounds": self.n_rounds,
                        "agent_picks": picks},
            ))
        return preds
