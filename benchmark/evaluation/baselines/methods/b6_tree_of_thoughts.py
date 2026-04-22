"""
B6: Tree of Thoughts. Breadth B, depth D.

Structure (batch-friendly approximation):
  Depth 0: B independent "thought seeds" per FD (high temp).
  Depth 1: for each seed, B expansions that each terminate with a probability.
The final probability is the (self-score-weighted) mean over all B*B leaves.

All requests submittable in up to D rounds; runner iterates rounds.
"""

from __future__ import annotations

import numpy as np

from .._shim import BatchResult

from ..base import Baseline
from ..prompts import INSTRUCTIONS_COT, INSTRUCTIONS_TOT_NODE


class B6TreeOfThoughts(Baseline):
    name = "b6_tree_of_thoughts"
    multi_round = True

    @property
    def breadth(self) -> int:
        return int(self.cfg.get("breadth", 3))

    @property
    def depth(self) -> int:
        return int(self.cfg.get("depth", 2))

    def build_requests(self, fds, articles):
        # Depth 0: seed thoughts
        instr = INSTRUCTIONS_TOT_NODE.format(breadth=self.breadth) + "\n\n" + INSTRUCTIONS_COT
        reqs = []
        for fd in fds:
            user = self.render_user(fd, articles, instructions=instr)
            for b in range(self.breadth):
                reqs.append(self.make_request(
                    custom_id=f"{fd['id']}::{self.name}::d0::b{b}",
                    user=user,
                    temperature=max(0.7, self.temperature),
                ))
        return reqs

    def build_requests_round(self, d: int, fds, articles, prior: dict):
        reqs = []
        instr_parent = INSTRUCTIONS_COT + "\n\nBuild on the following prior thought:\n{thought}"
        for fd in fds:
            for pb in range(self.breadth):
                pid = f"{fd['id']}::{self.name}::d{d-1}::b{pb}"
                res = prior.get(pid)
                parent = res.content[:1200] if isinstance(res, BatchResult) else ""
                user = self.render_user(fd, articles, instructions=instr_parent.format(thought=parent))
                for cb in range(self.breadth):
                    reqs.append(self.make_request(
                        custom_id=f"{fd['id']}::{self.name}::d{d}::b{pb}_{cb}",
                        user=user,
                        temperature=max(0.5, self.temperature),
                    ))
        return reqs

    def parse_responses(self, results, fds):
        preds = []
        final = self.depth - 1
        for fd in fds:
            hs = fd["hypothesis_set"]
            leaf_picks = []
            if final == 0:
                for b in range(self.breadth):
                    res = results.get(f"{fd['id']}::{self.name}::d0::b{b}")
                    content = res.content if isinstance(res, BatchResult) else ""
                    leaf_picks.append(self.parse_pick(content, hs))
            else:
                for pb in range(self.breadth):
                    for cb in range(self.breadth):
                        res = results.get(f"{fd['id']}::{self.name}::d{final}::b{pb}_{cb}")
                        content = res.content if isinstance(res, BatchResult) else ""
                        leaf_picks.append(self.parse_pick(content, hs))
            final_pick = self.plurality(leaf_picks, hs)
            preds.append(self.prediction_row(
                fd, final_pick,
                extras={"breadth": self.breadth, "depth": self.depth,
                        "leaf_picks": leaf_picks},
            ))
        return preds
