"""
ForecastBench data loader (geopolitics/conflict subset).

Expected file: data/forecastbench_geopolitics.jsonl
See data/README.md for format.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.config import get_config

# Binary hypotheses for ForecastBench adaptation
FB_HYPOTHESES = ["Yes", "No"]


@dataclass
class ForecastBenchQuery:
    id: str
    question: str
    resolution_date: str
    ground_truth: int           # 1 = Yes resolved, 0 = No resolved
    crowd_probability: float    # aggregate crowd probability of "Yes"
    category: str = "geopolitics"

    @property
    def label(self) -> str:
        return "Yes" if self.ground_truth == 1 else "No"

    @property
    def label_index(self) -> int:
        return 0 if self.ground_truth == 1 else 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "resolution_date": self.resolution_date,
            "ground_truth": self.ground_truth,
            "crowd_probability": self.crowd_probability,
            "category": self.category,
        }


class ForecastBenchDataset:
    def __init__(self, config=None):
        self.cfg = config or get_config()
        self._queries: list[ForecastBenchQuery] | None = None

    # GPT-4o training cutoff (approximate); questions resolving after this are leakage-free
    MODEL_CUTOFF = "2024-04-01"

    def queries(
        self,
        n: int | None = None,
        post_cutoff_only: bool = True,
    ) -> list[ForecastBenchQuery]:
        if self._queries is None:
            self._queries = self._load()
        qs = self._queries
        if post_cutoff_only:
            qs = [q for q in qs if q.resolution_date >= self.MODEL_CUTOFF]
        if n is not None:
            qs = qs[:n]
        return qs

    def queries_all(self, n: int | None = None) -> list[ForecastBenchQuery]:
        """Return all questions including pre-cutoff (use only for diagnostics)."""
        return self.queries(n=n, post_cutoff_only=False)

    def __len__(self) -> int:
        return len(self.queries())

    def __iter__(self) -> Iterator[ForecastBenchQuery]:
        return iter(self.queries())

    def crowd_brier_score(self) -> float:
        """Baseline Brier score from crowd forecasters."""
        qs = self.queries()
        return sum((q.crowd_probability - q.ground_truth) ** 2 for q in qs) / len(qs)

    def _load(self) -> list[ForecastBenchQuery]:
        path = self.cfg.forecastbench_path
        if not path.exists():
            raise FileNotFoundError(
                f"ForecastBench data not found at {path}.\n"
                "See data/README.md for download instructions."
            )
        queries = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                queries.append(ForecastBenchQuery(
                    id=obj["id"],
                    question=obj["question"],
                    resolution_date=obj["resolution_date"],
                    ground_truth=int(obj["ground_truth"]),
                    crowd_probability=float(obj["crowd_probability"]),
                    category=obj.get("category", "geopolitics"),
                ))
        print(f"[forecastbench] Loaded {len(queries)} queries.")
        return queries


def make_mock_fb_queries(n: int = 5) -> list[ForecastBenchQuery]:
    templates = [
        ("Will Russia launch new military operations in Ukraine in the next 30 days?", 1, 0.65),
        ("Will US-China relations improve significantly by end of 2024?", 0, 0.18),
        ("Will a ceasefire be reached in Gaza by Dec 31, 2024?", 0, 0.35),
        ("Will India and Pakistan engage in diplomatic talks this quarter?", 1, 0.55),
        ("Will new sanctions be imposed on Iran before year-end?", 1, 0.72),
    ]
    return [
        ForecastBenchQuery(
            id=f"mock_fb_{i:03d}",
            question=q,
            resolution_date="2024-12-31",
            ground_truth=gt,
            crowd_probability=cp,
        )
        for i, (q, gt, cp) in enumerate(templates[:n])
    ]


if __name__ == "__main__":
    cfg = get_config()
    if cfg.forecastbench_path.exists():
        ds = ForecastBenchDataset()
        print(f"Loaded {len(ds)} ForecastBench queries")
        print(f"Crowd Brier score: {ds.crowd_brier_score():.4f}")
    else:
        print("Real data not available. Mock queries:")
        for q in make_mock_fb_queries():
            print(" ", q)
