"""
MIRAI benchmark data loader.

Expected files (see data/README.md for format):
  data/mirai_test_queries.jsonl
  data/mirai_articles.jsonl
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.config import get_config

HYPOTHESES = ["VC", "MC", "VK", "MK"]
HYPOTHESIS_NAMES = {
    "VC": "Verbal Cooperation",
    "MC": "Material Cooperation",
    "VK": "Verbal Conflict",
    "MK": "Material Conflict",
}
# CAMEO code ranges per category
CAMEO_RANGES = {
    "VC": range(1, 5),    # 01-04
    "MC": range(5, 9),    # 05-08
    "VK": range(9, 17),   # 09-16
    "MK": range(17, 21),  # 17-20
}


@dataclass
class MiraiQuery:
    id: str
    timestamp: str          # "YYYY-MM-DD"
    subject: str            # e.g. "Israel"
    relation: str           # e.g. "Accuse"
    object: str             # e.g. "Palestine"
    label: str              # "VC" | "MC" | "VK" | "MK"
    doc_ids: list[str] = field(default_factory=list)
    label_full: str = ""

    @property
    def query_text(self) -> str:
        return f"{self.subject} — {self.object} ({self.timestamp})"

    @property
    def label_index(self) -> int:
        return HYPOTHESES.index(self.label)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "label": self.label,
            "label_full": self.label_full,
            "doc_ids": self.doc_ids,
        }


@dataclass
class MiraiArticle:
    id: str
    title: str
    abstract: str
    text: str = ""
    date: str = ""
    source: str = ""
    country_mentions: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        """Return abstract if available, else first 2000 chars of text."""
        if self.abstract:
            return self.abstract
        return self.text[:2000]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "date": self.date,
            "source": self.source,
        }


class MiraiDataset:
    def __init__(self, config=None):
        self.cfg = config or get_config()
        self._queries: list[MiraiQuery] | None = None
        self._articles: dict[str, MiraiArticle] | None = None

    def queries(self, n: int | None = None) -> list[MiraiQuery]:
        if self._queries is None:
            self._queries = self._load_queries()
        qs = self._queries
        if n is not None:
            qs = qs[:n]
        return qs

    def articles(self) -> dict[str, MiraiArticle]:
        if self._articles is None:
            self._articles = self._load_articles()
        return self._articles

    def get_article(self, article_id: str) -> MiraiArticle | None:
        return self.articles().get(article_id)

    def get_articles_for_query(self, query: MiraiQuery) -> list[MiraiArticle]:
        arts = self.articles()
        seen: set[str] = set()
        result = []
        for did in query.doc_ids:
            if did in arts and did not in seen:
                result.append(arts[did])
                seen.add(did)
        return result

    def label_distribution(self) -> dict[str, int]:
        counts = {h: 0 for h in HYPOTHESES}
        for q in self.queries():
            counts[q.label] += 1
        return counts

    def __len__(self) -> int:
        return len(self.queries())

    def __iter__(self) -> Iterator[MiraiQuery]:
        return iter(self.queries())

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_queries(self) -> list[MiraiQuery]:
        path = self.cfg.mirai_queries_path
        if not path.exists():
            raise FileNotFoundError(
                f"MIRAI queries not found at {path}.\n"
                "See data/README.md for download instructions."
            )
        queries = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                queries.append(MiraiQuery(
                    id=obj["id"],
                    timestamp=obj["timestamp"],
                    subject=obj["subject"],
                    relation=obj.get("relation", ""),
                    object=obj["object"],
                    label=obj["label"],
                    label_full=obj.get("label_full", HYPOTHESIS_NAMES.get(obj["label"], "")),
                    doc_ids=obj.get("doc_ids", []),
                ))
        print(f"[mirai] Loaded {len(queries)} queries.")
        return queries

    def _load_articles(self) -> dict[str, MiraiArticle]:
        path = self.cfg.mirai_articles_path
        if not path.exists():
            raise FileNotFoundError(
                f"MIRAI articles not found at {path}.\n"
                "See data/README.md for download instructions."
            )
        articles = {}
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                articles[obj["id"]] = MiraiArticle(
                    id=obj["id"],
                    title=obj.get("title", ""),
                    abstract=obj.get("abstract", ""),
                    text=obj.get("text", ""),
                    date=obj.get("date", ""),
                    source=obj.get("source", ""),
                    country_mentions=obj.get("country_mentions", []),
                )
        print(f"[mirai] Loaded {len(articles)} articles.")
        return articles


# Smoke-test mock: generate synthetic queries when real data is unavailable
def make_mock_queries(n: int = 5) -> list[MiraiQuery]:
    import random
    rng = random.Random(42)
    pairs = [
        ("Israel", "Palestine"), ("USA", "Russia"), ("China", "Taiwan"),
        ("India", "Pakistan"), ("Saudi Arabia", "Iran"), ("Turkey", "Greece"),
    ]
    labels = HYPOTHESES
    queries = []
    for i in range(n):
        s, o = pairs[i % len(pairs)]
        label = labels[i % len(labels)]
        queries.append(MiraiQuery(
            id=f"mock_{i:03d}",
            timestamp=f"2023-{(i % 12)+1:02d}-15",
            subject=s,
            relation="mock",
            object=o,
            label=label,
        ))
    return queries


def make_mock_articles(n: int = 10) -> list[MiraiArticle]:
    templates = [
        "Officials from {s} and {o} held diplomatic talks today.",
        "{s} military forces conducted exercises near {o} border.",
        "{s} foreign minister issued a statement criticizing {o}.",
        "Humanitarian aid from {s} arrived in {o} territory.",
        "{s} and {o} signed a memorandum of understanding.",
        "Protests erupted in {s} over {o} government's recent actions.",
        "{s} imposed new economic sanctions on {o}.",
        "Ceasefire negotiations between {s} and {o} stalled.",
        "{s} and {o} agreed to resume trade relations.",
        "Military confrontation near {s}-{o} border reported by observers.",
    ]
    articles = []
    for i, tmpl in enumerate(templates[:n]):
        text = tmpl.format(s="CountryA", o="CountryB")
        articles.append(MiraiArticle(
            id=f"mock_art_{i:03d}",
            title=text,
            abstract=text + " More details to follow.",
            date="2023-10-15",
        ))
    return articles


if __name__ == "__main__":
    cfg = get_config()
    if cfg.mirai_queries_path.exists():
        ds = MiraiDataset()
        print(f"Loaded {len(ds)} queries")
        print("Label distribution:", ds.label_distribution())
        print("First query:", ds.queries()[0])
    else:
        print("Real data not available. Mock queries:")
        for q in make_mock_queries():
            print(" ", q)
