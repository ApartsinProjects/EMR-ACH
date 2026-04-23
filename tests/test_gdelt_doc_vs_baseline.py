"""Side-by-side acceptance test for the v2.2 GDELT DOC index (C2).

For each of 20 sampled GDELT-CAMEO FDs, retrieve the top-10 articles via
``scripts/query_gdelt_doc_index.py`` and compare them against the v2.1
parent-pool retrievals already pinned to the FD via ``article_ids``.
The acceptance bar is loose: at least 3 of the 10 DOC-index hits must
overlap (by canonical URL) with the parent-pool retrievals. The test
serves as scaffolding; it SKIPS cleanly until the v2.2 GDELT DOC index
is built locally at ``data/gdelt_doc/index/`` (it is not built in the
v2.1 tree).
"""

from __future__ import annotations

import importlib.util as _iu
import json
import random
from pathlib import Path
from urllib.parse import urlparse

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_INDEX_DIR = REPO_ROOT / "data" / "gdelt_doc" / "index"
PARENT_BUNDLE = REPO_ROOT / "benchmark" / "data" / "2026-01-01"


def _canonical_url(url: str) -> str:
    """Lower-case host, drop scheme + query + fragment + trailing slash.

    Matches the canonicalization the v2.1 unifier uses for dedupe.
    """
    if not url:
        return ""
    p = urlparse(url)
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (p.path or "").rstrip("/")
    return f"{host}{path}"


def _load_query_module():
    spec = _iu.spec_from_file_location(
        "_qgdi_acc",
        REPO_ROOT / "scripts" / "query_gdelt_doc_index.py",
    )
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_parent_bundle():
    fds_path = PARENT_BUNDLE / "forecasts.jsonl"
    arts_path = PARENT_BUNDLE / "articles.jsonl"
    if not fds_path.exists() or not arts_path.exists():
        pytest.skip(
            f"v2.1 parent bundle missing at {PARENT_BUNDLE}; "
            "this test exercises the v2.2 DOC index against it."
        )
    fds = [json.loads(l) for l in fds_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    arts = {a["id"]: a for a in (
        json.loads(l) for l in arts_path.read_text(encoding="utf-8").splitlines() if l.strip()
    )}
    return fds, arts


def test_doc_index_overlaps_parent_pool_for_gdelt_fds():
    if not DOC_INDEX_DIR.exists():
        pytest.skip(
            f"GDELT DOC index not built at {DOC_INDEX_DIR}; v2.2 [A2] "
            "scripts/build_gdelt_doc_index.py has not been run locally."
        )

    qmod = _load_query_module()
    if not hasattr(qmod, "query_index"):
        pytest.skip(
            "query_gdelt_doc_index.py does not yet expose query_index(); "
            "the v2.2 [A3] FAISS query path is still behind --dry-run."
        )

    fds, arts_by_id = _load_parent_bundle()
    gdelt_fds = [fd for fd in fds if fd.get("benchmark") == "gdelt_cameo"]
    if len(gdelt_fds) < 20:
        pytest.skip(
            f"Only {len(gdelt_fds)} GDELT-CAMEO FDs in parent bundle; "
            "test wants >=20 to sample from."
        )

    rng = random.Random(20260423)
    sample = rng.sample(gdelt_fds, 20)

    failures = []
    for fd in sample:
        parent_urls = {
            _canonical_url(arts_by_id[aid]["url"])
            for aid in fd.get("article_ids", [])
            if aid in arts_by_id and arts_by_id[aid].get("url")
        }
        if not parent_urls:
            continue
        try:
            doc_hits = qmod.query_index(fd, top_k=10)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - fail clearly
            pytest.fail(f"query_index raised on fd={fd.get('id')}: {exc!r}")
        doc_urls = {_canonical_url(h.get("url", "")) for h in doc_hits}
        overlap = len(parent_urls & doc_urls)
        if overlap < 3:
            failures.append((fd.get("id"), overlap))

    # Allow a small minority to fall short (long-tail FDs with very narrow
    # parent pools); flag if more than 1/4 of the sample misses the bar.
    if len(failures) > 5:
        pytest.fail(
            f"{len(failures)}/20 sampled GDELT FDs had <3 URL overlaps "
            f"between DOC index and parent pool: {failures[:5]}"
        )
