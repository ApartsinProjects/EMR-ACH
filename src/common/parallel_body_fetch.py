"""src.common.parallel_body_fetch: ThreadPoolExecutor wrapper for
trafilatura body extraction (v2.2 [A5]).

Wraps the single-URL body-fetch function callers already have in
``scripts/fetch_article_text.py`` / ``scripts/retry_article_text.py`` in
a :class:`concurrent.futures.ThreadPoolExecutor` loop with a bounded
queue, per-URL timeout, and graceful tolerance of per-URL failures.

The single-threaded path is the second hottest wall-clock cost after
Google News (per V2_2_END_TO_END_AUDIT.md); a bounded-parallel rewrite
cuts the fetch stage from hours to minutes on a 218k-article pool
without going over API rate limits.

Callers inject the per-URL fetch callable at construction time so this
module has no import-time dependency on trafilatura. Example:

    from src.common.parallel_body_fetch import fetch_bodies_parallel

    results = fetch_bodies_parallel(
        urls, fetch_fn=my_trafilatura_fetch, max_workers=24, timeout=20,
    )

Default worker count is 24 (the v2.1 observed sweet spot).
"""

from __future__ import annotations

import concurrent.futures as _cf
from dataclasses import dataclass
from typing import Callable, Iterable

__all__ = [
    "BodyFetchResult",
    "fetch_bodies_parallel",
    "DEFAULT_WORKERS",
]

DEFAULT_WORKERS = 24


@dataclass(frozen=True)
class BodyFetchResult:
    url: str
    body: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.body is not None


def _run_one(
    url: str, fetch_fn: Callable[[str], str | None]
) -> BodyFetchResult:
    try:
        body = fetch_fn(url)
    except Exception as exc:  # noqa: BLE001  deliberate broad catch
        return BodyFetchResult(url=url, body=None, error=type(exc).__name__ + ": " + str(exc))
    return BodyFetchResult(url=url, body=body, error=None)


def fetch_bodies_parallel(
    urls: Iterable[str],
    *,
    fetch_fn: Callable[[str], str | None],
    max_workers: int = DEFAULT_WORKERS,
    per_url_timeout: float | None = 20.0,
) -> list[BodyFetchResult]:
    """Run ``fetch_fn`` across ``urls`` in a thread pool and return
    results in input order. Per-URL exceptions are captured as
    ``BodyFetchResult(error=...)`` rather than raised; the caller
    inspects ``.ok`` to filter.

    ``per_url_timeout`` is applied via ``future.result(timeout=...)``;
    timed-out URLs come back with a ``TimeoutError`` entry.
    """
    url_list = list(urls)
    results: list[BodyFetchResult | None] = [None] * len(url_list)
    if not url_list:
        return []

    with _cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_run_one, url, fetch_fn): i
            for i, url in enumerate(url_list)
        }
        for fut in _cf.as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                results[i] = fut.result(timeout=per_url_timeout)
            except _cf.TimeoutError:
                results[i] = BodyFetchResult(
                    url=url_list[i], body=None, error="TimeoutError"
                )
            except Exception as exc:  # noqa: BLE001
                results[i] = BodyFetchResult(
                    url=url_list[i], body=None,
                    error=type(exc).__name__ + ": " + str(exc),
                )
    # Every slot must be filled by now.
    return [r for r in results if r is not None]
