"""Bounded, ordered thread-pool mapping with a deadline.

``ThreadPoolExecutor.map`` creates one future per item up front, so a dataset with millions of
images costs memory before any work is done. :func:`bounded_map` pulls items lazily and keeps at
most ``window`` futures alive, yields results in input order, stops submitting at a deadline and
cancels what is still queued when the consumer stops or a call raises.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def bounded_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int,
    window: int | None = None,
    deadline: float | None = None,
) -> Iterator[R]:
    """Yield ``fn(item)`` for a prefix of ``items`` in order.

    The prefix is all of ``items`` unless ``deadline`` (a ``time.monotonic()`` value) passes first;
    callers detect that by counting the results. Exceptions from ``fn`` propagate.
    """
    if workers <= 1:
        for item in items:
            if deadline is not None and time.monotonic() >= deadline:
                return
            yield fn(item)
        return
    limit = window if window is not None else workers * 4
    pending: deque[Future[R]] = deque()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            for item in items:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                pending.append(executor.submit(fn, item))
                while len(pending) >= limit:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()
