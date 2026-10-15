"""Deprecation policy as code (docs/engineering/workflow.md).

Before 1.0 a public name that is replaced keeps working for one more minor version and emits a
:class:`DeprecationWarning` that names the replacement and the version that removes it. Wrap the
old callable with :func:`deprecated`; the warning points at the caller's line.
"""

from __future__ import annotations

import functools
import re
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

__all__ = ["deprecated"]

P = ParamSpec("P")
R = TypeVar("R")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?$")


def deprecated(
    *, since: str, removal: str, replacement: str
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator: calling the function warns that ``replacement`` should be used instead.

    Args:
        since: the version that deprecated the name (``"0.2"``).
        removal: the version that removes it; it must be later than ``since``.
        replacement: the name to use instead (shown in the message).

    Raises:
        ValueError: malformed versions, or ``removal`` not after ``since``.
    """
    if not (_VERSION.fullmatch(since) and _VERSION.fullmatch(removal)):
        raise ValueError("versions look like 0.2 or 0.2.1")
    if _key(removal) <= _key(since):
        raise ValueError("removal must be a later version than since")
    if not replacement.strip():
        raise ValueError("name the replacement")

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        message = (
            f"{func.__module__}.{func.__qualname__} is deprecated since {since} and will be "
            f"removed in {removal}; use {replacement} instead"
        )

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorate


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))
