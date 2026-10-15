"""Structured JSON logging that follows the org logging standard."""

from __future__ import annotations

import json
import math
import sys
import uuid
from collections.abc import Mapping, MutableMapping
from typing import Any, TextIO

import structlog

from datacopilot.settings import Settings

REDACTED = "[REDACTED]"
TRUNCATED = "[TRUNCATED]"
_MAX_DEPTH = 8
_SENSITIVE = ("token", "password", "api_key", "jwt", "authorization")
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}


def _safe_str(value: object) -> str:
    """str() that can never raise: bytes are decoded, failures fall back to the type name."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        return str(value)
    except Exception:  # noqa: BLE001  # logging must never raise
        return f"<{type(value).__name__}>"


def _is_sensitive(key: object) -> bool:
    name = _safe_str(key).lower().replace("-", "_")
    return any(name == s or name.endswith(f"_{s}") for s in _SENSITIVE)


def _redact_item(item: Any, depth: int, path: frozenset[int]) -> Any:
    """Redact one list/set item; a (sensitive_key, value) pair has its value replaced."""
    if (
        isinstance(item, (list, tuple))
        and len(item) == 2
        and isinstance(item[0], (str, bytes))
        and _is_sensitive(item[0])
    ):
        return [_safe_str(item[0]), REDACTED]
    return _redact(item, depth, path)


def _redact(value: Any, depth: int = 1, path: frozenset[int] = frozenset()) -> Any:
    """Key-based redaction; deep or self-referencing containers become [TRUNCATED]."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if not isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return value
    if depth > _MAX_DEPTH or id(value) in path:
        return TRUNCATED
    path = path | {id(value)}
    if isinstance(value, Mapping):
        return {
            _safe_str(k): REDACTED if _is_sensitive(k) else _redact(v, depth + 1, path)
            for k, v in value.items()
        }
    return [_redact_item(v, depth + 1, path) for v in value]


def _redact_secrets(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    return {
        _safe_str(k): REDACTED if _is_sensitive(k) else _redact(v) for k, v in event_dict.items()
    }


def _reassert_context(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Bound context (service, env, request_id) wins over caller kwargs."""
    event_dict.update(structlog.contextvars.get_contextvars())
    return event_dict


def _uppercase_level(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    event_dict["level"] = str(event_dict["level"]).upper()
    return event_dict


_FALLBACK_FIELDS = ("@timestamp", "level", "service", "env", "request_id", "logger")


def _dumps(obj: Any, **kwargs: Any) -> str:
    # JSONRenderer already passes its own `default`; ours must win.
    kwargs.update(default=str, ensure_ascii=False, allow_nan=False)
    try:
        return json.dumps(obj, **kwargs)
    except Exception:  # noqa: BLE001  # logging must never raise
        fallback = {k: _safe_str(obj[k]) for k in _FALLBACK_FIELDS if k in obj}
        fallback["message"] = "log_serialization_failed"
        return json.dumps(fallback, ensure_ascii=False)


def configure_logging(service: str, settings: Settings, stream: TextIO | None = None) -> str:
    """Configure structlog for one CLI invocation; return its request_id."""
    request_id = str(uuid.uuid4())
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service, env=settings.env, request_id=request_id)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _reassert_context,
            structlog.processors.add_log_level,
            _uppercase_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="@timestamp"),
            structlog.processors.EventRenamer("message"),
            _redact_secrets,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(serializer=_dumps),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_LEVELS[settings.effective_log_level]),
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stderr),
        cache_logger_on_first_use=False,
    )
    return request_id


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger().bind(logger=name)  # type: ignore[no-any-return]


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()


def current_request_id() -> str | None:
    """The ``request_id`` bound for this invocation, or None when logging is not configured."""
    value = structlog.contextvars.get_contextvars().get("request_id")
    return value if isinstance(value, str) else None
