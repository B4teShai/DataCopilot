"""Runtime settings read from the environment."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    SecretStr,
    ValidationError,
    field_validator,
)

Env = Literal["dev", "staging", "prod"]
Level = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

_PREFIX = "DATACOPILOT_"
DEFAULT_MAX_TABLE_ROWS = 5_000_000
DEFAULT_MAX_TABLE_BYTES = 4 * 1024**3
_FIELDS: dict[str, tuple[str, Callable[[str], str]]] = {
    "env": ("ENV", str.lower),
    "log_level": ("LOG_LEVEL", str.upper),
    "max_table_rows": ("MAX_TABLE_ROWS", str.strip),
    "max_table_bytes": ("MAX_TABLE_BYTES", str.strip),
    "workers": ("WORKERS", str.strip),
    "hash_cache": ("HASH_CACHE", lambda v: v.strip().lower()),
    "cache_dir": ("CACHE_DIR", str.strip),
    "cache_key": ("CACHE_KEY", str.strip),
}
MIN_CACHE_KEY_CHARS = 16
MAX_CACHE_KEY_CHARS = 1024
MAX_WORKERS = 256
"""Upper bound of ``DATACOPILOT_WORKERS``; a typo must not start thousands of threads."""


def _default_cache_dir(source: Mapping[str, str]) -> Path:
    base = source.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "datacopilot"
    return Path(source.get("HOME") or Path.home()) / ".cache" / "datacopilot"


class SettingsError(ValueError):
    """An environment variable holds an invalid value."""


class Settings(BaseModel, frozen=True):
    env: Env = "prod"
    log_level: Level = "INFO"
    max_table_rows: PositiveInt = DEFAULT_MAX_TABLE_ROWS
    max_table_bytes: PositiveInt = DEFAULT_MAX_TABLE_BYTES
    workers: int = Field(
        default_factory=lambda: min(os.cpu_count() or 1, MAX_WORKERS), ge=1, le=MAX_WORKERS
    )
    hash_cache: bool = True
    cache_key: SecretStr | None = None
    ci: bool = False
    cache_dir: Path = Field(default_factory=lambda: _default_cache_dir(os.environ))

    @field_validator("cache_key")
    @classmethod
    def _long_enough(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not (
            MIN_CACHE_KEY_CHARS <= len(value.get_secret_value()) <= MAX_CACHE_KEY_CHARS
        ):
            raise ValueError("invalid key length")
        return value

    @field_validator("cache_dir")
    @classmethod
    def _plausible_path(cls, value: Path) -> Path:
        if not str(value) or str(value) == "." or "\x00" in str(value):
            raise ValueError("invalid directory")
        return value

    @property
    def effective_log_level(self) -> Level:
        """DEBUG is honoured only in dev; elsewhere it is capped to INFO."""
        if self.log_level == "DEBUG" and self.env != "dev":
            return "INFO"
        return self.log_level


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    source = os.environ if environ is None else environ
    raw: dict[str, str] = {}
    for field, (suffix, normalise) in _FIELDS.items():
        value = source.get(_PREFIX + suffix)
        if value is not None:
            raw[field] = normalise(value.strip())
    raw["ci"] = source.get("CI", "").strip().lower() not in ("", "0", "false", "no")  # type: ignore[assignment]
    if "cache_dir" not in raw:
        raw["cache_dir"] = str(_default_cache_dir(source))
    try:
        return Settings(**raw)  # type: ignore[arg-type]
    except ValidationError as exc:
        names = sorted({_PREFIX + _FIELDS[str(err["loc"][0])][0] for err in exc.errors()})
        raise SettingsError(f"invalid value for {', '.join(names)}") from None
