"""Explicit process-local cache override; legacy locations remain the default."""

import os
from pathlib import Path

_cache_root: Path | None = None


def configure_cache(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Configure before constructing splitters or features; None resets the root."""
    global _cache_root
    _cache_root = (
        Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    )
    return get_cache_root()


def get_cache_root() -> Path:
    """Return the explicit root, or the original package-relative location."""
    return (
        _cache_root if _cache_root is not None else Path(__file__).resolve().parents[1]
    )


def validate_cache_component(value: str) -> str:
    """Keep custom namespaces inside their root without changing legacy names."""
    if _cache_root is not None and (
        not value.strip()
        or value in {".", ".."}
        or any(char in value for char in ("/", "\\", "\0"))
    ):
        raise ValueError("Cache namespace must be one nonempty path component")
    return value
