"""Pluralistic Alignment package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pluralistic_alignment")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"
