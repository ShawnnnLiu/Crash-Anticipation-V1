"""Crash anticipation baseline package."""

from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("crash-anticipation")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.1.0"

__all__ = ["__version__"]

