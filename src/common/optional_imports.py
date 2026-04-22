"""Deferred optional-import helper.

Replaces the common anti-pattern:

    try:
        import yfinance as yf
    except ImportError:
        yf = None

    # Later, somewhere deep in a hot loop:
    items = yf.Ticker(symbol).news    # <-- AttributeError: NoneType has no attribute 'Ticker'

The real error ("yfinance is not installed") is buried and the traceback points
at the wrong line. With this helper:

    yf = optional("yfinance", install_hint="pip install yfinance")

    items = yf.Ticker(symbol).news    # <-- clear: ImportError: yfinance required for
                                     #     this path. Install with: pip install yfinance

A `MissingOptional` is raised only on first *access* — scripts that don't touch
the optional path (e.g. `--source gdelt` when yfinance isn't installed) run fine.
"""
from __future__ import annotations

import importlib
from typing import Any


class MissingOptional(ImportError):
    """Raised when an optional dependency is used but not installed."""


class _LazyOptional:
    """Proxy object that imports the target module on first attribute access.
    If the module is missing, raises MissingOptional with a helpful hint."""

    __slots__ = ("_name", "_hint", "_mod")

    def __init__(self, name: str, install_hint: str):
        self._name = name
        self._hint = install_hint
        self._mod: Any = None

    def _resolve(self) -> Any:
        if self._mod is None:
            try:
                self._mod = importlib.import_module(self._name)
            except ImportError as e:
                raise MissingOptional(
                    f"{self._name!r} is required for this code path but not installed. "
                    f"Install with: {self._hint}. Original import error: {e}"
                ) from e
        return self._mod

    def __getattr__(self, item: str) -> Any:
        return getattr(self._resolve(), item)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    def __bool__(self) -> bool:
        """`if yf:` returns True only if the module is actually importable.
        Allows `if module: ... else: skip` patterns without triggering errors.
        """
        try:
            self._resolve()
            return True
        except MissingOptional:
            return False


def optional(name: str, install_hint: str | None = None) -> _LazyOptional:
    """Return a lazy proxy for an optional module.

    Args:
        name:         module to import (e.g. "yfinance").
        install_hint: optional pip command (defaults to "pip install {name}").

    Example:
        yf = optional("yfinance")
        if yf:                           # cheap probe, no import yet
            items = yf.Ticker("AAPL").news   # real import happens here
    """
    return _LazyOptional(name, install_hint or f"pip install {name}")


def require(name: str, install_hint: str | None = None) -> Any:
    """Eager version: import now, raise MissingOptional on failure.
    Use when the caller will always need the module."""
    proxy = optional(name, install_hint)
    return proxy._resolve()
