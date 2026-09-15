"""Core functions for mdsp.

Only the standard library is used here -- this package declares no runtime
dependencies, so anything imported below must ship with CPython.
"""

from __future__ import annotations

__all__ = ["add", "greet"]


def add(a: int, b: int) -> int:
    """Return the sum of *a* and *b*."""
    return a + b


def greet(name: str) -> str:
    """Return a greeting addressed to *name*."""
    return f"Hello, {name}!"
