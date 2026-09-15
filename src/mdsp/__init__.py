"""
mdsp - a pure-Python package with no runtime dependencies.

Example usage:
    >>> import mdsp
    >>> mdsp.add(1, 2)
    3
    >>> mdsp.greet("World")
    'Hello, World!'
"""

from mdsp.core import add, greet

__all__ = ["add", "greet"]
__version__ = "0.1.0"
