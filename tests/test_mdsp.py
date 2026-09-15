"""Tests for mdsp."""

import mdsp


def test_add():
    """Test add function."""
    assert mdsp.add(1, 2) == 3
    assert mdsp.add(-1, 1) == 0
    assert mdsp.add(0, 0) == 0


def test_greet():
    """Test greet function."""
    assert mdsp.greet("World") == "Hello, World!"
    assert mdsp.greet("Python") == "Hello, Python!"


def test_has_no_runtime_dependencies():
    """The distribution must not declare runtime dependencies."""
    from importlib.metadata import requires

    assert not (requires("mdsp") or [])
