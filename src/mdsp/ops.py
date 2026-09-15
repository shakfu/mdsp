"""Stateless per-sample operations."""

from __future__ import annotations

from mdsp import _core
from mdsp._base import Param, Processor

__all__ = ["Gain"]


class Gain(Processor):
    """Multiply by a linear *gain*."""

    _kernel = _core.Gain
    gain = Param(0)

    def __init__(
        self, gain: float = 1.0, *, sample_rate: float = 48000.0, channels: int = 1
    ) -> None:
        super().__init__(sample_rate, channels)
        self.gain = gain
