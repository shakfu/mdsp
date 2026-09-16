"""Stateless per-sample operations."""

from __future__ import annotations

from mdsp import _core
from mdsp._base import Param, Processor

__all__ = ["Gain"]


class Gain(Processor):
    """Multiply by a linear *gain*. Modulation input: ``gain``.

    >>> import numpy as np
    >>> from mdsp import AudioBuffer
    >>> env = AudioBuffer([0.0, 0.5, 1.0, 0.5])
    >>> Gain().process(AudioBuffer(np.ones(4)), gain=env).data.tolist()
    [[0.0, 0.5, 1.0, 0.5]]
    """

    _kernel = _core.Gain
    gain = Param(0)

    def __init__(
        self, gain: float = 1.0, *, sample_rate: float = 48000.0, channels: int = 1
    ) -> None:
        super().__init__(sample_rate, channels, gain=gain)
