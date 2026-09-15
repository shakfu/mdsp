"""Recursive filters."""

from __future__ import annotations

import math
from typing import Literal

from mdsp import _core
from mdsp._base import Param, Processor, _Unit, below_nyquist

__all__ = ["Biquad", "BiquadMode", "OnePole"]

BiquadMode = Literal["lowpass", "highpass", "bandpass", "notch"]
_MODES: tuple[BiquadMode, ...] = ("lowpass", "highpass", "bandpass", "notch")


def _positive(unit: _Unit, value: float) -> None:
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")


class OnePole(Processor):
    """One-pole lowpass. *cutoff* in Hz sets ``a = 1 - exp(-2 pi cutoff / sr)``."""

    _kernel = _core.OnePole
    cutoff = Param(0, below_nyquist)

    def __init__(
        self,
        cutoff: float = 1000.0,
        *,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(sample_rate, channels)
        self.cutoff = cutoff


class Biquad(Processor):
    """RBJ cookbook biquad.

    Args:
        mode: Filter response. ``bandpass`` has 0 dB gain at *cutoff*.
        cutoff: Cutoff or centre frequency in Hz.
        q: Quality factor. ``1/sqrt(2)`` gives a Butterworth low- or highpass.
    """

    _kernel = _core.Biquad
    cutoff = Param(1, below_nyquist)
    q = Param(2, _positive)

    def __init__(
        self,
        mode: BiquadMode = "lowpass",
        cutoff: float = 1000.0,
        q: float = 1 / math.sqrt(2),
        *,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(sample_rate, channels)
        self.mode = mode
        self.cutoff = cutoff
        self.q = q

    @property
    def mode(self) -> BiquadMode:
        return _MODES[int(self._params["mode"])]

    @mode.setter
    def mode(self, value: BiquadMode) -> None:
        if value not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {value!r}")
        index = float(_MODES.index(value))
        self._impl.set(0, index)
        self._params["mode"] = index
