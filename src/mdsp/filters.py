"""Recursive filters."""

from __future__ import annotations

import math
from typing import Literal

from mdsp import _core
from mdsp._base import Param, Processor, _Unit, below_nyquist

__all__ = ["Biquad", "BiquadMode", "OnePole", "Svf"]

BiquadMode = Literal["lowpass", "highpass", "bandpass", "notch"]
_MODES: tuple[BiquadMode, ...] = ("lowpass", "highpass", "bandpass", "notch")


def _positive(unit: _Unit, value: float) -> None:
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")


class OnePole(Processor):
    """One-pole lowpass. *cutoff* in Hz sets ``a = 1 - exp(-2 pi cutoff / sr)``.

    Modulation input: ``cutoff`` in Hz.
    """

    _kernel = _core.OnePole
    cutoff = Param(0, below_nyquist)

    def __init__(
        self,
        cutoff: float = 1000.0,
        *,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(sample_rate, channels, cutoff=cutoff)


class _TwoPoleFilter(Processor):
    """Shared parameters of the two-pole filters: mode, cutoff and q."""

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
        super().__init__(sample_rate, channels, mode=mode, cutoff=cutoff, q=q)

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


class Biquad(_TwoPoleFilter):
    """RBJ cookbook biquad. No modulation input: use `Svf` for that.

    Args:
        mode: Filter response. ``bandpass`` has 0 dB gain at *cutoff*.
        cutoff: Cutoff or centre frequency in Hz.
        q: Quality factor. ``1/sqrt(2)`` gives a Butterworth low- or highpass.
    """

    _kernel = _core.Biquad


class Svf(_TwoPoleFilter):
    """Topology-preserving state-variable filter, stable under fast modulation.

    Same parameters as `Biquad`, plus a ``cutoff`` modulation input in Hz. Use
    this rather than `Biquad` when the cutoff moves; see
    ``docs/dev/spikes/2026-09-16-interface``.

    Source: A. Simper, `Linear Trapezoidal Integrated SVF
    <https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf>`_.
    """

    _kernel = _core.Svf
