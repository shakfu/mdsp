"""Oscillators.

All start at phase 0; negative *freq* runs the phase backwards. Every
oscillator accepts a ``freq`` modulation input in Hz.
"""

from __future__ import annotations

from mdsp import _core
from mdsp._base import Generator, Param, _Unit

__all__ = ["Phasor", "Saw", "Sine", "Square"]


def _below_nyquist(unit: _Unit, value: float) -> None:
    if not abs(value) < unit.sample_rate / 2:
        raise ValueError(f"|freq| must be below {unit.sample_rate / 2}, got {value}")


class _Oscillator(Generator):
    freq = Param(0)

    def __init__(
        self, freq: float = 440.0, *, sample_rate: float = 48000.0, channels: int = 1
    ) -> None:
        super().__init__(sample_rate, channels, freq=freq)


class Phasor(_Oscillator):
    """Ramp from 0 to 1 at *freq* Hz. Not band-limited.

    >>> Phasor(freq=12000.0).generate(4).data.tolist()
    [[0.0, 0.25, 0.5, 0.75]]
    """

    _kernel = _core.Phasor


class Sine(_Oscillator):
    """Sine at *freq* Hz."""

    _kernel = _core.Sine


class Saw(_Oscillator):
    """Band-limited (PolyBLEP) rising sawtooth from -1 to 1. ``|freq|`` < Nyquist."""

    _kernel = _core.Saw
    freq = Param(0, _below_nyquist)


class Square(_Oscillator):
    """Band-limited (PolyBLEP) square: +1, then -1. ``|freq|`` < Nyquist."""

    _kernel = _core.Square
    freq = Param(0, _below_nyquist)
