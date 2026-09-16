"""Oscillators.

All start at phase 0; negative *freq* runs the phase backwards. Every
oscillator accepts a ``freq`` modulation input in Hz.
"""

from __future__ import annotations

from typing import Literal

import numpy as np  # noqa: F401  (used by the Noise doctest)

from mdsp import _core
from mdsp._base import Generator, Param, _Unit

__all__ = ["Color", "Noise", "Phasor", "Saw", "Sine", "Square"]


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


Color = Literal["white", "pink"]
COLORS: tuple[Color, ...] = ("white", "pink")


class Noise(Generator):
    """Noise from an xorshift generator, white or pink.

    ``pink`` falls at about 3 dB per octave, using Paul Kellet's economy
    filter, and is quieter than ``white``: 0.17 RMS against 0.58, with peaks
    inside [-1, 1]. Channels of one unit share the *seed* and so produce the
    same samples.

    >>> first = Noise(seed=7).generate(4).data
    >>> (Noise(seed=7).generate(4).data == first).all()
    np.True_
    """

    _kernel = _core.Noise
    seed = Param(0)

    def __init__(
        self,
        seed: float = 22222,
        color: Color = "white",
        *,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(sample_rate, channels, seed=seed, color=color)

    @property
    def color(self) -> Color:
        return COLORS[int(self._params["color"])]

    @color.setter
    def color(self, value: Color) -> None:
        if value not in COLORS:
            raise ValueError(f"color must be one of {COLORS}, got {value!r}")
        index = float(COLORS.index(value))
        self._impl.set(1, index)
        self._params["color"] = index
