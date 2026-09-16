"""Stateless per-sample operations."""

from __future__ import annotations

from typing import Literal

from mdsp import _core
from mdsp._base import Param, Processor, _Unit


def _positive_drive(unit: _Unit, value: float) -> None:
    if value <= 0:
        raise ValueError(f"drive must be positive, got {value}")


__all__ = ["Curve", "Gain", "Mix", "Scale", "Shape", "Shaper"]


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


Curve = Literal["linear", "exponential"]
CURVES: tuple[Curve, ...] = ("linear", "exponential")


class Scale(Processor):
    """Map [-1, 1] onto [*lo*, *hi*], turning an oscillator into a modulator.

    Args:
        lo: Output for an input of -1.
        hi: Output for an input of +1.
        curve: ``exponential`` is the musical mapping for frequencies; it needs
            both bounds positive.

    >>> from mdsp import AudioBuffer
    >>> Scale(100.0, 200.0).process(AudioBuffer([-1.0, 0.0, 1.0])).data.tolist()
    [[100.0, 150.0, 200.0]]
    """

    _kernel = _core.Scale
    lo = Param(0)
    hi = Param(1)

    def __init__(
        self,
        lo: float = -1.0,
        hi: float = 1.0,
        curve: Curve = "linear",
        *,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(sample_rate, channels, lo=lo, hi=hi, curve=curve)

    @property
    def curve(self) -> Curve:
        return CURVES[int(self._params["curve"])]

    @curve.setter
    def curve(self, value: Curve) -> None:
        if value not in CURVES:
            raise ValueError(f"curve must be one of {CURVES}, got {value!r}")
        index = float(CURVES.index(value))
        self._impl.set(2, index)
        self._params["curve"] = index


class Mix(Processor):
    """Sum up to four inputs, each with its own gain.

    Inputs ``in``, ``in2``, ``in3`` and ``in4`` are all audio; unconnected ones
    contribute nothing. In a `Graph` this is how voices or effect returns are
    combined; as a unit, pass the extra inputs as buffers.

    >>> from mdsp import AudioBuffer
    >>> a, b = AudioBuffer([1.0, 1.0]), AudioBuffer([0.5, 0.5])
    >>> Mix(0.5, 0.25).process(a, in2=b).data.tolist()
    [[0.625, 0.625]]
    """

    _kernel = _core.Mix
    gain = Param(0)
    gain2 = Param(1)
    gain3 = Param(2)
    gain4 = Param(3)

    def __init__(
        self,
        gain: float = 1.0,
        gain2: float = 1.0,
        gain3: float = 1.0,
        gain4: float = 1.0,
        *,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(
            sample_rate,
            channels,
            gain=gain,
            gain2=gain2,
            gain3=gain3,
            gain4=gain4,
        )


Shape = Literal["tanh", "soft", "hard"]
SHAPES: tuple[Shape, ...] = ("tanh", "soft", "hard")


class Shaper(Processor):
    """Waveshaping distortion.

    *drive* multiplies the input before shaping, and the result is scaled so a
    full-scale input stays full-scale: *drive* changes the character, not the
    level. Modulation input: ``drive``.

    >>> from mdsp import AudioBuffer
    >>> round(float(Shaper(drive=5.0).process(AudioBuffer([1.0])).data[0, 0]), 3)
    1.0
    >>> round(float(Shaper(drive=5.0).process(AudioBuffer([0.5])).data[0, 0]), 3)
    0.987
    """

    _kernel = _core.Shaper
    drive = Param(0, _positive_drive)

    def __init__(
        self,
        drive: float = 1.0,
        shape: Shape = "tanh",
        *,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(sample_rate, channels, drive=drive, shape=shape)

    @property
    def shape(self) -> Shape:
        return SHAPES[int(self._params["shape"])]

    @shape.setter
    def shape(self, value: Shape) -> None:
        if value not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, got {value!r}")
        index = float(SHAPES.index(value))
        self._impl.set(1, index)
        self._params["shape"] = index
