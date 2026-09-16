"""Units that mix channels together.

Every other unit is mono and runs once per channel. These see all channels at
once, which is what panning and stereo width need.
"""

from __future__ import annotations

from mdsp import _core
from mdsp._base import Param, Processor, _Unit

__all__ = ["Pan", "Width"]


def _in_unit_range(unit: _Unit, value: float) -> None:
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"pan must be in [-1, 1], got {value}")


def _width_range(unit: _Unit, value: float) -> None:
    if not 0.0 <= value <= 4.0:
        raise ValueError(f"width must be in [0, 4], got {value}")


class Pan(Processor):
    """Place a signal in the stereo field: -1 left, 0 centre, +1 right.

    Gains follow a constant-power law, so a sound holds its loudness as it
    moves. Channel 0 of the input is the source. Modulation input: ``pan``.

    >>> import numpy as np
    >>> from mdsp import AudioBuffer
    >>> mono = AudioBuffer(np.array([[1.0, 1.0], [0.0, 0.0]]))
    >>> Pan(-1.0, channels=2).process(mono).data.tolist()
    [[1.0, 1.0], [0.0, 0.0]]
    """

    _kernel = _core.Pan
    pan = Param(0, _in_unit_range)

    def __init__(
        self, pan: float = 0.0, *, sample_rate: float = 48000.0, channels: int = 2
    ) -> None:
        super().__init__(sample_rate, channels, pan=pan)


class Width(Processor):
    """Widen or narrow a stereo image by scaling its side signal.

    0 collapses to mono, 1 leaves the image alone, above 1 pushes the sides
    out. Anything other than two channels passes through unchanged.
    """

    _kernel = _core.Width
    width = Param(0, _width_range)

    def __init__(
        self, width: float = 1.0, *, sample_rate: float = 48000.0, channels: int = 2
    ) -> None:
        super().__init__(sample_rate, channels, width=width)
