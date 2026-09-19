"""Offline time stretching."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from mdsp import _core
from mdsp._base import check_planar
from mdsp.buffer import AudioBuffer

__all__ = ["MAX_FACTOR", "MIN_FACTOR", "Mode", "timestretch"]

Mode = Literal["wsola", "vocoder"]
MODES: tuple[Mode, ...] = ("wsola", "vocoder")

#: Shortest and longest a buffer may be stretched to.
MIN_FACTOR = 0.1
MAX_FACTOR = 10.0


def timestretch(
    buf: AudioBuffer,
    factor: float,
    *,
    window: float = 0.03,
    search: float | None = None,
    mode: Mode = "wsola",
) -> AudioBuffer:
    """Stretch *buf* to *factor* times its length, keeping its pitch.

    Two algorithms, both overlap-add, differing in how they keep overlapping
    frames coherent:

    ``wsola`` moves the splice. Each segment is taken from wherever it best
    continues the one before, so periodic material does not cancel. It keeps
    transients intact and goes rough on dense or polyphonic material.

    ``vocoder`` moves the phase. Frames are transformed, every bin is advanced
    by the frequency it actually holds rather than the one its bin stands for,
    and they are transformed back. It holds pitch on tonal and polyphonic
    material and smears transients, which is the opposite trade.

    Either way channel 0 decides for all channels, which holds the stereo
    image together.

    Offline only. The output has a different length from the input, so this
    cannot be a `Processor` or a `Graph` node; for stretching in real time,
    play a `Player` at ``1 / factor`` and correct the pitch with `Shift`.

    Args:
        buf: Source. Its samples are read, not modified.
        factor: Output length over input length, in [0.1, 10]. 2 is twice as
            long at the same pitch.
        window: Segment length in seconds. Long windows smear transients;
            short ones cannot hold low frequencies. 0.02 to 0.05 is usual.
        search: Seconds either side of the ideal splice to look for a better
            one. Defaults to a quarter window. 0 skips the search, which is
            plain overlap-add: faster, and phasey on tonal material. Ignored
            by ``vocoder``.
        mode: ``wsola`` or ``vocoder``. ``vocoder`` rounds *window* up to a
            power of two, since it transforms it.

    Returns:
        A new buffer of ``round(buf.frames * factor)`` frames.

    Raises:
        ValueError: If *factor*, *window* or *search* is out of range, or
            *mode* is not a known one.

    >>> from mdsp import Sine
    >>> longer = timestretch(Sine(440.0).generate(48000), 1.5)
    >>> longer.frames, longer.sample_rate
    (72000, 48000.0)
    """
    value = float(factor)
    if not math.isfinite(value) or not MIN_FACTOR <= value <= MAX_FACTOR:
        raise ValueError(
            f"factor must be in [{MIN_FACTOR}, {MAX_FACTOR}], got {factor}"
        )
    if not math.isfinite(window) or not 0.0 < window <= 1.0:
        raise ValueError(f"window must be in (0, 1] seconds, got {window}")
    reach = window / 4 if search is None else float(search)
    if not math.isfinite(reach) or reach < 0.0:
        raise ValueError(f"search must not be negative, got {search}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    # An even window: the hop is half of it, and periodic Hann at 50% overlap
    # sums to 1.
    samples = max(2, 2 * round(window * buf.sample_rate / 2))
    frames_out = max(1, round(buf.frames * value))
    out = AudioBuffer(
        np.zeros((buf.channels, frames_out), np.float32), buf.sample_rate, copy=False
    )
    check_planar([buf.data], (buf.channels, buf.frames))
    shape = (buf.channels, buf.frames, frames_out)
    if mode == "vocoder":
        _core.vocoder(buf.address, out.address, shape, samples)
    else:
        _core.timestretch(
            buf.address, out.address, shape, samples, round(reach * buf.sample_rate)
        )
    return out
