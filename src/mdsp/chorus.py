"""Chorus and flanging."""

from __future__ import annotations

from mdsp import _core
from mdsp._base import Param, Processor, _Unit

__all__ = ["MAX_DELAY_SECONDS", "Chorus"]

#: Base delay plus depth cannot exceed this; matches `_mojo/dsp/chorus.mojo`.
MAX_DELAY_SECONDS = 0.1


def _non_negative(unit: _Unit, value: float) -> None:
    if value < 0:
        raise ValueError(f"value must not be negative, got {value}")


def _within_line(unit: _Unit, value: float) -> None:
    if not 0.0 < value <= MAX_DELAY_SECONDS:
        raise ValueError(
            f"value must be in (0, {MAX_DELAY_SECONDS}] seconds, got {value}"
        )


def _unit_interval(unit: _Unit, value: float) -> None:
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"value must be in [-1, 1], got {value}")


class Chorus(Processor):
    """Delay line swept by its own LFO.

    A short *delay* with shallow *depth* flanges; a longer one with more depth
    choruses. Sweeping the delay shifts pitch while it moves, which is where
    the effect comes from.

    Args:
        rate: Sweep frequency in Hz.
        depth: How far the delay swings, in seconds.
        delay: Delay the sweep is centred on, in seconds.
        feedback: Delayed signal fed back into the line, in [-1, 1].
        mix: 0 is dry, 1 is swept signal only.

    Modulation input: ``rate`` in Hz.
    """

    _kernel = _core.Chorus
    rate = Param(0, _non_negative)
    depth = Param(1, _non_negative)
    delay = Param(2, _within_line)
    feedback = Param(3, _unit_interval)
    mix = Param(4)

    def __init__(
        self,
        rate: float = 0.5,
        depth: float = 0.002,
        delay: float = 0.01,
        feedback: float = 0.0,
        mix: float = 0.5,
        *,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(
            sample_rate,
            channels,
            rate=rate,
            depth=depth,
            delay=delay,
            feedback=feedback,
            mix=mix,
        )
