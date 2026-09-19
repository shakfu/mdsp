"""Chorus and flanging."""

from __future__ import annotations

from mdsp import _core
from mdsp._base import Param, Processor, _Unit

__all__ = ["MAX_DELAY_SECONDS", "MAX_VOICES", "Chorus"]

#: Base delay plus depth cannot exceed this; matches `_mojo/dsp/chorus.mojo`.
MAX_DELAY_SECONDS = 0.1

#: Most taps one `Chorus` reads from its line.
MAX_VOICES = 4


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


def _zero_to_one(unit: _Unit, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"value must be in [0, 1], got {value}")


def _voice_count(unit: _Unit, value: float) -> None:
    if value != int(value) or not 1 <= value <= MAX_VOICES:
        raise ValueError(f"voices must be a whole number in [1, {MAX_VOICES}]")


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
        voices: Taps per channel, up to `MAX_VOICES`, spaced evenly around the
            LFO cycle and averaged. More taps thicken the sound; 1 flanges.
        spread: Phase offset between channels, in [0, 1] of an LFO cycle
            across the whole unit. 0 sweeps every channel together; with two
            channels, 1 puts them in antiphase, which is what widens the image.

    One LFO drives every channel and voice, so a stereo `Chorus` is one moving
    image rather than two. Modulation input: ``rate`` in Hz, read from the
    first channel.

    >>> wide = Chorus(rate=0.8, voices=3, spread=1.0, channels=2)
    >>> wide.voices, wide.spread
    (3.0, 1.0)
    """

    _kernel = _core.Chorus
    rate = Param(0, _non_negative)
    depth = Param(1, _non_negative)
    delay = Param(2, _within_line)
    feedback = Param(3, _unit_interval)
    mix = Param(4)
    voices = Param(5, _voice_count)
    spread = Param(6, _zero_to_one)

    def __init__(
        self,
        rate: float = 0.5,
        depth: float = 0.002,
        delay: float = 0.01,
        feedback: float = 0.0,
        mix: float = 0.5,
        voices: int = 1,
        spread: float = 0.0,
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
            voices=voices,
            spread=spread,
        )
