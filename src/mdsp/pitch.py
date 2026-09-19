"""Pitch shifting."""

from __future__ import annotations

from mdsp import _core
from mdsp._base import Param, Processor, _Unit

__all__ = ["MAX_WINDOW_SECONDS", "Shift"]

#: Longest window the kernel's line holds; matches `_mojo/dsp/shift.mojo`.
MAX_WINDOW_SECONDS = 0.1


def _within_line(unit: _Unit, value: float) -> None:
    if not 0.0 < value <= MAX_WINDOW_SECONDS:
        raise ValueError(
            f"value must be in (0, {MAX_WINDOW_SECONDS}] seconds, got {value}"
        )


def _unit_interval(unit: _Unit, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"value must be in [0, 1], got {value}")


def _shift_range(unit: _Unit, value: float) -> None:
    if not -24.0 <= value <= 24.0:
        raise ValueError(f"semitones must be in [-24, 24], got {value}")


class Shift(Processor):
    """Pitch shifter: two heads crossfading on one delay line.

    Length is unchanged, so this shifts pitch without stretching time, and
    formants move with the pitch. The output lags the input by up to *window*.

    Args:
        semitones: How far to shift, in [-24, 24]. 12 is an octave up.
        window: Seconds the heads take to cross the line, up to
            `MAX_WINDOW_SECONDS`. Short windows chirp, long ones smear;
            0.03 to 0.08 suits most material.
        mix: 0 is dry, 1 is shifted only. Detuning by a few cents against the
            dry signal is what thickens a sound.

    At 0 semitones the heads stop moving and their two fixed taps comb the
    input. Bypass with ``mix=0``, not with ``semitones=0``.

    Each head steps its phase by a whole window when it wraps, which is a
    whole number of input cycles only when *window* happens to hold one. The
    rest of the time the step spreads energy into sidebands spaced by the
    grain rate, ``abs(1 - ratio) / window``, so a shifted tone sits within
    that much of its target rather than on it. Pitch-synchronous or spectral
    methods are what remove this; see `timestretch` for the offline path.

    Modulation input: ``semitones``.

    >>> shift = Shift(semitones=7.0)
    >>> shift.semitones, shift.inputs
    (7.0, ('semitones',))
    """

    _kernel = _core.Shift
    semitones = Param(0, _shift_range)
    window = Param(1, _within_line)
    mix = Param(2, _unit_interval)

    def __init__(
        self,
        semitones: float = 0.0,
        window: float = 0.05,
        mix: float = 1.0,
        *,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(
            sample_rate, channels, semitones=semitones, window=window, mix=mix
        )
