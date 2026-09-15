"""Delay lines."""

from __future__ import annotations

from collections.abc import Callable

from mdsp import _core
from mdsp._base import Param, Processor, _Unit

__all__ = ["MAX_DELAY_SECONDS", "Delay"]

#: Upper bound on ``max_delay``; matches the clamp in ``_mojo/dsp/delay.mojo``.
MAX_DELAY_SECONDS = 600.0


def _delay_in_range(unit: _Unit, value: float) -> None:
    assert isinstance(unit, Delay)
    lo, hi = 1 / unit.sample_rate, unit.max_delay
    if not lo <= value <= hi:
        raise ValueError(f"delay must be in [{lo}, {hi}] seconds, got {value}")


def _in_range(lo: float, hi: float) -> Callable[[_Unit, float], None]:
    def check(unit: _Unit, value: float) -> None:
        if not lo <= value <= hi:
            raise ValueError(f"value must be in [{lo}, {hi}], got {value}")

    return check


class Delay(Processor):
    """Delay line with linear interpolation, feedback and dry/wet mix.

    Output is ``(1 - mix) * x + mix * d``, where ``d`` is the signal *delay*
    seconds ago; ``x + feedback * d`` is written into the line.

    Args:
        delay: Delay in seconds, from one sample up to *max_delay*.
        feedback: Gain of the delayed signal fed back into the line, in [-1, 1].
        mix: 0 is dry only, 1 is delayed signal only.
        max_delay: Line length in seconds, fixed at construction. Defaults to
            ``max(delay, 1.0)``. Memory is ``4 * max_delay * sample_rate``
            bytes per channel.

    >>> from mdsp import AudioBuffer
    >>> Delay(2 / 48000).process(AudioBuffer([1, 2, 3, 4])).data.tolist()
    [[0.0, 0.0, 1.0, 2.0]]
    """

    _kernel = _core.Delay
    delay = Param(1, _delay_in_range)
    feedback = Param(2, _in_range(-1.0, 1.0))
    mix = Param(3, _in_range(0.0, 1.0))

    def __init__(
        self,
        delay: float = 0.25,
        feedback: float = 0.0,
        mix: float = 1.0,
        *,
        max_delay: float | None = None,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(sample_rate, channels)
        length = max(float(delay), 1.0) if max_delay is None else float(max_delay)
        if not 0.0 < length <= MAX_DELAY_SECONDS:
            raise ValueError(
                f"max_delay must be in (0, {MAX_DELAY_SECONDS}] seconds, got {length}"
            )
        self._impl.set(0, length)
        self._params["max_delay"] = length
        self.delay = delay
        self.feedback = feedback
        self.mix = mix

    @property
    def max_delay(self) -> float:
        """Line length in seconds."""
        return self._params["max_delay"]
