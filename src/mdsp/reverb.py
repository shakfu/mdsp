"""Reverberation."""

from __future__ import annotations

from mdsp import _core
from mdsp._base import Param, Processor, _Unit

__all__ = ["Reverb"]


def _unit_range(unit: _Unit, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"value must be in [0, 1], got {value}")


class Reverb(Processor):
    """Schroeder reverb in the Freeverb arrangement.

    Eight damped comb filters feed four allpass filters. Delay lengths scale
    with the sample rate, so the reverb time does not change with it.

    Args:
        room_size: How long the tail rings, 0 to 1.
        damping: How quickly its high frequencies fade, 0 to 1.
        mix: Blend of dry and reverberated signal, 0 to 1.

    Tuning from Jezar at Dreampoint's public-domain Freeverb.
    """

    _kernel = _core.Reverb
    room_size = Param(0, _unit_range)
    damping = Param(1, _unit_range)
    mix = Param(2, _unit_range)

    def __init__(
        self,
        room_size: float = 0.5,
        damping: float = 0.5,
        mix: float = 0.3,
        *,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(
            sample_rate, channels, room_size=room_size, damping=damping, mix=mix
        )
