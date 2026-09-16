"""Envelopes."""

from __future__ import annotations

from mdsp import _core
from mdsp._base import Generator, Param, _Unit

__all__ = ["Adsr"]


def _non_negative(unit: _Unit, value: float) -> None:
    if value < 0:
        raise ValueError(f"value must not be negative, got {value}")


def _unit_range(unit: _Unit, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"value must be in [0, 1], got {value}")


class Adsr(Generator):
    """Attack-decay-sustain-release envelope, output in [0, 1].

    Raising *gate* above 0.5 starts the attack; lowering it starts the release.
    The ``gate`` input lets another node do that at audio rate. Segments are
    linear, and *attack*, *decay* and *release* are the seconds a full segment
    takes.

    The envelope is a control signal: multiply audio by it, for example by
    connecting it to a `Gain`'s ``gain`` input in a `Graph`.

    >>> env = Adsr(attack=0.0, decay=0.0, sustain=1.0, gate=1.0, sample_rate=4.0)
    >>> env.generate(3).data.tolist()
    [[1.0, 1.0, 1.0]]
    """

    _kernel = _core.Adsr
    attack = Param(0, _non_negative)
    decay = Param(1, _non_negative)
    sustain = Param(2, _unit_range)
    release = Param(3, _non_negative)
    gate = Param(4)

    def __init__(
        self,
        attack: float = 0.01,
        decay: float = 0.1,
        sustain: float = 0.7,
        release: float = 0.2,
        gate: float = 0.0,
        *,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(
            sample_rate,
            channels,
            attack=attack,
            decay=decay,
            sustain=sustain,
            release=release,
            gate=gate,
        )
