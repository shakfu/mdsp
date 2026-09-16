"""Dynamics processing."""

from __future__ import annotations

from mdsp import _core
from mdsp._base import Param, Processor, _Unit

__all__ = ["Compressor", "Limiter"]


def _positive(unit: _Unit, value: float) -> None:
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")


def _non_negative(unit: _Unit, value: float) -> None:
    if value < 0:
        raise ValueError(f"value must not be negative, got {value}")


def _at_least_one(unit: _Unit, value: float) -> None:
    if value < 1.0:
        raise ValueError(f"ratio must be at least 1, got {value}")


class Compressor(Processor):
    """Feed-forward peak compressor with a hard knee.

    Args:
        threshold: Level in dB above which gain reduction starts.
        ratio: Input-to-output ratio above the threshold; 1 is no compression.
        attack: Seconds for the reduction to deepen.
        release: Seconds for it to ease off.
        makeup: Output gain in dB, applied after compression.
        knee: Width in dB around the threshold over which compression eases in.
            0 is a hard knee.

    Modulation input ``sidechain`` drives the detector while the audio passes
    through, which is how one signal ducks another.

    >>> from mdsp import AudioBuffer
    >>> quiet = Compressor(threshold=-20.0, ratio=4.0, attack=0.0)
    >>> loud = AudioBuffer([1.0] * 8)
    >>> round(float(quiet.process(loud).data[0, -1]), 3)
    0.178
    """

    _kernel = _core.Compressor
    threshold = Param(0)
    ratio = Param(1, _at_least_one)
    attack = Param(2, _non_negative)
    release = Param(3, _non_negative)
    makeup = Param(4)
    knee = Param(5, _non_negative)

    def __init__(
        self,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack: float = 0.005,
        release: float = 0.1,
        makeup: float = 0.0,
        knee: float = 0.0,
        *,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(
            sample_rate,
            channels,
            threshold=threshold,
            ratio=ratio,
            attack=attack,
            release=release,
            makeup=makeup,
            knee=knee,
        )


class Limiter(Compressor):
    """A `Compressor` set to limit: high ratio, fast attack.

    It is the same kernel, so it also takes a ``sidechain`` input.
    """

    def __init__(
        self,
        threshold: float = -1.0,
        release: float = 0.05,
        *,
        sample_rate: float = 48000.0,
        channels: int = 1,
    ) -> None:
        super().__init__(
            threshold=threshold,
            ratio=50.0,
            attack=0.0005,
            release=release,
            sample_rate=sample_rate,
            channels=channels,
        )
