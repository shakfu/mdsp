"""Buffer playback at variable speed."""

from __future__ import annotations

from typing import cast

from mdsp import _core
from mdsp._base import Generator, Param, _Unit, check_planar
from mdsp.buffer import AudioBuffer

__all__ = ["MAX_RATE", "Player"]

#: Fastest playback either way; matches `_mojo/dsp/player.mojo`.
MAX_RATE = 16.0


def _within_rate(unit: _Unit, value: float) -> None:
    if not -MAX_RATE <= value <= MAX_RATE:
        raise ValueError(f"rate must be in [-{MAX_RATE}, {MAX_RATE}], got {value}")


def _non_negative(unit: _Unit, value: float) -> None:
    if value < 0:
        raise ValueError(f"value must not be negative, got {value}")


class Player(Generator):
    """Plays a buffer it owns, at any rate, forwards or backwards.

    Rate carries pitch and duration together, as tape speed does. Pair it with
    `Shift` at ``-12 * log2(rate)`` semitones to hold the pitch while the
    duration changes, which is time stretching that runs in real time.

    Args:
        buf: Source samples. They are copied, so later edits to *buf* do not
            reach the player; call `load` to replace them.
        rate: Source frames per output frame, in [-16, 16]. 0.5 is half speed
            an octave down, negative plays backwards.
        loop: Wrap at the ends instead of falling silent.
        position: Where to start, in seconds.

    Modulation input: ``rate``, read from the first channel. Sample rate and
    channel count come from *buf*. ``position`` reports the last value set,
    not where the head has reached since.

    >>> from mdsp import AudioBuffer, Sine
    >>> player = Player(Sine(440.0).generate(4800), rate=0.5)
    >>> player.duration, player.generate(9600).frames
    (0.1, 9600)
    """

    _kernel = _core.Player
    rate = Param(0, _within_rate)
    position = Param(2, _non_negative)

    def __init__(
        self,
        buf: AudioBuffer,
        rate: float = 1.0,
        loop: bool = False,
        position: float = 0.0,
    ) -> None:
        super().__init__(buf.sample_rate, buf.channels)
        self.load(buf)
        self.rate = rate
        self.loop = loop
        self.position = position

    @property
    def _player(self) -> _core.Player:
        """`_impl` narrowed: only this kernel has `load` and `frames`."""
        return cast("_core.Player", self._impl)

    def load(self, buf: AudioBuffer) -> None:
        """Replace the source. Playback restarts from the beginning.

        Raises:
            ValueError: If *buf* does not match the player's sample rate,
                channel count or layout.
        """
        self._check_buffer(buf, "buffer")
        check_planar([buf.data], (self._channels, buf.frames))
        self._player.load(buf.address, buf.frames)
        self._params["position"] = 0.0

    @property
    def loop(self) -> bool:
        return bool(self._params["loop"])

    @loop.setter
    def loop(self, value: bool) -> None:
        self._impl.set(1, float(bool(value)))
        self._params["loop"] = float(bool(value))

    @property
    def frames(self) -> int:
        """Length of the source in frames."""
        return int(self._player.frames())

    @property
    def duration(self) -> float:
        """Length of the source in seconds."""
        return self.frames / self._sample_rate
