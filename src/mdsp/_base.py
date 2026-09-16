"""Base classes binding Python objects to Mojo kernels in ``mdsp._core``.

Every kernel instance processes planar ``[channels, frames]`` float32 buffers
in one Mojo call with the GIL released. This module is the trust boundary:
``_core`` receives raw addresses and cannot validate them.

Instances are not thread-safe. Distinct instances may run in parallel threads.
"""

from __future__ import annotations

import math
import sysconfig
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Protocol, overload

import numpy as np
from numpy.typing import NDArray

from mdsp.buffer import AudioBuffer


def _check_interpreter(gil_disabled: bool) -> None:
    # mdsp._core segfaults on import under free-threaded CPython (tested 3.14t).
    if gil_disabled:
        raise ImportError("mdsp does not support free-threaded Python builds")


_check_interpreter(bool(sysconfig.get_config_var("Py_GIL_DISABLED")))

try:
    from mdsp import _core
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mdsp._core is not built. Run `make build` in the source tree."
    ) from exc

__all__ = [
    "Chain",
    "Generator",
    "Param",
    "Processor",
    "SupportsProcess",
    "check_planar",
]


class Param:
    """A kernel parameter, stored in Python and forwarded to ``_core``.

    Args:
        index: Position in the kernel's ``param_names()``.
        validate: Called as ``validate(unit, value)``; raises ``ValueError``.
    """

    def __init__(
        self,
        index: int,
        validate: Callable[[_Unit, float], None] | None = None,
    ) -> None:
        self.index = index
        self.validate = validate
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> Param: ...
    @overload
    def __get__(self, obj: _Unit, objtype: type | None = None) -> float: ...
    def __get__(self, obj: _Unit | None, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return obj._params[self.name]

    def __set__(self, obj: _Unit, value: float) -> None:
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"{self.name} must be finite, got {v}")
        if self.validate is not None:
            self.validate(obj, v)
        obj._impl.set(self.index, v)
        obj._params[self.name] = v


def below_nyquist(unit: _Unit, value: float) -> None:
    """Require ``0 < value < sample_rate / 2``."""
    if not 0.0 < value < unit.sample_rate / 2:
        raise ValueError(
            f"frequency must be in (0, {unit.sample_rate / 2}), got {value}"
        )


def check_planar(
    buffers: Sequence[NDArray[np.float32]], shape: tuple[int, int]
) -> None:
    """Every buffer must be C-contiguous float32 of *shape*.

    Re-checked on every call: `AudioBuffer.data` is a mutable ndarray whose
    shape and dtype can be reassigned in place.
    """
    for arr in buffers:
        if arr.dtype != np.float32 or not arr.flags.c_contiguous or arr.shape != shape:
            raise ValueError(
                f"expected C-contiguous float32 array of shape {shape}, "
                f"got {arr.dtype} {arr.shape}"
            )


class _Unit:
    _kernel: ClassVar[type[_core._Bank]]

    def __init__(
        self, sample_rate: float, channels: int, **params: float | str
    ) -> None:
        sr = float(sample_rate)
        if not (math.isfinite(sr) and sr > 0):
            raise ValueError(f"sample_rate must be positive and finite, got {sr}")
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError(f"channels must be an int >= 1, got {channels!r}")
        self._sample_rate = sr
        self._channels = channels
        self._impl = type(self)._kernel(sr, channels)
        self._params: dict[str, float] = {}
        #: Names of the modulation inputs, in the order `_core` expects them.
        self._mod_names: tuple[str, ...] = tuple(self._impl.input_names()[1:])
        for name, value in params.items():
            setattr(self, name, value)
        # Parameter changes ramp over 10 ms; starting values take effect at once.
        self._impl.reset()

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def inputs(self) -> tuple[str, ...]:
        """Names of the modulation inputs this unit accepts."""
        return self._mod_names

    def reset(self) -> None:
        """Clear internal state. Parameters are kept."""
        self._impl.reset()

    def _check_buffer(
        self, buf: AudioBuffer, name: str, frames: int | None = None
    ) -> None:
        if buf.sample_rate != self._sample_rate:
            raise ValueError(
                f"{name} sample_rate {buf.sample_rate} != {self._sample_rate}"
            )
        if buf.channels != self._channels:
            raise ValueError(
                f"{name} has {buf.channels} channels, expected {self._channels}"
            )
        if frames is not None and buf.frames != frames:
            raise ValueError(f"{name} has {buf.frames} frames, expected {frames}")

    def _run(
        self,
        src: AudioBuffer,
        dst: AudioBuffer,
        mods: dict[str, AudioBuffer] | None = None,
    ) -> None:
        """Validate every buffer, then hand `_core` their addresses."""
        mods = mods or {}
        unknown = set(mods) - set(self._mod_names)
        if unknown:
            raise TypeError(
                f"{type(self).__name__} has no modulation input(s) "
                f"{sorted(unknown)}; expected {list(self._mod_names)}"
            )
        buffers = [src.data, dst.data]
        addresses = []
        for name in self._mod_names:
            buf = mods.get(name)
            if buf is None:
                addresses.append(0)
                continue
            if buf.sample_rate != self._sample_rate:
                raise ValueError(
                    f"{name} sample_rate {buf.sample_rate} != {self._sample_rate}"
                )
            buffers.append(buf.data)
            addresses.append(buf.address)
        frames = dst.frames
        check_planar(buffers, (self._channels, frames))
        if not dst.data.flags.writeable:
            raise ValueError("output array is read-only")
        self._impl.process(src.address, dst.address, frames, addresses)

    def __repr__(self) -> str:
        params = "".join(f"{k}={v!r}, " for k, v in self._params.items())
        return (
            f"{type(self).__name__}({params}sample_rate={self._sample_rate}, "
            f"channels={self._channels})"
        )


class Processor(_Unit):
    """A unit that transforms a buffer."""

    def process(
        self, buf: AudioBuffer, out: AudioBuffer | None = None, **mods: AudioBuffer
    ) -> AudioBuffer:
        """Return the processed buffer. State carries over to the next call.

        Args:
            out: Write here instead of allocating, and return it. It must match
                *buf* in sample rate, channels and frames. Passing *buf* itself
                processes in place.
            mods: Modulation inputs by name (see ``inputs``). Each buffer must
                match *buf*, and replaces that parameter for every sample.

        Raises:
            ValueError: If a buffer has a different sample rate or shape.
            TypeError: If a keyword does not name a modulation input.
        """
        self._check_buffer(buf, "buffer")
        if out is None:
            result = AudioBuffer(np.empty_like(buf.data), self._sample_rate, copy=False)
        else:
            self._check_buffer(out, "out")
            result = out
        self._run(buf, result, mods)
        return result


class Generator(_Unit):
    """A unit that produces a buffer from its internal state."""

    def generate(
        self, frames: int, out: AudioBuffer | None = None, **mods: AudioBuffer
    ) -> AudioBuffer:
        """Return the next *frames* samples.

        Args:
            out: Write here instead of allocating, and return it. It must hold
                *frames* frames at this unit's sample rate and channel count.
            mods: Modulation inputs by name, as in `Processor.process`.
        """
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
            raise ValueError(f"frames must be an int >= 0, got {frames!r}")
        if out is None:
            result = AudioBuffer(
                np.empty((self._channels, frames), np.float32),
                self._sample_rate,
                copy=False,
            )
        else:
            self._check_buffer(out, "out", frames)
            result = out
        self._run(result, result, mods)
        return result


class SupportsProcess(Protocol):
    def process(
        self, buf: AudioBuffer, out: AudioBuffer | None = None
    ) -> AudioBuffer: ...
    def reset(self) -> None: ...


class Chain:
    """Apply processors in order.

    >>> from mdsp import Gain, OnePole
    >>> chain = Chain(OnePole(cutoff=500.0), Gain(gain=0.5))
    >>> chain.process(AudioBuffer(np.ones(4))).frames
    4
    """

    def __init__(self, *processors: SupportsProcess) -> None:
        self.processors = list(processors)
        self._scratch: AudioBuffer | None = None

    def process(self, buf: AudioBuffer, out: AudioBuffer | None = None) -> AudioBuffer:
        """Apply every processor in turn.

        Args:
            out: Write the result here instead of allocating. Stages alternate
                between *out* and one reused scratch buffer, so a steady stream
                of same-sized blocks allocates nothing after the first call.
        """
        if out is None:
            for p in self.processors:
                buf = p.process(buf)
            return buf
        stages = len(self.processors)
        if stages == 0:
            np.copyto(out.data, buf.data)
            return out
        if stages > 1 and (
            self._scratch is None
            or self._scratch.data.shape != buf.data.shape
            or self._scratch.sample_rate != buf.sample_rate
        ):
            self._scratch = AudioBuffer.zeros(buf.channels, buf.frames, buf.sample_rate)
        for index, p in enumerate(self.processors):
            # The last stage lands in `out`; earlier ones alternate with scratch.
            target = out if (stages - 1 - index) % 2 == 0 else self._scratch
            buf = p.process(buf, target)
        return buf

    def reset(self) -> None:
        for p in self.processors:
            p.reset()

    def __repr__(self) -> str:
        return f"Chain({', '.join(map(repr, self.processors))})"
