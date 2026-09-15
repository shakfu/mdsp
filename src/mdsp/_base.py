"""Base classes binding Python objects to Mojo kernels in ``mdsp._core``.

Every kernel instance processes planar ``[channels, frames]`` float32 buffers
in one Mojo call with the GIL released. This module is the trust boundary:
``_core`` receives raw addresses and cannot validate them.

Instances are not thread-safe. Distinct instances may run in parallel threads.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar, Protocol, overload

import numpy as np
from numpy.typing import NDArray

from mdsp.buffer import AudioBuffer

try:
    from mdsp import _core
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mdsp._core is not built. Run `make build` in the source tree."
    ) from exc

__all__ = ["Chain", "Generator", "Param", "Processor", "SupportsProcess"]


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


class _Unit:
    _kernel: ClassVar[type[_core._Bank]]

    def __init__(self, sample_rate: float, channels: int) -> None:
        sr = float(sample_rate)
        if not (math.isfinite(sr) and sr > 0):
            raise ValueError(f"sample_rate must be positive and finite, got {sr}")
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError(f"channels must be an int >= 1, got {channels!r}")
        self._sample_rate = sr
        self._channels = channels
        self._impl = type(self)._kernel(sr, channels)
        self._params: dict[str, float] = {}

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def reset(self) -> None:
        """Clear internal state. Parameters are kept."""
        self._impl.reset()

    def _run(self, src: NDArray[np.float32], dst: NDArray[np.float32]) -> None:
        # Re-checked on every call: AudioBuffer.data is a mutable ndarray whose
        # shape and dtype can be reassigned in place.
        shape = (self._channels, dst.shape[-1])
        for arr in (src, dst):
            if (
                arr.dtype != np.float32
                or not arr.flags.c_contiguous
                or arr.shape != shape
            ):
                raise ValueError(
                    f"expected C-contiguous float32 array of shape {shape}, "
                    f"got {arr.dtype} {arr.shape}"
                )
        if not dst.flags.writeable:
            raise ValueError("output array is read-only")
        self._impl.process(src.ctypes.data, dst.ctypes.data, shape[1])

    def __repr__(self) -> str:
        params = "".join(f"{k}={v!r}, " for k, v in self._params.items())
        return (
            f"{type(self).__name__}({params}sample_rate={self._sample_rate}, "
            f"channels={self._channels})"
        )


class Processor(_Unit):
    """A unit that transforms a buffer."""

    def process(self, buf: AudioBuffer) -> AudioBuffer:
        """Return the processed buffer. State carries over to the next call.

        Raises:
            ValueError: If *buf* has a different sample rate or channel count.
        """
        if buf.sample_rate != self._sample_rate:
            raise ValueError(
                f"buffer sample_rate {buf.sample_rate} != {self._sample_rate}"
            )
        if buf.channels != self._channels:
            raise ValueError(
                f"buffer has {buf.channels} channels, expected {self._channels}"
            )
        out = np.empty_like(buf.data)
        self._run(buf.data, out)
        return AudioBuffer(out, self._sample_rate, copy=False)


class Generator(_Unit):
    """A unit that produces a buffer from its internal state."""

    def generate(self, frames: int) -> AudioBuffer:
        """Return the next *frames* samples."""
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
            raise ValueError(f"frames must be an int >= 0, got {frames!r}")
        out = np.empty((self._channels, frames), np.float32)
        self._run(out, out)
        return AudioBuffer(out, self._sample_rate, copy=False)


class SupportsProcess(Protocol):
    def process(self, buf: AudioBuffer) -> AudioBuffer: ...
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

    def process(self, buf: AudioBuffer) -> AudioBuffer:
        for p in self.processors:
            buf = p.process(buf)
        return buf

    def reset(self) -> None:
        for p in self.processors:
            p.reset()

    def __repr__(self) -> str:
        return f"Chain({', '.join(map(repr, self.processors))})"
