"""AudioBuffer: planar ``[channels, frames]`` float32 samples with a sample rate."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["AudioBuffer"]


class AudioBuffer:
    """A 2D ``[channels, frames]`` C-contiguous float32 buffer.

    Args:
        data: Samples. 1D input becomes ``[1, frames]``.
        sample_rate: Sample rate in Hz. Must be positive and finite.
        copy: Copy *data* into storage the buffer owns. Pass ``False`` only for
            an array nothing else references. Without a copy, writes through
            the buffer would alias the caller's array only when no dtype or
            layout conversion happened.

    Raises:
        ValueError: If *data* is not 1D or 2D, or *sample_rate* is invalid.

    >>> buf = AudioBuffer(np.zeros((2, 480)), sample_rate=48000)
    >>> buf.channels, buf.frames, buf.duration
    (2, 480, 0.01)
    """

    __slots__ = ("_ctypes", "_data", "_sample_rate", "_view")

    def __init__(
        self, data: ArrayLike, sample_rate: float = 48000.0, copy: bool = True
    ) -> None:
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"AudioBuffer requires 1D or 2D data, got {arr.ndim}D")
        sr = float(sample_rate)
        if not (np.isfinite(sr) and sr > 0):
            raise ValueError(f"sample_rate must be positive and finite, got {sr}")
        self._data: NDArray[np.float32] = np.array(
            arr, dtype=np.float32, order="C", copy=True if copy else None
        )
        # `data` hands out a view, which numpy refuses to resize. The storage
        # therefore cannot move, so its address can be cached for the kernels.
        self._view: NDArray[np.float32] = self._data.view()
        self._ctypes = self._data.ctypes
        self._sample_rate = sr

    @classmethod
    def zeros(
        cls, channels: int, frames: int, sample_rate: float = 48000.0
    ) -> AudioBuffer:
        """Return a silent buffer."""
        return cls(np.zeros((channels, frames), np.float32), sample_rate, copy=False)

    @property
    def data(self) -> NDArray[np.float32]:
        """A view of the ``[channels, frames]`` samples.

        Writes go straight to the buffer's storage; the view exists so the
        storage cannot be reallocated out from under a running kernel.
        """
        return self._view

    @property
    def address(self) -> int:
        """Address of the first sample, for `mdsp._core`."""
        return self._ctypes.data

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return int(self._data.shape[0])

    @property
    def frames(self) -> int:
        return int(self._data.shape[1])

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.frames / self._sample_rate

    def __repr__(self) -> str:
        return (
            f"AudioBuffer(channels={self.channels}, frames={self.frames}, "
            f"sample_rate={self._sample_rate})"
        )
