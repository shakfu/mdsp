"""WAV reading and writing.

Covers PCM 8/16/24/32 and IEEE float 32/64, including WAVE_FORMAT_EXTENSIBLE.
For other formats read the file with `soundfile <https://pypi.org/project/soundfile/>`_
and pass the samples to `AudioBuffer`.

File contents are untrusted: every chunk header is checked against the bytes
actually present, and a malformed file raises `ValueError`.
"""

from __future__ import annotations

import os
import struct
from typing import BinaryIO, Literal

import numpy as np
from numpy.typing import NDArray

from mdsp.buffer import AudioBuffer

__all__ = ["SampleFormat", "read_wav", "write_wav"]

SampleFormat = Literal["float32", "float64", "int16", "int24", "int32"]

_PCM = 0x0001
_IEEE_FLOAT = 0x0003
_EXTENSIBLE = 0xFFFE
# First two bytes of the WAVE_FORMAT_EXTENSIBLE GUID carry the format tag.
_GUID_TAIL = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"

#: Divisor that maps each integer format to [-1, 1).
_SCALE = {16: 32768.0, 24: 8388608.0, 32: 2147483648.0}

_WRITE_FORMATS: dict[str, tuple[int, int]] = {
    "int16": (_PCM, 16),
    "int24": (_PCM, 24),
    "int32": (_PCM, 32),
    "float32": (_IEEE_FLOAT, 32),
    "float64": (_IEEE_FLOAT, 64),
}


def _read_exactly(stream: BinaryIO, size: int, what: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(
            f"truncated WAV file: {what} needs {size} bytes, got {len(data)}"
        )
    return data


def _parse_fmt(chunk: bytes) -> tuple[int, int, int, float]:
    """Return (format tag, channels, bits per sample, sample rate)."""
    if len(chunk) < 16:
        raise ValueError(f"fmt chunk is {len(chunk)} bytes, need at least 16")
    tag, channels, rate, _bytes_per_second, _align, bits = struct.unpack_from(
        "<HHIIHH", chunk
    )
    if tag == _EXTENSIBLE:
        if len(chunk) < 40:
            raise ValueError("WAVE_FORMAT_EXTENSIBLE fmt chunk is too short")
        if chunk[26:40] != _GUID_TAIL:
            raise ValueError("unsupported WAVE_FORMAT_EXTENSIBLE subformat")
        tag = struct.unpack_from("<H", chunk, 24)[0]
    if tag not in (_PCM, _IEEE_FLOAT):
        raise ValueError(f"unsupported WAV format tag 0x{tag:04x}")
    if channels < 1:
        raise ValueError("WAV file declares no channels")
    if tag == _PCM and bits not in (8, 16, 24, 32):
        raise ValueError(f"unsupported PCM bit depth {bits}")
    if tag == _IEEE_FLOAT and bits not in (32, 64):
        raise ValueError(f"unsupported float bit depth {bits}")
    if rate == 0:
        raise ValueError("WAV file declares a sample rate of 0")
    return tag, channels, bits, float(rate)


def _decode(raw: bytes, tag: int, bits: int) -> NDArray[np.float32]:
    """Interleaved bytes to float32 samples in [-1, 1]."""
    if tag == _IEEE_FLOAT:
        dtype = "<f4" if bits == 32 else "<f8"
        return np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if bits == 8:  # unsigned, midpoint 128
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if bits == 24:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = np.where(values >= 1 << 23, values - (1 << 24), values)
        return values.astype(np.float32) / _SCALE[24]
    dtype = "<i2" if bits == 16 else "<i4"
    return np.frombuffer(raw, dtype=dtype).astype(np.float32) / _SCALE[bits]


def read_wav(
    path: str | os.PathLike[str], *, start: int = 0, frames: int | None = None
) -> AudioBuffer:
    """Read *path* into an `AudioBuffer` of float32 samples.

    Args:
        start: First frame to return.
        frames: How many frames to return; all remaining frames by default.

    Raises:
        ValueError: If the file is not a WAV file this module supports, or if
            its chunk sizes disagree with the bytes present.

    >>> import numpy as np
    >>> from mdsp import AudioBuffer
    >>> write_wav("/tmp/mdsp-doctest.wav", AudioBuffer(np.zeros((2, 16)), 44100))
    >>> buf = read_wav("/tmp/mdsp-doctest.wav")
    >>> buf.channels, buf.frames, buf.sample_rate
    (2, 16, 44100.0)
    """
    if start < 0 or (frames is not None and frames < 0):
        raise ValueError("start and frames must not be negative")
    with open(path, "rb") as stream:
        header = _read_exactly(stream, 12, "RIFF header")
        if header[:4] == b"RIFX":
            raise ValueError("big-endian (RIFX) WAV files are not supported")
        if header[:4] == b"RF64":
            raise ValueError("RF64 (over 4 GB) WAV files are not supported")
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError("not a WAV file")

        fmt: tuple[int, int, int, float] | None = None
        while True:
            head = stream.read(8)
            if len(head) < 8:
                raise ValueError("WAV file ends before its data chunk")
            name, size = struct.unpack("<4sI", head)
            if name == b"data":
                break
            if name == b"fmt ":
                fmt = _parse_fmt(_read_exactly(stream, size, "fmt chunk"))
            else:  # LIST, fact, cue and friends
                stream.seek(size, os.SEEK_CUR)
            if size & 1:
                stream.seek(1, os.SEEK_CUR)  # chunks are word-aligned

        if fmt is None:
            raise ValueError("WAV file has no fmt chunk before its data")
        tag, channels, bits, rate = fmt
        frame_bytes = channels * bits // 8
        # Trust the file only as far as its bytes: a declared size larger than
        # what is present would otherwise ask for that much memory.
        present = os.fstat(stream.fileno()).st_size - stream.tell()
        if size > present:
            raise ValueError(
                f"data chunk declares {size} bytes but only {present} remain"
            )
        available = size // frame_bytes
        if start > available:
            raise ValueError(
                f"start {start} is past the end of the file ({available} frames)"
            )
        wanted = available - start if frames is None else min(frames, available - start)
        stream.seek(start * frame_bytes, os.SEEK_CUR)
        raw = _read_exactly(stream, wanted * frame_bytes, "data chunk")

    samples = _decode(raw, tag, bits)
    planar = np.ascontiguousarray(samples.reshape(-1, channels).T)
    if not planar.flags.writeable:  # a view of the read-only input bytes
        planar = planar.copy()
    return AudioBuffer(planar, rate, copy=False)


def _encode(data: NDArray[np.float32], fmt: SampleFormat) -> bytes:
    interleaved = np.ascontiguousarray(data.T)
    if fmt == "float32":
        return interleaved.astype("<f4").tobytes()
    if fmt == "float64":
        return interleaved.astype("<f8").tobytes()
    bits = {"int16": 16, "int24": 24, "int32": 32}[fmt]
    scale = _SCALE[bits]
    scaled = np.clip(np.rint(interleaved.astype(np.float64) * scale), -scale, scale - 1)
    if bits == 24:
        values = scaled.astype(np.int32).reshape(-1)
        packed = np.empty((values.size, 3), np.uint8)
        packed[:, 0] = values & 0xFF
        packed[:, 1] = (values >> 8) & 0xFF
        packed[:, 2] = (values >> 16) & 0xFF
        return packed.tobytes()
    return scaled.astype("<i2" if bits == 16 else "<i4").tobytes()


def write_wav(
    path: str | os.PathLike[str], buf: AudioBuffer, *, fmt: SampleFormat = "float32"
) -> None:
    """Write *buf* to *path*.

    Args:
        fmt: Sample format in the file. Integer formats clip samples outside
            [-1, 1); ``float32`` stores the buffer unchanged.

    Raises:
        ValueError: If *fmt* is not supported, or the data exceeds the 4 GB
            that the WAV format can address.
    """
    if fmt not in _WRITE_FORMATS:
        raise ValueError(f"fmt must be one of {sorted(_WRITE_FORMATS)}, got {fmt!r}")
    tag, bits = _WRITE_FORMATS[fmt]
    payload = _encode(buf.data, fmt)
    if len(payload) + 36 > 0xFFFFFFFF:
        raise ValueError("WAV files cannot exceed 4 GB; write fewer frames")
    channels = buf.channels
    rate = round(buf.sample_rate)
    align = channels * bits // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(payload) + (len(payload) & 1),
        b"WAVE",
        b"fmt ",
        16,
        tag,
        channels,
        rate,
        rate * align,
        align,
        bits,
        b"data",
        len(payload),
    )
    with open(path, "wb") as stream:
        stream.write(header)
        stream.write(payload)
        if len(payload) & 1:
            stream.write(b"\x00")  # chunks are word-aligned
