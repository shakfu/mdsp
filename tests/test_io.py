"""WAV reading and writing, including malformed files."""

import struct

import numpy as np
import pytest

from mdsp import AudioBuffer, read_wav, write_wav

SR = 44100.0


def _signal(channels=2, frames=1000, seed=0):
    rng = np.random.default_rng(seed)
    return AudioBuffer(rng.uniform(-0.9, 0.9, (channels, frames)), SR)


def _wav_bytes(fmt_chunk, data, extra_chunks=b"", riff=b"RIFF", wave=b"WAVE"):
    body = b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
    body += extra_chunks
    body += b"data" + struct.pack("<I", len(data)) + data
    return riff + struct.pack("<I", 4 + len(body)) + wave + body


def _pcm_fmt(channels=1, rate=44100, bits=16, tag=1):
    align = channels * bits // 8
    return struct.pack("<HHIIHH", tag, channels, rate, rate * align, align, bits)


@pytest.mark.parametrize(
    ("fmt", "tolerance"),
    [
        ("float32", 0.0),
        ("float64", 0.0),
        ("int16", 1 / 32768),
        ("int24", 1 / 8388608),
        ("int32", 1 / 2147483648),
    ],
)
@pytest.mark.parametrize("channels", [1, 2, 5])
def test_round_trip(tmp_path, fmt, tolerance, channels):
    x = _signal(channels=channels)
    path = tmp_path / "out.wav"
    write_wav(path, x, fmt=fmt)
    y = read_wav(path)
    assert (y.channels, y.frames, y.sample_rate) == (channels, 1000, SR)
    assert y.data.dtype == np.float32
    np.testing.assert_allclose(y.data, x.data, atol=tolerance)


def test_read_result_is_writable(tmp_path):
    path = tmp_path / "mono.wav"
    write_wav(path, _signal(channels=1))
    buf = read_wav(path)
    buf.data[0, 0] = 0.5  # would raise if it aliased the read-only file bytes
    assert buf.data[0, 0] == 0.5


def test_integer_formats_clip(tmp_path):
    path = tmp_path / "loud.wav"
    write_wav(path, AudioBuffer([2.0, -2.0, 0.0], SR), fmt="int16")
    np.testing.assert_allclose(read_wav(path).data[0], [1.0 - 1 / 32768, -1.0, 0.0])


def test_known_24_bit_values(tmp_path):
    path = tmp_path / "24.wav"
    write_wav(path, AudioBuffer([0.0, 0.5, -0.5], SR), fmt="int24")
    raw = path.read_bytes()[:-1]  # drop the word-alignment pad
    assert raw[-9:] == b"\x00\x00\x00" + b"\x00\x00\x40" + b"\x00\x00\xc0"
    np.testing.assert_allclose(read_wav(path).data[0], [0.0, 0.5, -0.5])


def test_odd_length_data_is_padded(tmp_path):
    path = tmp_path / "odd.wav"
    write_wav(path, AudioBuffer([0.25, -0.25, 0.75], SR), fmt="int24")  # 9 bytes
    assert len(path.read_bytes()) % 2 == 0
    assert read_wav(path).frames == 3


def test_start_and_frames(tmp_path):
    x = _signal(channels=2, frames=100)
    path = tmp_path / "slice.wav"
    write_wav(path, x)
    np.testing.assert_array_equal(
        read_wav(path, start=10, frames=5).data, x.data[:, 10:15]
    )
    np.testing.assert_array_equal(read_wav(path, start=90).data, x.data[:, 90:])
    assert read_wav(path, start=100).frames == 0
    assert read_wav(path, frames=1000).frames == 100


@pytest.mark.parametrize(("start", "frames"), [(-1, None), (0, -5)])
def test_negative_slice_rejected(tmp_path, start, frames):
    path = tmp_path / "s.wav"
    write_wav(path, _signal(frames=10))
    with pytest.raises(ValueError, match="negative"):
        read_wav(path, start=start, frames=frames)


def test_start_past_end(tmp_path):
    path = tmp_path / "s.wav"
    write_wav(path, _signal(frames=10))
    with pytest.raises(ValueError, match="past the end"):
        read_wav(path, start=11)


def test_extra_chunks_are_skipped(tmp_path):
    data = struct.pack("<3h", 0, 16384, -16384)
    extra = b"LIST" + struct.pack("<I", 5) + b"INFOx" + b"\x00"  # odd size, padded
    path = tmp_path / "chunks.wav"
    path.write_bytes(_wav_bytes(_pcm_fmt(), data, extra))
    np.testing.assert_allclose(read_wav(path).data[0], [0.0, 0.5, -0.5])


def test_extensible_format(tmp_path):
    guid = (
        struct.pack("<H", 1)
        + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    )
    fmt = _pcm_fmt(tag=0xFFFE) + struct.pack("<HHI", 22, 16, 0x3) + guid
    path = tmp_path / "ext.wav"
    path.write_bytes(_wav_bytes(fmt, struct.pack("<2h", 16384, -16384)))
    np.testing.assert_allclose(read_wav(path).data[0], [0.5, -0.5])


def test_8_bit_pcm_is_unsigned(tmp_path):
    path = tmp_path / "8bit.wav"
    path.write_bytes(_wav_bytes(_pcm_fmt(bits=8), bytes([128, 255, 0, 64])))
    np.testing.assert_allclose(read_wav(path).data[0], [0.0, 127 / 128, -1.0, -0.5])


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (b"", "truncated"),
        (b"NOPE" + struct.pack("<I", 4) + b"WAVE", "not a WAV"),
        (b"RIFX" + struct.pack("<I", 4) + b"WAVE", "big-endian"),
        (b"RF64" + struct.pack("<I", 4) + b"WAVE", "RF64"),
        (b"RIFF" + struct.pack("<I", 4) + b"WAVE", "ends before its data"),
    ],
)
def test_bad_headers(tmp_path, content, match):
    path = tmp_path / "bad.wav"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=match):
        read_wav(path)


@pytest.mark.parametrize(
    ("fmt_chunk", "match"),
    [
        (_pcm_fmt()[:10], "at least 16"),
        (_pcm_fmt(tag=0x0055), "format tag"),
        (_pcm_fmt(bits=12), "bit depth"),
        (_pcm_fmt(channels=0), "no channels"),
        (_pcm_fmt(rate=0), "sample rate of 0"),
        (_pcm_fmt(tag=3, bits=16), "float bit depth"),
        (_pcm_fmt(tag=0xFFFE), "EXTENSIBLE"),
    ],
)
def test_bad_fmt_chunk(tmp_path, fmt_chunk, match):
    path = tmp_path / "bad.wav"
    path.write_bytes(_wav_bytes(fmt_chunk, b"\x00\x00"))
    with pytest.raises(ValueError, match=match):
        read_wav(path)


def test_data_chunk_larger_than_the_file(tmp_path):
    path = tmp_path / "lying.wav"
    body = _wav_bytes(_pcm_fmt(), b"\x00" * 8)
    # Claim 4 GB of samples in a file holding 8 bytes of them.
    lying = body[:-12] + b"data" + struct.pack("<I", 0xFFFFFFF0) + body[-8:]
    path.write_bytes(lying)
    with pytest.raises(ValueError, match="only .* remain"):
        read_wav(path)


def test_missing_fmt_chunk(tmp_path):
    path = tmp_path / "nofmt.wav"
    body = b"data" + struct.pack("<I", 2) + b"\x00\x00"
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    with pytest.raises(ValueError, match="no fmt chunk"):
        read_wav(path)


def test_unknown_write_format(tmp_path):
    with pytest.raises(ValueError, match="fmt must be one of"):
        write_wav(tmp_path / "x.wav", _signal(frames=2), fmt="int8")  # type: ignore[arg-type]


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_wav(tmp_path / "nope.wav")


def test_round_trip_through_a_processing_chain(tmp_path):
    from mdsp import Chain, Gain, Svf

    source = tmp_path / "in.wav"
    result = tmp_path / "out.wav"
    write_wav(source, _signal(channels=2, frames=4800), fmt="int24")
    buf = read_wav(source)
    chain = Chain(
        Svf("lowpass", 2000.0, sample_rate=SR, channels=2),
        Gain(0.5, sample_rate=SR, channels=2),
    )
    write_wav(result, chain.process(buf), fmt="int24")
    assert read_wav(result).frames == 4800
