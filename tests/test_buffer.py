import numpy as np
import pytest

from mdsp import AudioBuffer


def test_1d_becomes_mono():
    buf = AudioBuffer(np.arange(5), sample_rate=8000)
    assert (buf.channels, buf.frames) == (1, 5)
    assert buf.data.dtype == np.float32
    assert buf.data.flags.c_contiguous


def test_non_contiguous_input_is_made_contiguous():
    src = np.arange(12, dtype=np.float32).reshape(3, 4).T  # [4, 3], F-order
    buf = AudioBuffer(src)
    assert buf.data.flags.c_contiguous
    np.testing.assert_array_equal(buf.data, src)


def test_copies_by_default():
    src = np.zeros((1, 4), np.float32)
    buf = AudioBuffer(src)
    buf.data[0, 0] = 1.0
    assert src[0, 0] == 0.0


def test_copy_false_adopts_compatible_array():
    src = np.zeros((2, 4), np.float32)
    buf = AudioBuffer(src, copy=False)
    assert buf.data.base is src  # a view of the caller's array, not a copy
    buf.data[0, 0] = 1.0
    assert src[0, 0] == 1.0


def test_data_view_cannot_be_reallocated():
    # The kernels cache this address, so the storage must not move.
    buf = AudioBuffer(np.zeros((1, 8)), 48000)
    before = buf.address
    with pytest.raises(ValueError, match="does not own its data"):
        buf.data.resize((1, 64), refcheck=False)
    assert buf.address == before


def test_properties_and_zeros():
    buf = AudioBuffer.zeros(2, 4800, sample_rate=48000)
    assert (buf.channels, buf.frames, buf.sample_rate) == (2, 4800, 48000.0)
    assert buf.duration == pytest.approx(0.1)
    assert not buf.data.any()
    assert repr(buf) == "AudioBuffer(channels=2, frames=4800, sample_rate=48000.0)"


@pytest.mark.parametrize("data", [np.float32(1.0), np.zeros((1, 2, 3))])
def test_rejects_bad_ndim(data):
    with pytest.raises(ValueError, match="1D or 2D"):
        AudioBuffer(data)


@pytest.mark.parametrize("sr", [0, -1, float("nan"), float("inf")])
def test_rejects_bad_sample_rate(sr):
    with pytest.raises(ValueError, match="sample_rate"):
        AudioBuffer(np.zeros(4), sample_rate=sr)
