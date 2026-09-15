import numpy as np
import pytest

from mdsp import AudioBuffer, Delay
from mdsp.delay import MAX_DELAY_SECONDS

SR = 48000.0


def _impulse(frames, channels=1):
    x = np.zeros((channels, frames), np.float32)
    x[:, 0] = 1.0
    return AudioBuffer(x, SR)


def test_integer_delay_shifts_input():
    x = AudioBuffer(np.random.default_rng(0).uniform(-1, 1, 1000), SR)
    y = Delay(37 / SR, sample_rate=SR).process(x).data[0]
    np.testing.assert_array_equal(y[:37], 0.0)
    np.testing.assert_array_equal(y[37:], x.data[0, :-37])


def test_fractional_delay_interpolates_linearly():
    y = Delay(2.25 / SR, sample_rate=SR).process(_impulse(6)).data[0]
    np.testing.assert_allclose(y, [0, 0, 0.75, 0.25, 0, 0], atol=1e-7)


def test_feedback_produces_decaying_echoes():
    y = Delay(100 / SR, feedback=-0.5, sample_rate=SR).process(_impulse(450)).data[0]
    expected = np.zeros(450, np.float32)
    expected[[100, 200, 300, 400]] = [1.0, -0.5, 0.25, -0.125]
    np.testing.assert_array_equal(y, expected)


def test_mix_blends_dry_and_delayed():
    y = Delay(1 / SR, mix=0.25, sample_rate=SR).process(_impulse(3)).data[0]
    np.testing.assert_allclose(y, [0.75, 0.25, 0.0])


def test_echo_survives_block_boundary_and_reset_clears_line():
    d = Delay(10 / SR, sample_rate=SR, channels=2)
    first = d.process(_impulse(5, channels=2)).data
    second = d.process(AudioBuffer(np.zeros((2, 10)), SR)).data
    np.testing.assert_array_equal(first, 0.0)
    np.testing.assert_array_equal(second[:, 5], [1.0, 1.0])
    d.process(_impulse(5, channels=2))
    d.reset()
    np.testing.assert_array_equal(
        d.process(AudioBuffer(np.zeros((2, 20)), SR)).data, 0.0
    )


def test_max_delay_defaults_and_is_fixed():
    assert Delay(0.1).max_delay == 1.0
    assert Delay(2.5).max_delay == 2.5
    d = Delay(0.1, max_delay=0.2)
    d.delay = 0.2
    with pytest.raises(ValueError, match="delay"):
        d.delay = 0.2001
    with pytest.raises(AttributeError):
        d.max_delay = 5.0  # type: ignore[misc]
    assert d.delay == 0.2


@pytest.mark.parametrize("max_delay", [0.0, -1.0, MAX_DELAY_SECONDS + 1, float("nan")])
def test_rejects_bad_max_delay(max_delay):
    with pytest.raises(ValueError, match="max_delay"):
        Delay(0.1, max_delay=max_delay)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"delay": 0.0}, "delay"),
        ({"delay": 0.5 / SR}, "delay"),
        ({"feedback": 1.01}, "value"),
        ({"feedback": -1.01}, "value"),
        ({"mix": -0.1}, "value"),
        ({"mix": 1.1}, "value"),
    ],
)
def test_rejects_out_of_range_params(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Delay(**{"delay": 0.1, **kwargs}, sample_rate=SR)


def test_minimum_delay_is_one_sample():
    y = Delay(1 / SR, sample_rate=SR).process(_impulse(3)).data[0]
    np.testing.assert_array_equal(y, [0.0, 1.0, 0.0])


def test_repr():
    assert repr(Delay(0.5, 0.25)) == (
        "Delay(max_delay=1.0, delay=0.5, feedback=0.25, mix=1.0, "
        "sample_rate=48000.0, channels=1)"
    )
