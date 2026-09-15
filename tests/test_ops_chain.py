import numpy as np
import pytest

from mdsp import AudioBuffer, Biquad, Chain, Gain, OnePole

SR = 48000.0


def test_gain():
    x = AudioBuffer(np.linspace(-1, 1, 64).reshape(2, 32), SR)
    np.testing.assert_array_equal(
        Gain(-0.5, channels=2).process(x).data, x.data * np.float32(-0.5)
    )


def test_process_does_not_modify_input():
    x = AudioBuffer(np.ones(16), SR)
    Gain(3.0).process(x)
    assert (x.data == 1.0).all()


def test_chain_equals_sequential_processing():
    x = AudioBuffer(np.random.default_rng(1).uniform(-1, 1, (2, 1000)), SR)
    kwargs = {"sample_rate": SR, "channels": 2}
    chain = Chain(
        OnePole(2000.0, **kwargs),
        Biquad("highpass", 100.0, **kwargs),
        Gain(0.3, **kwargs),
    )
    expected = Gain(0.3, **kwargs).process(
        Biquad("highpass", 100.0, **kwargs).process(
            OnePole(2000.0, **kwargs).process(x)
        )
    )
    np.testing.assert_array_equal(chain.process(x).data, expected.data)


def test_chain_reset_and_nesting():
    x = AudioBuffer(np.random.default_rng(2).uniform(-1, 1, 500), SR)
    chain = Chain(Chain(OnePole(300.0)), Gain(2.0))
    first = chain.process(x).data
    chain.reset()
    np.testing.assert_array_equal(chain.process(x).data, first)
    assert repr(chain).startswith("Chain(Chain(OnePole(")


def test_empty_chain_is_identity():
    x = AudioBuffer(np.ones(4), SR)
    assert Chain().process(x) is x


def test_chain_propagates_mismatch_errors():
    with pytest.raises(ValueError, match="channels"):
        Chain(Gain(channels=2)).process(AudioBuffer(np.ones(4), SR))
