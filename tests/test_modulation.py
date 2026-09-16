"""Modulation inputs and parameter smoothing."""

import numpy as np
import pytest

from mdsp import AudioBuffer, Biquad, Delay, Gain, OnePole, Phasor, Saw, Sine, Svf

SR = 48000.0


def _const(value, frames=4800, channels=1):
    return AudioBuffer(np.full((channels, frames), value, np.float32), SR)


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        (Sine(), ("freq",)),
        (Saw(), ("freq",)),
        (Phasor(), ("freq",)),
        (Gain(), ("gain",)),
        (OnePole(), ("cutoff",)),
        (Svf(), ("cutoff",)),
        (Delay(0.01), ()),
        (Biquad(), ()),
    ],
)
def test_declared_modulation_inputs(unit, expected):
    assert unit.inputs == expected


@pytest.mark.parametrize("make", [lambda: OnePole(1234.0), lambda: Svf(cutoff=1234.0)])
def test_constant_modulation_equals_the_parameter(make):
    x = AudioBuffer(np.random.default_rng(0).uniform(-1, 1, 4800), SR)
    constant = make().process(x).data
    modulated = make().process(x, cutoff=_const(1234.0)).data
    np.testing.assert_array_equal(modulated, constant)


def test_cutoff_modulation_sweeps_the_filter():
    x = AudioBuffer(np.random.default_rng(1).uniform(-1, 1, 4800), SR)
    sweep = AudioBuffer(np.linspace(200, 8000, 4800), SR)
    swept = Svf(cutoff=200.0).process(x, cutoff=sweep).data
    fixed = Svf(cutoff=200.0).process(x).data
    # A rising cutoff passes more signal as it opens.
    assert np.abs(swept[:, -1000:]).mean() > 3 * np.abs(fixed[:, -1000:]).mean()


def test_gain_modulation_applies_an_envelope():
    ones = AudioBuffer(np.ones(8), SR)
    env = AudioBuffer(np.linspace(0, 1, 8), SR)
    out = Gain(0.0).process(ones, gain=env).data
    np.testing.assert_allclose(out[0], np.linspace(0, 1, 8), atol=1e-7)


def test_freq_modulation_is_vibrato():
    frames = 4800
    vibrato = Sine(5.0).generate(frames)
    freq = AudioBuffer(440.0 + 50.0 * vibrato.data, SR)
    modulated = Sine(440.0).generate(frames, freq=freq).data
    plain = Sine(440.0).generate(frames).data
    assert np.abs(modulated).max() <= 1.0
    assert not np.allclose(modulated, plain)


def test_modulation_buffer_must_match():
    unit = Svf(channels=2)
    x = AudioBuffer(np.zeros((2, 64)), SR)
    with pytest.raises(ValueError, match="shape"):
        unit.process(x, cutoff=AudioBuffer(np.zeros((2, 32)), SR))
    with pytest.raises(ValueError, match="sample_rate"):
        unit.process(x, cutoff=AudioBuffer(np.zeros((2, 64)), 44100))
    with pytest.raises(ValueError, match="shape"):
        unit.process(x, cutoff=AudioBuffer(np.zeros((1, 64)), SR))


def test_unknown_modulation_input_is_rejected():
    with pytest.raises(TypeError, match="no modulation input"):
        Svf().process(AudioBuffer(np.zeros(8), SR), resonance=_const(1.0, 8))
    with pytest.raises(TypeError, match="no modulation input"):
        Gain().process(AudioBuffer(np.zeros(8), SR), cutoff=_const(1.0, 8))


def test_parameter_change_is_smoothed():
    ones = AudioBuffer(np.ones(2400), SR)
    gain = Gain(1.0)
    gain.process(ones)
    gain.gain = 0.0
    out = gain.process(ones).data[0]
    ramp_samples = int(0.01 * SR)
    assert np.abs(np.diff(out)).max() < 2.0 / ramp_samples  # no step
    assert out[0] > 0.99
    np.testing.assert_allclose(out[ramp_samples:], 0.0, atol=1e-7)


def test_smoothing_is_independent_of_block_size():
    x = AudioBuffer(np.random.default_rng(2).uniform(-1, 1, 3000), SR)
    whole = Gain(1.0)
    whole.gain = 0.25
    expected = whole.process(x).data

    chunked = Gain(1.0)
    chunked.gain = 0.25
    parts = [
        chunked.process(AudioBuffer(x.data[:, a:b], SR)).data
        for a, b in ((0, 1), (1, 100), (100, 1500), (1500, 3000))
    ]
    np.testing.assert_array_equal(np.concatenate(parts, axis=1), expected)
