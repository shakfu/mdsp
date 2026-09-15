import numpy as np
import pytest

from mdsp import Phasor, Saw, Sine, Square

SR = 48000.0


def test_sine_matches_closed_form():
    freq = 441.7
    out = Sine(freq, sample_rate=SR).generate(4800)
    n = np.arange(4800)
    np.testing.assert_allclose(
        out.data[0], np.sin(2 * np.pi * freq * n / SR), atol=1e-5
    )


def test_sine_is_continuous_across_blocks():
    whole = Sine(1234.5, sample_rate=SR).generate(3000).data
    osc = Sine(1234.5, sample_rate=SR)
    parts = np.concatenate([osc.generate(k).data for k in (1, 999, 0, 2000)], axis=1)
    np.testing.assert_array_equal(parts, whole)


def test_phasor_exact_ramp():
    out = Phasor(12000.0, sample_rate=SR).generate(9)
    np.testing.assert_array_equal(out.data[0], [0, 0.25, 0.5, 0.75] * 2 + [0])


@pytest.mark.parametrize("freq", [333.3, -333.3, 60000.0])
def test_phasor_wraps_into_unit_interval(freq):
    out = Phasor(freq, sample_rate=SR).generate(4800).data[0]
    assert out.min() >= 0.0
    assert out.max() < 1.0
    expected = (np.arange(4800) * freq / SR) % 1.0
    diff = np.abs(out - expected)
    np.testing.assert_allclose(np.minimum(diff, 1 - diff), 0, atol=1e-5)


def test_frequency_change_keeps_phase():
    osc = Phasor(12000.0, sample_rate=SR)
    osc.generate(1)  # phase 0.25
    osc.freq = 6000.0
    np.testing.assert_array_equal(osc.generate(3).data[0], [0.25, 0.375, 0.5])


def test_channels_and_reset():
    osc = Sine(100.0, sample_rate=SR, channels=3)
    first = osc.generate(64)
    assert first.data.shape == (3, 64)
    np.testing.assert_array_equal(first.data[0], first.data[2])
    osc.reset()
    np.testing.assert_array_equal(osc.generate(64).data, first.data)


def test_generate_zero_frames():
    assert Sine().generate(0).data.shape == (1, 0)


@pytest.mark.parametrize("frames", [-1, 1.5, True])
def test_generate_rejects_bad_frames(frames):
    with pytest.raises(ValueError, match="frames"):
        Sine().generate(frames)


def _naive(kind, freq, frames):
    p = (np.arange(frames) * freq / SR) % 1.0
    return 2 * p - 1 if kind is Saw else np.where(p < 0.5, 1.0, -1.0)


def _alias_db(y, f0):
    """Power outside harmonics of *f0* relative to harmonic power (1 Hz bins)."""
    power = np.abs(np.fft.rfft(y)) ** 2
    bins = np.arange(len(power))
    harmonic = (bins % f0 == 0) & (bins > 0)
    return 10 * np.log10(power[~harmonic & (bins > 0)].sum() / power[harmonic].sum())


@pytest.mark.parametrize("kind", [Saw, Square])
@pytest.mark.parametrize("f0", [440, 3517, 7919])
def test_band_limited_reduces_aliasing(kind, f0):
    y = kind(float(f0), sample_rate=SR).generate(int(SR)).data[0].astype(np.float64)
    blep, naive = _alias_db(y, f0), _alias_db(_naive(kind, f0, int(SR)), f0)
    # Measured: 14.5-30 dB better than naive, and at most -24 dB.
    assert blep < naive - 12
    assert blep < -20


@pytest.mark.parametrize(
    ("kind", "ideal"),
    [
        (Saw, [2 / (np.pi * h) for h in range(1, 4)]),
        (Square, [4 / np.pi, 0.0, 4 / (3 * np.pi)]),
    ],
)
def test_band_limited_harmonic_amplitudes(kind, ideal):
    y = kind(1000.0, sample_rate=SR).generate(int(SR)).data[0]
    amp = np.abs(np.fft.rfft(y.astype(np.float64))) * 2 / len(y)
    got = amp[[1000, 2000, 3000]]
    np.testing.assert_allclose(got, ideal, rtol=0.02, atol=1e-6)
    assert abs(y.mean()) < 1e-6


@pytest.mark.parametrize("kind", [Saw, Square])
def test_band_limited_negative_freq_is_negation(kind):
    up = kind(1733.3, sample_rate=SR).generate(4800).data
    down = kind(-1733.3, sample_rate=SR).generate(4800).data
    np.testing.assert_allclose(down, -up, atol=1e-6)
    assert np.abs(up).max() <= 1.0


@pytest.mark.parametrize("kind", [Saw, Square])
@pytest.mark.parametrize("freq", [24000.0, -24000.0, 30000.0])
def test_band_limited_rejects_freq_at_or_above_nyquist(kind, freq):
    with pytest.raises(ValueError, match="freq"):
        kind(freq, sample_rate=SR)
