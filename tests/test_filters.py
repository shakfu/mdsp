import math

import numpy as np
import pytest

from mdsp import AudioBuffer, Biquad, OnePole, Sine

SR = 48000.0


def _noise(channels, frames, seed=0):
    rng = np.random.default_rng(seed)
    return AudioBuffer(rng.uniform(-1, 1, (channels, frames)), SR)


def _rbj_reference(mode, cutoff, q, x):
    """Direct form I, float64, from the RBJ Audio EQ Cookbook."""
    w0 = 2 * math.pi * cutoff / SR
    cw, alpha = math.cos(w0), math.sin(w0) / (2 * q)
    b = {
        "lowpass": ((1 - cw) / 2, 1 - cw, (1 - cw) / 2),
        "highpass": ((1 + cw) / 2, -(1 + cw), (1 + cw) / 2),
        "bandpass": (alpha, 0.0, -alpha),
        "notch": (1.0, -2 * cw, 1.0),
    }[mode]
    a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    b0, b1, b2 = (v / a0 for v in b)
    a1, a2 = a1 / a0, a2 / a0
    y = np.zeros(len(x))
    x1 = x2 = y1 = y2 = 0.0
    for i, xi in enumerate(x.astype(np.float64)):
        y[i] = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1, y2, y1 = x1, xi, y1, y[i]
    return y


def _gain_db_at(filt, freq):
    """Steady-state gain of *filt* for a sine at *freq*."""
    x = Sine(freq, sample_rate=SR).generate(int(SR))
    y = filt.process(x).data[0, int(SR) // 2 :]
    ref = x.data[0, int(SR) // 2 :]
    return 20 * math.log10(np.sqrt(np.mean(y**2)) / np.sqrt(np.mean(ref**2)))


def test_onepole_matches_difference_equation():
    x = _noise(1, 2000)
    y = OnePole(1500.0, sample_rate=SR).process(x).data[0]
    a = np.float32(1 - math.exp(-2 * math.pi * 1500.0 / SR))
    z = np.float32(0)
    ref = np.empty(2000, np.float32)
    for i, xi in enumerate(x.data[0]):
        z = z + a * (xi - z)
        ref[i] = z
    np.testing.assert_allclose(y, ref, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("mode", ["lowpass", "highpass", "bandpass", "notch"])
def test_biquad_matches_rbj_reference(mode):
    x = _noise(1, 3000)
    y = Biquad(mode, cutoff=2500.0, q=3.0, sample_rate=SR).process(x).data[0]
    np.testing.assert_allclose(
        y, _rbj_reference(mode, 2500.0, 3.0, x.data[0]), atol=1e-5
    )


@pytest.mark.parametrize(
    ("mode", "expected_db"),
    [("lowpass", -3.0103), ("highpass", -3.0103), ("bandpass", 0.0)],
)
def test_biquad_gain_at_cutoff(mode, expected_db):
    filt = Biquad(mode, cutoff=1000.0, sample_rate=SR)
    assert _gain_db_at(filt, 1000.0) == pytest.approx(expected_db, abs=0.05)


def test_biquad_notch_rejects_centre():
    assert (
        _gain_db_at(Biquad("notch", cutoff=1000.0, q=1.0, sample_rate=SR), 1000.0) < -40
    )


def test_lowpass_dc_gain_is_unity_highpass_is_zero():
    ones = AudioBuffer(np.ones(48000), SR)
    assert Biquad("lowpass", 200.0, sample_rate=SR).process(ones).data[
        0, -1
    ] == pytest.approx(1.0, abs=1e-5)
    assert Biquad("highpass", 200.0, sample_rate=SR).process(ones).data[
        0, -1
    ] == pytest.approx(0.0, abs=1e-5)
    assert OnePole(200.0, sample_rate=SR).process(ones).data[0, -1] == pytest.approx(
        1.0, abs=1e-5
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda **kw: OnePole(900.0, **kw),
        lambda **kw: Biquad("bandpass", 900.0, 4.0, **kw),
    ],
    ids=["onepole", "biquad"],
)
def test_state_carries_across_blocks(make):
    x = _noise(2, 5000)
    whole = make(sample_rate=SR, channels=2).process(x).data
    filt = make(sample_rate=SR, channels=2)
    bounds = ((0, 1), (1, 1000), (1000, 1000), (1000, 5000))
    parts = [filt.process(AudioBuffer(x.data[:, a:b], SR)).data for a, b in bounds]
    np.testing.assert_array_equal(np.concatenate(parts, axis=1), whole)


def test_channels_are_independent():
    x = _noise(2, 1000)
    stereo = Biquad("lowpass", 3000.0, sample_rate=SR, channels=2).process(x).data
    for c in range(2):
        mono = (
            Biquad("lowpass", 3000.0, sample_rate=SR)
            .process(AudioBuffer(x.data[c], SR))
            .data[0]
        )
        np.testing.assert_array_equal(stereo[c], mono)


def test_reset_clears_state_keeps_params():
    x = _noise(1, 500)
    filt = Biquad("highpass", 700.0, 2.0, sample_rate=SR)
    first = filt.process(x).data
    filt.reset()
    np.testing.assert_array_equal(filt.process(x).data, first)
    assert (filt.mode, filt.cutoff, filt.q) == ("highpass", 700.0, 2.0)


def test_parameter_change_takes_effect():
    x = _noise(1, 2000)
    filt = Biquad("lowpass", 500.0, sample_rate=SR)
    filt.mode = "highpass"
    filt.cutoff = 4000.0
    filt.q = 0.5
    expected = Biquad("highpass", 4000.0, 0.5, sample_rate=SR).process(x).data
    np.testing.assert_array_equal(filt.process(x).data, expected)


@pytest.mark.parametrize("cutoff", [0.0, -5.0, 24000.0, float("nan"), float("inf")])
def test_invalid_cutoff_rejected_and_previous_value_kept(cutoff):
    filt = OnePole(1000.0, sample_rate=SR)
    with pytest.raises(ValueError):
        filt.cutoff = cutoff
    assert filt.cutoff == 1000.0


def test_invalid_q_and_mode_rejected():
    with pytest.raises(ValueError):
        Biquad(q=0.0)
    with pytest.raises(ValueError, match="mode"):
        Biquad("allpass")  # type: ignore[arg-type]


def test_repr():
    assert (
        repr(OnePole(100.0)) == "OnePole(cutoff=100.0, sample_rate=48000.0, channels=1)"
    )
