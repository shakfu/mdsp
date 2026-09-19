"""Shift: pitch shifting on a two-head delay line."""

import numpy as np
import pytest

from mdsp import AudioBuffer, Graph, Input, Scale, Shift, Sine

SR = 48000.0


def _peak_hz(samples, sample_rate=SR):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.fft.rfftfreq(len(samples), 1 / sample_rate)[np.argmax(spectrum)])


def _tone(freq=440.0, frames=48000):
    return Sine(freq, sample_rate=SR).generate(frames)


@pytest.mark.parametrize("semitones", [-12.0, -7.0, -1.0, 5.0, 12.0])
def test_shift_moves_the_pitch_by_the_ratio(semitones):
    out = Shift(semitones=semitones, sample_rate=SR).process(_tone()).data[0, 24000:]
    assert _peak_hz(out) == pytest.approx(440.0 * 2 ** (semitones / 12), rel=0.01)


def test_shift_leaves_the_length_alone():
    tone = _tone(frames=4800)
    assert Shift(semitones=7.0, sample_rate=SR).process(tone).frames == 4800


def test_mix_zero_is_the_dry_signal():
    x = AudioBuffer(np.random.default_rng(0).uniform(-1, 1, 2048), SR)
    out = Shift(semitones=7.0, mix=0.0, sample_rate=SR).process(x).data
    np.testing.assert_allclose(out, x.data, atol=1e-6)


def test_no_shift_combs_rather_than_bypassing():
    """Two fixed taps, so 0 semitones is a comb; mix=0 is the bypass."""
    x = AudioBuffer(np.random.default_rng(1).uniform(-1, 1, 4096), SR)
    combed = Shift(semitones=0.0, sample_rate=SR).process(x).data
    assert not np.allclose(combed, x.data, atol=1e-3)


def test_shifted_output_stays_bounded():
    x = AudioBuffer(np.random.default_rng(2).uniform(-1, 1, 24000), SR)
    for semitones in (-24.0, -5.0, 3.0, 24.0):
        out = Shift(semitones=semitones, sample_rate=SR).process(x).data
        assert np.isfinite(out).all()
        assert np.abs(out).max() <= 1.5  # crossfade is constant power


@pytest.mark.parametrize("window", [0.02, 0.03, 0.08])
@pytest.mark.parametrize("semitones", [-12.0, 7.0, 12.0])
def test_a_wrap_holds_the_pitch_within_one_grain_rate(window, semitones):
    """The step at a grain boundary spreads energy by abs(1 - ratio) / window."""
    ratio = 2 ** (semitones / 12)
    out = (
        Shift(semitones=semitones, window=window, sample_rate=SR)
        .process(_tone(frames=96000))
        .data[0, 48000:]
    )
    tolerance = abs(1 - ratio) / window + SR / len(out)  # sidebands plus a bin
    assert abs(_peak_hz(out) - 440.0 * ratio) <= tolerance


def test_window_changes_the_result():
    tone = _tone(frames=4800)
    short = Shift(semitones=12.0, window=0.02, sample_rate=SR).process(tone).data
    long = Shift(semitones=12.0, window=0.08, sample_rate=SR).process(tone).data
    assert not np.allclose(short, long)


def test_a_modulation_buffer_replaces_the_parameter():
    tone = _tone()
    octave = AudioBuffer(np.full(48000, 12.0), SR)
    out = Shift(semitones=0.0, sample_rate=SR).process(tone, semitones=octave).data
    assert _peak_hz(out[0, 24000:]) == pytest.approx(880.0, rel=0.01)


def test_a_modulation_ramp_raises_the_pitch_over_time():
    tone = _tone()
    ramp = AudioBuffer(np.linspace(0.0, 12.0, 48000), SR)
    out = Shift(sample_rate=SR).process(tone, semitones=ramp).data[0]
    thirds = [_peak_hz(out[a : a + 16000]) for a in (0, 16000, 32000)]
    assert thirds[0] < thirds[1] < thirds[2]
    assert thirds[2] > 700.0  # well past the 440 Hz it started from


def test_reset_clears_the_line():
    x = AudioBuffer(np.random.default_rng(3).uniform(-1, 1, 2048), SR)
    shift = Shift(semitones=4.0, sample_rate=SR)
    first = shift.process(x).data.copy()
    shift.reset()
    np.testing.assert_array_equal(shift.process(x).data, first)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"semitones": 25.0}, r"\[-24, 24\]"),
        ({"window": 0.0}, "seconds"),
        ({"window": 0.5}, "seconds"),
        ({"mix": 1.5}, r"\[0, 1\]"),
    ],
)
def test_shift_rejects_bad_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Shift(**kwargs)


def test_shift_is_a_graph_node():
    g = Graph(SR, block=64)
    src = g.add(Input)
    lfo = g.add(Sine, freq=0.5)
    depth = g.add(Scale, lo=-2.0, hi=2.0)
    shift = g.add(Shift, semitones=7.0, window=0.04)
    g.connect(src, shift)
    g.connect(lfo, depth)
    g.connect(depth, shift, "semitones")
    g.output = shift
    out = g.process(_tone(frames=4800)).data
    assert np.isfinite(out).all()
    assert g.params(shift) == ("semitones", "window", "mix")
