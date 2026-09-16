"""Mix, Adsr and Noise, as units and as graph nodes."""

import numpy as np
import pytest

from mdsp import Adsr, AudioBuffer, Gain, Graph, Input, Mix, Noise, Saw, Sine

SR = 48000.0


# ----------------------------------------------------------------- Mix


def test_mix_sums_weighted_inputs():
    a = AudioBuffer(np.ones((1, 8)), SR)
    b = AudioBuffer(np.full((1, 8), 0.5), SR)
    c = AudioBuffer(np.full((1, 8), -1.0), SR)
    out = Mix(1.0, 2.0, 0.5, sample_rate=SR).process(a, in2=b, in3=c).data
    np.testing.assert_allclose(out, 1.0 + 1.0 - 0.5)


def test_mix_ignores_unconnected_inputs():
    a = AudioBuffer(np.ones((1, 4)), SR)
    np.testing.assert_allclose(Mix(0.25, sample_rate=SR).process(a).data, 0.25)


def test_mix_inputs_are_declared():
    assert Mix().inputs == ("in2", "in3", "in4")
    assert Mix()._impl.param_names() == ["gain", "gain2", "gain3", "gain4"]


def test_mix_in_a_graph_combines_two_voices():
    g = Graph(SR, block=64)
    a = g.add(Saw, freq=110.0)
    b = g.add(Saw, freq=220.0)
    mix = g.add(Mix, gain=0.5, gain2=0.5)
    g.connect(a, mix)
    g.connect(b, mix, "in2")
    g.output = mix
    mixed = g.generate(480).data

    expected = (
        0.5 * Saw(110.0, sample_rate=SR).generate(480).data
        + 0.5 * Saw(220.0, sample_rate=SR).generate(480).data
    )
    np.testing.assert_allclose(mixed, expected, atol=1e-7)


# ----------------------------------------------------------------- Adsr


def _stages(sample_rate=1000.0, **kwargs):
    defaults = {"attack": 0.01, "decay": 0.01, "sustain": 0.5, "release": 0.01}
    return Adsr(**{**defaults, **kwargs}, sample_rate=sample_rate)


def test_envelope_is_silent_until_gated():
    np.testing.assert_array_equal(_stages().generate(100).data, 0.0)


def test_envelope_stage_lengths_are_exact():
    env = _stages(gate=1.0)  # 10 samples per segment at 1 kHz
    out = env.generate(30).data[0]
    assert out[9] == pytest.approx(1.0)  # attack ends
    assert out[19] == pytest.approx(0.5)  # decay reaches sustain
    assert out[29] == pytest.approx(0.5)  # holds
    env.gate = 0.0
    release = env.generate(15).data[0]
    assert release[9] == pytest.approx(0.0)  # release ends
    np.testing.assert_allclose(release[10:], 0.0)


def test_envelope_rises_and_falls_monotonically():
    env = _stages(gate=1.0)
    attack = env.generate(10).data[0]
    assert np.all(np.diff(attack) > 0)
    decay = env.generate(10).data[0]
    assert np.all(np.diff(decay) < 0)


def test_gate_input_drives_the_envelope():
    env = _stages(sample_rate=1000.0)
    gate = AudioBuffer(np.concatenate([np.ones(20), np.zeros(20)]), 1000.0)
    out = env.generate(40, gate=gate).data[0]
    assert out[19] == pytest.approx(0.5, abs=1e-6)  # gated: reached sustain
    assert out[-1] == pytest.approx(0.0)  # released after the gate fell


def test_retrigger_starts_from_the_current_level():
    env = _stages(gate=1.0)
    env.generate(30)
    env.gate = 0.0
    env.generate(5)  # part-way through the release
    partial = env.generate(1).data[0, 0]
    env.gate = 1.0
    rising = env.generate(3).data[0]
    assert rising[0] > partial  # climbs again rather than jumping to zero


def test_zero_length_segments_take_one_sample():
    env = Adsr(attack=0.0, decay=0.0, sustain=1.0, gate=1.0, sample_rate=SR)
    assert env.generate(2).data[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"attack": -1.0}, "negative"),
        ({"release": -0.5}, "negative"),
        ({"sustain": 1.5}, r"\[0, 1\]"),
        ({"sustain": -0.1}, r"\[0, 1\]"),
        ({"gate": float("nan")}, "finite"),
    ],
)
def test_envelope_rejects_bad_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Adsr(**kwargs)


def test_envelope_shapes_a_tone_in_a_graph():
    g = Graph(SR, block=64)
    tone = g.add(Sine, freq=220.0)
    env = g.add(Adsr, attack=0.005, decay=0.005, sustain=0.5, release=0.005, gate=1.0)
    amp = g.add(Gain, gain=1.0)
    g.connect(tone, amp)
    g.connect(env, amp, "gain")
    g.output = amp
    out = g.generate(4800).data[0]
    assert abs(out[0]) < 0.01  # starts from silence
    assert np.abs(out[-480:]).max() == pytest.approx(0.5, abs=0.02)  # at sustain


# ----------------------------------------------------------------- Noise


def test_noise_is_deterministic_and_bounded():
    first = Noise(seed=7, sample_rate=SR).generate(1000).data
    np.testing.assert_array_equal(
        Noise(seed=7, sample_rate=SR).generate(1000).data, first
    )
    assert first.min() >= -1.0
    assert first.max() < 1.0
    assert abs(float(first.mean())) < 0.05


def test_different_seeds_differ():
    a = Noise(seed=1, sample_rate=SR).generate(500).data
    b = Noise(seed=2, sample_rate=SR).generate(500).data
    assert not np.array_equal(a, b)


def test_noise_reset_repeats_the_sequence():
    noise = Noise(seed=3, sample_rate=SR)
    first = noise.generate(100).data.copy()
    noise.reset()
    np.testing.assert_array_equal(noise.generate(100).data, first)


def test_noise_is_broadband():
    spectrum = np.abs(np.fft.rfft(Noise(seed=5, sample_rate=SR).generate(4096).data[0]))
    bands = np.array_split(spectrum[1:], 4)
    energies = [float((band**2).sum()) for band in bands]
    assert max(energies) / min(energies) < 2.0  # roughly flat across the spectrum


def test_noise_through_a_graph_matches_the_unit():
    g = Graph(SR, block=64)
    g.add(Noise, seed=11)
    np.testing.assert_array_equal(
        g.generate(512).data, Noise(seed=11, sample_rate=SR).generate(512).data
    )


def test_input_node_feeds_a_mix():
    g = Graph(SR, block=32)
    src = g.add(Input)
    tone = g.add(Sine, freq=100.0)
    mix = g.add(Mix, gain=1.0, gain2=1.0)
    g.connect(src, mix)
    g.connect(tone, mix, "in2")
    g.output = mix
    buf = AudioBuffer(np.ones((1, 256)), SR)
    expected = 1.0 + Sine(100.0, sample_rate=SR).generate(256).data
    np.testing.assert_allclose(g.process(buf).data, expected, atol=1e-7)
