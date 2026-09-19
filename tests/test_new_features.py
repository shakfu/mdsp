"""Node removal, EQ shapes, pink noise, chorus and stream capture."""

import ctypes.util
import math

import numpy as np
import pytest

from mdsp import (
    AudioBuffer,
    Biquad,
    Chorus,
    Compressor,
    Delay,
    Gain,
    Graph,
    Input,
    Mix,
    Noise,
    Saw,
    Scale,
    Sine,
    Stream,
    input_devices,
)

SR = 48000.0


def _db(value):
    return 20 * math.log10(abs(value) + 1e-12)


def _response_db(unit, freq, frames=24000):
    """Steady-state gain of *unit* at *freq*."""
    tone = Sine(freq, sample_rate=SR).generate(frames)
    out = unit.process(tone).data[0, frames // 2 :]
    reference = tone.data[0, frames // 2 :]
    return _db(np.sqrt((out**2).mean())) - _db(np.sqrt((reference**2).mean()))


# ------------------------------------------------------- node removal


def _three_node_graph():
    g = Graph(SR, block=64)
    tone = g.add(Sine, freq=100.0)
    quiet = g.add(Gain, gain=0.5)
    mix = g.add(Mix, gain=1.0, gain2=1.0)
    g.connect(tone, quiet)
    g.connect(quiet, mix)
    g.connect(tone, mix, "in2")
    g.output = mix
    return g, tone, quiet, mix


def test_removing_a_node_drops_its_contribution():
    g, _tone, quiet, _mix = _three_node_graph()
    g.reset()
    g.remove(quiet)
    expected = Sine(100.0, sample_rate=SR).generate(128).data  # only the direct path
    np.testing.assert_allclose(g.generate(128).data, expected, atol=1e-7)


def test_handles_stay_valid_after_a_removal():
    g, tone, quiet, _mix = _three_node_graph()
    g.remove(quiet)
    assert g.removed(quiet)
    assert not g.removed(tone)
    g.set(tone, "freq", 200.0)  # untouched handles keep working
    assert g.kind(tone) == "sine"


def test_a_removed_node_cannot_be_used():
    g, tone, quiet, mix = _three_node_graph()
    g.remove(quiet)
    for call in (
        lambda: g.set(quiet, "gain", 1.0),
        lambda: g.connect(tone, quiet),
        lambda: g.connect(quiet, mix),
        lambda: setattr(g, "output", quiet),
    ):
        with pytest.raises(ValueError, match="was removed"):
            call()


def test_the_output_node_cannot_be_removed():
    g, _, _, mix = _three_node_graph()
    with pytest.raises(ValueError, match="is the output"):
        g.remove(mix)


def test_removing_a_feedback_source_clears_the_loop():
    g = Graph(SR, block=64)
    src = g.add(Input)
    mix = g.add(Mix, gain=1.0, gain2=0.9)
    tap = g.add(Gain, gain=1.0)
    g.connect(src, mix)
    g.connect(mix, tap)
    g.connect(tap, mix, "in2", delay=64)
    g.output = tap
    impulse = np.zeros((1, 512), np.float32)
    impulse[0, 0] = 1.0
    g.process(AudioBuffer(impulse, SR))
    g.output = mix
    g.remove(tap)
    silence = g.process(AudioBuffer(np.zeros((1, 512)), SR)).data
    np.testing.assert_allclose(silence, 0.0, atol=1e-7)


# ----------------------------------------------------------- EQ shapes


@pytest.mark.parametrize("gain", [-12.0, -6.0, 6.0, 12.0])
def test_low_shelf_lifts_bass_and_leaves_treble(gain):
    shelf = Biquad("lowshelf", cutoff=500.0, gain=gain, sample_rate=SR)
    assert _response_db(shelf, 50.0) == pytest.approx(gain, abs=0.6)
    assert _response_db(
        Biquad("lowshelf", 500.0, gain=gain, sample_rate=SR), 8000.0
    ) == (pytest.approx(0.0, abs=0.3))


@pytest.mark.parametrize("gain", [-9.0, 9.0])
def test_high_shelf_lifts_treble_and_leaves_bass(gain):
    assert _response_db(
        Biquad("highshelf", 2000.0, gain=gain, sample_rate=SR), 16000.0
    ) == pytest.approx(gain, abs=0.6)
    assert _response_db(
        Biquad("highshelf", 2000.0, gain=gain, sample_rate=SR), 60.0
    ) == pytest.approx(0.0, abs=0.3)


def test_peaking_boosts_only_around_its_centre():
    peak = Biquad("peaking", 1000.0, q=2.0, gain=10.0, sample_rate=SR)
    assert _response_db(peak, 1000.0) == pytest.approx(10.0, abs=0.3)
    assert _response_db(
        Biquad("peaking", 1000.0, q=2.0, gain=10.0, sample_rate=SR), 100.0
    ) == pytest.approx(0.0, abs=0.5)


def test_gain_is_ignored_by_the_plain_modes():
    plain = Biquad("lowpass", 1000.0, gain=12.0, sample_rate=SR)
    reference = Biquad("lowpass", 1000.0, sample_rate=SR)
    x = AudioBuffer(np.random.default_rng(0).uniform(-1, 1, 512), SR)
    np.testing.assert_array_equal(plain.process(x).data, reference.process(x).data)


# ------------------------------------------------------ compressor knee


def test_soft_knee_eases_compression_in():
    hard = Compressor(threshold=-20.0, ratio=4.0, attack=0.0, sample_rate=SR)
    soft = Compressor(threshold=-20.0, ratio=4.0, attack=0.0, knee=12.0, sample_rate=SR)

    def out_db(comp, level_db):
        level = 10 ** (level_db / 20)
        return _db(comp.process(AudioBuffer(np.full((1, 480), level), SR)).data[0, -1])

    assert out_db(soft, -26.0) == pytest.approx(-26.0, abs=0.01)  # below the knee
    assert out_db(soft, -20.0) == pytest.approx(-21.125, abs=0.05)  # slope * knee / 8
    assert out_db(hard, -20.0) == pytest.approx(
        -20.0, abs=0.01
    )  # hard knee: nothing yet
    assert out_db(soft, 0.0) == pytest.approx(out_db(hard, 0.0), abs=0.05)  # far above


def test_knee_must_not_be_negative():
    with pytest.raises(ValueError, match="negative"):
        Compressor(knee=-1.0)


# --------------------------------------------------------- pink noise


def test_pink_noise_falls_with_frequency():
    pink = Noise(seed=3, color="pink", sample_rate=SR).generate(1 << 16).data[0]
    spectrum = np.abs(np.fft.rfft(pink)) ** 2
    freqs = np.fft.rfftfreq(len(pink), 1 / SR)

    def band(low, high):
        mask = (freqs >= low) & (freqs < high)
        return float(spectrum[mask].mean())

    # Pink noise halves its power per octave: about -3 dB.
    assert 10 * math.log10(band(2000, 4000) / band(1000, 2000)) == pytest.approx(
        -3.0, abs=1.5
    )
    white = Noise(seed=3, sample_rate=SR).generate(1 << 16).data[0]
    white_spectrum = np.abs(np.fft.rfft(white)) ** 2
    flat = float(white_spectrum[(freqs >= 2000) & (freqs < 4000)].mean()) / float(
        white_spectrum[(freqs >= 1000) & (freqs < 2000)].mean()
    )
    assert 10 * math.log10(flat) == pytest.approx(0.0, abs=1.0)


def test_pink_noise_is_bounded_and_repeatable():
    first = Noise(seed=9, color="pink", sample_rate=SR).generate(4800).data
    np.testing.assert_array_equal(
        Noise(seed=9, color="pink", sample_rate=SR).generate(4800).data, first
    )
    assert np.abs(first).max() <= 1.0  # stays inside full scale
    assert Noise(color="pink").color == "pink"


def test_unknown_colour_is_rejected():
    with pytest.raises(ValueError, match="color must be one of"):
        Noise(color="brown")  # type: ignore[arg-type]


# ------------------------------------------------------ chorus and flanging


def test_chorus_modulates_the_delay():
    tone = AudioBuffer(Saw(220.0, sample_rate=SR).generate(24000).data, SR)
    swept = Chorus(rate=2.0, depth=0.003, delay=0.008, mix=1.0, sample_rate=SR)
    still = Chorus(rate=0.0, depth=0.0, delay=0.008, mix=1.0, sample_rate=SR)
    moving = swept.process(tone).data
    fixed = still.process(tone).data
    assert np.isfinite(moving).all()
    assert not np.allclose(moving, fixed)  # the sweep changes the signal


@pytest.mark.parametrize("rate", [0.0, 3.0])
def test_chorus_with_no_depth_is_a_plain_delay(rate):
    """Depth 0 holds the line still however fast the LFO runs."""
    x = AudioBuffer(np.random.default_rng(1).uniform(-1, 1, 2048), SR)
    chorus = Chorus(
        rate=rate, depth=0.0, delay=0.01, feedback=0.0, mix=1.0, sample_rate=SR
    )
    delay = Delay(0.01, feedback=0.0, mix=1.0, max_delay=0.1, sample_rate=SR)
    np.testing.assert_allclose(chorus.process(x).data, delay.process(x).data, atol=1e-6)


def test_delay_time_can_be_modulated():
    """A flanger: an LFO sweeping a short delay."""
    g = Graph(SR, block=64)
    src = g.add(Input)
    lfo = g.add(Sine, freq=0.5)
    sweep = g.add(Scale, lo=0.001, hi=0.004)
    flanger = g.add(Delay, max_delay=0.02, delay=0.002, feedback=0.4, mix=0.5)
    g.connect(lfo, sweep)
    g.connect(src, flanger)
    g.connect(sweep, flanger, "delay")
    g.output = flanger
    x = AudioBuffer(Saw(150.0, sample_rate=SR).generate(4800).data, SR)
    out = g.process(x).data
    assert np.isfinite(out).all()
    assert not np.allclose(out, x.data)
    assert "delay" in g.inputs(flanger)


def test_chorus_rejects_delays_beyond_its_line():
    with pytest.raises(ValueError, match="seconds"):
        Chorus(delay=0.5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"voices": 0}, "whole number"),
        ({"voices": 5}, "whole number"),
        ({"voices": 1.5}, "whole number"),
        ({"spread": 1.5}, r"\[0, 1\]"),
    ],
)
def test_chorus_rejects_bad_voices_and_spread(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Chorus(**kwargs)


def _stereo_noise(frames=4800, seed=4):
    mono = np.random.default_rng(seed).uniform(-1, 1, frames)
    return AudioBuffer(np.stack([mono, mono]), SR)


def test_spread_puts_the_channels_out_of_phase():
    x = _stereo_noise()
    together = (
        Chorus(rate=1.0, depth=0.004, delay=0.008, mix=1.0, channels=2, sample_rate=SR)
        .process(x)
        .data
    )
    apart = (
        Chorus(
            rate=1.0,
            depth=0.004,
            delay=0.008,
            mix=1.0,
            spread=1.0,
            channels=2,
            sample_rate=SR,
        )
        .process(x)
        .data
    )
    np.testing.assert_array_equal(together[0], together[1])  # one shared LFO
    assert np.corrcoef(apart[0], apart[1])[0, 1] < 0.9  # antiphase sweeps


def test_spread_needs_a_sweep_to_do_anything():
    """With no depth the LFO moves nothing, so the offset cannot either."""
    x = _stereo_noise()
    kwargs = {
        "rate": 1.0,
        "depth": 0.0,
        "delay": 0.008,
        "mix": 1.0,
        "channels": 2,
        "sample_rate": SR,
    }
    np.testing.assert_array_equal(
        Chorus(spread=1.0, **kwargs).process(x).data,
        Chorus(**kwargs).process(x).data,
    )


def test_voices_read_one_line_at_different_phases():
    x = AudioBuffer(np.random.default_rng(5).uniform(-1, 1, 4800), SR)
    kwargs = {"rate": 1.0, "delay": 0.008, "mix": 1.0, "sample_rate": SR}
    one = Chorus(depth=0.004, voices=1, **kwargs).process(x).data
    three = Chorus(depth=0.004, voices=3, **kwargs).process(x).data
    assert not np.allclose(one, three)
    # Voices are averaged, so more of them cannot make the output louder.
    assert np.abs(three).max() <= np.abs(one).max() + 1e-6
    # Without a sweep every voice reads the same place and they collapse.
    flat = {"depth": 0.0, **kwargs}
    np.testing.assert_allclose(
        Chorus(voices=4, **flat).process(x).data,
        Chorus(voices=1, **flat).process(x).data,
        atol=1e-6,
    )


def test_chorus_is_a_stereo_graph_node():
    g = Graph(SR, channels=2, block=64)
    src = g.add(Input)
    wide = g.add(Chorus, rate=2.0, depth=0.003, delay=0.008, mix=1.0, spread=1.0)
    g.connect(src, wide)
    g.output = wide
    out = g.process(_stereo_noise(frames=2048)).data
    assert np.isfinite(out).all()
    assert np.corrcoef(out[0], out[1])[0, 1] < 0.9
    assert g.params(wide)[-2:] == ("voices", "spread")


# ------------------------------------------------------- stream capture

has_portaudio = ctypes.util.find_library("portaudio") is not None


def _has_input_device() -> bool:
    """Capture only works where a stream with input really opens."""
    if not has_portaudio:
        return False
    try:
        graph = Graph(SR, block=64)
        graph.add(Gain, gain=0.0)
        stream = Stream(graph, input_device=True)
        stream.start()
        stream.stop()
    except Exception:  # noqa: BLE001 - any failure means no usable device
        return False
    return True


@pytest.mark.skipif(not has_portaudio, reason="PortAudio not installed")
def test_input_devices_are_listed_separately():
    for device in input_devices():
        assert device["max_input_channels"] >= 1


@pytest.mark.skipif(not has_portaudio, reason="PortAudio not installed")
def test_input_device_argument_is_validated():
    g = Graph(SR, block=64)
    g.add(Sine, freq=100.0)
    with pytest.raises(TypeError, match="input_device must be"):
        Stream(g, input_device="default")  # type: ignore[arg-type]


@pytest.mark.skipif(not _has_input_device(), reason="no audio input device")
def test_capture_feeds_the_input_node():
    g = Graph(SR, block=64)
    src = g.add(Input)
    monitor = g.add(Gain, gain=0.0)  # silent: this test should make no noise
    g.connect(src, monitor)
    g.output = monitor
    import time

    with Stream(g, input_device=True) as stream:
        time.sleep(0.3)
        stats = stream.stats
    assert stats["callbacks"] > 0
    assert stats["underruns"] == 0
