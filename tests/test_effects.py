"""Compressor, Limiter, Shaper and Reverb."""

import math

import numpy as np
import pytest

from mdsp import AudioBuffer, Compressor, Graph, Input, Limiter, Reverb, Shaper, Sine

SR = 48000.0


def _tone(amplitude=1.0, frames=4800, sample_rate=SR):
    return AudioBuffer(
        Sine(440.0, sample_rate=sample_rate).generate(frames).data * amplitude,
        sample_rate,
    )


def _db(value):
    return 20 * math.log10(abs(value) + 1e-12)


# ----------------------------------------------------------- Compressor


@pytest.mark.parametrize(
    ("level_db", "ratio"), [(0.0, 4.0), (-10.0, 2.0), (-6.0, 8.0), (0.0, 1.0)]
)
def test_static_gain_reduction_follows_the_formula(level_db, ratio):
    level = 10 ** (level_db / 20)
    comp = Compressor(threshold=-20.0, ratio=ratio, attack=0.0, sample_rate=SR)
    steady = comp.process(AudioBuffer(np.full((1, 480), level), SR)).data[0, -1]
    over = level_db - (-20.0)
    expected_db = level_db - (over * (1 - 1 / ratio) if over > 0 else 0.0)
    assert _db(steady) == pytest.approx(expected_db, abs=0.05)


def test_signal_below_the_threshold_is_untouched():
    quiet = AudioBuffer(np.full((1, 480), 0.01), SR)  # -40 dB
    out = Compressor(threshold=-20.0, attack=0.0, sample_rate=SR).process(quiet).data
    np.testing.assert_allclose(out, quiet.data, rtol=1e-6)


def test_makeup_gain_is_applied():
    loud = AudioBuffer(np.ones((1, 480)), SR)
    plain = Compressor(threshold=-20.0, attack=0.0, sample_rate=SR).process(loud)
    boosted = Compressor(
        threshold=-20.0, attack=0.0, makeup=6.0, sample_rate=SR
    ).process(loud)
    assert _db(boosted.data[0, -1]) - _db(plain.data[0, -1]) == pytest.approx(
        6.0, abs=0.01
    )


def test_attack_and_release_take_their_time():
    loud = AudioBuffer(np.ones((1, 4800)), SR)
    comp = Compressor(
        threshold=-20.0, ratio=4.0, attack=0.01, release=0.01, sample_rate=SR
    )
    out = comp.process(loud).data[0]
    assert out[0] > 0.9  # not yet reduced
    attack_samples = int(0.01 * SR)
    # One time constant covers about 63% of the 15 dB the reduction travels.
    assert _db(out[attack_samples]) == pytest.approx(-15 * 0.63, abs=1.5)
    assert _db(out[-1]) == pytest.approx(-15.0, abs=0.2)  # settled

    quiet = comp.process(AudioBuffer(np.full((1, 4800), 0.01), SR)).data[0]
    assert _db(quiet[-1]) - _db(0.01) == pytest.approx(0.0, abs=0.2)  # released


def test_sidechain_ducks_the_audio():
    audio = AudioBuffer(np.full((1, 480), 0.1), SR)  # -20 dB, at the threshold
    key = AudioBuffer(np.ones((1, 480)), SR)  # 0 dB
    comp = Compressor(threshold=-20.0, ratio=4.0, attack=0.0, sample_rate=SR)
    ducked = comp.process(audio, sidechain=key).data[0, -1]
    assert _db(ducked) == pytest.approx(_db(0.1) - 15.0, abs=0.1)


def test_limiter_holds_the_threshold():
    loud = AudioBuffer(np.ones((1, 4800)), SR)
    out = Limiter(threshold=-6.0, sample_rate=SR).process(loud).data[0]
    assert _db(out[-1]) == pytest.approx(-6.0, abs=0.3)
    assert Limiter().inputs == ("sidechain",)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"ratio": 0.5}, "at least 1"), ({"attack": -1.0}, "negative")],
)
def test_compressor_rejects_bad_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Compressor(**kwargs)


# --------------------------------------------------------------- Shaper


@pytest.mark.parametrize("shape", ["tanh", "soft", "hard"])
def test_full_scale_stays_full_scale(shape):
    out = Shaper(drive=8.0, shape=shape, sample_rate=SR).process(AudioBuffer([1.0]))
    assert out.data[0, 0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("shape", ["tanh", "soft", "hard"])
def test_shaping_is_odd_symmetric(shape):
    x = AudioBuffer([-0.7, -0.3, 0.3, 0.7], SR)
    out = Shaper(drive=4.0, shape=shape, sample_rate=SR).process(x).data[0]
    np.testing.assert_allclose(out[:2], -out[:1:-1], atol=1e-7)


def test_drive_adds_harmonics():
    tone = _tone(0.5)

    def harmonic_ratio(drive):
        y = Shaper(drive=drive, sample_rate=SR).process(tone).data[0]
        spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        fundamental = spectrum[: len(spectrum) // 8].max()
        return float(spectrum[len(spectrum) // 8 :].sum() / fundamental)

    assert harmonic_ratio(20.0) > harmonic_ratio(1.001) * 5


def test_hard_clip_bounds_the_output():
    x = AudioBuffer(np.linspace(-2, 2, 64), SR)
    out = Shaper(drive=3.0, shape="hard", sample_rate=SR).process(x).data
    assert out.max() <= 1.0 and out.min() >= -1.0


def test_drive_modulation_input():
    x = AudioBuffer(np.full((1, 8), 0.5), SR)
    drive = AudioBuffer(np.full((1, 8), 5.0), SR)
    modulated = Shaper(drive=1.0, sample_rate=SR).process(x, drive=drive).data
    fixed = Shaper(drive=5.0, sample_rate=SR).process(x).data
    np.testing.assert_allclose(modulated, fixed, atol=1e-7)


def test_shaper_rejects_bad_parameters():
    with pytest.raises(ValueError, match="drive must be positive"):
        Shaper(drive=0.0)
    with pytest.raises(ValueError, match="shape must be one of"):
        Shaper(shape="fold")  # type: ignore[arg-type]


# --------------------------------------------------------------- Reverb


def _impulse_tail(frames=int(SR), **kwargs):
    impulse = np.zeros((1, frames), np.float32)
    impulse[0, 0] = 1.0
    return Reverb(sample_rate=SR, **kwargs).process(AudioBuffer(impulse, SR)).data[0]


def _energy(tail, start, stop):
    return float((tail[int(start * SR) : int(stop * SR)] ** 2).sum())


def test_reverb_tail_decays():
    tail = _impulse_tail(room_size=0.8, mix=1.0)
    early = _energy(tail, 0.05, 0.15)
    late = _energy(tail, 0.6, 0.7)
    assert early > 0.0
    assert late < early / 10


def test_bigger_rooms_ring_longer():
    small = _energy(_impulse_tail(room_size=0.1, mix=1.0), 0.4, 0.6)
    large = _energy(_impulse_tail(room_size=0.9, mix=1.0), 0.4, 0.6)
    assert large > small * 10


def test_damping_removes_high_frequencies():
    def centroid(damping):
        tail = _impulse_tail(room_size=0.8, damping=damping, mix=1.0)[int(0.1 * SR) :]
        spectrum = np.abs(np.fft.rfft(tail))
        freqs = np.fft.rfftfreq(len(tail), 1 / SR)
        return float((spectrum * freqs).sum() / spectrum.sum())

    assert centroid(0.9) < centroid(0.0)


def test_mix_zero_passes_the_input_through():
    x = _tone(0.5, frames=480)
    out = Reverb(mix=0.0, sample_rate=SR).process(x).data
    np.testing.assert_array_equal(out, x.data)


def test_reset_clears_the_tail():
    reverb = Reverb(room_size=0.9, mix=1.0, sample_rate=SR)
    impulse = np.zeros((1, 4800), np.float32)
    impulse[0, 0] = 1.0
    reverb.process(AudioBuffer(impulse, SR))
    ringing = reverb.process(AudioBuffer(np.zeros((1, 4800)), SR)).data
    assert np.abs(ringing).max() > 0.0
    reverb.reset()
    silent = reverb.process(AudioBuffer(np.zeros((1, 4800)), SR)).data
    np.testing.assert_array_equal(silent, 0.0)


def test_tail_length_does_not_depend_on_sample_rate():
    def half_life(sample_rate):
        frames = int(sample_rate)
        impulse = np.zeros((1, frames), np.float32)
        impulse[0, 0] = 1.0
        tail = (
            Reverb(room_size=0.8, mix=1.0, sample_rate=sample_rate)
            .process(AudioBuffer(impulse, sample_rate))
            .data[0]
        )
        energy = np.cumsum(tail**2)
        return float(np.searchsorted(energy, energy[-1] / 2) / sample_rate)

    assert half_life(96000.0) == pytest.approx(half_life(48000.0), rel=0.15)


def test_effects_work_as_graph_nodes():
    g = Graph(SR, block=64)
    src = g.add(Input)
    drive = g.add(Shaper, drive=4.0, shape="soft")
    comp = g.add(Compressor, threshold=-12.0, ratio=3.0)
    space = g.add(Reverb, room_size=0.6, mix=0.25)
    g.connect(src, drive)
    g.connect(drive, comp)
    g.connect(comp, space)
    g.output = space
    out = g.process(_tone(0.8, frames=4800))
    assert np.isfinite(out.data).all()
    assert np.abs(out.data).max() > 0.0
