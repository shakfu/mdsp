"""timestretch: WSOLA and the phase vocoder."""

import numpy as np
import pytest

from mdsp import AudioBuffer, Sine, timestretch

SR = 48000.0
MODES = ["wsola", "vocoder"]


def _peak_hz(samples, sample_rate=SR):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.fft.rfftfreq(len(samples), 1 / sample_rate)[np.argmax(spectrum)])


def _tone(frames=48000, freq=440.0):
    return Sine(freq, sample_rate=SR).generate(frames)


def _noise(frames=24000, channels=1, seed=0):
    rng = np.random.default_rng(seed)
    mono = rng.uniform(-1, 1, frames)
    return AudioBuffer(np.stack([mono * (1 - 0.5 * c) for c in range(channels)]), SR)


def _body(samples):
    """The interior, past the window either end that the edges disturb."""
    edge = len(samples) // 8
    return samples[edge:-edge]


# ------------------------------------------------------- both algorithms


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("factor", [0.25, 0.5, 1.5, 2.0, 4.0])
def test_length_follows_the_factor(mode, factor):
    out = timestretch(_tone(frames=12000), factor, mode=mode)
    assert out.frames == round(12000 * factor)
    assert out.sample_rate == SR and out.channels == 1


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("factor", [0.5, 1.5, 2.0])
def test_pitch_and_level_survive_the_stretch(mode, factor):
    out = _body(timestretch(_tone(), factor, mode=mode).data[0])
    assert _peak_hz(out) == pytest.approx(440.0, rel=0.01)
    assert out.std() == pytest.approx(0.707, rel=0.02)


@pytest.mark.parametrize("mode", MODES)
def test_factor_one_reproduces_the_input(mode):
    src = _noise()
    np.testing.assert_allclose(
        timestretch(src, 1.0, mode=mode).data, src.data, atol=1e-5
    )


@pytest.mark.parametrize("mode", MODES)
def test_the_source_is_left_alone(mode):
    src = _noise()
    before = src.data.copy()
    timestretch(src, 2.0, mode=mode)
    np.testing.assert_array_equal(src.data, before)


@pytest.mark.parametrize("mode", MODES)
def test_channels_stay_together(mode):
    out = timestretch(_noise(channels=2), 1.7, mode=mode).data
    np.testing.assert_allclose(out[1], out[0] * 0.5, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_a_source_shorter_than_the_window_stays_bounded(mode):
    out = timestretch(_noise(frames=64), 2.0, mode=mode)
    assert out.frames == 128
    assert np.isfinite(out.data).all() and np.abs(out.data).max() <= 1.5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"factor": 0.0}, "factor"),
        ({"factor": 20.0}, "factor"),
        ({"factor": float("nan")}, "factor"),
        ({"factor": 2.0, "window": 0.0}, "window"),
        ({"factor": 2.0, "window": 2.0}, "window"),
        ({"factor": 2.0, "search": -0.1}, "negative"),
        ({"factor": 2.0, "mode": "psola"}, "mode must be one of"),
    ],
)
def test_timestretch_rejects_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        timestretch(_noise(frames=4800), **kwargs)


# -------------------------------------------------------------- WSOLA


@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_events_move_in_proportion(factor):
    x = np.zeros((1, 48000), np.float32)
    x[0, 16000:32000] = _tone(frames=16000).data[0]  # a burst in the middle
    out = timestretch(AudioBuffer(x, SR), factor).data[0]
    loud = np.nonzero(np.abs(out) > 0.1)[0]
    window = int(0.03 * SR)
    assert abs(loud[0] - 16000 * factor) < window
    assert abs(loud[-1] - 32000 * factor) < window


def test_the_tail_is_the_end_of_the_source():
    x = _tone().data.copy()
    x[0, 47000:] = 0.0  # the source ends in silence, so the output must too
    out = timestretch(AudioBuffer(x, SR), 2.0).data[0]
    assert np.abs(out[-1500:]).max() == 0.0


def test_search_zero_is_plain_overlap_add():
    src = _noise()
    plain = timestretch(src, 2.0, search=0.0)
    searched = timestretch(src, 2.0)
    assert plain.frames == searched.frames
    assert not np.allclose(plain.data, searched.data)


def test_the_source_shorter_than_the_window_is_copied():
    src = _noise(frames=64)
    out = timestretch(src, 2.0)
    np.testing.assert_allclose(out.data[:, :64], src.data, atol=1e-6)
    np.testing.assert_array_equal(out.data[:, 64:], 0.0)


# ------------------------------------------------------- phase vocoder


@pytest.mark.parametrize("factor", [0.2, 0.33, 0.5, 0.75, 1.5, 2.0, 4.0])
def test_the_vocoder_holds_its_level_at_any_factor(factor):
    """Compression shortens the synthesis hop; a longer analysis hop would
    make a bin's phase step ambiguous and cost up to 2 dB."""
    out = _body(timestretch(_tone(), factor, mode="vocoder").data[0])
    assert out.std() == pytest.approx(0.707, rel=0.01)


def test_the_vocoder_is_flat_across_the_interior():
    out = timestretch(_tone(), 2.0, mode="vocoder").data[0]
    blocks = np.array([out[i : i + 1024].std() for i in range(0, len(out), 1024)])
    interior = blocks[5:-5]  # a window either end is approximate
    assert interior.min() == pytest.approx(0.707, rel=0.01)
    assert interior.max() == pytest.approx(0.707, rel=0.01)


def test_the_vocoder_keeps_two_tones_apart():
    """What WSOLA cannot do: stretch a chord without the partials beating."""
    mixed = _tone(freq=440.0).data + _tone(freq=660.0).data
    out = _body(timestretch(AudioBuffer(mixed / 2, SR), 2.0, mode="vocoder").data[0])
    spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)

    def level(target):
        band = (freqs > target - 15) & (freqs < target + 15)
        return float(spectrum[band].max())

    assert level(440.0) > 0.2 * spectrum.max()
    assert level(660.0) > 0.2 * spectrum.max()


def test_the_vocoder_window_is_rounded_to_a_power_of_two():
    """0.03 s at 48 kHz is 1440 samples; the transform needs 2048."""
    rounded = timestretch(_tone(), 2.0, window=0.03, mode="vocoder").data
    explicit = timestretch(_tone(), 2.0, window=2048 / SR, mode="vocoder").data
    np.testing.assert_allclose(rounded, explicit, atol=1e-6)
