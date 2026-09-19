"""Player: buffer playback at variable speed."""

import math

import numpy as np
import pytest

from mdsp import AudioBuffer, Graph, Player, Shift, Sine

SR = 48000.0


def _peak_hz(samples, sample_rate=SR):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.fft.rfftfreq(len(samples), 1 / sample_rate)[np.argmax(spectrum)])


def _source(frames=4800, freq=440.0, channels=1):
    mono = Sine(freq, sample_rate=SR).generate(frames).data[0]
    return AudioBuffer(np.stack([mono * (1 - 0.5 * c) for c in range(channels)]), SR)


def test_rate_one_returns_the_source():
    src = _source()
    np.testing.assert_allclose(Player(src).generate(4800).data, src.data, atol=1e-6)


def test_the_source_is_copied_not_referenced():
    src = _source()
    player = Player(src)
    src.data[:] = 0.0
    assert np.abs(player.generate(4800).data).max() > 0.5


@pytest.mark.parametrize(("rate", "expected"), [(0.5, 220.0), (2.0, 880.0)])
def test_rate_carries_pitch_and_duration_together(rate, expected):
    out = Player(_source(frames=9600), rate=rate).generate(9600).data[0]
    played = out[: int(9600 / rate)] if rate > 1 else out
    assert _peak_hz(played) == pytest.approx(expected, rel=0.02)


def test_playing_past_the_end_is_silent():
    out = Player(_source()).generate(9600).data
    np.testing.assert_array_equal(out[:, 4800:], 0.0)


def test_looping_repeats_the_source():
    out = Player(_source(), loop=True).generate(9600).data
    np.testing.assert_allclose(out[:, :4800], out[:, 4800:], atol=1e-6)


def test_negative_rate_plays_backwards():
    src = _source()
    out = Player(src, rate=-1.0, position=0.1).generate(4800).data[0]
    np.testing.assert_allclose(out[1:], src.data[0, ::-1][:-1], atol=1e-3)


def test_position_seeks_in_seconds():
    src = _source()
    out = Player(src, position=0.05).generate(2400).data
    np.testing.assert_allclose(out, src.data[:, 2400:], atol=1e-6)


def test_channels_stay_together():
    src = _source(channels=2)
    out = Player(src, rate=0.5).generate(9600).data
    np.testing.assert_allclose(out[1], out[0] * 0.5, atol=1e-6)


def test_rate_can_be_modulated():
    src = _source(frames=48000)
    sweep = AudioBuffer(np.linspace(0.25, 2.0, 24000), SR)
    out = Player(src).generate(24000, rate=sweep).data[0]
    assert _peak_hz(out[:6000]) < _peak_hz(out[-6000:])  # speeding up raises pitch


def test_load_replaces_the_source():
    player = Player(_source(freq=440.0))
    player.load(_source(freq=880.0))
    assert _peak_hz(player.generate(4800).data[0]) == pytest.approx(880.0, rel=0.02)
    assert player.position == 0.0


def test_load_rejects_a_mismatched_buffer():
    player = Player(_source())
    with pytest.raises(ValueError, match="channels"):
        player.load(_source(channels=2))
    with pytest.raises(ValueError, match="sample_rate"):
        player.load(AudioBuffer(np.zeros((1, 100)), 44100.0))


def test_reset_rewinds():
    player = Player(_source())
    first = player.generate(2400).data.copy()
    player.reset()
    np.testing.assert_array_equal(player.generate(2400).data, first)


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"rate": 32.0}, "rate"), ({"position": -1.0}, "negative")]
)
def test_player_rejects_bad_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Player(_source(), **kwargs)


@pytest.mark.parametrize("rate", [0.5, 1.5])
def test_a_player_and_a_shift_stretch_time_at_constant_pitch(rate):
    """Play slower, shift back up: the length changes and the pitch does not."""
    src = _source(frames=48000)
    played = Player(src, rate=rate).generate(int(48000 / rate))
    semitones = -12 * math.log2(rate)
    shifted = Shift(semitones=semitones, sample_rate=SR).process(played).data[0]
    assert len(shifted) == pytest.approx(48000 / rate, abs=1)
    body = shifted[len(shifted) // 4 : 3 * len(shifted) // 4]
    grain_rate = abs(1 - 2 ** (semitones / 12)) / 0.05
    tolerance = grain_rate + SR / len(body)  # plus one FFT bin
    assert abs(_peak_hz(body) - 440.0) <= tolerance


# --------------------------------------------------------- as a graph node


def test_a_player_node_needs_its_source_loaded():
    g = Graph(SR, block=64)
    node = g.add(Player, rate=1.0)
    g.output = node
    np.testing.assert_array_equal(g.generate(512).data, 0.0)  # empty until loaded
    g.load(node, _source())
    np.testing.assert_allclose(g.generate(512).data, _source().data[:, :512], atol=1e-6)


def test_only_a_player_takes_a_source():
    g = Graph(SR, block=64)
    gain = g.add("gain", gain=1.0)
    g.output = gain
    with pytest.raises(ValueError, match="not a player"):
        g.load(gain, _source())


def test_load_checks_the_buffer_against_the_graph():
    g = Graph(SR, channels=2, block=64)
    node = g.add(Player)
    g.output = node
    with pytest.raises(ValueError, match="channels"):
        g.load(node, _source(channels=1))
    with pytest.raises(ValueError, match="sample_rate"):
        g.load(node, AudioBuffer(np.zeros((2, 128)), 44100.0))


def test_a_graph_can_stretch_time_while_it_runs():
    """Player at half speed into Shift an octave up: twice as long, same pitch."""
    g = Graph(SR, block=64)
    player = g.add(Player, rate=0.5)
    shift = g.add(Shift, semitones=12.0, window=0.05)
    g.connect(player, shift)
    g.output = shift
    g.load(player, _source(frames=48000))
    out = g.generate(96000).data[0]
    body = out[24000:72000]
    assert abs(_peak_hz(body) - 440.0) <= (1 - 2) ** 2 / 0.05 + SR / len(body)
