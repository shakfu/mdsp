"""Feedback edges and channel-aware nodes."""

import numpy as np
import pytest

from mdsp import AudioBuffer, Gain, Graph, Input, Mix, Pan, Sine, Svf, Width

SR = 48000.0


def _echo_graph(block=64, delay=4800, feedback=0.6, channels=1):
    """input -> mix -> gain, with the gain fed back into the mix."""
    g = Graph(SR, channels=channels, block=block)
    src = g.add(Input)
    mix = g.add(Mix, gain=1.0, gain2=feedback)
    tap = g.add(Gain, gain=1.0)
    g.connect(src, mix)
    g.connect(mix, tap)
    g.connect(tap, mix, "in2", delay=delay)
    g.output = tap
    return g


def _impulse(frames, channels=1):
    data = np.zeros((channels, frames), np.float32)
    data[:, 0] = 1.0
    return AudioBuffer(data, SR)


# ------------------------------------------------------------- feedback


def test_feedback_loop_repeats_and_decays():
    out = _echo_graph().process(_impulse(48000)).data[0]
    peaks = [float(out[i * 4800]) for i in range(5)]
    np.testing.assert_allclose(peaks, [0.6**i for i in range(5)], atol=1e-6)


@pytest.mark.parametrize("block", [1, 64, 512, 4800])
def test_feedback_does_not_depend_on_block_size(block):
    reference = _echo_graph(block=4800).process(_impulse(24000)).data
    np.testing.assert_allclose(
        _echo_graph(block=block).process(_impulse(24000)).data, reference, atol=1e-6
    )


def test_a_node_can_feed_itself():
    g = Graph(SR, block=64)
    src = g.add(Input)
    mix = g.add(Mix, gain=1.0, gain2=0.5)
    g.connect(src, mix)
    g.connect(mix, mix, "in2", delay=64)  # its own output, one block late
    g.output = mix
    out = g.process(_impulse(512)).data[0]
    np.testing.assert_allclose(
        [float(out[i * 64]) for i in range(4)], [0.5**i for i in range(4)], atol=1e-6
    )


def test_feedback_is_cleared_by_reset():
    g = _echo_graph()
    g.process(_impulse(9600))
    ringing = g.process(AudioBuffer(np.zeros((1, 9600)), SR)).data
    assert np.abs(ringing).max() > 0.0
    g.reset()
    silence = g.process(AudioBuffer(np.zeros((1, 9600)), SR)).data
    np.testing.assert_array_equal(silence, 0.0)


def test_feedback_runs_per_channel():
    g = _echo_graph(channels=2)
    data = np.zeros((2, 24000), np.float32)
    data[0, 0] = 1.0  # only the left channel is struck
    out = g.process(AudioBuffer(data, SR)).data
    np.testing.assert_allclose(
        [float(out[0, i * 4800]) for i in range(3)], [1.0, 0.6, 0.36], atol=1e-6
    )
    np.testing.assert_array_equal(out[1], 0.0)


def test_feedback_shorter_than_a_block_renders_in_smaller_chunks():
    g = Graph(SR, block=64)
    src = g.add(Input)
    mix = g.add(Mix, gain=1.0, gain2=0.5)
    tap = g.add(Gain, gain=1.0)
    g.connect(src, mix)
    g.connect(mix, tap)
    g.connect(tap, mix, "in2", delay=16)  # a quarter of the block
    g.output = tap
    out = g.process(_impulse(256)).data[0]
    np.testing.assert_allclose(
        [float(out[i * 16]) for i in range(5)], [0.5**i for i in range(5)], atol=1e-6
    )


def test_feedback_delay_is_validated():
    g = Graph(SR, block=512)
    a = g.add(Sine, freq=100.0)
    b = g.add(Gain, gain=1.0)
    g.connect(a, b)
    with pytest.raises(ValueError, match="at least 1 sample"):
        g.connect(b, a, "in", delay=0)
    with pytest.raises(TypeError, match="delay must be an int"):
        g.connect(b, a, "in", delay=1.5)  # type: ignore[arg-type]


def test_backwards_connection_still_needs_a_delay():
    g = Graph(SR, block=64)
    a = g.add(Sine, freq=100.0)
    b = g.add(Gain, gain=1.0)
    with pytest.raises(ValueError, match="pass delay="):
        g.connect(b, a)


# -------------------------------------------------------- channel-aware


def test_pan_holds_constant_power():
    mono = AudioBuffer(np.array([[1.0, 1.0], [0.0, 0.0]]), SR)
    for position in (-1.0, -0.5, 0.0, 0.5, 1.0):
        out = Pan(position, sample_rate=SR).process(mono).data[:, 0]
        assert float((out**2).sum()) == pytest.approx(1.0, abs=1e-6)
    left = Pan(-1.0, sample_rate=SR).process(mono).data[:, 0]
    right = Pan(1.0, sample_rate=SR).process(mono).data[:, 0]
    np.testing.assert_allclose(left, [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(right, [0.0, 1.0], atol=1e-6)


def test_pan_modulation_input():
    mono = AudioBuffer(np.array([[1.0] * 4, [0.0] * 4]), SR)
    sweep = AudioBuffer(np.array([[-1.0, 0.0, 1.0, 0.0]] * 2), SR)
    out = Pan(0.0, sample_rate=SR).process(mono, pan=sweep).data
    np.testing.assert_allclose(out[0], [1.0, 0.70710677, 0.0, 0.70710677], atol=1e-6)


def test_pan_rejects_out_of_range():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        Pan(2.0)


@pytest.mark.parametrize(
    ("width", "expected"), [(0.0, [0.0, 0.0]), (1.0, [1.0, -1.0]), (2.0, [2.0, -2.0])]
)
def test_width_scales_the_side_signal(width, expected):
    stereo = AudioBuffer(np.array([[1.0, 1.0], [-1.0, -1.0]]), SR)
    out = Width(width, sample_rate=SR).process(stereo).data[:, 0]
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_width_keeps_the_mid_signal():
    stereo = AudioBuffer(np.array([[1.0, 1.0], [1.0, 1.0]]), SR)  # identical channels
    out = Width(0.0, sample_rate=SR).process(stereo).data
    np.testing.assert_allclose(out, 1.0, atol=1e-6)


def test_width_passes_non_stereo_through():
    mono = AudioBuffer(np.array([[0.25, -0.5, 0.75]]), SR)
    out = Width(2.0, sample_rate=SR, channels=1).process(mono).data
    np.testing.assert_array_equal(out, mono.data)


def test_pan_and_width_as_graph_nodes():
    g = Graph(SR, channels=2, block=64)
    tone = g.add(Sine, freq=200.0)
    pan = g.add(Pan, pan=-0.5)
    wide = g.add(Width, width=1.5)
    g.connect(tone, pan)
    g.connect(pan, wide)
    g.output = wide
    out = g.generate(256).data
    assert np.abs(out[0]).max() > np.abs(out[1]).max()  # still leaning left
    assert g.inputs(pan) == ("in", "pan")
    assert g.params(wide) == ("width",)


def test_mono_nodes_stay_independent_per_channel():
    g = Graph(SR, channels=2, block=64)
    src = g.add(Input)
    filt = g.add(Svf, cutoff=1000.0)
    g.connect(src, filt)
    g.output = filt
    data = np.zeros((2, 512), np.float32)
    data[0, 0] = 1.0  # impulse only on the left
    out = g.process(AudioBuffer(data, SR)).data
    assert np.abs(out[0]).max() > 0.0
    np.testing.assert_array_equal(out[1], 0.0)


def test_channel_aware_node_sees_both_channels():
    """Width mixes the channels, so a left-only signal reaches the right."""
    g = Graph(SR, channels=2, block=64)
    src = g.add(Input)
    wide = g.add(Width, width=0.0)  # collapse to mono
    g.connect(src, wide)
    g.output = wide
    data = np.zeros((2, 128), np.float32)
    data[0, :] = 1.0
    out = g.process(AudioBuffer(data, SR)).data
    np.testing.assert_allclose(out, 0.5, atol=1e-6)
