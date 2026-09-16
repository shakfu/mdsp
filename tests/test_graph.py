"""Runtime graphs: wiring, rendering, and equivalence with single units."""

import numpy as np
import pytest

from mdsp import (
    AudioBuffer,
    Chain,
    Delay,
    Gain,
    Graph,
    Input,
    OnePole,
    Saw,
    Scale,
    Sine,
    Svf,
)

SR = 48000.0


def _noise(channels=1, frames=4800, seed=0):
    rng = np.random.default_rng(seed)
    return AudioBuffer(rng.uniform(-1, 1, (channels, frames)), SR)


def test_matches_an_equivalent_chain():
    x = _noise(channels=2)
    g = Graph(SR, channels=2)
    src = g.add(Input)
    filt = g.add(Svf, mode="lowpass", cutoff=900.0, q=2.0)
    out = g.add(Gain, gain=0.5)
    g.connect(src, filt)
    g.connect(filt, out)
    g.output = out

    kwargs = {"sample_rate": SR, "channels": 2}
    chain = Chain(Svf("lowpass", 900.0, 2.0, **kwargs), Gain(0.5, **kwargs))
    np.testing.assert_array_equal(g.process(x).data, chain.process(x).data)


def test_matches_a_unit_with_a_modulation_buffer():
    x = _noise()
    g = Graph(SR)
    src = g.add(Input)
    lfo = g.add(Sine, freq=3.0)
    sweep = g.add(Scale, lo=300.0, hi=4000.0, curve="exponential")
    filt = g.add(Svf, mode="lowpass", cutoff=1000.0, q=3.0)
    g.connect(lfo, sweep)
    g.connect(src, filt)
    g.connect(sweep, filt, "cutoff")
    g.output = filt

    cutoff = Scale(300.0, 4000.0, "exponential", sample_rate=SR).process(
        Sine(3.0, sample_rate=SR).generate(x.frames)
    )
    expected = Svf("lowpass", 1000.0, 3.0, sample_rate=SR).process(x, cutoff=cutoff)
    np.testing.assert_array_equal(g.process(x).data, expected.data)


def test_generate_matches_a_unit():
    g = Graph(SR)
    g.add(Saw, freq=110.0)
    np.testing.assert_array_equal(
        g.generate(4800).data, Saw(110.0, sample_rate=SR).generate(4800).data
    )


@pytest.mark.parametrize("block", [1, 7, 64, 4096])
def test_output_does_not_depend_on_block_size(block):
    x = _noise(frames=3000)
    expected = None
    for size in (4096, block):
        g = Graph(SR, block=size)
        src = g.add(Input)
        filt = g.add(OnePole, cutoff=800.0)
        echo = g.add(Delay, max_delay=0.05, delay=0.01, feedback=0.4, mix=0.5)
        g.connect(src, filt)
        g.connect(filt, echo)
        g.output = echo
        result = g.process(x).data
        if expected is None:
            expected = result
        else:
            np.testing.assert_array_equal(result, expected)


def test_channels_are_independent():
    x = _noise(channels=2)
    stereo = Graph(SR, channels=2)
    src = stereo.add(Input)
    filt = stereo.add(Svf, cutoff=1500.0)
    stereo.connect(src, filt)
    stereo.output = filt
    out = stereo.process(x).data

    for channel in range(2):
        mono = Graph(SR)
        m_src = mono.add(Input)
        m_filt = mono.add(Svf, cutoff=1500.0)
        mono.connect(m_src, m_filt)
        mono.output = m_filt
        expected = mono.process(AudioBuffer(x.data[channel], SR)).data[0]
        np.testing.assert_array_equal(out[channel], expected)


def test_input_node_reads_silence_in_generate():
    g = Graph(SR)
    src = g.add(Input)
    gain = g.add(Gain, gain=1.0)
    g.connect(src, gain)
    g.output = gain
    np.testing.assert_array_equal(g.generate(128).data, np.zeros((1, 128), np.float32))


def test_unconnected_input_of_a_processor_is_silent():
    g = Graph(SR)
    g.add(OnePole, cutoff=1000.0)  # nothing connected to its audio input
    np.testing.assert_array_equal(g.generate(64).data, np.zeros((1, 64), np.float32))


def test_output_defaults_to_the_last_node_and_can_be_moved():
    g = Graph(SR)
    tone = g.add(Sine, freq=100.0)
    quiet = g.add(Gain, gain=0.25)
    g.connect(tone, quiet)
    assert g.output == quiet
    g.output = tone
    np.testing.assert_array_equal(
        g.generate(64).data, Sine(100.0, sample_rate=SR).generate(64).data
    )


def test_reset_one_node_and_all_nodes():
    x = _noise(frames=500)
    g = Graph(SR)
    src = g.add(Input)
    echo = g.add(Delay, max_delay=0.05, delay=0.01, feedback=0.5)
    g.connect(src, echo)
    first = g.process(x).data
    g.reset(echo)
    np.testing.assert_array_equal(g.process(x).data, first)
    g.process(x)
    g.reset()
    np.testing.assert_array_equal(g.process(x).data, first)


def test_introspection():
    g = Graph(SR, channels=2, block=64)
    src = g.add(Input)
    filt = g.add(Svf)
    assert len(g) == 2
    assert g.kind(src) == "input" and g.kind(filt) == "svf"
    assert g.params(filt) == ("mode", "cutoff", "q")
    assert g.inputs(filt) == ("in", "cutoff")
    assert g.params(src) == ()
    assert "nodes=2" in repr(g) and "channels=2" in repr(g)
    assert "svf" in Graph.KINDS and "input" in Graph.KINDS


def test_parameters_can_change_after_building():
    x = _noise(frames=2000)
    g = Graph(SR)
    src = g.add(Input)
    filt = g.add(Svf, mode="lowpass", cutoff=5000.0)
    g.connect(src, filt)
    g.set(filt, "cutoff", 300.0)
    g.set(filt, "mode", "highpass")
    g.reset(filt)  # end the ramps
    expected = Svf("highpass", 300.0, sample_rate=SR).process(x)
    np.testing.assert_array_equal(g.process(x).data, expected.data)


def test_kind_accepts_names_as_well_as_classes():
    g = Graph(SR)
    assert g.kind(g.add("saw", freq=200.0)) == "saw"
    assert g.kind(g.add(Saw, freq=200.0)) == "saw"


@pytest.mark.parametrize(
    ("build", "error", "match"),
    [
        (lambda g: g.add(object), ValueError, "unknown node kind"),
        (lambda g: g.add("vocoder"), ValueError, "unknown node kind"),
        (lambda g: g.add(Sine, cutoff=1.0), ValueError, "no parameter"),
        (lambda g: g.add(Svf, mode="allpass"), ValueError, "mode must be one of"),
        (lambda g: g.add(Sine, freq=float("nan")), ValueError, "finite"),
        (lambda g: g.set(0, "freq", 1.0), ValueError, "no node with handle"),
        (lambda g: g.set(True, "freq", 1.0), TypeError, "node handles are ints"),
    ],
)
def test_build_errors(build, error, match):
    with pytest.raises(error, match=match):
        build(Graph(SR))


def test_connection_errors():
    g = Graph(SR)
    a = g.add(Sine, freq=1.0)
    b = g.add(Svf)
    with pytest.raises(ValueError, match="has no input"):
        g.connect(a, b, "resonance")
    with pytest.raises(ValueError, match="added before it"):
        g.connect(b, a)
    with pytest.raises(ValueError, match="added before it"):
        g.connect(a, a)
    with pytest.raises(ValueError, match="no node with handle"):
        g.connect(a, 7)


def test_render_errors():
    with pytest.raises(ValueError, match="no nodes"):
        Graph(SR).generate(8)
    g = Graph(SR, channels=2)
    g.add(Sine, freq=100.0)
    with pytest.raises(ValueError, match="channels"):
        g.process(AudioBuffer(np.zeros(8), SR))
    with pytest.raises(ValueError, match="sample_rate"):
        g.process(AudioBuffer(np.zeros((2, 8)), 44100))
    with pytest.raises(ValueError, match="frames"):
        g.generate(-1)


def test_empty_render():
    g = Graph(SR)
    g.add(Sine, freq=100.0)
    assert g.generate(0).frames == 0


def test_graph_construction_errors():
    with pytest.raises(ValueError, match="sample_rate"):
        Graph(0.0)
    with pytest.raises(ValueError, match="channels"):
        Graph(SR, channels=0)
    with pytest.raises(ValueError, match="block"):
        Graph(SR, block=0)


def test_render_into_out():
    x = _noise(channels=2, frames=256)
    g = Graph(SR, channels=2)
    src = g.add(Input)
    filt = g.add(Svf, cutoff=1200.0)
    g.connect(src, filt)
    g.output = filt
    expected = g.process(x).data.copy()
    g.reset()

    out = AudioBuffer.zeros(2, 256, SR)
    assert g.process(x, out) is out
    np.testing.assert_array_equal(out.data, expected)

    g.reset()
    tone = Graph(SR)
    tone.add(Sine, freq=220.0)
    generated = AudioBuffer.zeros(1, 256, SR)
    assert tone.generate(256, generated) is generated
    np.testing.assert_array_equal(
        generated.data, Sine(220.0, sample_rate=SR).generate(256).data
    )


def test_out_is_checked():
    g = Graph(SR)
    g.add(Sine, freq=100.0)
    with pytest.raises(ValueError, match="out has 32 frames"):
        g.generate(64, AudioBuffer.zeros(1, 32, SR))
    with pytest.raises(ValueError, match="out sample_rate"):
        g.generate(64, AudioBuffer.zeros(1, 64, 44100.0))
    read_only = AudioBuffer.zeros(1, 64, SR)
    read_only.data.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        g.generate(64, read_only)
