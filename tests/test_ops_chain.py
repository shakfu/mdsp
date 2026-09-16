import numpy as np
import pytest

from mdsp import AudioBuffer, Biquad, Chain, Gain, OnePole, Scale, Sine

SR = 48000.0


def test_gain():
    x = AudioBuffer(np.linspace(-1, 1, 64).reshape(2, 32), SR)
    np.testing.assert_array_equal(
        Gain(-0.5, channels=2).process(x).data, x.data * np.float32(-0.5)
    )


def test_process_does_not_modify_input():
    x = AudioBuffer(np.ones(16), SR)
    Gain(3.0).process(x)
    assert (x.data == 1.0).all()


def test_chain_equals_sequential_processing():
    x = AudioBuffer(np.random.default_rng(1).uniform(-1, 1, (2, 1000)), SR)
    kwargs = {"sample_rate": SR, "channels": 2}
    chain = Chain(
        OnePole(2000.0, **kwargs),
        Biquad("highpass", 100.0, **kwargs),
        Gain(0.3, **kwargs),
    )
    expected = Gain(0.3, **kwargs).process(
        Biquad("highpass", 100.0, **kwargs).process(
            OnePole(2000.0, **kwargs).process(x)
        )
    )
    np.testing.assert_array_equal(chain.process(x).data, expected.data)


def test_chain_reset_and_nesting():
    x = AudioBuffer(np.random.default_rng(2).uniform(-1, 1, 500), SR)
    chain = Chain(Chain(OnePole(300.0)), Gain(2.0))
    first = chain.process(x).data
    chain.reset()
    np.testing.assert_array_equal(chain.process(x).data, first)
    assert repr(chain).startswith("Chain(Chain(OnePole(")


def test_empty_chain_is_identity():
    x = AudioBuffer(np.ones(4), SR)
    assert Chain().process(x) is x


def test_chain_propagates_mismatch_errors():
    with pytest.raises(ValueError, match="channels"):
        Chain(Gain(channels=2)).process(AudioBuffer(np.ones(4), SR))


def test_scale_linear_endpoints_and_midpoint():
    x = AudioBuffer([-1.0, -0.5, 0.0, 0.5, 1.0], SR)
    out = Scale(100.0, 200.0, sample_rate=SR).process(x).data[0]
    np.testing.assert_allclose(out, [100.0, 125.0, 150.0, 175.0, 200.0])


def test_scale_exponential_is_geometric():
    x = AudioBuffer([-1.0, 0.0, 1.0], SR)
    out = Scale(100.0, 1600.0, "exponential", sample_rate=SR).process(x).data[0]
    np.testing.assert_allclose(out, [100.0, 400.0, 1600.0], rtol=1e-5)


def test_scale_exponential_clamps_non_positive_bounds():
    out = (
        Scale(0.0, 100.0, "exponential", sample_rate=SR)
        .process(AudioBuffer([-1.0, 1.0], SR))
        .data[0]
    )
    assert out[0] > 0.0 and out[1] == pytest.approx(100.0, rel=1e-5)


def test_scale_rejects_unknown_curve():
    with pytest.raises(ValueError, match="curve must be one of"):
        Scale(0.0, 1.0, "logarithmic")  # type: ignore[arg-type]


def test_scale_drives_a_filter_cutoff():
    lfo = Sine(2.0, sample_rate=SR).generate(4800)
    cutoff = Scale(200.0, 4000.0, "exponential", sample_rate=SR).process(lfo)
    assert cutoff.data.min() >= 200.0 - 1e-3
    assert cutoff.data.max() <= 4000.0 + 1e-3


def test_out_matches_the_allocating_version():
    x = AudioBuffer(np.random.default_rng(3).uniform(-1, 1, (2, 500)), SR)
    kwargs = {"sample_rate": SR, "channels": 2}
    expected = Chain(OnePole(700.0, **kwargs), Gain(0.4, **kwargs)).process(x).data
    chain = Chain(OnePole(700.0, **kwargs), Gain(0.4, **kwargs))
    out = AudioBuffer.zeros(2, 500, SR)
    result = chain.process(x, out)
    assert result is out
    np.testing.assert_array_equal(out.data, expected)


def test_chain_reuses_one_scratch_buffer():
    x = AudioBuffer(np.ones((1, 64)), SR)
    chain = Chain(Gain(2.0), Gain(3.0), Gain(0.5))
    out = AudioBuffer.zeros(1, 64, SR)
    chain.process(x, out)
    scratch = chain._scratch
    for _ in range(3):
        chain.process(x, out)
    assert chain._scratch is scratch  # no new allocation per call
    np.testing.assert_allclose(out.data, 3.0)


@pytest.mark.parametrize("stages", [0, 1, 2, 3])
def test_chain_out_for_any_stage_count(stages):
    x = AudioBuffer(np.ones((1, 16)), SR)
    chain = Chain(*[Gain(2.0) for _ in range(stages)])
    out = AudioBuffer.zeros(1, 16, SR)
    np.testing.assert_allclose(chain.process(x, out).data, 2.0**stages)


def test_unit_processes_in_place():
    data = np.random.default_rng(4).uniform(-1, 1, (1, 256))
    x, copy = AudioBuffer(data, SR), AudioBuffer(data, SR)
    expected = OnePole(500.0, sample_rate=SR).process(copy).data
    result = OnePole(500.0, sample_rate=SR).process(x, x)
    assert result is x
    np.testing.assert_array_equal(x.data, expected)


def test_generate_into_out():
    out = AudioBuffer.zeros(1, 128, SR)
    osc = Sine(440.0, sample_rate=SR)
    assert osc.generate(128, out) is out
    np.testing.assert_array_equal(
        out.data, Sine(440.0, sample_rate=SR).generate(128).data
    )


@pytest.mark.parametrize(
    ("out", "match"),
    [
        (AudioBuffer.zeros(1, 64, 44100.0), "sample_rate"),
        (AudioBuffer.zeros(2, 64, SR), "channels"),
        (AudioBuffer.zeros(1, 32, SR), "shape"),
    ],
)
def test_out_must_match(out, match):
    x = AudioBuffer(np.zeros((1, 64)), SR)
    with pytest.raises(ValueError, match=match):
        Gain().process(x, out)


def test_generate_out_frame_count_is_checked():
    with pytest.raises(ValueError, match="frames"):
        Sine().generate(64, AudioBuffer.zeros(1, 32, 48000.0))
