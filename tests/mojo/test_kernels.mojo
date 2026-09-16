"""Checks the `Processor` contract for every kernel.

Run: mojo run -I src/mdsp/_mojo tests/mojo/test_kernels.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from dsp.filters import PEAKING
from dsp import (
    Adsr,
    Compressor,
    Reverb,
    Shaper,
    Mix,
    Noise,
    Port,
    MAX_INPUTS,
    Biquad,
    Delay,
    Gain,
    OnePole,
    Phasor,
    Ports,
    Processor,
    SamplePtr,
    Saw,
    Sine,
    Square,
    Svf,
)
from dsp.delay import read_position

comptime SR = 48000.0
comptime N = 4096


def _input() -> List[Float32]:
    # Deterministic LCG noise in [-1, 1).
    var out = List[Float32](capacity=N)
    var state: UInt32 = 12345
    for _ in range(N):
        state = state * 1664525 + 1013904223
        out.append(Float32(state >> 8) / Float32(1 << 23) - 1.0)
    return out^


def _ptr(mut buf: List[Float32], offset: Int) -> SamplePtr:
    return SamplePtr(unsafe_from_address=Int(buf.unsafe_ptr()) + offset * 4)


def _run[P: Processor](mut unit: P, src: Int, dst: SamplePtr, n: Int, mod: Int = 0):
    """Process with the audio input at `src` and an optional modulation input."""
    var ports = InlineArray[Port, MAX_INPUTS](fill=Port())
    ports[0] = Port(src, 1)
    ports[1] = Port(mod, 1)
    unit.process(Ports(unsafe_from_address=Int(Pointer(to=ports))), dst, n)


def _check_contract[P: Processor](var unit: P) raises:
    unit.reset()  # start settled: `set` ramps, `reset` snaps
    var x = _input()

    var ticked = List[Float32](length=N, fill=0.0)
    var a = unit.copy()
    for i in range(N):
        ticked[i] = a.tick(x[i])

    # Out-of-place, one block.
    var block = List[Float32](length=N, fill=0.0)
    var b = unit.copy()
    _run(b, Int(_ptr(x, 0)), _ptr(block, 0), N)

    # In-place, irregular block sizes.
    var inplace = x.copy()
    var c = unit.copy()
    var sizes: List[Int] = [1, 7, 64, 3, 513, 0, 1024]
    var pos = 0
    var k = 0
    while pos < N:
        var n = min(sizes[k % len(sizes)], N - pos)
        _run(c, Int(_ptr(inplace, pos)), _ptr(inplace, pos), n)
        pos += n
        k += 1

    for i in range(N):
        assert_equal(block[i], ticked[i], String("process != tick at ", i))
        assert_equal(inplace[i], ticked[i], String("in-place blocks != tick at ", i))

    # reset() restores the post-construction output.
    b.reset()
    var again = List[Float32](length=N, fill=0.0)
    _run(b, Int(_ptr(x, 0)), _ptr(again, 0), N)
    for i in range(N):
        assert_equal(again[i], ticked[i], String("reset did not clear state at ", i))


def test_phasor() raises:
    var u = Phasor(SR)
    u.set(Phasor.FREQ, 441.3)
    _check_contract(u^)


def test_sine() raises:
    var u = Sine(SR)
    u.set(Sine.FREQ, -997.0)
    _check_contract(u^)


def test_saw_and_square_both_directions() raises:
    for sign in range(2):
        var freq = 1733.3 if sign == 0 else -1733.3
        var saw = Saw(SR)
        saw.set(Saw.FREQ, freq)
        _check_contract(saw^)
        var square = Square(SR)
        square.set(Square.FREQ, freq)
        _check_contract(square^)


def test_onepole() raises:
    var u = OnePole(SR)
    u.set(OnePole.CUTOFF, 2500.0)
    _check_contract(u^)


def test_biquad_modes() raises:
    for mode in range(4):
        var u = Biquad(SR)
        u.set(Biquad.MODE, Float64(mode))
        u.set(Biquad.CUTOFF, 3000.0)
        u.set(Biquad.Q, 2.0)
        _check_contract(u^)


def test_gain() raises:
    var u = Gain(SR)
    u.set(Gain.GAIN, -0.25)
    _check_contract(u^)


def test_delay_fractional_with_feedback() raises:
    var u = Delay(SR)
    u.set(Delay.MAX_DELAY, 0.01)
    u.set(Delay.DELAY, 37.25 / SR)
    u.set(Delay.FEEDBACK, -0.7)
    u.set(Delay.MIX, 0.6)
    _check_contract(u^)


def test_delay_wraps_short_line() raises:
    # Line of 3 slots: the read index wraps on almost every sample.
    var u = Delay(SR)
    u.set(Delay.MAX_DELAY, 0.5 / SR)
    u.set(Delay.DELAY, 1.0e9)
    u.set(Delay.FEEDBACK, 0.5)
    _check_contract(u^)


def test_delay_read_position_stays_below_size() raises:
    # 1 - (1 + 1e-12) + 48002 rounds to exactly 48002 in Float64.
    var size = 48002
    assert_equal(Float64(1) - (1.0 + 1.0e-12) + Float64(size), Float64(size))
    var r = read_position(1, 1.0 + 1.0e-12, size)
    assert_true(r >= 0.0 and r < Float64(size), String("r = ", r))
    assert_equal(read_position(5, 2.5, size), 2.5)
    assert_equal(read_position(0, 1.0, size), Float64(size - 1))


def test_mix() raises:
    var u = Mix(SR)
    u.set(Mix.GAIN, 0.5)
    u.set(Mix.GAIN2, 0.25)
    _check_contract(u^)


def test_noise() raises:
    var u = Noise(SR)
    u.set(Noise.SEED, 99.0)
    _check_contract(u^)


def test_adsr() raises:
    var u = Adsr(SR)
    u.set(Adsr.ATTACK_TIME, 0.001)
    u.set(Adsr.DECAY_TIME, 0.002)
    u.set(Adsr.SUSTAIN_LEVEL, 0.4)
    u.set(Adsr.RELEASE_TIME, 0.003)
    u.set(Adsr.GATE, 1.0)
    _check_contract(u^)


def test_mix_sums_its_inputs() raises:
    var ones = List[Float32](length=N, fill=1.0)
    var halves = List[Float32](length=N, fill=0.5)
    var out = List[Float32](length=N, fill=0.0)
    var u = Mix(SR)
    u.set(Mix.GAIN, 2.0)
    u.set(Mix.GAIN2, 4.0)
    u.reset()
    var ports = InlineArray[Port, MAX_INPUTS](fill=Port())
    ports[0] = Port(Int(_ptr(ones, 0)), 1)
    ports[1] = Port(Int(_ptr(halves, 0)), 1)
    u.process(Ports(unsafe_from_address=Int(Pointer(to=ports))), _ptr(out, 0), N)
    for i in range(N):
        assert_equal(out[i], Float32(4.0), String("mix at ", i))


def test_adsr_stages() raises:
    var u = Adsr(SR)
    u.set(Adsr.ATTACK_TIME, 10.0 / SR)  # 10 samples
    u.set(Adsr.DECAY_TIME, 10.0 / SR)
    u.set(Adsr.SUSTAIN_LEVEL, 0.5)
    u.set(Adsr.RELEASE_TIME, 10.0 / SR)
    u.reset()
    assert_equal(u.tick(0.0), Float32(0.0))  # gate closed: silent
    u.set(Adsr.GATE, 1.0)
    for _ in range(10):
        _ = u.tick(0.0)
    assert_equal(u.level, 1.0)  # attack reached the top
    for _ in range(10):
        _ = u.tick(0.0)
    assert_equal(u.level, 0.5)  # decayed to sustain
    _ = u.tick(0.0)
    assert_equal(u.level, 0.5)  # holds while gated
    u.set(Adsr.GATE, 0.0)
    for _ in range(10):
        _ = u.tick(0.0)
    assert_equal(u.level, 0.0)  # released


def test_compressor() raises:
    var u = Compressor(SR)
    u.set(Compressor.THRESHOLD, -18.0)
    u.set(Compressor.RATIO, 6.0)
    u.set(Compressor.ATTACK, 0.002)
    u.set(Compressor.RELEASE, 0.05)
    _check_contract(u^)


def test_shaper() raises:
    for shape in range(3):
        var u = Shaper(SR)
        u.set(Shaper.DRIVE, 3.0)
        u.set(Shaper.SHAPE, Float64(shape))
        _check_contract(u^)


def test_reverb() raises:
    var u = Reverb(SR)
    u.set(Reverb.ROOM_SIZE, 0.7)
    u.set(Reverb.DAMPING, 0.3)
    u.set(Reverb.MIX, 0.4)
    _check_contract(u^)


def test_unknown_param_is_noop() raises:
    var u = OnePole(SR)
    var before = u.a
    u.set(99, 5.0)
    assert_equal(u.a, before)


def test_params_are_clamped() raises:
    var u = Biquad(SR)
    u.set(Biquad.CUTOFF, 1.0e9)
    u.set(Biquad.Q, 0.0)
    u.set(Biquad.MODE, 42.0)
    assert_equal(u.cutoff.target, 0.4999 * SR)
    assert_equal(u.q.target, 1.0e-3)
    assert_equal(u.mode, PEAKING)  # clamped to the last mode
    var d = Delay(SR)
    d.set(Delay.MAX_DELAY, 1.0e9)
    assert_equal(d.max_delay, 600.0)
    d.set(Delay.MAX_DELAY, 0.001)
    d.set(Delay.DELAY, 10.0)
    assert_equal(d.delay_samples.target, Float64(len(d.line) - 2))


def _process_split[P: Processor](var unit: P, mut x: List[Float32], sizes: List[Int]) -> List[Float32]:
    var out = List[Float32](length=len(x), fill=0.0)
    var pos = 0
    var k = 0
    while pos < len(x):
        var n = min(sizes[k % len(sizes)], len(x) - pos)
        _run(unit, Int(_ptr(x, pos)), _ptr(out, pos), n)
        pos += n
        k += 1
    return out^


def test_svf_modes() raises:
    for mode in range(4):
        var u = Svf(SR)
        u.set(Svf.MODE, Float64(mode))
        u.set(Svf.CUTOFF, 2200.0)
        u.set(Svf.Q, 3.0)
        u.reset()
        _check_contract(u^)


def test_modulation_input_matches_constant_parameter() raises:
    var x = _input()
    var cutoff = List[Float32](length=N, fill=1234.0)

    var svf_const = Svf(SR)
    svf_const.set(Svf.CUTOFF, 1234.0)
    svf_const.reset()
    var svf_mod = Svf(SR)
    var a = List[Float32](length=N, fill=0.0)
    var b = List[Float32](length=N, fill=0.0)
    _run(svf_const, Int(_ptr(x, 0)), _ptr(a, 0), N)
    _run(svf_mod, Int(_ptr(x, 0)), _ptr(b, 0), N, Int(_ptr(cutoff, 0)))
    for i in range(N):
        assert_equal(b[i], a[i], String("svf cutoff modulation differs at ", i))

    var lp_const = OnePole(SR)
    lp_const.set(OnePole.CUTOFF, 1234.0)
    lp_const.reset()
    var lp_mod = OnePole(SR)
    var c = List[Float32](length=N, fill=0.0)
    var d = List[Float32](length=N, fill=0.0)
    _run(lp_const, Int(_ptr(x, 0)), _ptr(c, 0), N)
    _run(lp_mod, Int(_ptr(x, 0)), _ptr(d, 0), N, Int(_ptr(cutoff, 0)))
    for i in range(N):
        assert_equal(d[i], c[i], String("one-pole cutoff modulation differs at ", i))


def test_smoothing_is_block_size_independent() raises:
    var x = _input()
    var whole: List[Int] = [N]
    var irregular: List[Int] = [1, 7, 64, 3, 513, 1024]

    var gain = Gain(SR)
    gain.reset()
    gain.set(Gain.GAIN, 0.25)  # ramps from 1.0
    var svf = Svf(SR)
    svf.reset()
    svf.set(Svf.CUTOFF, 6000.0)

    var g1 = _process_split(gain.copy(), x, whole)
    var g2 = _process_split(gain.copy(), x, irregular)
    var s1 = _process_split(svf.copy(), x, whole)
    var s2 = _process_split(svf.copy(), x, irregular)
    for i in range(N):
        assert_equal(g2[i], g1[i], String("gain ramp depends on block size at ", i))
        assert_equal(s2[i], s1[i], String("cutoff ramp depends on block size at ", i))


def test_reset_ends_the_ramp() raises:
    var x = _input()
    var ramped = Gain(SR)
    ramped.reset()
    ramped.set(Gain.GAIN, 0.25)
    ramped.reset()
    var direct = Gain(SR)
    direct.set(Gain.GAIN, 0.25)
    direct.reset()
    var a = List[Float32](length=N, fill=0.0)
    var b = List[Float32](length=N, fill=0.0)
    _run(ramped, Int(_ptr(x, 0)), _ptr(a, 0), N)
    _run(direct, Int(_ptr(x, 0)), _ptr(b, 0), N)
    for i in range(N):
        assert_equal(a[i], b[i], String("reset did not end the ramp at ", i))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
