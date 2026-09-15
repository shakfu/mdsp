"""Checks the `Processor` contract for every kernel.

Run: mojo run -I src/mdsp/_mojo tests/mojo/test_kernels.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from dsp import (
    Biquad,
    Delay,
    Gain,
    OnePole,
    Phasor,
    Processor,
    SamplePtr,
    Saw,
    Sine,
    Square,
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


def _check_contract[P: Processor](unit: P) raises:
    var x = _input()

    var ticked = List[Float32](length=N, fill=0.0)
    var a = unit.copy()
    for i in range(N):
        ticked[i] = a.tick(x[i])

    # Out-of-place, one block.
    var block = List[Float32](length=N, fill=0.0)
    var b = unit.copy()
    b.process(_ptr(x, 0), _ptr(block, 0), N)

    # In-place, irregular block sizes.
    var inplace = x.copy()
    var c = unit.copy()
    var sizes: List[Int] = [1, 7, 64, 3, 513, 0, 1024]
    var pos = 0
    var k = 0
    while pos < N:
        var n = min(sizes[k % len(sizes)], N - pos)
        c.process(_ptr(inplace, pos), _ptr(inplace, pos), n)
        pos += n
        k += 1

    for i in range(N):
        assert_equal(block[i], ticked[i], String("process != tick at ", i))
        assert_equal(inplace[i], ticked[i], String("in-place blocks != tick at ", i))

    # reset() restores the post-construction output.
    b.reset()
    var again = List[Float32](length=N, fill=0.0)
    b.process(_ptr(x, 0), _ptr(again, 0), N)
    for i in range(N):
        assert_equal(again[i], ticked[i], String("reset did not clear state at ", i))


def test_phasor() raises:
    var u = Phasor(SR)
    u.set(Phasor.FREQ, 441.3)
    _check_contract(u)


def test_sine() raises:
    var u = Sine(SR)
    u.set(Sine.FREQ, -997.0)
    _check_contract(u)


def test_saw_and_square_both_directions() raises:
    for sign in range(2):
        var freq = 1733.3 if sign == 0 else -1733.3
        var saw = Saw(SR)
        saw.set(Saw.FREQ, freq)
        _check_contract(saw)
        var square = Square(SR)
        square.set(Square.FREQ, freq)
        _check_contract(square)


def test_onepole() raises:
    var u = OnePole(SR)
    u.set(OnePole.CUTOFF, 2500.0)
    _check_contract(u)


def test_biquad_modes() raises:
    for mode in range(4):
        var u = Biquad(SR)
        u.set(Biquad.MODE, Float64(mode))
        u.set(Biquad.CUTOFF, 3000.0)
        u.set(Biquad.Q, 2.0)
        _check_contract(u)


def test_gain() raises:
    var u = Gain(SR)
    u.set(Gain.GAIN, -0.25)
    _check_contract(u)


def test_delay_fractional_with_feedback() raises:
    var u = Delay(SR)
    u.set(Delay.MAX_DELAY, 0.01)
    u.set(Delay.DELAY, 37.25 / SR)
    u.set(Delay.FEEDBACK, -0.7)
    u.set(Delay.MIX, 0.6)
    _check_contract(u)


def test_delay_wraps_short_line() raises:
    # Line of 3 slots: the read index wraps on almost every sample.
    var u = Delay(SR)
    u.set(Delay.MAX_DELAY, 0.5 / SR)
    u.set(Delay.DELAY, 1.0e9)
    u.set(Delay.FEEDBACK, 0.5)
    _check_contract(u)


def test_delay_read_position_stays_below_size() raises:
    # 1 - (1 + 1e-12) + 48002 rounds to exactly 48002 in Float64.
    var size = 48002
    assert_equal(Float64(1) - (1.0 + 1.0e-12) + Float64(size), Float64(size))
    var r = read_position(1, 1.0 + 1.0e-12, size)
    assert_true(r >= 0.0 and r < Float64(size), String("r = ", r))
    assert_equal(read_position(5, 2.5, size), 2.5)
    assert_equal(read_position(0, 1.0, size), Float64(size - 1))


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
    assert_equal(u.cutoff, 0.4999 * SR)
    assert_equal(u.q, 1.0e-3)
    assert_equal(u.mode, Biquad.NOTCH)
    var d = Delay(SR)
    d.set(Delay.MAX_DELAY, 1.0e9)
    assert_equal(d.max_delay, 600.0)
    d.set(Delay.MAX_DELAY, 0.001)
    d.set(Delay.DELAY, 10.0)
    assert_equal(d.delay_samples, Float64(len(d.line) - 2))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
