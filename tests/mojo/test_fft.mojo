"""Checks the FFT against a direct DFT, and the real transform against it.

Run: mojo run -I src/mdsp/_mojo tests/mojo/test_fft.mojo
"""

from std.math import cos, pi, sin, sqrt
from std.testing import TestSuite, assert_true

from dsp import Fft, RealFft, SamplePtr, round_up_pow2


def _noise(n: Int, seed: UInt32) -> List[Float32]:
    var out = List[Float32](capacity=n)
    var state = seed
    for _ in range(n):
        state = state * 1664525 + 1013904223
        out.append(Float32(state >> 8) / Float32(1 << 23) - 1.0)
    return out^


def _ptr(mut buf: List[Float32]) -> SamplePtr:
    return SamplePtr(unsafe_from_address=Int(buf.unsafe_ptr()))


def _dft(re: List[Float32], im: List[Float32]) -> List[Float64]:
    """Direct transform, interleaved re/im, as the reference."""
    var n = len(re)
    var out = List[Float64](length=2 * n, fill=0.0)
    for k in range(n):
        var sr = 0.0
        var si = 0.0
        for i in range(n):
            var angle = -2.0 * pi * Float64(k) * Float64(i) / Float64(n)
            var c = cos(angle)
            var s = sin(angle)
            sr += Float64(re[i]) * c - Float64(im[i]) * s
            si += Float64(re[i]) * s + Float64(im[i]) * c
        out[2 * k] = sr
        out[2 * k + 1] = si
    return out^


def _worst(a: Float64, b: Float64) -> Float64:
    return a if a > b else b


def test_round_up_pow2() raises:
    var cases: List[Int] = [1, 2, 3, 5, 8, 9, 1000, 1024, 1025]
    var wanted: List[Int] = [2, 2, 4, 8, 8, 16, 1024, 1024, 2048]
    for i in range(len(cases)):
        assert_true(
            round_up_pow2(cases[i]) == wanted[i],
            String("round_up_pow2(", cases[i], ") = ", round_up_pow2(cases[i])),
        )


def test_forward_matches_a_direct_dft() raises:
    var sizes: List[Int] = [2, 8, 64, 256]
    for size in sizes:
        var re = _noise(size, 12345)
        var im = _noise(size, 999)
        var reference = _dft(re, im)
        var fft = Fft(size)
        fft.forward(_ptr(re), _ptr(im))
        var error = 0.0
        for k in range(size):
            error = _worst(error, abs(Float64(re[k]) - reference[2 * k]))
            error = _worst(error, abs(Float64(im[k]) - reference[2 * k + 1]))
        # Float32 over log2(size) stages, against a Float64 reference.
        var tolerance = 1.0e-5 * Float64(size)
        assert_true(
            error < tolerance,
            String("size ", size, ": error ", error, " >= ", tolerance),
        )


def test_inverse_undoes_forward() raises:
    var size = 1024
    var re = _noise(size, 7)
    var im = _noise(size, 8)
    var want_re = re.copy()
    var want_im = im.copy()
    var fft = Fft(size)
    fft.forward(_ptr(re), _ptr(im))
    fft.inverse(_ptr(re), _ptr(im))
    var error = 0.0
    for i in range(size):
        error = _worst(error, abs(Float64(re[i]) - Float64(want_re[i])))
        error = _worst(error, abs(Float64(im[i]) - Float64(want_im[i])))
    assert_true(error < 1.0e-4, String("round trip error ", error))


def test_a_sinusoid_lands_in_one_bin() raises:
    var size = 256
    var re = List[Float32](length=size, fill=0.0)
    var im = List[Float32](length=size, fill=0.0)
    for i in range(size):
        re[i] = Float32(cos(2.0 * pi * 5.0 * Float64(i) / Float64(size)))
    var fft = Fft(size)
    fft.forward(_ptr(re), _ptr(im))
    for k in range(size):
        var magnitude = sqrt(
            Float64(re[k]) * Float64(re[k]) + Float64(im[k]) * Float64(im[k])
        )
        var wanted = Float64(size) / 2.0 if (k == 5 or k == size - 5) else 0.0
        assert_true(
            abs(magnitude - wanted) < 1.0e-2,
            String("bin ", k, ": ", magnitude, " wanted ", wanted),
        )


def test_real_transform_matches_the_complex_one() raises:
    var sizes: List[Int] = [4, 16, 128, 1024]
    for size in sizes:
        var x = _noise(size, 4242)
        var re = x.copy()
        var im = List[Float32](length=size, fill=0.0)
        var complex = Fft(size)
        complex.forward(_ptr(re), _ptr(im))

        var real = RealFft(size)
        var out_re = List[Float32](length=size // 2 + 1, fill=0.0)
        var out_im = List[Float32](length=size // 2 + 1, fill=0.0)
        real.forward(_ptr(x), _ptr(out_re), _ptr(out_im))
        assert_true(real.bins() == size // 2 + 1, "bin count")
        var error = 0.0
        for k in range(size // 2 + 1):
            error = _worst(error, abs(Float64(out_re[k]) - Float64(re[k])))
            error = _worst(error, abs(Float64(out_im[k]) - Float64(im[k])))
        var tolerance = 1.0e-5 * Float64(size)
        assert_true(
            error < tolerance,
            String("size ", size, ": real vs complex ", error),
        )


def test_real_inverse_undoes_real_forward() raises:
    var sizes: List[Int] = [4, 64, 2048]
    for size in sizes:
        var x = _noise(size, 31337)
        var want = x.copy()
        var re = List[Float32](length=size // 2 + 1, fill=0.0)
        var im = List[Float32](length=size // 2 + 1, fill=0.0)
        var real = RealFft(size)
        real.forward(_ptr(x), _ptr(re), _ptr(im))
        for i in range(size):
            x[i] = 0.0
        real.inverse(_ptr(re), _ptr(im), _ptr(x))
        var error = 0.0
        for i in range(size):
            error = _worst(error, abs(Float64(x[i]) - Float64(want[i])))
        assert_true(error < 1.0e-4, String("size ", size, ": round trip ", error))


def test_parseval_holds() raises:
    var size = 512
    var x = _noise(size, 2024)
    var time_energy = 0.0
    for i in range(size):
        time_energy += Float64(x[i]) * Float64(x[i])
    var re = List[Float32](length=size // 2 + 1, fill=0.0)
    var im = List[Float32](length=size // 2 + 1, fill=0.0)
    RealFft(size).forward(_ptr(x), _ptr(re), _ptr(im))
    var band_energy = 0.0
    for k in range(size // 2 + 1):
        var power = Float64(re[k]) * Float64(re[k]) + Float64(im[k]) * Float64(im[k])
        # Every bin but DC and Nyquist stands for a conjugate pair.
        band_energy += power if (k == 0 or k == size // 2) else 2.0 * power
    band_energy /= Float64(size)
    assert_true(
        abs(band_energy - time_energy) < 1.0e-3 * time_energy,
        String("Parseval: ", band_energy, " vs ", time_energy),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
