"""Radix-2 FFT, and the real-input transform built on it.

Sizes are powers of two. Twiddles and the bit-reversal permutation are built
once, at construction, so a transform allocates nothing: these run on the
audio thread. Buffers are split real and imaginary rather than interleaved,
which is the layout a SIMD butterfly wants.
"""

from std.math import ceil, cos, log2, pi, sin

from dsp.processor import SamplePtr


@always_inline
def round_up_pow2(n: Int) -> Int:
    """Smallest power of two at least *n*, and at least 2."""
    var size = 2
    while size < n:
        size <<= 1
    return size


struct Fft(Copyable, Movable, Writable):
    """In-place complex transform of `size` points, decimation in time."""

    var size: Int
    var wr: List[Float32]  # cos of the twiddle angles, size / 2 of them
    var wi: List[Float32]  # sin of them
    var rev: List[Int32]  # bit-reversal permutation

    def __init__(out self, size: Int):
        self.size = round_up_pow2(size)
        self.wr = List[Float32](length=self.size // 2, fill=0.0)
        self.wi = List[Float32](length=self.size // 2, fill=0.0)
        for k in range(self.size // 2):
            var angle = -2.0 * pi * Float64(k) / Float64(self.size)
            self.wr[k] = Float32(cos(angle))
            self.wi[k] = Float32(sin(angle))
        self.rev = List[Int32](length=self.size, fill=0)
        var bits = Int(log2(Float64(self.size)) + 0.5)
        for i in range(self.size):
            var r = 0
            for b in range(bits):
                if i & (1 << b) != 0:
                    r |= 1 << (bits - 1 - b)
            self.rev[i] = Int32(r)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Fft(size=", self.size, ")")

    def forward(self, re: SamplePtr, im: SamplePtr):
        """Unscaled transform of `size` complex points, in place."""
        var rev = self.rev.unsafe_ptr()
        for i in range(self.size):
            var j = Int(rev[unsafe_offset=i])
            if j > i:
                var tr = re[unsafe_offset=i]
                re[unsafe_offset=i] = re[unsafe_offset=j]
                re[unsafe_offset=j] = tr
                var ti = im[unsafe_offset=i]
                im[unsafe_offset=i] = im[unsafe_offset=j]
                im[unsafe_offset=j] = ti
        var wr = self.wr.unsafe_ptr()
        var wi = self.wi.unsafe_ptr()
        var span = 2
        while span <= self.size:
            var half = span // 2
            var stride = self.size // span
            var base = 0
            while base < self.size:
                for j in range(half):
                    var k = j * stride
                    var cr = wr[unsafe_offset=k]
                    var ci = wi[unsafe_offset=k]
                    var a = base + j
                    var b = a + half
                    var br = re[unsafe_offset=b]
                    var bi = im[unsafe_offset=b]
                    var tr = br * cr - bi * ci
                    var ti = br * ci + bi * cr
                    re[unsafe_offset=b] = re[unsafe_offset=a] - tr
                    im[unsafe_offset=b] = im[unsafe_offset=a] - ti
                    re[unsafe_offset=a] = re[unsafe_offset=a] + tr
                    im[unsafe_offset=a] = im[unsafe_offset=a] + ti
                base += span
            span <<= 1

    def inverse(self, re: SamplePtr, im: SamplePtr):
        """Transform back, scaled by 1 / size.

        Conjugating either side of a forward transform is the inverse, so
        there is no second twiddle table.
        """
        for i in range(self.size):
            im[unsafe_offset=i] = -im[unsafe_offset=i]
        self.forward(re, im)
        var scale = Float32(1.0 / Float64(self.size))
        for i in range(self.size):
            re[unsafe_offset=i] = re[unsafe_offset=i] * scale
            im[unsafe_offset=i] = -im[unsafe_offset=i] * scale


struct RealFft(Copyable, Movable, Writable):
    """Transform of `size` real points through a complex one of half that.

    Real input has a conjugate-symmetric spectrum, so half the transform is
    redundant. Packing the even samples into the real part and the odd ones
    into the imaginary part, transforming at `size / 2` and untangling the two
    halves costs one extra pass and saves the other half of the work.

    `forward` writes the `size / 2 + 1` bins from DC to Nyquist. The imaginary
    parts of those two are zero and are written as such.
    """

    var size: Int
    var half: Fft
    var tr: List[Float32]  # exp(-2 pi i k / size), k < size / 2
    var ti: List[Float32]

    def __init__(out self, size: Int):
        self.size = max(round_up_pow2(size), 4)
        self.half = Fft(self.size // 2)
        var quarter = self.size // 2
        self.tr = List[Float32](length=quarter, fill=0.0)
        self.ti = List[Float32](length=quarter, fill=0.0)
        for k in range(quarter):
            var angle = -2.0 * pi * Float64(k) / Float64(self.size)
            self.tr[k] = Float32(cos(angle))
            self.ti[k] = Float32(sin(angle))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("RealFft(size=", self.size, ")")

    def bins(self) -> Int:
        return self.size // 2 + 1

    def forward(self, x: SamplePtr, re: SamplePtr, im: SamplePtr):
        """Read `size` reals from *x*; write `bins()` complex to *re*, *im*."""
        var m = self.size // 2
        for k in range(m):
            re[unsafe_offset=k] = x[unsafe_offset=2 * k]
            im[unsafe_offset=k] = x[unsafe_offset=2 * k + 1]
        self.half.forward(re, im)
        # Split the half-size spectrum into the even and odd spectra of the
        # real signal, then recombine them with a full-size twiddle.
        var zero_r = re[unsafe_offset=0]
        var zero_i = im[unsafe_offset=0]
        var tr = self.tr.unsafe_ptr()
        var ti = self.ti.unsafe_ptr()
        for k in range(1, m // 2 + 1):
            var j = m - k
            var kr = re[unsafe_offset=k]
            var ki = im[unsafe_offset=k]
            var jr = re[unsafe_offset=j]
            var ji = im[unsafe_offset=j]
            var er = 0.5 * (kr + jr)  # E[k], even-sample spectrum
            var ei = 0.5 * (ki - ji)
            var or_ = 0.5 * (ki + ji)  # O[k], odd-sample spectrum, times -i
            var oi = 0.5 * (jr - kr)
            var cr = tr[unsafe_offset=k]
            var ci = ti[unsafe_offset=k]
            var br = or_ * cr - oi * ci
            var bi = or_ * ci + oi * cr
            re[unsafe_offset=k] = er + br
            im[unsafe_offset=k] = ei + bi
            re[unsafe_offset=j] = er - br  # X[m - k] is the conjugate
            im[unsafe_offset=j] = -(ei - bi)
        re[unsafe_offset=0] = zero_r + zero_i  # DC
        im[unsafe_offset=0] = 0.0
        re[unsafe_offset=m] = zero_r - zero_i  # Nyquist
        im[unsafe_offset=m] = 0.0

    def inverse(self, re: SamplePtr, im: SamplePtr, x: SamplePtr):
        """Read `bins()` complex from *re*, *im*; write `size` reals to *x*.

        The spectrum is consumed: *re* and *im* are left holding the half-size
        complex signal, not what was passed in.
        """
        var m = self.size // 2
        var dc = re[unsafe_offset=0]
        var nyquist = re[unsafe_offset=m]
        var tr = self.tr.unsafe_ptr()
        var ti = self.ti.unsafe_ptr()
        for k in range(1, m // 2 + 1):
            var j = m - k
            var kr = re[unsafe_offset=k]
            var ki = im[unsafe_offset=k]
            var jr = re[unsafe_offset=j]
            var ji = im[unsafe_offset=j]
            var er = 0.5 * (kr + jr)  # E[k] again, from X[k] and X[m - k]
            var ei = 0.5 * (ki - ji)
            var br = 0.5 * (kr - jr)  # W[k] * O[k]
            var bi = 0.5 * (ki + ji)
            var cr = tr[unsafe_offset=k]
            var ci = -ti[unsafe_offset=k]  # conjugate twiddle undoes it
            var or_ = br * cr - bi * ci
            var oi = br * ci + bi * cr
            re[unsafe_offset=k] = er - oi  # Z[k] = E[k] + i * O[k]
            im[unsafe_offset=k] = ei + or_
            re[unsafe_offset=j] = er + oi
            im[unsafe_offset=j] = -(ei - or_)
        re[unsafe_offset=0] = 0.5 * (dc + nyquist)
        im[unsafe_offset=0] = 0.5 * (dc - nyquist)
        self.half.inverse(re, im)
        for k in range(m):
            x[unsafe_offset=2 * k] = re[unsafe_offset=k]
            x[unsafe_offset=2 * k + 1] = im[unsafe_offset=k]
