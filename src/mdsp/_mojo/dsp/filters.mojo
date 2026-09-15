from std.math import clamp, cos, exp, pi, sin

from dsp.processor import Processor, SamplePtr


def _clamp_freq(hz: Float64, sample_rate: Float64) -> Float64:
    return clamp(hz, 1.0e-3, 0.4999 * sample_rate)


struct OnePole(Processor, Writable):
    """One-pole lowpass: `y += a * (x - y)`, with `a` set from `cutoff` in Hz."""

    comptime CUTOFF = 0

    var sample_rate: Float64
    var cutoff: Float64
    var a: Float32
    var z: Float32

    def __init__(out self, sample_rate: Float64):
        self.sample_rate = sample_rate
        self.cutoff = 0.0
        self.a = 0.0
        self.z = 0.0
        self.set(Self.CUTOFF, 1000.0)

    @staticmethod
    def param_names() -> List[String]:
        return ["cutoff"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.CUTOFF:
            self.cutoff = _clamp_freq(value, self.sample_rate)
            self.a = Float32(1.0 - exp(-2.0 * pi * self.cutoff / self.sample_rate))

    def reset(mut self):
        self.z = 0.0

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        self.z += self.a * (x - self.z)
        return self.z

    def process(mut self, src: SamplePtr, dst: SamplePtr, n: Int):
        var z = self.z
        var a = self.a
        for i in range(n):
            z += a * (src[unsafe_offset=i] - z)
            dst[unsafe_offset=i] = z
        self.z = z


struct Biquad(Processor, Writable):
    """RBJ cookbook biquad in transposed direct form II.

    Coefficients and state are Float64: Float32 state loses precision at low
    cutoffs.
    """

    comptime MODE = 0
    comptime CUTOFF = 1
    comptime Q = 2

    comptime LOWPASS = 0
    comptime HIGHPASS = 1
    comptime BANDPASS = 2
    comptime NOTCH = 3

    var sample_rate: Float64
    var mode: Int
    var cutoff: Float64
    var q: Float64
    var b0: Float64
    var b1: Float64
    var b2: Float64
    var a1: Float64
    var a2: Float64
    var s1: Float64
    var s2: Float64

    def __init__(out self, sample_rate: Float64):
        self.sample_rate = sample_rate
        self.mode = Self.LOWPASS
        self.cutoff = _clamp_freq(1000.0, sample_rate)
        self.q = 0.7071067811865476
        self.b0 = 1.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.s1 = 0.0
        self.s2 = 0.0
        self._update()

    @staticmethod
    def param_names() -> List[String]:
        return ["mode", "cutoff", "q"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.MODE:
            self.mode = Int(clamp(value, 0.0, 3.0))
        elif param == Self.CUTOFF:
            self.cutoff = _clamp_freq(value, self.sample_rate)
        elif param == Self.Q:
            self.q = clamp(value, 1.0e-3, 1.0e3)
        else:
            return
        self._update()

    def _update(mut self):
        var w0 = 2.0 * pi * self.cutoff / self.sample_rate
        var cw = cos(w0)
        var alpha = sin(w0) / (2.0 * self.q)
        var a0 = 1.0 + alpha
        var b0: Float64
        var b1: Float64
        var b2: Float64
        if self.mode == Self.LOWPASS:
            b0 = (1.0 - cw) / 2.0
            b1 = 1.0 - cw
            b2 = b0
        elif self.mode == Self.HIGHPASS:
            b0 = (1.0 + cw) / 2.0
            b1 = -(1.0 + cw)
            b2 = b0
        elif self.mode == Self.BANDPASS:
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
        else:
            b0 = 1.0
            b1 = -2.0 * cw
            b2 = 1.0
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = -2.0 * cw / a0
        self.a2 = (1.0 - alpha) / a0

    def reset(mut self):
        self.s1 = 0.0
        self.s2 = 0.0

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        var xd = Float64(x)
        var y = self.b0 * xd + self.s1
        self.s1 = self.b1 * xd - self.a1 * y + self.s2
        self.s2 = self.b2 * xd - self.a2 * y
        return Float32(y)

    def process(mut self, src: SamplePtr, dst: SamplePtr, n: Int):
        var b0 = self.b0
        var b1 = self.b1
        var b2 = self.b2
        var a1 = self.a1
        var a2 = self.a2
        var s1 = self.s1
        var s2 = self.s2
        for i in range(n):
            var x = Float64(src[unsafe_offset=i])
            var y = b0 * x + s1
            s1 = b1 * x - a1 * y + s2
            s2 = b2 * x - a2 * y
            dst[unsafe_offset=i] = Float32(y)
        self.s1 = s1
        self.s2 = s2
