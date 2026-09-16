from std.math import clamp, cos, exp, pi, sin, tan

from dsp.processor import Ports, Processor, SamplePtr, audio_input, input_address
from dsp.smooth import Smoothed

comptime LOWPASS = 0
comptime HIGHPASS = 1
comptime BANDPASS = 2
comptime NOTCH = 3
comptime LOWSHELF = 4
comptime HIGHSHELF = 5
comptime PEAKING = 6


def _clamp_freq(hz: Float64, sample_rate: Float64) -> Float64:
    return clamp(hz, 1.0e-3, 0.4999 * sample_rate)


struct OnePole(Processor, Writable):
    """One-pole lowpass: `y += a * (x - y)`, with `a` set from `cutoff` in Hz.

    Ports: audio, `cutoff` in Hz.
    """

    comptime CUTOFF = 0

    var sample_rate: Float64
    var cutoff: Smoothed
    var coefficient_for: Float64
    var a: Float32
    var z: Float32

    def __init__(out self, sample_rate: Float64):
        self.sample_rate = sample_rate
        self.cutoff = Smoothed(1000.0, sample_rate)
        self.coefficient_for = 0.0
        self.a = 0.0
        self.z = 0.0
        self._update(self.cutoff.value)

    @staticmethod
    def param_names() -> List[String]:
        return ["cutoff"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in", "cutoff"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.CUTOFF:
            self.cutoff.set(_clamp_freq(value, self.sample_rate))

    def reset(mut self):
        self.cutoff.snap()
        self._update(self.cutoff.value)
        self.z = 0.0

    @always_inline
    def _update(mut self, cutoff: Float64):
        self.coefficient_for = cutoff
        self.a = Float32(1.0 - exp(-2.0 * pi * cutoff / self.sample_rate))

    @always_inline
    def _update_if_changed(mut self, cutoff: Float64):
        if cutoff != self.coefficient_for:
            self._update(cutoff)

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        self._update_if_changed(self.cutoff.next())
        self.z += self.a * (x - self.z)
        return self.z

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var src = audio_input(ins, dst)
        var cutoff_mod = input_address(ins, 1)
        var z = self.z
        if cutoff_mod != 0:
            var cutoff = SamplePtr(unsafe_from_address=cutoff_mod)
            for i in range(n):
                self._update_if_changed(
                    _clamp_freq(Float64(cutoff[unsafe_offset=i]), self.sample_rate)
                )
                z += self.a * (src[unsafe_offset=i] - z)
                dst[unsafe_offset=i] = z
            self.z = z
            return
        var i = 0
        var ramping = self.cutoff.ramping(n)
        while i < ramping:
            self._update_if_changed(self.cutoff.next())
            z += self.a * (src[unsafe_offset=i] - z)
            dst[unsafe_offset=i] = z
            i += 1
        var a = self.a
        while i < n:
            z += a * (src[unsafe_offset=i] - z)
            dst[unsafe_offset=i] = z
            i += 1
        self.z = z


struct Biquad(Processor, Writable):
    """RBJ cookbook biquad in transposed direct form II.

    No modulation input: coefficients cost a `cos` and `sin` per sample, and
    this form overshoots badly when they change fast. Use `Svf` for modulation
    (docs/dev/spikes/2026-09-16-interface). Coefficients and state are Float64;
    Float32 state loses precision at low cutoffs.
    """

    comptime MODE = 0
    comptime CUTOFF = 1
    comptime Q = 2
    comptime GAIN = 3

    var sample_rate: Float64
    var mode: Int
    var cutoff: Smoothed
    var q: Smoothed
    var gain: Float64  # dB, used by the shelf and peaking modes
    var cutoff_for: Float64
    var q_for: Float64
    var b0: Float64
    var b1: Float64
    var b2: Float64
    var a1: Float64
    var a2: Float64
    var s1: Float64
    var s2: Float64

    def __init__(out self, sample_rate: Float64):
        self.sample_rate = sample_rate
        self.mode = LOWPASS
        self.cutoff = Smoothed(_clamp_freq(1000.0, sample_rate), sample_rate)
        self.q = Smoothed(0.7071067811865476, sample_rate)
        self.gain = 0.0
        self.cutoff_for = 0.0
        self.q_for = 0.0
        self.b0 = 1.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.s1 = 0.0
        self.s2 = 0.0
        self._update(self.cutoff.value, self.q.value)

    @staticmethod
    def param_names() -> List[String]:
        return ["mode", "cutoff", "q", "gain"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.MODE:
            self.mode = Int(clamp(value, 0.0, 6.0))
            self._update(self.cutoff.value, self.q.value)
        elif param == Self.GAIN:
            self.gain = clamp(value, -48.0, 48.0)
            self._update(self.cutoff.value, self.q.value)
        elif param == Self.CUTOFF:
            self.cutoff.set(_clamp_freq(value, self.sample_rate))
        elif param == Self.Q:
            self.q.set(clamp(value, 1.0e-3, 1.0e3))

    def reset(mut self):
        self.cutoff.snap()
        self.q.snap()
        self._update(self.cutoff.value, self.q.value)
        self.s1 = 0.0
        self.s2 = 0.0

    def _update(mut self, cutoff: Float64, q: Float64):
        self.cutoff_for = cutoff
        self.q_for = q
        var w0 = 2.0 * pi * cutoff / self.sample_rate
        var cw = cos(w0)
        var alpha = sin(w0) / (2.0 * q)
        var a0 = 1.0 + alpha
        var b0: Float64
        var b1: Float64
        var b2: Float64
        if self.mode >= LOWSHELF:
            # Shelves and peaking scale by A = sqrt(linear gain), as in the
            # RBJ cookbook, and set their own a0/a1/a2.
            var a = 10.0 ** (self.gain / 40.0)
            var two_sqrt_a_alpha = 2.0 * (a**0.5) * alpha
            if self.mode == PEAKING:
                self.b0 = (1.0 + alpha * a) / (1.0 + alpha / a)
                self.b1 = (-2.0 * cw) / (1.0 + alpha / a)
                self.b2 = (1.0 - alpha * a) / (1.0 + alpha / a)
                self.a1 = (-2.0 * cw) / (1.0 + alpha / a)
                self.a2 = (1.0 - alpha / a) / (1.0 + alpha / a)
                return
            var shelf_a0: Float64
            if self.mode == LOWSHELF:
                shelf_a0 = (a + 1.0) + (a - 1.0) * cw + two_sqrt_a_alpha
                self.b0 = a * ((a + 1.0) - (a - 1.0) * cw + two_sqrt_a_alpha) / shelf_a0
                self.b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cw) / shelf_a0
                self.b2 = a * ((a + 1.0) - (a - 1.0) * cw - two_sqrt_a_alpha) / shelf_a0
                self.a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cw) / shelf_a0
                self.a2 = ((a + 1.0) + (a - 1.0) * cw - two_sqrt_a_alpha) / shelf_a0
            else:
                shelf_a0 = (a + 1.0) - (a - 1.0) * cw + two_sqrt_a_alpha
                self.b0 = a * ((a + 1.0) + (a - 1.0) * cw + two_sqrt_a_alpha) / shelf_a0
                self.b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cw) / shelf_a0
                self.b2 = a * ((a + 1.0) + (a - 1.0) * cw - two_sqrt_a_alpha) / shelf_a0
                self.a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cw) / shelf_a0
                self.a2 = ((a + 1.0) - (a - 1.0) * cw - two_sqrt_a_alpha) / shelf_a0
            return
        if self.mode == LOWPASS:
            b0 = (1.0 - cw) / 2.0
            b1 = 1.0 - cw
            b2 = b0
        elif self.mode == HIGHPASS:
            b0 = (1.0 + cw) / 2.0
            b1 = -(1.0 + cw)
            b2 = b0
        elif self.mode == BANDPASS:
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

    @always_inline
    def _advance(mut self) -> Bool:
        """Step both ramps; True while either is still ramping."""
        var ramping = self.cutoff.remaining > 0 or self.q.remaining > 0
        var cutoff = self.cutoff.next()
        var q = self.q.next()
        if ramping:
            self._update(cutoff, q)
        return ramping

    @always_inline
    def _sample(mut self, x: Float32) -> Float32:
        var xd = Float64(x)
        var y = self.b0 * xd + self.s1
        self.s1 = self.b1 * xd - self.a1 * y + self.s2
        self.s2 = self.b2 * xd - self.a2 * y
        return Float32(y)

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        _ = self._advance()
        return self._sample(x)

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var src = audio_input(ins, dst)
        var i = 0
        var ramping = max(self.cutoff.ramping(n), self.q.ramping(n))
        while i < ramping:
            _ = self._advance()
            dst[unsafe_offset=i] = self._sample(src[unsafe_offset=i])
            i += 1
        var b0 = self.b0
        var b1 = self.b1
        var b2 = self.b2
        var a1 = self.a1
        var a2 = self.a2
        var s1 = self.s1
        var s2 = self.s2
        while i < n:
            var x = Float64(src[unsafe_offset=i])
            var y = b0 * x + s1
            s1 = b1 * x - a1 * y + s2
            s2 = b2 * x - a2 * y
            dst[unsafe_offset=i] = Float32(y)
            i += 1
        self.s1 = s1
        self.s2 = s2


struct Svf(Processor, Writable):
    """Topology-preserving state-variable filter.

    Stable under fast cutoff changes, so this is the filter to modulate.
    Ports: audio, `cutoff` in Hz. Source: A. Simper, "Linear Trapezoidal
    Integrated SVF", https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf
    """

    comptime MODE = 0
    comptime CUTOFF = 1
    comptime Q = 2

    var sample_rate: Float64
    var mode: Int
    var cutoff: Smoothed
    var q: Smoothed
    var cutoff_for: Float64
    var q_for: Float64
    var k: Float64
    var a1: Float64
    var a2: Float64
    var a3: Float64
    var ic1: Float64
    var ic2: Float64

    def __init__(out self, sample_rate: Float64):
        self.sample_rate = sample_rate
        self.mode = LOWPASS
        self.cutoff = Smoothed(_clamp_freq(1000.0, sample_rate), sample_rate)
        self.q = Smoothed(0.7071067811865476, sample_rate)
        self.cutoff_for = 0.0
        self.q_for = 0.0
        self.k = 1.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.a3 = 0.0
        self.ic1 = 0.0
        self.ic2 = 0.0
        self._update(self.cutoff.value, self.q.value)

    @staticmethod
    def param_names() -> List[String]:
        return ["mode", "cutoff", "q"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in", "cutoff"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.MODE:
            self.mode = Int(clamp(value, 0.0, 3.0))
        elif param == Self.CUTOFF:
            self.cutoff.set(_clamp_freq(value, self.sample_rate))
        elif param == Self.Q:
            self.q.set(clamp(value, 1.0e-3, 1.0e3))

    def reset(mut self):
        self.cutoff.snap()
        self.q.snap()
        self._update(self.cutoff.value, self.q.value)
        self.ic1 = 0.0
        self.ic2 = 0.0

    @always_inline
    def _update(mut self, cutoff: Float64, q: Float64):
        self.cutoff_for = cutoff
        self.q_for = q
        self.k = 1.0 / q
        var g = tan(pi * cutoff / self.sample_rate)
        self.a1 = 1.0 / (1.0 + g * (g + self.k))
        self.a2 = g * self.a1
        self.a3 = g * self.a2

    @always_inline
    def _update_if_changed(mut self, cutoff: Float64, q: Float64):
        if cutoff != self.cutoff_for or q != self.q_for:
            self._update(cutoff, q)

    @always_inline
    def _sample(mut self, x: Float32) -> Float32:
        var v0 = Float64(x)
        var v3 = v0 - self.ic2
        var v1 = self.a1 * self.ic1 + self.a2 * v3
        var v2 = self.ic2 + self.a2 * self.ic1 + self.a3 * v3
        self.ic1 = 2.0 * v1 - self.ic1
        self.ic2 = 2.0 * v2 - self.ic2
        if self.mode == LOWPASS:
            return Float32(v2)
        if self.mode == HIGHPASS:
            return Float32(v0 - self.k * v1 - v2)
        if self.mode == BANDPASS:
            return Float32(self.k * v1)  # 0 dB at the centre frequency
        return Float32(v0 - self.k * v1)

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        self._update_if_changed(self.cutoff.next(), self.q.next())
        return self._sample(x)

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var src = audio_input(ins, dst)
        var cutoff_mod = input_address(ins, 1)
        if cutoff_mod != 0:
            var cutoff = SamplePtr(unsafe_from_address=cutoff_mod)
            for i in range(n):
                self._update_if_changed(
                    _clamp_freq(Float64(cutoff[unsafe_offset=i]), self.sample_rate),
                    self.q.next(),
                )
                dst[unsafe_offset=i] = self._sample(src[unsafe_offset=i])
            return
        var i = 0
        var ramping = max(self.cutoff.ramping(n), self.q.ramping(n))
        while i < ramping:
            self._update_if_changed(self.cutoff.next(), self.q.next())
            dst[unsafe_offset=i] = self._sample(src[unsafe_offset=i])
            i += 1
        while i < n:
            dst[unsafe_offset=i] = self._sample(src[unsafe_offset=i])
            i += 1
