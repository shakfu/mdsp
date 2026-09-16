from std.math import clamp, floor, pi, sin

from dsp.processor import Ports, Processor, SamplePtr, input_address


trait Shape:
    """A periodic waveform over phase in [0, 1)."""

    @staticmethod
    def value(phase: Float64, inc: Float64) -> Float64:
        """Output at `phase`; `inc` is the phase increment per sample."""
        ...


struct Osc[S: Shape](Processor, Writable):
    """Phase accumulator at `freq` Hz driving a `Shape`, starting at phase 0.

    Ports: audio (ignored), `freq` in Hz. Frequency is not smoothed: a new
    frequency takes effect on the next sample.
    """

    comptime FREQ = 0

    var sample_rate: Float64
    var freq: Float64
    var phase: Float64

    def __init__(out self, sample_rate: Float64):
        self.sample_rate = sample_rate
        self.freq = 0.0
        self.phase = 0.0

    @staticmethod
    def param_names() -> List[String]:
        return ["freq"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in", "freq"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.FREQ:
            self.freq = value

    def reset(mut self):
        self.phase = 0.0

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        var inc = self.freq / self.sample_rate
        var y = Float32(Self.S.value(self.phase, inc))
        self.phase += inc
        self.phase -= floor(self.phase)
        return y

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var phase = self.phase
        var freq_mod = input_address(ins, 1)
        if freq_mod == 0:
            var inc = self.freq / self.sample_rate
            for i in range(n):
                dst[unsafe_offset=i] = Float32(Self.S.value(phase, inc))
                phase += inc
                phase -= floor(phase)
        else:
            var freq = SamplePtr(unsafe_from_address=freq_mod)
            for i in range(n):
                var inc = Float64(freq[unsafe_offset=i]) / self.sample_rate
                dst[unsafe_offset=i] = Float32(Self.S.value(phase, inc))
                phase += inc
                phase -= floor(phase)
        self.phase = phase


struct PhasorShape(Shape):
    """Ramp from 0 to 1. Not band-limited."""

    @staticmethod
    @always_inline
    def value(phase: Float64, inc: Float64) -> Float64:
        return phase


struct SineShape(Shape):
    @staticmethod
    @always_inline
    def value(phase: Float64, inc: Float64) -> Float64:
        return sin(2.0 * pi * phase)


@always_inline
def poly_blep(t: Float64, dt: Float64) -> Float64:
    """Two-sample polynomial correction for a unit step at phase 0."""
    if t < dt:
        var u = t / dt
        return u + u - u * u - 1.0
    if t > 1.0 - dt:
        var u = (t - 1.0) / dt
        return u * u + u + u + 1.0
    return 0.0


@always_inline
def _mirror(phase: Float64) -> Float64:
    # Phase running backwards, as a forward phase: 1 - phase wrapped into [0, 1).
    var p = 1.0 - phase
    return p - floor(p)


struct SawShape(Shape):
    """PolyBLEP sawtooth from -1 to 1."""

    @staticmethod
    @always_inline
    def value(phase: Float64, inc: Float64) -> Float64:
        if inc < 0.0:  # odd symmetry: saw(p) == -saw(1 - p)
            return -Self._forward(_mirror(phase), min(-inc, 0.5))
        return Self._forward(phase, min(inc, 0.5))

    @staticmethod
    @always_inline
    def _forward(phase: Float64, dt: Float64) -> Float64:
        return 2.0 * phase - 1.0 - poly_blep(phase, dt)


struct SquareShape(Shape):
    """PolyBLEP square, +1 for the first half-period, -1 for the second."""

    @staticmethod
    @always_inline
    def value(phase: Float64, inc: Float64) -> Float64:
        if inc < 0.0:  # odd symmetry: square(p) == -square(1 - p)
            return -Self._forward(_mirror(phase), min(-inc, 0.5))
        return Self._forward(phase, min(inc, 0.5))

    @staticmethod
    @always_inline
    def _forward(phase: Float64, dt: Float64) -> Float64:
        var half = phase + 0.5
        half -= floor(half)
        var naive = 1.0 if phase < 0.5 else -1.0
        return naive + poly_blep(phase, dt) - poly_blep(half, dt)


comptime Phasor = Osc[PhasorShape]
comptime Sine = Osc[SineShape]
comptime Saw = Osc[SawShape]
comptime Square = Osc[SquareShape]


struct Noise(Processor, Writable):
    """White noise in [-1, 1) from an xorshift generator.

    Every channel of one unit shares the `seed`, so they produce the same
    samples; generate mono noise per channel if you need them decorrelated.
    """

    comptime SEED = 0
    comptime COLOR = 1

    comptime WHITE = 0
    comptime PINK = 1

    var seed: UInt32
    var state: UInt32
    var color: Int
    var pink_a: Float32
    var pink_b: Float32
    var pink_c: Float32

    def __init__(out self, sample_rate: Float64):
        self.seed = 22222
        self.state = self.seed
        self.color = Self.WHITE
        self.pink_a = 0.0
        self.pink_b = 0.0
        self.pink_c = 0.0

    @staticmethod
    def param_names() -> List[String]:
        return ["seed", "color"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.SEED:
            self.seed = UInt32(Int(max(value, 1.0)) & 0xFFFFFFFF)
            self.state = self.seed
        elif param == Self.COLOR:
            self.color = Int(clamp(value, 0.0, 1.0))

    def reset(mut self):
        self.state = self.seed
        self.pink_a = 0.0
        self.pink_b = 0.0
        self.pink_c = 0.0

    @always_inline
    def _next(mut self) -> Float32:
        self.state ^= self.state << 13
        self.state ^= self.state >> 17
        self.state ^= self.state << 5
        var white = Float32(self.state >> 8) / Float32(1 << 23) - 1.0
        if self.color == Self.WHITE:
            return white
        # Paul Kellet's economy pink filter: three one-poles summed, which
        # tracks -3 dB per octave closely enough for audio.
        self.pink_a = 0.99765 * self.pink_a + white * 0.0990460
        self.pink_b = 0.96300 * self.pink_b + white * 0.2965164
        self.pink_c = 0.57000 * self.pink_c + white * 1.0526913
        # 0.1 keeps peaks inside [-1, 1] over long runs (measured 0.8 peak,
        # 0.17 RMS); pink noise is quieter than white at the same peak.
        return (self.pink_a + self.pink_b + self.pink_c + white * 0.1848) * 0.1

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        return self._next()

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        for i in range(n):
            dst[unsafe_offset=i] = self._next()
