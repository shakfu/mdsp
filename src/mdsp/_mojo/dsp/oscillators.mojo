from std.math import floor, pi, sin

from dsp.processor import Processor, SamplePtr


trait Shape:
    """A periodic waveform over phase in [0, 1)."""

    @staticmethod
    def value(phase: Float64, inc: Float64) -> Float64:
        """Output at `phase`; `inc` is the phase increment per sample."""
        ...


struct Osc[S: Shape](Processor, Writable):
    """Phase accumulator at `freq` Hz driving a `Shape`, starting at phase 0."""

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

    def process(mut self, src: SamplePtr, dst: SamplePtr, n: Int):
        var phase = self.phase
        var inc = self.freq / self.sample_rate
        for i in range(n):
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
