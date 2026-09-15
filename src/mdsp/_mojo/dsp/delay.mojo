from std.math import ceil, clamp, floor

from dsp.processor import Processor, SamplePtr

comptime MAX_DELAY_SECONDS = 600.0


struct Delay(Processor, Writable):
    """Delay line with linear interpolation, feedback and dry/wet mix.

    `y = (1 - mix) * x + mix * d`, where `d` is the line read `delay` seconds
    back, and `x + feedback * d` is written. The minimum delay is one sample.
    Setting `max_delay` reallocates and clears the line.
    """

    comptime MAX_DELAY = 0
    comptime DELAY = 1
    comptime FEEDBACK = 2
    comptime MIX = 3

    var sample_rate: Float64
    var line: List[Float32]
    var write: Int
    var max_delay: Float64
    var delay_samples: Float64
    var feedback: Float32
    var mix: Float32

    def __init__(out self, sample_rate: Float64):
        self.sample_rate = sample_rate
        self.line = List[Float32]()
        self.write = 0
        self.max_delay = 0.0
        self.delay_samples = 1.0
        self.feedback = 0.0
        self.mix = 1.0
        self.set(Self.MAX_DELAY, 1.0)

    @staticmethod
    def param_names() -> List[String]:
        return ["max_delay", "delay", "feedback", "mix"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.MAX_DELAY:
            self.max_delay = clamp(value, 0.0, MAX_DELAY_SECONDS)
            # +2: one slot for the minimum delay, one for the interpolation neighbour.
            var size = Int(ceil(self.max_delay * self.sample_rate)) + 2
            self.line = List[Float32](length=size, fill=0.0)
            self.write = 0
            self.delay_samples = self._clamp_delay(self.delay_samples)
        elif param == Self.DELAY:
            self.delay_samples = self._clamp_delay(value * self.sample_rate)
        elif param == Self.FEEDBACK:
            self.feedback = Float32(clamp(value, -1.0, 1.0))
        elif param == Self.MIX:
            self.mix = Float32(clamp(value, 0.0, 1.0))

    def _clamp_delay(self, samples: Float64) -> Float64:
        return clamp(samples, 1.0, Float64(len(self.line) - 2))

    def reset(mut self):
        for ref s in self.line:
            s = 0.0
        self.write = 0

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        var n = len(self.line)
        var base = self.line.unsafe_ptr()
        var y = _step(base, n, self.write, self.delay_samples, self.feedback, self.mix, x)
        self.write += 1
        if self.write == n:
            self.write = 0
        return y

    def process(mut self, src: SamplePtr, dst: SamplePtr, n: Int):
        var size = len(self.line)
        var base = self.line.unsafe_ptr()
        var w = self.write
        var d = self.delay_samples
        var fb = self.feedback
        var mix = self.mix
        for i in range(n):
            dst[unsafe_offset=i] = _step(base, size, w, d, fb, mix, src[unsafe_offset=i])
            w += 1
            if w == size:
                w = 0
        self.write = w


@always_inline
def _step[origin: MutOrigin](
    base: Pointer[Float32, origin],
    size: Int,
    w: Int,
    delay: Float64,
    feedback: Float32,
    mix: Float32,
    x: Float32,
) -> Float32:
    """Read `delay` samples behind `w`, write the input plus feedback at `w`."""
    var r = read_position(w, delay, size)
    var i0 = Int(floor(r))
    var frac = Float32(r - Float64(i0))
    var i1 = i0 + 1
    if i1 == size:
        i1 = 0
    var d = base[unsafe_offset=i0] * (1.0 - frac) + base[unsafe_offset=i1] * frac
    base[unsafe_offset=w] = x + feedback * d
    return (1.0 - mix) * x + mix * d


@always_inline
def read_position(w: Int, delay: Float64, size: Int) -> Float64:
    """Position `delay` samples behind `w`, wrapped into `[0, size)`."""
    var r = Float64(w) - delay
    if r < 0.0:
        r += Float64(size)
        # A tiny negative `r` rounds up to `size`; position 0 is within one ulp.
        if r >= Float64(size):
            r = 0.0
    return r
