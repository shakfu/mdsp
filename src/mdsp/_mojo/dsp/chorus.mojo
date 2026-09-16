from std.math import ceil, clamp, floor, pi, sin

from dsp.delay import read_position
from dsp.processor import Ports, Processor, SamplePtr, audio_input, input_address
from dsp.smooth import Smoothed

comptime MAX_SECONDS = 0.1  # base delay plus depth


struct Chorus(Processor, Writable):
    """Delay line whose length is swept by its own LFO.

    Short base delays with shallow depth give flanging; longer ones with more
    depth give chorus. Ports: audio, `rate` in Hz. Sweeping the delay shifts
    pitch as it moves, which is where the effect comes from.
    """

    comptime RATE = 0
    comptime DEPTH = 1
    comptime DELAY = 2
    comptime FEEDBACK = 3
    comptime MIX = 4

    var sample_rate: Float64
    var line: List[Float32]
    var write: Int
    var phase: Float64
    var rate: Float64
    var depth: Smoothed  # samples
    var delay: Smoothed  # samples
    var feedback: Float32
    var mix: Float32

    def __init__(out self, sample_rate: Float64):
        self.sample_rate = sample_rate
        self.line = List[Float32](
            length=Int(ceil(MAX_SECONDS * sample_rate)) + 2, fill=0.0
        )
        self.write = 0
        self.phase = 0.0
        self.rate = 0.5
        self.depth = Smoothed(0.002 * sample_rate, sample_rate)
        self.delay = Smoothed(0.01 * sample_rate, sample_rate)
        self.feedback = 0.0
        self.mix = 0.5

    @staticmethod
    def param_names() -> List[String]:
        return ["rate", "depth", "delay", "feedback", "mix"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in", "rate"]

    @always_inline
    def _clamp_samples(self, samples: Float64) -> Float64:
        return clamp(samples, 1.0, Float64(len(self.line) - 2))

    def set(mut self, param: Int, value: Float64):
        if param == Self.RATE:
            self.rate = clamp(value, 0.0, 100.0)
        elif param == Self.DEPTH:
            self.depth.set(self._clamp_samples(max(value, 0.0) * self.sample_rate))
        elif param == Self.DELAY:
            self.delay.set(self._clamp_samples(value * self.sample_rate))
        elif param == Self.FEEDBACK:
            self.feedback = Float32(clamp(value, -1.0, 1.0))
        elif param == Self.MIX:
            self.mix = Float32(clamp(value, 0.0, 1.0))

    def reset(mut self):
        for ref sample in self.line:
            sample = 0.0
        self.write = 0
        self.phase = 0.0
        self.depth.snap()
        self.delay.snap()

    @always_inline
    def _advance(mut self, x: Float32, rate: Float64) -> Float32:
        var sweep = sin(2.0 * pi * self.phase)
        self.phase += rate / self.sample_rate
        self.phase -= floor(self.phase)
        var samples = self._clamp_samples(self.delay.next() + self.depth.next() * sweep)

        var size = len(self.line)
        var base = self.line.unsafe_ptr()
        var r = read_position(self.write, samples, size)
        var i0 = Int(floor(r))
        var frac = Float32(r - Float64(i0))
        var i1 = i0 + 1
        if i1 == size:
            i1 = 0
        var delayed = base[unsafe_offset=i0] * (1.0 - frac) + base[
            unsafe_offset=i1
        ] * frac
        base[unsafe_offset=self.write] = x + self.feedback * delayed
        self.write += 1
        if self.write == size:
            self.write = 0
        return (1.0 - self.mix) * x + self.mix * delayed

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        return self._advance(x, self.rate)

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var src = audio_input(ins, dst)
        var modulation = input_address(ins, 1)
        if modulation != 0:
            var rate = SamplePtr(unsafe_from_address=modulation)
            for i in range(n):
                dst[unsafe_offset=i] = self._advance(
                    src[unsafe_offset=i], Float64(rate[unsafe_offset=i])
                )
            return
        var rate = self.rate
        for i in range(n):
            dst[unsafe_offset=i] = self._advance(src[unsafe_offset=i], rate)
