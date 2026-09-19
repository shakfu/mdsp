from std.math import ceil, clamp, floor, pi, sin

from dsp.delay import read_position
from dsp.processor import Ports, SamplePtr, WideProcessor, input_address
from dsp.smooth import Smoothed

comptime MAX_SECONDS = 0.1  # base delay plus depth
comptime MAX_VOICES = 4


struct Chorus(WideProcessor, Writable):
    """Delay lines swept by one shared LFO.

    Short base delays with shallow depth give flanging; longer ones with more
    depth give chorus. Ports: audio, `rate` in Hz. Sweeping the delay shifts
    pitch as it moves, which is where the effect comes from.

    The LFO runs once for every channel and voice; each reads it at its own
    phase offset, `spread * channel / channels + voice / voices`. One shared
    phase is what makes the channels sound like one moving image rather than
    several. Channel-aware because the offset needs the channel index, which a
    per-channel kernel does not have. `rate` modulation is read from the first
    channel.
    """

    comptime RATE = 0
    comptime DEPTH = 1
    comptime DELAY = 2
    comptime FEEDBACK = 3
    comptime MIX = 4
    comptime VOICES = 5
    comptime SPREAD = 6

    var sample_rate: Float64
    var channels: Int
    var stride: Int  # samples in one channel's line
    var line: List[Float32]  # `channels` lines end to end
    var write: Int
    var phase: Float64
    var rate: Float64
    var depth: Smoothed  # samples
    var delay: Smoothed  # samples
    var feedback: Float32
    var mix: Float32
    var voices: Int
    var spread: Float64

    def __init__(out self, sample_rate: Float64, channels: Int):
        self.sample_rate = sample_rate
        self.channels = max(channels, 1)
        self.stride = Int(ceil(MAX_SECONDS * sample_rate)) + 2
        self.line = List[Float32](length=self.stride * self.channels, fill=0.0)
        self.write = 0
        self.phase = 0.0
        self.rate = 0.5
        self.depth = Smoothed(0.002 * sample_rate, sample_rate)
        self.delay = Smoothed(0.01 * sample_rate, sample_rate)
        self.feedback = 0.0
        self.mix = 0.5
        self.voices = 1
        self.spread = 0.0

    @staticmethod
    def param_names() -> List[String]:
        return ["rate", "depth", "delay", "feedback", "mix", "voices", "spread"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in", "rate"]

    @always_inline
    def _clamp_samples(self, samples: Float64) -> Float64:
        return clamp(samples, 1.0, Float64(self.stride - 2))

    def set(mut self, param: Int, value: Float64):
        if param == Self.RATE:
            self.rate = clamp(value, 0.0, 100.0)
        elif param == Self.DEPTH:
            # The read position is clamped where it is used, so depth needs no
            # floor of its own: 0 must mean a still delay line.
            self.depth.set(
                clamp(value * self.sample_rate, 0.0, Float64(self.stride - 2))
            )
        elif param == Self.DELAY:
            self.delay.set(self._clamp_samples(value * self.sample_rate))
        elif param == Self.FEEDBACK:
            self.feedback = Float32(clamp(value, -1.0, 1.0))
        elif param == Self.MIX:
            self.mix = Float32(clamp(value, 0.0, 1.0))
        elif param == Self.VOICES:
            self.voices = Int(clamp(value, 1.0, Float64(MAX_VOICES)))
        elif param == Self.SPREAD:
            self.spread = clamp(value, 0.0, 1.0)

    def reset(mut self):
        for ref sample in self.line:
            sample = 0.0
        self.write = 0
        self.phase = 0.0
        self.depth.snap()
        self.delay.snap()

    @always_inline
    def _voice(
        self, channel: Int, phase: Float64, delay: Float64, depth: Float64
    ) -> Float32:
        """Sum this channel's voices, reading its line at each voice phase."""
        var base = self.line.unsafe_ptr()
        var start = channel * self.stride
        var wet = Float32(0.0)
        for v in range(self.voices):
            var offset = Float64(v) / Float64(self.voices)
            var sweep = sin(2.0 * pi * (phase + offset))
            var samples = self._clamp_samples(delay + depth * sweep)
            var r = read_position(self.write, samples, self.stride)
            var i0 = Int(floor(r))
            var frac = Float32(r - Float64(i0))
            var i1 = i0 + 1
            if i1 == self.stride:
                i1 = 0
            wet += base[unsafe_offset=start + i0] * (1.0 - frac) + base[
                unsafe_offset=start + i1
            ] * frac
        return wet / Float32(self.voices)

    def process(mut self, ins: Ports, dst: SamplePtr, frames: Int, channels: Int):
        var source = input_address(ins, 0)
        var input = SamplePtr(unsafe_from_address=source) if source != 0 else dst
        var modulation = input_address(ins, 1)
        # A caller with more channels than the lines were built for would read
        # past them; the extra channels pass through untouched.
        var active = min(channels, self.channels)
        var base = self.line.unsafe_ptr()
        for i in range(frames):
            var rate = self.rate
            if modulation != 0:
                rate = Float64(
                    SamplePtr(unsafe_from_address=modulation)[unsafe_offset=i]
                )
            var depth = self.depth.next()
            var delay = self.delay.next()
            var phase = self.phase
            self.phase += rate / self.sample_rate
            self.phase -= floor(self.phase)
            for c in range(active):
                var offset = self.spread * Float64(c) / Float64(self.channels)
                var x = input[unsafe_offset=c * frames + i]
                var wet = self._voice(c, phase + offset, delay, depth)
                base[unsafe_offset=c * self.stride + self.write] = (
                    x + self.feedback * wet
                )
                dst[unsafe_offset=c * frames + i] = (1.0 - self.mix) * x + self.mix * wet
            self.write += 1
            if self.write == self.stride:
                self.write = 0
        for c in range(active, channels):
            for i in range(frames):
                dst[unsafe_offset=c * frames + i] = input[unsafe_offset=c * frames + i]
