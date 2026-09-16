"""Schroeder reverb in the Freeverb arrangement.

Eight damped comb filters in parallel feed four allpass filters in series, the
tuning taken from Jezar at Dreampoint's public-domain Freeverb. Delay lengths
are scaled from their original 44.1 kHz values so other sample rates keep the
same reverb time.
"""

from std.math import clamp

from dsp.processor import Ports, Processor, SamplePtr, audio_input
from dsp.smooth import Smoothed

comptime COMB_COUNT = 8
comptime ALLPASS_COUNT = 4
comptime TUNING_RATE = 44100.0


def comb_lengths() -> List[Int]:
    return [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]


def allpass_lengths() -> List[Int]:
    return [556, 441, 341, 225]


struct Comb(Copyable, Movable, Writable):
    """Feedback comb with a one-pole lowpass in the loop, which damps the tail."""

    var line: List[Float32]
    var index: Int
    var filtered: Float32

    def __init__(out self, length: Int):
        self.line = List[Float32](length=max(length, 1), fill=0.0)
        self.index = 0
        self.filtered = 0.0

    @always_inline
    def tick(mut self, x: Float32, feedback: Float32, damping: Float32) -> Float32:
        var y = self.line[self.index]
        self.filtered = y * (1.0 - damping) + self.filtered * damping
        self.line[self.index] = x + self.filtered * feedback
        self.index += 1
        if self.index == len(self.line):
            self.index = 0
        return y

    def reset(mut self):
        for ref sample in self.line:
            sample = 0.0
        self.filtered = 0.0


struct Allpass(Copyable, Movable, Writable):
    var line: List[Float32]
    var index: Int

    def __init__(out self, length: Int):
        self.line = List[Float32](length=max(length, 1), fill=0.0)
        self.index = 0

    @always_inline
    def tick(mut self, x: Float32, feedback: Float32) -> Float32:
        var buffered = self.line[self.index]
        var y = buffered - x
        self.line[self.index] = x + buffered * feedback
        self.index += 1
        if self.index == len(self.line):
            self.index = 0
        return y

    def reset(mut self):
        for ref sample in self.line:
            sample = 0.0
        self.index = 0


struct Reverb(Processor, Writable):
    """Room reverb. Ports: audio only.

    `room_size` sets how long the tail rings, `damping` how fast its highs
    fade, and `mix` the blend of dry and reverberated signal.
    """

    comptime ROOM_SIZE = 0
    comptime DAMPING = 1
    comptime MIX = 2

    var combs: List[Comb]
    var allpasses: List[Allpass]
    var room_size: Smoothed
    var damping: Smoothed
    var mix: Smoothed

    def __init__(out self, sample_rate: Float64):
        var scale = sample_rate / TUNING_RATE
        self.combs = List[Comb]()
        for length in comb_lengths():
            self.combs.append(Comb(Int(Float64(length) * scale)))
        self.allpasses = List[Allpass]()
        for length in allpass_lengths():
            self.allpasses.append(Allpass(Int(Float64(length) * scale)))
        self.room_size = Smoothed(0.5, sample_rate)
        self.damping = Smoothed(0.5, sample_rate)
        self.mix = Smoothed(0.3, sample_rate)

    @staticmethod
    def param_names() -> List[String]:
        return ["room_size", "damping", "mix"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in"]

    def set(mut self, param: Int, value: Float64):
        var unit = clamp(value, 0.0, 1.0)
        if param == Self.ROOM_SIZE:
            self.room_size.set(unit)
        elif param == Self.DAMPING:
            self.damping.set(unit)
        elif param == Self.MIX:
            self.mix.set(unit)

    def reset(mut self):
        for ref comb in self.combs:
            comb.reset()
        for ref allpass in self.allpasses:
            allpass.reset()
        self.room_size.snap()
        self.damping.snap()
        self.mix.snap()

    @always_inline
    def _apply(mut self, x: Float32, room_size: Float64, damping: Float64, mix: Float64) -> Float32:
        # Feedback below 1 keeps the tail decaying; 0.98 is Freeverb's longest.
        var feedback = Float32(0.7 + 0.28 * room_size)
        var damp = Float32(0.4 * damping)
        var input = x * 0.015  # Freeverb's input gain, which keeps combs stable
        var wet: Float32 = 0.0
        for i in range(len(self.combs)):
            wet += self.combs[i].tick(input, feedback, damp)
        for i in range(len(self.allpasses)):
            wet = self.allpasses[i].tick(wet, 0.5)
        return Float32(Float64(x) * (1.0 - mix) + Float64(wet) * mix)

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        return self._apply(
            x, self.room_size.next(), self.damping.next(), self.mix.next()
        )

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var src = audio_input(ins, dst)
        var i = 0
        var ramping = max(
            self.room_size.ramping(n), max(self.damping.ramping(n), self.mix.ramping(n))
        )
        while i < ramping:
            dst[unsafe_offset=i] = self._apply(
                src[unsafe_offset=i],
                self.room_size.next(),
                self.damping.next(),
                self.mix.next(),
            )
            i += 1
        var room_size = self.room_size.value
        var damping = self.damping.value
        var mix = self.mix.value
        while i < n:
            dst[unsafe_offset=i] = self._apply(
                src[unsafe_offset=i], room_size, damping, mix
            )
            i += 1
