from std.math import ceil, clamp, floor, pi, sin

from dsp.delay import read_position
from dsp.processor import Ports, Processor, SamplePtr, audio_input, input_address
from dsp.smooth import Smoothed

comptime MAX_WINDOW = 0.1  # seconds; the line is allocated for this much


struct Shift(Processor, Writable):
    """Pitch shifter: two read heads crossfading on one delay line.

    The heads run at `ratio = 2 ** (semitones / 12)` relative to the write
    head, so the delay between them drifts and the pitch moves with it. Each
    head wraps once per pass through the window; the pair sits half a window
    apart and is crossfaded by `sin(pi * phase)`, whose squares sum to 1, so a
    head is silent exactly where it wraps.

    Pitch only: the window is fixed, so time is not stretched and formants
    move with the pitch. Output lags the input by up to `window`.

    At 0 semitones the heads stop rather than the effect stopping: two fixed
    taps sum to a comb, which is what a micro-detune box does. Bypass with
    `mix` instead.

    A wrap steps the head by a whole window, which is a whole number of input
    cycles only by coincidence; otherwise the step spreads energy into
    sidebands spaced by `abs(1 - ratio) / window`. Ports: audio, `semitones`.
    """

    comptime SEMITONES = 0
    comptime WINDOW = 1
    comptime MIX = 2

    var sample_rate: Float64
    var line: List[Float32]
    var write: Int
    var phase: Float64  # of the first head, in [0, 1) of a window
    var semitones: Smoothed
    var ratio_for: Float64
    var ratio: Float64
    var window: Float64  # samples
    var mix: Float32

    def __init__(out self, sample_rate: Float64):
        self.sample_rate = sample_rate
        self.line = List[Float32](
            length=Int(ceil(MAX_WINDOW * sample_rate)) + 2, fill=0.0
        )
        self.write = 0
        self.phase = 0.0
        self.semitones = Smoothed(0.0, sample_rate)
        self.ratio_for = 0.0
        self.ratio = 1.0
        self.window = 1.0
        self.mix = 1.0
        self.set(Self.WINDOW, 0.05)

    @staticmethod
    def param_names() -> List[String]:
        return ["semitones", "window", "mix"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in", "semitones"]

    @always_inline
    def _clamp_window(self, samples: Float64) -> Float64:
        return clamp(samples, 2.0, Float64(len(self.line) - 2))

    def set(mut self, param: Int, value: Float64):
        if param == Self.SEMITONES:
            self.semitones.set(clamp(value, -24.0, 24.0))
        elif param == Self.WINDOW:
            self.window = self._clamp_window(value * self.sample_rate)
        elif param == Self.MIX:
            self.mix = Float32(clamp(value, 0.0, 1.0))

    def reset(mut self):
        for ref sample in self.line:
            sample = 0.0
        self.write = 0
        self.phase = 0.0
        self.semitones.snap()
        self._update(self.semitones.value)

    @always_inline
    def _update(mut self, semitones: Float64):
        self.ratio_for = semitones
        self.ratio = 2.0 ** (semitones / 12.0)

    @always_inline
    def _update_if_changed(mut self, semitones: Float64):
        if semitones != self.ratio_for:
            self._update(semitones)

    @always_inline
    def _tap(self, delay: Float64) -> Float32:
        var size = len(self.line)
        var base = self.line.unsafe_ptr()
        var r = read_position(self.write, delay, size)
        var i0 = Int(floor(r))
        var frac = Float32(r - Float64(i0))
        var i1 = i0 + 1
        if i1 == size:
            i1 = 0
        return base[unsafe_offset=i0] * (1.0 - frac) + base[unsafe_offset=i1] * frac

    @always_inline
    def _advance(mut self, x: Float32) -> Float32:
        self.line.unsafe_ptr()[unsafe_offset=self.write] = x
        var first = self.phase
        var second = first + 0.5
        if second >= 1.0:
            second -= 1.0
        var wet = Float32(sin(pi * first)) * self._tap(first * self.window) + Float32(
            sin(pi * second)
        ) * self._tap(second * self.window)
        # The delay between write and read heads drifts at 1 - ratio samples
        # per sample; over a window that is one wrap.
        self.phase += (1.0 - self.ratio) / self.window
        self.phase -= floor(self.phase)
        self.write += 1
        if self.write == len(self.line):
            self.write = 0
        return (1.0 - self.mix) * x + self.mix * wet

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        self._update_if_changed(self.semitones.next())
        return self._advance(x)

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var src = audio_input(ins, dst)
        var modulation = input_address(ins, 1)
        if modulation != 0:
            var semitones = SamplePtr(unsafe_from_address=modulation)
            for i in range(n):
                self._update_if_changed(
                    clamp(Float64(semitones[unsafe_offset=i]), -24.0, 24.0)
                )
                dst[unsafe_offset=i] = self._advance(src[unsafe_offset=i])
            return
        var i = 0
        var ramping = self.semitones.ramping(n)
        while i < ramping:
            self._update_if_changed(self.semitones.next())
            dst[unsafe_offset=i] = self._advance(src[unsafe_offset=i])
            i += 1
        while i < n:
            dst[unsafe_offset=i] = self._advance(src[unsafe_offset=i])
            i += 1
