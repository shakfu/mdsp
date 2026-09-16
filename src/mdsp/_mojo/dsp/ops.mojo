from std.math import clamp, exp, log, tanh

from dsp.processor import Ports, Processor, SamplePtr, audio_input, input_address
from dsp.smooth import Smoothed


struct Gain(Processor, Writable):
    """Multiply by a linear gain. Ports: audio, `gain`."""

    comptime GAIN = 0

    var gain: Smoothed

    def __init__(out self, sample_rate: Float64):
        self.gain = Smoothed(1.0, sample_rate)

    @staticmethod
    def param_names() -> List[String]:
        return ["gain"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in", "gain"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.GAIN:
            self.gain.set(value)

    def reset(mut self):
        self.gain.snap()

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        return x * Float32(self.gain.next())

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var src = audio_input(ins, dst)
        var gain_mod = input_address(ins, 1)
        if gain_mod != 0:
            var g = SamplePtr(unsafe_from_address=gain_mod)
            for i in range(n):
                dst[unsafe_offset=i] = src[unsafe_offset=i] * g[unsafe_offset=i]
            return
        var i = 0
        var ramping = self.gain.ramping(n)
        while i < ramping:
            dst[unsafe_offset=i] = src[unsafe_offset=i] * Float32(self.gain.next())
            i += 1
        var g = Float32(self.gain.value)
        while i < n:
            dst[unsafe_offset=i] = src[unsafe_offset=i] * g
            i += 1


struct Scale(Processor, Writable):
    """Map [-1, 1] onto [`lo`, `hi`], linearly or exponentially.

    Turns an oscillator into a modulation source: a sine into a cutoff sweep.
    The exponential curve is the musical one for frequencies; it needs both
    bounds positive, so they are clamped away from zero. Ports: audio.
    """

    comptime LO = 0
    comptime HI = 1
    comptime CURVE = 2

    comptime LINEAR = 0
    comptime EXPONENTIAL = 1

    var lo: Smoothed
    var hi: Smoothed
    var curve: Int

    def __init__(out self, sample_rate: Float64):
        self.lo = Smoothed(-1.0, sample_rate)
        self.hi = Smoothed(1.0, sample_rate)
        self.curve = Self.LINEAR

    @staticmethod
    def param_names() -> List[String]:
        return ["lo", "hi", "curve"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.LO:
            self.lo.set(value)
        elif param == Self.HI:
            self.hi.set(value)
        elif param == Self.CURVE:
            self.curve = Int(clamp(value, 0.0, 1.0))

    def reset(mut self):
        self.lo.snap()
        self.hi.snap()

    @always_inline
    def _map(self, x: Float32, lo: Float64, hi: Float64) -> Float32:
        var unit = (Float64(x) + 1.0) * 0.5
        if self.curve == Self.EXPONENTIAL:
            var low = max(lo, 1.0e-6)
            return Float32(low * exp(unit * log(max(hi, 1.0e-6) / low)))
        return Float32(lo + (hi - lo) * unit)

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        return self._map(x, self.lo.next(), self.hi.next())

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var src = audio_input(ins, dst)
        var i = 0
        var ramping = max(self.lo.ramping(n), self.hi.ramping(n))
        while i < ramping:
            dst[unsafe_offset=i] = self._map(
                src[unsafe_offset=i], self.lo.next(), self.hi.next()
            )
            i += 1
        var lo = self.lo.value
        var hi = self.hi.value
        while i < n:
            dst[unsafe_offset=i] = self._map(src[unsafe_offset=i], lo, hi)
            i += 1


struct Mix(Processor, Writable):
    """Sum up to four inputs, each with its own gain.

    Ports are all audio: `in`, `in2`, `in3`, `in4`. Unconnected inputs
    contribute nothing, so a mixer can be wired up as voices arrive.
    """

    comptime GAIN = 0
    comptime GAIN2 = 1
    comptime GAIN3 = 2
    comptime GAIN4 = 3

    var gains: InlineArray[Smoothed, 4]

    def __init__(out self, sample_rate: Float64):
        self.gains = InlineArray[Smoothed, 4](fill=Smoothed(1.0, sample_rate))

    @staticmethod
    def param_names() -> List[String]:
        return ["gain", "gain2", "gain3", "gain4"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in", "in2", "in3", "in4"]

    def set(mut self, param: Int, value: Float64):
        if 0 <= param and param < 4:
            self.gains[param].set(value)

    def reset(mut self):
        for i in range(4):
            self.gains[i].snap()

    @always_inline
    def _advance(mut self, mut gains: InlineArray[Float32, 4]):
        for i in range(4):
            gains[i] = Float32(self.gains[i].next())

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        # Only the first input exists per-sample; the others advance in step so
        # that `process` with one input connected matches this exactly.
        var gains = InlineArray[Float32, 4](fill=0.0)
        self._advance(gains)
        return x * gains[0]

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var sources = InlineArray[Int, 4](fill=0)
        for port in range(4):
            sources[port] = input_address(ins, port)
        if sources[0] == 0:  # nothing connected to the first port: silence
            sources[0] = Int(dst)
            for i in range(n):
                dst[unsafe_offset=i] = 0.0
        var gains = InlineArray[Float32, 4](fill=0.0)
        for i in range(n):
            self._advance(gains)
            var total = SamplePtr(unsafe_from_address=sources[0])[
                unsafe_offset=i
            ] * gains[0]
            for port in range(1, 4):
                if sources[port] != 0:
                    total += (
                        SamplePtr(unsafe_from_address=sources[port])[unsafe_offset=i]
                        * gains[port]
                    )
            dst[unsafe_offset=i] = total


comptime SHAPE_TANH = 0
comptime SHAPE_SOFT = 1
comptime SHAPE_HARD = 2


struct Shaper(Processor, Writable):
    """Waveshaping distortion.

    `drive` multiplies the input before shaping, and the output is scaled so a
    full-scale input stays full-scale, which keeps `drive` from doubling as a
    volume control. Shapes: `tanh`, `soft` (cubic soft clip), `hard` (clip).
    Ports: audio, `drive`.
    """

    comptime DRIVE = 0
    comptime SHAPE = 1

    var drive: Smoothed
    var shape: Int

    def __init__(out self, sample_rate: Float64):
        self.drive = Smoothed(1.0, sample_rate)
        self.shape = SHAPE_TANH

    @staticmethod
    def param_names() -> List[String]:
        return ["drive", "shape"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in", "drive"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.DRIVE:
            self.drive.set(clamp(value, 1.0e-3, 1000.0))
        elif param == Self.SHAPE:
            self.shape = Int(clamp(value, 0.0, 2.0))

    def reset(mut self):
        self.drive.snap()

    @always_inline
    def _curve(self, x: Float64) -> Float64:
        if self.shape == SHAPE_HARD:
            return clamp(x, -1.0, 1.0)
        if self.shape == SHAPE_SOFT:
            var clamped = clamp(x, -1.5, 1.5)
            return clamped - clamped * clamped * clamped / 6.75
        return tanh(x)

    @always_inline
    def _apply(self, x: Float32, drive: Float64) -> Float32:
        var normalise = self._curve(drive)
        var shaped = self._curve(drive * Float64(x))
        return Float32(shaped / normalise) if normalise != 0.0 else Float32(shaped)

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        return self._apply(x, self.drive.next())

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var src = audio_input(ins, dst)
        var drive_address = input_address(ins, 1)
        if drive_address != 0:
            var drive = SamplePtr(unsafe_from_address=drive_address)
            for i in range(n):
                dst[unsafe_offset=i] = self._apply(
                    src[unsafe_offset=i], Float64(drive[unsafe_offset=i])
                )
            return
        var i = 0
        var ramping = self.drive.ramping(n)
        while i < ramping:
            dst[unsafe_offset=i] = self._apply(src[unsafe_offset=i], self.drive.next())
            i += 1
        var drive = self.drive.value
        while i < n:
            dst[unsafe_offset=i] = self._apply(src[unsafe_offset=i], drive)
            i += 1
