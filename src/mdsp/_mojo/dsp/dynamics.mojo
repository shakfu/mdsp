from std.math import ceil, clamp, exp, log10

from dsp.processor import Ports, Processor, SamplePtr, audio_input, input_address

comptime TINY = 1.0e-12  # keeps log10 finite on silence
comptime MAX_LOOKAHEAD = 0.02  # seconds; the line is allocated for this much


struct Compressor(Processor, Writable):
    """Feed-forward compressor working in decibels.

    Gain reduction is computed from the detector level, then smoothed with
    separate attack and release times, which is the usual arrangement: the
    smoothing acts on the reduction rather than on the signal.

    Ports: audio, `sidechain`. A connected sidechain drives the detector while
    the audio passes through, which is how ducking is built. `knee` widens the
    region around the threshold where compression eases in. `rms` averages the
    detector over a window instead of following peaks, and `lookahead` delays
    the audio so reduction arrives with the transient that caused it.
    """

    comptime THRESHOLD = 0
    comptime RATIO = 1
    comptime ATTACK = 2
    comptime RELEASE = 3
    comptime MAKEUP = 4
    comptime KNEE = 5
    comptime RMS = 6
    comptime LOOKAHEAD = 7

    var sample_rate: Float64
    var threshold: Float64  # dB
    var ratio: Float64
    var makeup: Float64  # linear
    var attack_coefficient: Float64
    var release_coefficient: Float64
    var knee: Float64  # dB wide, centred on the threshold
    var rms_coefficient: Float64  # 0 follows peaks instead
    var reduction: Float64  # dB, always >= 0
    var mean_square: Float64
    var line: List[Float32]  # holds the audio while the detector runs ahead
    var write: Int
    var lookahead: Int  # samples

    def __init__(out self, sample_rate: Float64):
        self.sample_rate = sample_rate
        self.threshold = -20.0
        self.ratio = 4.0
        self.makeup = 1.0
        self.attack_coefficient = 0.0
        self.release_coefficient = 0.0
        self.knee = 0.0
        self.rms_coefficient = 0.0
        self.reduction = 0.0
        self.mean_square = 0.0
        self.line = List[Float32](
            length=Int(ceil(MAX_LOOKAHEAD * sample_rate)) + 1, fill=0.0
        )
        self.write = 0
        self.lookahead = 0
        self.set(Self.ATTACK, 0.005)
        self.set(Self.RELEASE, 0.1)

    @staticmethod
    def param_names() -> List[String]:
        return [
            "threshold",
            "ratio",
            "attack",
            "release",
            "makeup",
            "knee",
            "rms",
            "lookahead",
        ]

    @staticmethod
    def input_names() -> List[String]:
        return ["in", "sidechain"]

    @always_inline
    def _coefficient(self, seconds: Float64) -> Float64:
        """One-pole coefficient for a time constant, 0 when it is immediate."""
        var samples = max(seconds, 0.0) * self.sample_rate
        return 0.0 if samples < 1.0 else exp(-1.0 / samples)

    def set(mut self, param: Int, value: Float64):
        if param == Self.THRESHOLD:
            self.threshold = clamp(value, -120.0, 24.0)
        elif param == Self.RATIO:
            self.ratio = clamp(value, 1.0, 1000.0)
        elif param == Self.ATTACK:
            self.attack_coefficient = self._coefficient(value)
        elif param == Self.RELEASE:
            self.release_coefficient = self._coefficient(value)
        elif param == Self.MAKEUP:
            self.makeup = 10.0 ** (clamp(value, -24.0, 48.0) / 20.0)
        elif param == Self.KNEE:
            self.knee = clamp(value, 0.0, 48.0)
        elif param == Self.RMS:
            self.rms_coefficient = self._coefficient(value)
        elif param == Self.LOOKAHEAD:
            var samples = clamp(value, 0.0, MAX_LOOKAHEAD) * self.sample_rate
            self.lookahead = min(Int(samples + 0.5), len(self.line) - 1)

    def reset(mut self):
        self.reduction = 0.0
        self.mean_square = 0.0
        for ref sample in self.line:
            sample = 0.0
        self.write = 0

    @always_inline
    def _level(mut self, detector: Float32) -> Float64:
        """Detector level in dB: peak, or the root of a running mean square."""
        var d = Float64(detector)
        if self.rms_coefficient == 0.0:
            return 20.0 * log10(abs(d) + TINY)
        self.mean_square = (
            d * d + self.rms_coefficient * (self.mean_square - d * d)
        )
        return 10.0 * log10(self.mean_square + TINY)

    @always_inline
    def _delayed(mut self, x: Float32) -> Float32:
        """The sample `lookahead` frames back; *x* itself when it is 0."""
        var size = len(self.line)
        var base = self.line.unsafe_ptr()
        base[unsafe_offset=self.write] = x
        var read = self.write - self.lookahead
        if read < 0:
            read += size
        var delayed = base[unsafe_offset=read]
        self.write += 1
        if self.write == size:
            self.write = 0
        return delayed

    @always_inline
    def _apply(mut self, x: Float32, detector: Float32) -> Float32:
        var over = self._level(detector) - self.threshold
        var slope = 1.0 - 1.0 / self.ratio
        var half_knee = 0.5 * self.knee
        var target: Float64
        if over >= half_knee:
            target = over * slope
        elif over <= -half_knee:
            target = 0.0
        else:
            # Quadratic across the knee, so the curve and its slope stay
            # continuous where compression starts.
            var into_knee = over + half_knee
            target = slope * into_knee * into_knee / (2.0 * self.knee)
        # Attack when the reduction deepens, release when it eases.
        var coefficient = (
            self.attack_coefficient if target > self.reduction
            else self.release_coefficient
        )
        self.reduction = target + coefficient * (self.reduction - target)
        return self._delayed(x) * Float32(
            self.makeup * 10.0 ** (-self.reduction / 20.0)
        )

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        return self._apply(x, x)

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var src = audio_input(ins, dst)
        var sidechain_address = input_address(ins, 1)
        if sidechain_address == 0:
            for i in range(n):
                var x = src[unsafe_offset=i]
                dst[unsafe_offset=i] = self._apply(x, x)
            return
        var sidechain = SamplePtr(unsafe_from_address=sidechain_address)
        for i in range(n):
            dst[unsafe_offset=i] = self._apply(
                src[unsafe_offset=i], sidechain[unsafe_offset=i]
            )
