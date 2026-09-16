from std.math import clamp, exp, log10

from dsp.processor import Ports, Processor, SamplePtr, audio_input, input_address

comptime TINY = 1.0e-12  # keeps log10 finite on silence


struct Compressor(Processor, Writable):
    """Feed-forward peak compressor working in decibels.

    Gain reduction is computed from the detector level, then smoothed with
    separate attack and release times, which is the usual arrangement: the
    smoothing acts on the reduction rather than on the signal.

    Ports: audio, `sidechain`. A connected sidechain drives the detector while
    the audio passes through, which is how ducking is built. `knee` widens the
    region around the threshold where compression eases in.
    """

    comptime THRESHOLD = 0
    comptime RATIO = 1
    comptime ATTACK = 2
    comptime RELEASE = 3
    comptime MAKEUP = 4
    comptime KNEE = 5

    var sample_rate: Float64
    var threshold: Float64  # dB
    var ratio: Float64
    var makeup: Float64  # linear
    var attack_coefficient: Float64
    var release_coefficient: Float64
    var knee: Float64  # dB wide, centred on the threshold
    var reduction: Float64  # dB, always >= 0

    def __init__(out self, sample_rate: Float64):
        self.sample_rate = sample_rate
        self.threshold = -20.0
        self.ratio = 4.0
        self.makeup = 1.0
        self.attack_coefficient = 0.0
        self.release_coefficient = 0.0
        self.knee = 0.0
        self.reduction = 0.0
        self.set(Self.ATTACK, 0.005)
        self.set(Self.RELEASE, 0.1)

    @staticmethod
    def param_names() -> List[String]:
        return ["threshold", "ratio", "attack", "release", "makeup", "knee"]

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

    def reset(mut self):
        self.reduction = 0.0

    @always_inline
    def _apply(mut self, x: Float32, detector: Float32) -> Float32:
        var level = 20.0 * log10(Float64(abs(detector)) + TINY)
        var over = level - self.threshold
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
        return x * Float32(self.makeup * 10.0 ** (-self.reduction / 20.0))

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
