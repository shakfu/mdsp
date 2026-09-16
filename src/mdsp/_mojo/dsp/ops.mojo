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
