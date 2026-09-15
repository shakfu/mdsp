from dsp.processor import Processor, SamplePtr


struct Gain(Processor, Writable):
    """Multiply by a linear gain."""

    comptime GAIN = 0

    var gain: Float32

    def __init__(out self, sample_rate: Float64):
        self.gain = 1.0

    @staticmethod
    def param_names() -> List[String]:
        return ["gain"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.GAIN:
            self.gain = Float32(value)

    def reset(mut self):
        pass

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        return x * self.gain

    def process(mut self, src: SamplePtr, dst: SamplePtr, n: Int):
        var g = self.gain
        for i in range(n):
            dst[unsafe_offset=i] = src[unsafe_offset=i] * g
