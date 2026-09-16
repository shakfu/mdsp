"""Units that mix channels together."""

from std.math import clamp, cos, pi, sin

from dsp.processor import Ports, SamplePtr, WideProcessor, input_address
from dsp.smooth import Smoothed


@always_inline
def _channel(base: Int, channel: Int, frames: Int) -> SamplePtr:
    return SamplePtr(unsafe_from_address=base + channel * frames * 4)


struct Pan(WideProcessor, Writable):
    """Place a signal in the stereo field, -1 left to +1 right.

    Gains follow a constant-power law, so a sound keeps its loudness as it
    moves. The first input channel is the source; with one channel the output
    is that signal at the centre gain, and beyond two channels the extra
    channels are silent.
    """

    comptime PAN = 0

    var pan: Smoothed

    def __init__(out self, sample_rate: Float64):
        self.pan = Smoothed(0.0, sample_rate)

    @staticmethod
    def param_names() -> List[String]:
        return ["pan"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in", "pan"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.PAN:
            self.pan.set(clamp(value, -1.0, 1.0))

    def reset(mut self):
        self.pan.snap()

    def process(mut self, ins: Ports, dst: SamplePtr, frames: Int, channels: Int):
        var source = input_address(ins, 0)
        var modulation = input_address(ins, 1)
        var input = SamplePtr(unsafe_from_address=source) if source != 0 else dst
        var left = _channel(Int(dst), 0, frames)
        var right = _channel(Int(dst), 1, frames) if channels > 1 else left
        for i in range(frames):
            var position = self.pan.next()
            if modulation != 0:
                position = clamp(
                    Float64(
                        SamplePtr(unsafe_from_address=modulation)[unsafe_offset=i]
                    ),
                    -1.0,
                    1.0,
                )
            var angle = (position + 1.0) * 0.25 * pi  # 0 at hard left, pi/2 right
            var x = input[unsafe_offset=i] if source != 0 else Float32(0.0)
            if channels == 1:
                left[unsafe_offset=i] = x * Float32(cos(angle) + sin(angle)) * 0.7071
            else:
                left[unsafe_offset=i] = x * Float32(cos(angle))
                right[unsafe_offset=i] = x * Float32(sin(angle))
        for c in range(2, channels):
            var extra = _channel(Int(dst), c, frames)
            for i in range(frames):
                extra[unsafe_offset=i] = 0.0


struct Width(WideProcessor, Writable):
    """Widen or narrow a stereo image by scaling its side signal.

    `width` 0 collapses to mono, 1 leaves the image alone, and above 1 pushes
    the sides out. Anything other than two channels passes through unchanged.
    """

    comptime WIDTH = 0

    var width: Smoothed

    def __init__(out self, sample_rate: Float64):
        self.width = Smoothed(1.0, sample_rate)

    @staticmethod
    def param_names() -> List[String]:
        return ["width"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.WIDTH:
            self.width.set(clamp(value, 0.0, 4.0))

    def reset(mut self):
        self.width.snap()

    def process(mut self, ins: Ports, dst: SamplePtr, frames: Int, channels: Int):
        var source = input_address(ins, 0)
        if source == 0:
            for i in range(frames * channels):
                dst[unsafe_offset=i] = 0.0
            return
        if channels != 2:
            for i in range(frames * channels):
                dst[unsafe_offset=i] = SamplePtr(unsafe_from_address=source)[
                    unsafe_offset=i
                ]
            return
        var in_left = _channel(source, 0, frames)
        var in_right = _channel(source, 1, frames)
        var left = _channel(Int(dst), 0, frames)
        var right = _channel(Int(dst), 1, frames)
        for i in range(frames):
            var width = Float32(self.width.next())
            var mid = (in_left[unsafe_offset=i] + in_right[unsafe_offset=i]) * 0.5
            var side = (in_left[unsafe_offset=i] - in_right[unsafe_offset=i]) * 0.5
            side *= width
            left[unsafe_offset=i] = mid + side
            right[unsafe_offset=i] = mid - side
