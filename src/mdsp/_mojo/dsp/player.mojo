from std.math import clamp, floor

from dsp.processor import Ports, SamplePtr, WideProcessor, input_address


struct Player(WideProcessor, Writable):
    """Plays a buffer it owns, at any rate, forwards or backwards.

    One read position walks the source at `rate` samples per output sample,
    with linear interpolation between neighbours, so rate carries pitch and
    duration together, as tape speed does. A `Shift` of
    `-12 * log2(rate)` semitones after it puts the pitch back, which is time
    stretching that runs in real time.

    Channel-aware because one position serves every channel: they must stay
    together. Ports: ignored, `rate`. Past the end a player that does not loop
    is silent. `load` replaces the source and is not real-time safe.
    """

    comptime RATE = 0
    comptime LOOP = 1
    comptime POSITION = 2

    var sample_rate: Float64
    var channels: Int
    var source: List[Float32]  # `channels` blocks of `frames`
    var frames: Int
    var position: Float64  # in source frames
    var rate: Float64
    var loop: Bool

    def __init__(out self, sample_rate: Float64, channels: Int):
        self.sample_rate = sample_rate
        self.channels = max(channels, 1)
        self.source = List[Float32]()
        self.frames = 0
        self.position = 0.0
        self.rate = 1.0
        self.loop = False

    @staticmethod
    def param_names() -> List[String]:
        return ["rate", "loop", "position"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in", "rate"]

    def load(mut self, address: Int, frames: Int):
        """Copy `channels * frames` planar samples from *address*."""
        var count = frames * self.channels
        self.source = List[Float32](length=count, fill=0.0)
        if count > 0:
            var src = SamplePtr(unsafe_from_address=address)
            var dst = self.source.unsafe_ptr()
            for i in range(count):
                dst[unsafe_offset=i] = src[unsafe_offset=i]
        self.frames = frames
        self.position = 0.0

    def set(mut self, param: Int, value: Float64):
        if param == Self.RATE:
            self.rate = clamp(value, -16.0, 16.0)
        elif param == Self.LOOP:
            self.loop = value >= 0.5
        elif param == Self.POSITION:
            self.position = clamp(
                value * self.sample_rate, 0.0, Float64(max(self.frames, 0))
            )

    def reset(mut self):
        self.position = 0.0

    @always_inline
    def _sample(self, channel: Int, index: Int, frac: Float32) -> Float32:
        var next = index + 1
        if next >= self.frames:
            next = 0 if self.loop else self.frames - 1
        var base = self.source.unsafe_ptr()
        var start = channel * self.frames
        return base[unsafe_offset=start + index] * (1.0 - frac) + base[
            unsafe_offset=start + next
        ] * frac

    def process(mut self, ins: Ports, dst: SamplePtr, frames: Int, channels: Int):
        var modulation = input_address(ins, 1)
        var active = min(channels, self.channels)
        var length = Float64(self.frames)
        for i in range(frames):
            var rate = self.rate
            if modulation != 0:
                rate = clamp(
                    Float64(
                        SamplePtr(unsafe_from_address=modulation)[unsafe_offset=i]
                    ),
                    -16.0,
                    16.0,
                )
            var playing = (
                self.frames > 0
                and self.position >= 0.0
                and self.position < length
            )
            var index = Int(floor(self.position)) if playing else 0
            var frac = Float32(self.position - Float64(index)) if playing else 0.0
            for c in range(active):
                dst[unsafe_offset=c * frames + i] = (
                    self._sample(c, index, frac) if playing else Float32(0.0)
                )
            self.position += rate
            if self.loop and self.frames > 0:
                self.position -= floor(self.position / length) * length
            elif self.position < 0.0:
                self.position = -1.0  # parked before the start, silent
            elif self.position > length:
                self.position = length  # parked at the end, silent
        for c in range(active, channels):
            for i in range(frames):
                dst[unsafe_offset=c * frames + i] = 0.0
