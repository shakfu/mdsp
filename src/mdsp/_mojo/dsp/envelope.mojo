from std.math import clamp

from dsp.processor import Ports, Processor, SamplePtr, input_address

comptime IDLE = 0
comptime ATTACK = 1
comptime DECAY = 2
comptime SUSTAIN = 3
comptime RELEASE = 4


struct Adsr(Processor, Writable):
    """Attack-decay-sustain-release envelope, output in [0, 1].

    Port 0 is ignored: this generates a control signal, so connect its output
    to something that takes it, such as a `Gain`'s `gain` input. `gate` above
    0.5 starts the attack, below it starts the release; the `gate` port lets
    another node do that at audio rate.

    Segments are linear. `attack`, `decay` and `release` are the seconds a full
    0-to-1 (or level-to-0) segment takes.
    """

    comptime ATTACK_TIME = 0
    comptime DECAY_TIME = 1
    comptime SUSTAIN_LEVEL = 2
    comptime RELEASE_TIME = 3
    comptime GATE = 4

    var sample_rate: Float64
    var attack: Float64
    var decay: Float64
    var sustain: Float64
    var release: Float64
    var gate: Float64
    var stage: Int
    var level: Float64
    var step: Float64
    var remaining: Int  # samples left in the current segment

    def __init__(out self, sample_rate: Float64):
        self.sample_rate = sample_rate
        self.attack = 0.01
        self.decay = 0.1
        self.sustain = 0.7
        self.release = 0.2
        self.gate = 0.0
        self.stage = IDLE
        self.level = 0.0
        self.step = 0.0
        self.remaining = 0

    @staticmethod
    def param_names() -> List[String]:
        return ["attack", "decay", "sustain", "release", "gate"]

    @staticmethod
    def input_names() -> List[String]:
        return ["in", "gate"]

    def set(mut self, param: Int, value: Float64):
        if param == Self.ATTACK_TIME:
            self.attack = max(value, 0.0)
        elif param == Self.DECAY_TIME:
            self.decay = max(value, 0.0)
        elif param == Self.SUSTAIN_LEVEL:
            self.sustain = clamp(value, 0.0, 1.0)
        elif param == Self.RELEASE_TIME:
            self.release = max(value, 0.0)
        elif param == Self.GATE:
            self.gate = value

    def reset(mut self):
        self.stage = IDLE
        self.level = 0.0
        self.remaining = 0

    @always_inline
    def _begin(mut self, stage: Int, target: Float64, seconds: Float64):
        """Start a segment that reaches `target` after `seconds`.

        Counting samples keeps a segment's length and endpoint exact; adding a
        step repeatedly drifts. A time changed mid-segment applies to the next.
        """
        self.stage = stage
        self.remaining = max(Int(seconds * self.sample_rate), 1)
        self.step = (target - self.level) / Float64(self.remaining)

    @always_inline
    def _advance(mut self, gate: Float64) -> Float32:
        if gate > 0.5:
            if self.stage == IDLE or self.stage == RELEASE:
                self._begin(ATTACK, 1.0, self.attack)
        elif self.stage != IDLE and self.stage != RELEASE:
            self._begin(RELEASE, 0.0, self.release)

        if self.stage == SUSTAIN:
            self.level = self.sustain
        elif self.stage != IDLE:
            self.level += self.step
            self.remaining -= 1
            if self.remaining == 0:
                if self.stage == ATTACK:
                    self.level = 1.0
                    self._begin(DECAY, self.sustain, self.decay)
                elif self.stage == DECAY:
                    self.level = self.sustain
                    self.stage = SUSTAIN
                else:  # release finished
                    self.level = 0.0
                    self.stage = IDLE
        return Float32(self.level)

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        return self._advance(self.gate)

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var gate_mod = input_address(ins, 1)
        if gate_mod == 0:
            var gate = self.gate
            for i in range(n):
                dst[unsafe_offset=i] = self._advance(gate)
            return
        var gate = SamplePtr(unsafe_from_address=gate_mod)
        for i in range(n):
            dst[unsafe_offset=i] = self._advance(Float64(gate[unsafe_offset=i]))
