"""Parameter smoothing."""

comptime SMOOTH_SECONDS = 0.01


struct Smoothed(Copyable, Writable):
    """A parameter that ramps to its target over a fixed number of samples.

    Advancing per sample keeps output independent of how callers split blocks;
    a ramp spread over each block does not (docs/dev/spikes/2026-09-16-interface).
    """

    var value: Float64
    var target: Float64
    var step: Float64
    var remaining: Int
    var length: Int

    def __init__(out self, value: Float64, sample_rate: Float64):
        self.value = value
        self.target = value
        self.step = 0.0
        self.remaining = 0
        self.length = max(Int(SMOOTH_SECONDS * sample_rate), 1)

    def set(mut self, target: Float64):
        """Ramp to *target* from the current value."""
        self.target = target
        self.step = (target - self.value) / Float64(self.length)
        self.remaining = self.length

    def snap(mut self):
        """Jump to the target, ending any ramp."""
        self.value = self.target
        self.remaining = 0

    @always_inline
    def next(mut self) -> Float64:
        if self.remaining > 0:
            self.value += self.step
            self.remaining -= 1
            if self.remaining == 0:
                self.value = self.target
        return self.value

    @always_inline
    def ramping(self, n: Int) -> Int:
        """How many of the next *n* samples are still ramping."""
        return min(self.remaining, n)
