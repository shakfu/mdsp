from std.math import exp, pi

struct OnePole(Copyable, Movable, Writable):
    var a: Float32
    var z: Float32

    def __init__(out self):
        self.a = 1.0
        self.z = 0.0

    def set_cutoff(mut self, hz: Float64, sr: Float64):
        self.a = Float32(1.0 - exp(-2.0 * pi * hz / sr))

    @always_inline
    def next(mut self, x: Float32) -> Float32:
        self.z += self.a * (x - self.z)
        return self.z
