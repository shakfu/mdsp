from std.math import exp, pi, sin
from std.time import perf_counter_ns

trait Processor(Deinitable, Movable):
    def tick(mut self, x: Float32) -> Float32: ...

struct Sine(Movable, Writable):
    var phase: Float64
    var inc: Float64
    def __init__(out self, hz: Float64, sr: Float64):
        self.phase = 0.0
        self.inc = hz / sr

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        var y = Float32(sin(2.0 * pi * self.phase))
        self.phase += self.inc
        if self.phase >= 1.0:
            self.phase -= 1.0
        return y

struct OnePole(Processor, Writable):
    var a: Float32
    var z: Float32
    def __init__(out self, hz: Float64, sr: Float64):
        self.a = Float32(1.0 - exp(-2.0 * pi * hz / sr))
        self.z = 0.0
    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        self.z += self.a * (x - self.z)
        return self.z

struct Gain(Processor, Writable):
    var g: Float32
    def __init__(out self, g: Float32):
        self.g = g
    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        return x * self.g

struct Chain[A: Processor, B: Processor](Processor):
    var a: Self.A
    var b: Self.B
    def __init__(out self, var a: Self.A, var b: Self.B):
        self.a = a^
        self.b = b^
    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        return self.b.tick(self.a.tick(x))

def run[P: Processor](mut p: P, mut buf: List[Float32]):
    for i in range(len(buf)):
        buf[i] = p.tick(buf[i])

def main():
    var c = Chain[Chain[OnePole, Gain], OnePole](Chain[OnePole, Gain](OnePole(1000.0, 48000.0), Gain(0.5)), OnePole(500.0, 48000.0))
    var buf = List[Float32](length=480000, fill=1.0)
    var t = perf_counter_ns()
    run(c, buf)
    var dt = perf_counter_ns() - t
    print("last:", buf[len(buf)-1], " ns:", dt, " M samp/s:", Float64(480000) / Float64(dt) * 1000.0)
