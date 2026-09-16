"""E1-E3: parameter smoothing, modulation interface, filters under modulation.

Build: mojo build --fp-mode contract=off modulation.mojo -o modulation && ./modulation
"""

from std.math import exp, pi, sin, cos, tan, sqrt
from std.time import perf_counter_ns

comptime SR = 48000.0
comptime FPtr = Pointer[Float32, MutAnyOrigin]


def addr(mut l: List[Float32]) -> FPtr:
    return FPtr(unsafe_from_address=Int(l.unsafe_ptr()))


def noise(n: Int, seed: UInt32) -> List[Float32]:
    var out = List[Float32](capacity=n)
    var s = seed
    for _ in range(n):
        s = s * 1664525 + 1013904223
        out.append(Float32(s >> 8) / Float32(1 << 23) - 1.0)
    return out^


def maxdiff(a: List[Float32], b: List[Float32]) -> Float32:
    var m: Float32 = 0.0
    for i in range(len(a)):
        m = max(m, abs(a[i] - b[i]))
    return m


def msps(n: Int, ns: Int) -> Float64:
    return Float64(n) / Float64(ns) * 1.0e3


# ============================================================================
# E1. Smoothing: is output independent of how the caller splits blocks?
# A gain steps from 0 to 1 at the start; smoothers approach it.
# ============================================================================

trait BlockProc(Copyable, Deinitable):
    def process(mut self, src: FPtr, dst: FPtr, n: Int): ...


@fieldwise_init
struct BlockRamp(BlockProc):
    """Linear ramp from current to target across whatever block is processed."""
    var current: Float32
    var target: Float32

    def process(mut self, src: FPtr, dst: FPtr, n: Int):
        var step = (self.target - self.current) / Float32(max(n, 1))
        for i in range(n):
            self.current += step
            dst[unsafe_offset=i] = src[unsafe_offset=i] * self.current


@fieldwise_init
struct OnePoleSmoother(BlockProc):
    """Per-sample exponential approach; 10 ms time constant."""
    var current: Float32
    var target: Float32
    var k: Float32

    def process(mut self, src: FPtr, dst: FPtr, n: Int):
        var c = self.current
        for i in range(n):
            c += self.k * (self.target - c)
            dst[unsafe_offset=i] = src[unsafe_offset=i] * c
        self.current = c


@fieldwise_init
struct LinearSmoother(BlockProc):
    """Per-sample linear ramp over a fixed number of samples (10 ms)."""
    var current: Float32
    var step: Float32
    var remaining: Int

    def set_target(mut self, target: Float32, length: Int):
        self.step = (target - self.current) / Float32(length)
        self.remaining = length

    def process(mut self, src: FPtr, dst: FPtr, n: Int):
        var c = self.current
        var r = self.remaining
        var i = 0
        # Ramp segment, then a constant segment with no per-sample branch.
        var ramp = min(r, n)
        while i < ramp:
            c += self.step
            dst[unsafe_offset=i] = src[unsafe_offset=i] * c
            i += 1
        r -= ramp
        while i < n:
            dst[unsafe_offset=i] = src[unsafe_offset=i] * c
            i += 1
        self.current = c
        self.remaining = r


def run_split[S: BlockProc](var s: S, mut x: List[Float32], sizes: List[Int]) -> List[Float32]:
    var out = List[Float32](length=len(x), fill=0.0)
    var pos = 0
    var k = 0
    var base_in = Int(x.unsafe_ptr())
    var base_out = Int(out.unsafe_ptr())
    while pos < len(x):
        var n = min(sizes[k % len(sizes)], len(x) - pos)
        s.process(FPtr(unsafe_from_address=base_in + pos * 4), FPtr(unsafe_from_address=base_out + pos * 4), n)
        pos += n
        k += 1
    return out^


def e1() raises:
    print("E1 smoothing: max |whole block - irregular blocks| after a 0 -> 1 gain step")
    var x = noise(4096, 7)
    var whole: List[Int] = [4096]
    var irregular: List[Int] = [1, 7, 64, 3, 513, 1024]
    var length = Int(Float64(0.01) * SR)

    var br = BlockRamp(0.0, 1.0)
    print("  block ramp        ", maxdiff(run_split(br.copy(), x, whole), run_split(br.copy(), x, irregular)))
    var op = OnePoleSmoother(0.0, 1.0, Float32(1.0 - exp(-1.0 / (0.01 * SR))))
    print("  one-pole smoother ", maxdiff(run_split(op.copy(), x, whole), run_split(op.copy(), x, irregular)))
    var ls = LinearSmoother(0.0, 0.0, 0)
    ls.set_target(1.0, length)
    print("  fixed-length ramp ", maxdiff(run_split(ls.copy(), x, whole), run_split(ls.copy(), x, irregular)))

    # Cost once settled: the one-pole smoother never stops multiplying.
    var big = noise(4_800_000, 3)
    var out = List[Float32](length=len(big), fill=0.0)
    var settled_op = OnePoleSmoother(1.0, 1.0, op.k)
    var settled_ls = LinearSmoother(1.0, 0.0, 0)
    var t = perf_counter_ns()
    settled_op.process(addr(big), addr(out), len(big))
    var t_op = Int(perf_counter_ns() - t)
    t = perf_counter_ns()
    settled_ls.process(addr(big), addr(out), len(big))
    var t_ls = Int(perf_counter_ns() - t)
    print("  settled cost, M samples/s: one-pole", msps(len(big), t_op), " fixed-length ramp", msps(len(big), t_ls))


# ============================================================================
# E2/E3. Cutoff modulation interface and filter choice.
# TPT state-variable filter: A. Simper, "Linear Trapezoidal Integrated SVF",
# https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf
# ============================================================================

struct Svf(Copyable):
    var k: Float64
    var ic1: Float64
    var ic2: Float64
    var a1: Float64
    var a2: Float64
    var a3: Float64

    def __init__(out self, q: Float64):
        self.k = 1.0 / q
        self.ic1 = 0.0
        self.ic2 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.a3 = 0.0

    @always_inline
    def coeffs(mut self, cutoff: Float64):
        var g = tan(pi * cutoff / SR)
        self.a1 = 1.0 / (1.0 + g * (g + self.k))
        self.a2 = g * self.a1
        self.a3 = g * self.a2

    @always_inline
    def lowpass(mut self, x: Float32) -> Float32:
        var v0 = Float64(x)
        var v3 = v0 - self.ic2
        var v1 = self.a1 * self.ic1 + self.a2 * v3
        var v2 = self.ic2 + self.a2 * self.ic1 + self.a3 * v3
        self.ic1 = 2.0 * v1 - self.ic1
        self.ic2 = 2.0 * v2 - self.ic2
        return Float32(v2)

    # Interface B: constant parameter, coefficients once per block.
    def process_const(mut self, src: FPtr, dst: FPtr, n: Int, cutoff: Float64):
        self.coeffs(cutoff)
        for i in range(n):
            dst[unsafe_offset=i] = self.lowpass(src[unsafe_offset=i])

    # Interface A (and B when modulated): coefficients every sample from a buffer.
    def process_mod(mut self, src: FPtr, dst: FPtr, n: Int, cutoff: FPtr):
        for i in range(n):
            self.coeffs(Float64(cutoff[unsafe_offset=i]))
            dst[unsafe_offset=i] = self.lowpass(src[unsafe_offset=i])


struct BiquadLP(Copyable):
    """RBJ lowpass, transposed direct form II, as in mdsp today."""
    var q: Float64
    var b0: Float64
    var b1: Float64
    var b2: Float64
    var a1: Float64
    var a2: Float64
    var s1: Float64
    var s2: Float64

    def __init__(out self, q: Float64):
        self.q = q
        self.b0 = 0.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.s1 = 0.0
        self.s2 = 0.0

    @always_inline
    def coeffs(mut self, cutoff: Float64):
        var w0 = 2.0 * pi * cutoff / SR
        var cw = cos(w0)
        var alpha = sin(w0) / (2.0 * self.q)
        var a0 = 1.0 + alpha
        self.b0 = (1.0 - cw) / 2.0 / a0
        self.b1 = (1.0 - cw) / a0
        self.b2 = self.b0
        self.a1 = -2.0 * cw / a0
        self.a2 = (1.0 - alpha) / a0

    def process_mod(mut self, src: FPtr, dst: FPtr, n: Int, cutoff: FPtr):
        for i in range(n):
            self.coeffs(Float64(cutoff[unsafe_offset=i]))
            var x = Float64(src[unsafe_offset=i])
            var y = self.b0 * x + self.s1
            self.s1 = self.b1 * x - self.a1 * y + self.s2
            self.s2 = self.b2 * x - self.a2 * y
            dst[unsafe_offset=i] = Float32(y)


def e2_e3() raises:
    comptime N = 4_800_000
    var x = noise(N, 11)
    var y = List[Float32](length=N, fill=0.0)
    var const_cut = List[Float32](length=N, fill=1000.0)
    var lfo = List[Float32](length=N, fill=0.0)
    for i in range(N):
        lfo[i] = Float32(1000.0 * exp(2.0 * sin(2.0 * pi * 3.0 * Float64(i) / SR)))  # 135 Hz .. 7.4 kHz

    print("E2 cutoff interface, SVF lowpass, M samples/s (best of 5):")
    var px = addr(x)
    var py = addr(y)
    var pc = addr(const_cut)
    var pl = addr(lfo)
    var t = Int.MAX
    for _ in range(5):
        var f = Svf(0.707)
        var s = perf_counter_ns()
        f.process_const(px, py, N, 1000.0)
        t = min(t, Int(perf_counter_ns() - s))
    print("  B constant: coefficients per block      ", msps(N, t))
    t = Int.MAX
    for _ in range(5):
        var f = Svf(0.707)
        var s = perf_counter_ns()
        f.process_mod(px, py, N, pc)
        t = min(t, Int(perf_counter_ns() - s))
    print("  A constant: coefficients per sample     ", msps(N, t))
    t = Int.MAX
    for _ in range(5):
        var f = Svf(0.707)
        var s = perf_counter_ns()
        f.process_mod(px, py, N, pl)
        t = min(t, Int(perf_counter_ns() - s))
    print("  A/B modulated: SVF per-sample tan       ", msps(N, t))
    t = Int.MAX
    for _ in range(5):
        var f = BiquadLP(0.707)
        var s = perf_counter_ns()
        f.process_mod(px, py, N, pl)
        t = min(t, Int(perf_counter_ns() - s))
    print("  A/B modulated: biquad per-sample cos/sin", msps(N, t))

    # E3: hard modulation. Cutoff jumps 200 Hz <-> 12 kHz every 32 samples, Q = 8.
    comptime M = 480000
    var xin = noise(M, 5)
    var hard = List[Float32](length=M, fill=0.0)
    for i in range(M):
        hard[i] = 200.0 if (i // 32) % 2 == 0 else 12000.0
    var ys = List[Float32](length=M, fill=0.0)
    var yb = List[Float32](length=M, fill=0.0)
    var svf = Svf(8.0)
    svf.process_mod(addr(xin), addr(ys), M, addr(hard))
    var bq = BiquadLP(8.0)
    bq.process_mod(addr(xin), addr(yb), M, addr(hard))
    var ms: Float32 = 0.0
    var mb: Float32 = 0.0
    var es: Float64 = 0.0
    var eb: Float64 = 0.0
    for i in range(M):
        ms = max(ms, abs(ys[i]))
        mb = max(mb, abs(yb[i]))
        es += Float64(ys[i]) * Float64(ys[i])
        eb += Float64(yb[i]) * Float64(yb[i])
    print("E3 hard cutoff jumps (200 Hz <-> 12 kHz every 32 samples, Q=8, noise in [-1, 1]):")
    print("  SVF    peak", ms, " rms", sqrt(es / Float64(M)))
    print("  biquad peak", mb, " rms", sqrt(eb / Float64(M)))


def main() raises:
    e1()
    e2_e3()
