"""E4 benchmark: the demo patch through the Variant engine vs the same kernels called directly.

Patch: saw 110 Hz + saw 110.7 Hz -> mix -> SVF lowpass (cutoff <- sine 0.5 Hz scaled
to 200..4000 Hz) -> gain 0.5.

Build: mojo build --fp-mode contract=off -I ../../../../src/mdsp/_mojo -I . graph_bench.mojo -o graph_bench
"""

from std.time import perf_counter_ns

from dsp import Gain, Saw, Sine
from engine import AnyNode, ExpScale, FPtr, Graph, Mix2, Mono, SvfLP, Ports

comptime SR = 48000.0
comptime SECONDS = 20


def build_demo(block: Int) raises -> Graph:
    var g = Graph(SR, block)
    var s1 = g.add_kind("saw")
    var s2 = g.add_kind("saw")
    var mix = g.add_kind("mix2")
    var lfo = g.add_kind("sine")
    var scale = g.add_kind("expscale")
    var svf = g.add_kind("svf")
    var out = g.add_kind("gain")
    g.set(s1, 0, 110.0)
    g.set(s2, 0, 110.7)
    g.set(lfo, 0, 0.5)
    g.set(scale, 0, 200.0)
    g.set(scale, 1, 4000.0)
    g.set(svf, 1, 4.0)
    g.set(out, 0, 0.5)
    g.connect(s1, mix, 0)
    g.connect(s2, mix, 1)
    g.connect(lfo, scale, 0)
    g.connect(mix, svf, 0)
    g.connect(scale, svf, 1)
    g.connect(svf, out, 0)
    return g^


def run_static(block: Int, mut out: List[Float32]):
    """The same patch with the node types called directly: no Variant, no wiring table."""
    var s1 = Mono[Saw](SR)
    var s2 = Mono[Saw](SR)
    var mix = Mix2(SR)
    var lfo = Mono[Sine](SR)
    var scale = ExpScale(SR)
    var svf = SvfLP(SR)
    var gain = Mono[Gain](SR)
    s1.set(0, 110.0)
    s2.set(0, 110.7)
    lfo.set(0, 0.5)
    scale.set(0, 200.0)
    scale.set(1, 4000.0)
    svf.set(1, 4.0)
    gain.set(0, 0.5)
    var bufs = List[Float32](length=7 * block, fill=0.0)
    var base = Int(bufs.unsafe_ptr())
    var stride = block * 4
    var ports = InlineArray[Int, 4](fill=0)
    var p = Ports(unsafe_from_address=Int(Pointer(to=ports)))
    var total = len(out)
    var done = 0
    while done < total:
        var n = min(block, total - done)
        ports[0] = 0
        ports[1] = 0
        s1.process(p, FPtr(unsafe_from_address=base), n)
        s2.process(p, FPtr(unsafe_from_address=base + stride), n)
        ports[0] = base
        ports[1] = base + stride
        mix.process(p, FPtr(unsafe_from_address=base + 2 * stride), n)
        ports[0] = 0
        ports[1] = 0
        lfo.process(p, FPtr(unsafe_from_address=base + 3 * stride), n)
        ports[0] = base + 3 * stride
        scale.process(p, FPtr(unsafe_from_address=base + 4 * stride), n)
        ports[0] = base + 2 * stride
        ports[1] = base + 4 * stride
        svf.process(p, FPtr(unsafe_from_address=base + 5 * stride), n)
        ports[0] = base + 5 * stride
        ports[1] = 0
        gain.process(p, FPtr(unsafe_from_address=base + 6 * stride), n)
        var o = out.unsafe_ptr()
        for i in range(n):
            o[unsafe_offset=done + i] = bufs[6 * block + i]
        done += n


def main() raises:
    var total = 48000 * SECONDS
    for bi in range(3):
        var block = 64 if bi == 0 else (512 if bi == 1 else 4096)
        var a = List[Float32](length=total, fill=0.0)
        var b = List[Float32](length=total, fill=0.0)
        var best_static = Int.MAX
        var best_graph = Int.MAX
        for _ in range(3):
            var t = perf_counter_ns()
            run_static(block, a)
            best_static = min(best_static, Int(perf_counter_ns() - t))
            var g = build_demo(block)
            t = perf_counter_ns()
            g.process(FPtr(unsafe_from_address=Int(b.unsafe_ptr())), total)
            best_graph = min(best_graph, Int(perf_counter_ns() - t))
        var diff: Float32 = 0.0
        for i in range(total):
            diff = max(diff, abs(a[i] - b[i]))
        var per_block_us = Float64(best_graph) / Float64(total // block) / 1.0e3
        print(
            "block", block,
            "| static", Float64(total) / Float64(best_static) * 1.0e3, "M samples/s",
            "| graph", Float64(total) / Float64(best_graph) * 1.0e3, "M samples/s",
            "| graph per block", per_block_us, "us",
            "| realtime x", Float64(total) / SR / (Float64(best_graph) / 1.0e9),
            "| maxdiff", diff,
        )
