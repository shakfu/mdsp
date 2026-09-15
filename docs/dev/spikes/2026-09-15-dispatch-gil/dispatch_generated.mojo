from std.utils import Variant
from std.math import exp, pi
from std.time import perf_counter_ns

comptime SR = 48000.0
comptime TOTAL = 480000 * 4
comptime STAGES = 8          # graph: Saw -> (OnePole -> Gain) x STAGES  = 17 nodes

comptime FPtr = Pointer[Float32, MutAnyOrigin]

trait Processor(Copyable):
    def tick(mut self, x: Float32) -> Float32: ...
    def process(mut self, src: FPtr, dst: FPtr, n: Int): ...

struct Saw(Processor, Writable):
    var phase: Float32
    var inc: Float32
    def __init__(out self, hz: Float64):
        self.phase = 0.0
        self.inc = Float32(hz / SR)
    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        var y = self.phase * 2.0 - 1.0
        self.phase += self.inc
        if self.phase >= 1.0:
            self.phase -= 1.0
        return y
    def process(mut self, src: FPtr, dst: FPtr, n: Int):
        var p = self.phase          # local copy: dst may alias self
        var inc = self.inc
        for i in range(n):
            dst[unsafe_offset=i] = p * 2.0 - 1.0
            p += inc
            if p >= 1.0:
                p -= 1.0
        self.phase = p

struct OnePole(Processor, Writable):
    var a: Float32
    var z: Float32
    def __init__(out self, hz: Float64):
        self.a = Float32(1.0 - exp(-2.0 * pi * hz / SR))
        self.z = 0.0
    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        self.z += self.a * (x - self.z)
        return self.z
    def process(mut self, src: FPtr, dst: FPtr, n: Int):
        var z = self.z
        var a = self.a
        for i in range(n):
            z += a * (src[unsafe_offset=i] - z)
            dst[unsafe_offset=i] = z
        self.z = z

struct Gain(Processor, Writable):
    var g: Float32
    def __init__(out self, g: Float32):
        self.g = g
    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        return x * self.g
    def process(mut self, src: FPtr, dst: FPtr, n: Int):
        var g = self.g
        for i in range(n):
            dst[unsafe_offset=i] = src[unsafe_offset=i] * g

comptime Node = Variant[Saw, OnePole, Gain]

def make_nodes() -> List[Node]:
    var nodes = List[Node]()
    nodes.append(Node(Saw(110.0)))
    for s in range(STAGES):
        nodes.append(Node(OnePole(4000.0 - Float64(s) * 300.0)))
        nodes.append(Node(Gain(0.99)))
    return nodes^

@always_inline
def slot(base_addr: Int, j: Int, block: Int) -> FPtr:
    return FPtr(unsafe_from_address=base_addr + j * block * 4)

# 1. static, per-sample ticks, fully inlined (option B)
def run_static_tick(block: Int, mut out: List[Float32]):
    var s = Saw(110.0)
    var f = List[OnePole]()
    var g = List[Gain]()
    for k in range(STAGES):
        f.append(OnePole(4000.0 - Float64(k) * 300.0))
        g.append(Gain(0.99))
    var o = out.unsafe_ptr()
    var pos = 0
    while pos < TOTAL:
        for i in range(block):
            var y = s.tick(0.0)
            for k in range(STAGES):
                y = g[k].tick(f[k].tick(y))
            o[unsafe_offset=pos + i] = y
        pos += block

# 2. static types, block process, no dispatch (isolates dispatch from block effects)
def run_static_block(block: Int, mut out: List[Float32]):
    var s = Saw(110.0)
    var f = List[OnePole]()
    var g = List[Gain]()
    for k in range(STAGES):
        f.append(OnePole(4000.0 - Float64(k) * 300.0))
        g.append(Gain(0.99))
    var bufs = List[Float32](length=2 * block, fill=0.0)
    var a = slot(Int(bufs.unsafe_ptr()), 0, block)
    var b = slot(Int(bufs.unsafe_ptr()), 1, block)
    var o = out.unsafe_ptr()
    var pos = 0
    while pos < TOTAL:
        s.process(a, a, block)
        for k in range(STAGES):
            f[k].process(a, b, block)
            g[k].process(b, a, block)
        for i in range(block):
            o[unsafe_offset=pos + i] = a[unsafe_offset=i]
        pos += block

# 3. Variant, dispatch per node per block
@always_inline
def node_process[*Ts: Processor](mut n: Variant[*Ts], src: FPtr, dst: FPtr, k: Int):
    comptime for i in range(len(Ts)):
        if n.isa[Ts[i]]():
            n[Ts[i]].process(src, dst, k)
            return

def run_variant_block(block: Int, mut out: List[Float32]):
    var nodes = make_nodes()
    var nn = len(nodes)
    var bufs = List[Float32](length=(nn + 1) * block, fill=0.0)
    var base = Int(bufs.unsafe_ptr())
    var o = out.unsafe_ptr()
    var pos = 0
    while pos < TOTAL:
        for j in range(nn):
            node_process(nodes[j], slot(base, j, block), slot(base, j + 1, block), block)
        var last = slot(base, nn, block)
        for i in range(block):
            o[unsafe_offset=pos + i] = last[unsafe_offset=i]
        pos += block

# 4. Variant, dispatch per node per sample
@always_inline
def node_tick[*Ts: Processor](mut n: Variant[*Ts], x: Float32) -> Float32:
    comptime for i in range(len(Ts)):
        if n.isa[Ts[i]]():
            return n[Ts[i]].tick(x)
    return x

def run_variant_sample(block: Int, mut out: List[Float32]):
    var nodes = make_nodes()
    var nn = len(nodes)
    var o = out.unsafe_ptr()
    var pos = 0
    while pos < TOTAL:
        for i in range(block):
            var y: Float32 = 0.0
            for j in range(nn):
                y = node_tick(nodes[j], y)
            o[unsafe_offset=pos + i] = y
        pos += block

def maxdiff(a: List[Float32], b: List[Float32]) -> Float32:
    var m: Float32 = 0.0
    for i in range(len(a)):
        var d = abs(a[i] - b[i])
        if d > m:
            m = d
    return m

def report(name: String, ns: Int):
    print("  ", name, Float64(TOTAL) / Float64(ns) * 1.0e3, "M samp/s")

def main():
    print("nodes:", 1 + 2 * STAGES, " samples:", TOTAL)
    for bi in range(4):
        var block = 1 if bi == 0 else (16 if bi == 1 else (64 if bi == 2 else 512))
        print("block", block)
        var r = List[Float32](length=TOTAL, fill=0.0)
        var o2 = List[Float32](length=TOTAL, fill=0.0)
        var o3 = List[Float32](length=TOTAL, fill=0.0)
        var o4 = List[Float32](length=TOTAL, fill=0.0)
        var t = perf_counter_ns(); run_static_tick(block, r); report("static tick    ", Int(perf_counter_ns() - t))
        t = perf_counter_ns(); run_static_block(block, o2); report("static block   ", Int(perf_counter_ns() - t))
        t = perf_counter_ns(); run_variant_block(block, o3); report("variant block  ", Int(perf_counter_ns() - t))
        t = perf_counter_ns(); run_variant_sample(block, o4); report("variant sample ", Int(perf_counter_ns() - t))
        print("   maxdiff:", maxdiff(r, o2), maxdiff(r, o3), maxdiff(r, o4))
