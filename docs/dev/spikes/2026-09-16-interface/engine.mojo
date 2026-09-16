"""E4: port-based node interface and a Variant graph engine.

Existing mdsp kernels (`Processor`, 1 in / 1 out) become nodes through the
`Mono[P]` adapter. New node types take several inputs; a modulatable parameter
is an optional input port, so an unconnected port keeps the constant path (E2).
"""

from std.math import exp, log, pi, tan
from std.utils import Variant

from dsp import Gain, Processor, SamplePtr, Saw, Sine

comptime FPtr = Pointer[Float32, MutAnyOrigin]
comptime Ports = Pointer[Int, MutAnyOrigin]  # input buffer addresses; 0 = unconnected


@always_inline
def port(ins: Ports, k: Int) -> Int:
    return ins[unsafe_offset=k]


trait Node(Copyable, Deinitable):
    @staticmethod
    def num_inputs() -> Int: ...

    def set(mut self, param: Int, value: Float64): ...

    def process(mut self, ins: Ports, dst: FPtr, n: Int): ...


struct Mono[P: Processor](Node):
    """Adapter: an existing single-input kernel as a graph node."""

    var kernel: Self.P

    def __init__(out self, sample_rate: Float64):
        self.kernel = Self.P(sample_rate)

    @staticmethod
    def num_inputs() -> Int:
        return 1

    def set(mut self, param: Int, value: Float64):
        self.kernel.set(param, value)

    def process(mut self, ins: Ports, dst: FPtr, n: Int):
        var src = port(ins, 0)
        # Generators ignore their input; an unconnected port passes the output buffer.
        self.kernel.process(SamplePtr(unsafe_from_address=src if src != 0 else Int(dst)), dst, n)


struct Mix2(Node):
    var ga: Float32
    var gb: Float32

    def __init__(out self, sample_rate: Float64):
        self.ga = 0.5
        self.gb = 0.5

    @staticmethod
    def num_inputs() -> Int:
        return 2

    def set(mut self, param: Int, value: Float64):
        if param == 0:
            self.ga = Float32(value)
        elif param == 1:
            self.gb = Float32(value)

    def process(mut self, ins: Ports, dst: FPtr, n: Int):
        var a = FPtr(unsafe_from_address=port(ins, 0))
        var b = FPtr(unsafe_from_address=port(ins, 1))
        var ga = self.ga
        var gb = self.gb
        for i in range(n):
            dst[unsafe_offset=i] = ga * a[unsafe_offset=i] + gb * b[unsafe_offset=i]


struct ExpScale(Node):
    """Map [-1, 1] to [lo, hi] exponentially: a control-rate helper for LFOs."""

    var lo: Float64
    var hi: Float64

    def __init__(out self, sample_rate: Float64):
        self.lo = 100.0
        self.hi = 1000.0

    @staticmethod
    def num_inputs() -> Int:
        return 1

    def set(mut self, param: Int, value: Float64):
        if param == 0:
            self.lo = value
        elif param == 1:
            self.hi = value

    def process(mut self, ins: Ports, dst: FPtr, n: Int):
        var x = FPtr(unsafe_from_address=port(ins, 0))
        var lo = self.lo
        var half_log_ratio = 0.5 * log(self.hi / self.lo)
        for i in range(n):
            dst[unsafe_offset=i] = Float32(lo * exp((Float64(x[unsafe_offset=i]) + 1.0) * half_log_ratio))


struct SvfLP(Node):
    """TPT SVF lowpass. Inputs: audio, cutoff in Hz (optional; overrides `cutoff`)."""

    var sample_rate: Float64
    var cutoff: Float64
    var k: Float64
    var ic1: Float64
    var ic2: Float64

    def __init__(out self, sample_rate: Float64):
        self.sample_rate = sample_rate
        self.cutoff = 1000.0
        self.k = 1.0 / 0.707
        self.ic1 = 0.0
        self.ic2 = 0.0

    @staticmethod
    def num_inputs() -> Int:
        return 2

    def set(mut self, param: Int, value: Float64):
        if param == 0:
            self.cutoff = value
        elif param == 1:
            self.k = 1.0 / value

    def process(mut self, ins: Ports, dst: FPtr, n: Int):
        var x = FPtr(unsafe_from_address=port(ins, 0))
        var mod = port(ins, 1)
        var k = self.k
        var ic1 = self.ic1
        var ic2 = self.ic2
        var w = pi / self.sample_rate
        if mod == 0:
            var g = tan(w * self.cutoff)
            var a1 = 1.0 / (1.0 + g * (g + k))
            var a2 = g * a1
            var a3 = g * a2
            for i in range(n):
                var v3 = Float64(x[unsafe_offset=i]) - ic2
                var v1 = a1 * ic1 + a2 * v3
                var v2 = ic2 + a2 * ic1 + a3 * v3
                ic1 = 2.0 * v1 - ic1
                ic2 = 2.0 * v2 - ic2
                dst[unsafe_offset=i] = Float32(v2)
        else:
            var c = FPtr(unsafe_from_address=mod)
            var nyq = 0.49 * self.sample_rate
            for i in range(n):
                var g = tan(w * min(max(Float64(c[unsafe_offset=i]), 1.0), nyq))
                var a1 = 1.0 / (1.0 + g * (g + k))
                var a2 = g * a1
                var a3 = g * a2
                var v3 = Float64(x[unsafe_offset=i]) - ic2
                var v1 = a1 * ic1 + a2 * v3
                var v2 = ic2 + a2 * ic1 + a3 * v3
                ic1 = 2.0 * v1 - ic1
                ic2 = 2.0 * v2 - ic2
                dst[unsafe_offset=i] = Float32(v2)
        self.ic1 = ic1
        self.ic2 = ic2


comptime AnyNode = Variant[Mono[Saw], Mono[Sine], Mono[Gain], Mix2, ExpScale, SvfLP]


def _num_inputs[*Ts: Node](n: Variant[*Ts]) -> Int:
    comptime for i in range(len(Ts)):
        if n.isa[Ts[i]]():
            return Ts[i].num_inputs()
    return 0


def _set[*Ts: Node](mut n: Variant[*Ts], param: Int, value: Float64):
    comptime for i in range(len(Ts)):
        if n.isa[Ts[i]]():
            n[Ts[i]].set(param, value)
            return


def _process[*Ts: Node](mut n: Variant[*Ts], ins: Ports, dst: FPtr, frames: Int):
    comptime for i in range(len(Ts)):
        if n.isa[Ts[i]]():
            n[Ts[i]].process(ins, dst, frames)
            return


comptime MAX_INPUTS = 4


struct Graph(Movable):
    """Nodes in insertion order; a node may only read from earlier nodes.

    Each node owns one output buffer of `block` frames in `arena`. `process`
    splits any request into `block`-sized chunks, so one call can render a
    whole file (offline) or one audio callback (real time). No allocation
    happens after `add`.
    """

    var sample_rate: Float64
    var block: Int
    var nodes: List[AnyNode]
    var wiring: List[Int]  # MAX_INPUTS source indices per node, -1 = unconnected
    var arena: List[Float32]

    def __init__(out self, sample_rate: Float64, block: Int):
        self.sample_rate = sample_rate
        self.block = block
        self.nodes = List[AnyNode]()
        self.wiring = List[Int]()
        self.arena = List[Float32]()

    def add(mut self, var node: AnyNode) -> Int:
        self.nodes.append(node^)
        for _ in range(MAX_INPUTS):
            self.wiring.append(-1)
        self.arena.resize(len(self.nodes) * self.block, 0.0)
        return len(self.nodes) - 1

    def add_kind(mut self, kind: String) raises -> Int:
        var sr = self.sample_rate
        if kind == "saw":
            return self.add(AnyNode(Mono[Saw](sr)))
        if kind == "sine":
            return self.add(AnyNode(Mono[Sine](sr)))
        if kind == "gain":
            return self.add(AnyNode(Mono[Gain](sr)))
        if kind == "mix2":
            return self.add(AnyNode(Mix2(sr)))
        if kind == "expscale":
            return self.add(AnyNode(ExpScale(sr)))
        if kind == "svf":
            return self.add(AnyNode(SvfLP(sr)))
        raise Error("unknown node kind: " + kind)

    def connect(mut self, src: Int, dst: Int, port_index: Int) raises:
        if not (0 <= src and src < dst and dst < len(self.nodes)):
            raise Error("connections must run from an earlier node to a later one")
        if not (0 <= port_index and port_index < _num_inputs(self.nodes[dst])):
            raise Error("no such input port")
        self.wiring[dst * MAX_INPUTS + port_index] = src

    def set(mut self, node: Int, param: Int, value: Float64) raises:
        if not (0 <= node and node < len(self.nodes)):
            raise Error("no such node")
        _set(self.nodes[node], param, value)

    def process(mut self, dst: FPtr, frames: Int):
        """Render `frames` samples of the last node's output into `out`."""
        var ports = InlineArray[Int, MAX_INPUTS](fill=0)
        var ports_ptr = Ports(unsafe_from_address=Int(Pointer(to=ports)))
        var base = Int(self.arena.unsafe_ptr())
        var stride = self.block * 4
        var last = len(self.nodes) - 1
        var done = 0
        while done < frames:
            var n = min(self.block, frames - done)
            for j in range(len(self.nodes)):
                for k in range(MAX_INPUTS):
                    var src = self.wiring[j * MAX_INPUTS + k]
                    ports[k] = base + src * stride if src >= 0 else 0
                _process(self.nodes[j], ports_ptr, FPtr(unsafe_from_address=base + j * stride), n)
            var result = FPtr(unsafe_from_address=base + last * stride)
            for i in range(n):
                dst[unsafe_offset=done + i] = result[unsafe_offset=i]
            done += n
