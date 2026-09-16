"""A graph of kernels, rendered one block at a time.

Nodes are held in a `Variant`, so a patch is built at runtime with no compiler.
Dispatch costs nothing measurable next to calling the kernels directly
(docs/dev/spikes/2026-09-16-interface). One `Graph` is mono; `_core` runs one
per channel, as it does for single kernels.

A node may only read from nodes added before it, so no cycles are possible and
nodes run in the order they were added.
"""

from std.utils import Variant

from dsp.chorus import Chorus
from dsp.delay import Delay
from dsp.filters import Biquad, OnePole, Svf
from dsp.dynamics import Compressor
from dsp.envelope import Adsr
from dsp.reverb import Reverb
from dsp.ops import Gain, Mix, Scale, Shaper
from dsp.oscillators import Noise, Osc, PhasorShape, SawShape, SineShape, SquareShape
from dsp.processor import (
    MAX_INPUTS,
    Port,
    Ports,
    Processor,
    SamplePtr,
    WideProcessor,
    audio_input,
)
from dsp.stereo import Pan, Width


struct Passthrough(Processor, Writable):
    """The graph's input: copies port 0, or silence when nothing is connected."""

    def __init__(out self, sample_rate: Float64):
        pass

    @staticmethod
    def param_names() -> List[String]:
        return []

    @staticmethod
    def input_names() -> List[String]:
        return ["in"]

    def set(mut self, param: Int, value: Float64):
        pass

    def reset(mut self):
        pass

    @always_inline
    def tick(mut self, x: Float32) -> Float32:
        return x

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int):
        var src = audio_input(ins, dst)
        if Int(src) == Int(dst):  # unconnected
            for i in range(n):
                dst[unsafe_offset=i] = 0.0
            return
        for i in range(n):
            dst[unsafe_offset=i] = src[unsafe_offset=i]


struct FeedbackLine(Movable):
    """Carries a node's output back to an earlier input, delayed by `delay`.

    The delay counts samples, not blocks, so a patch sounds the same whatever
    block size renders it. It must cover a whole block: every node writes once
    per block, so a shorter loop could not be read before it was overwritten.
    Each channel has its own line, laid out one after another.
    """

    var source: Int
    var delay: Int
    var length: Int  # samples per channel
    var block: Int  # scratch samples per channel
    var line: List[Float32]
    var scratch: List[Float32]  # the delayed samples this block reads
    var write: Int

    def __init__(out self, source: Int, delay: Int, block: Int, channels: Int):
        self.source = source
        self.delay = delay
        self.length = delay + block
        self.block = block
        self.line = List[Float32](length=self.length * channels, fill=0.0)
        self.scratch = List[Float32](length=block * channels, fill=0.0)
        self.write = 0

    def read_into_scratch(mut self, n: Int, channels: Int):
        var start = self.write - self.delay
        if start < 0:
            start += self.length
        for c in range(channels):
            var read = start
            var offset = c * self.length
            for i in range(n):
                self.scratch[c * self.block + i] = self.line[offset + read]
                read += 1
                if read == self.length:
                    read = 0

    def append(mut self, samples: SamplePtr, n: Int, channels: Int, block: Int):
        var write = self.write
        for c in range(channels):
            write = self.write
            var offset = c * self.length
            for i in range(n):
                self.line[offset + write] = samples[unsafe_offset=c * block + i]
                write += 1
                if write == self.length:
                    write = 0
        self.write = write

    def clear(mut self):
        for ref sample in self.line:
            sample = 0.0
        self.write = 0


comptime AnyNode = Variant[
    Passthrough,
    Osc[PhasorShape],
    Osc[SineShape],
    Osc[SawShape],
    Osc[SquareShape],
    Gain,
    OnePole,
    Biquad,
    Svf,
    Delay,
    Scale,
    Mix,
    Adsr,
    Noise,
    Compressor,
    Shaper,
    Reverb,
    Chorus,
]

comptime AnyWideNode = Variant[Pan, Width]

comptime WIDE_BASE = 100  # kind numbers at or above this are channel-aware


def _wide_node(kind: Int, sample_rate: Float64) raises -> AnyWideNode:
    if kind == WIDE_BASE:
        return AnyWideNode(Pan(sample_rate))
    if kind == WIDE_BASE + 1:
        return AnyWideNode(Width(sample_rate))
    raise Error("unknown node kind")


def _wide_inputs[*Ts: WideProcessor](node: Variant[*Ts]) -> List[String]:
    comptime for i in range(len(Ts)):
        if node.isa[Ts[i]]():
            return Ts[i].input_names()
    return []


def _wide_params[*Ts: WideProcessor](node: Variant[*Ts]) -> List[String]:
    comptime for i in range(len(Ts)):
        if node.isa[Ts[i]]():
            return Ts[i].param_names()
    return []


def _wide_set[*Ts: WideProcessor](mut node: Variant[*Ts], param: Int, value: Float64):
    comptime for i in range(len(Ts)):
        if node.isa[Ts[i]]():
            node[Ts[i]].set(param, value)
            return


def _wide_reset[*Ts: WideProcessor](mut node: Variant[*Ts]):
    comptime for i in range(len(Ts)):
        if node.isa[Ts[i]]():
            node[Ts[i]].reset()
            return


def _wide_process[*Ts: WideProcessor](
    mut node: Variant[*Ts], ins: Ports, dst: SamplePtr, frames: Int, channels: Int
):
    comptime for i in range(len(Ts)):
        if node.isa[Ts[i]]():
            node[Ts[i]].process(ins, dst, frames, channels)
            return


def kind_names() -> List[String]:
    """Node kinds, indexed as `_node` expects; `_core` exposes these to Python."""
    return [
        "input",
        "phasor",
        "sine",
        "saw",
        "square",
        "gain",
        "onepole",
        "biquad",
        "svf",
        "delay",
        "scale",
        "mix",
        "adsr",
        "noise",
        "compressor",
        "shaper",
        "reverb",
        "chorus",
        "pan",
        "width",
    ]


def _node(kind: Int, sample_rate: Float64) raises -> AnyNode:
    if kind == 0:
        return AnyNode(Passthrough(sample_rate))
    if kind == 1:
        return AnyNode(Osc[PhasorShape](sample_rate))
    if kind == 2:
        return AnyNode(Osc[SineShape](sample_rate))
    if kind == 3:
        return AnyNode(Osc[SawShape](sample_rate))
    if kind == 4:
        return AnyNode(Osc[SquareShape](sample_rate))
    if kind == 5:
        return AnyNode(Gain(sample_rate))
    if kind == 6:
        return AnyNode(OnePole(sample_rate))
    if kind == 7:
        return AnyNode(Biquad(sample_rate))
    if kind == 8:
        return AnyNode(Svf(sample_rate))
    if kind == 9:
        return AnyNode(Delay(sample_rate))
    if kind == 10:
        return AnyNode(Scale(sample_rate))
    if kind == 11:
        return AnyNode(Mix(sample_rate))
    if kind == 12:
        return AnyNode(Adsr(sample_rate))
    if kind == 13:
        return AnyNode(Noise(sample_rate))
    if kind == 14:
        return AnyNode(Compressor(sample_rate))
    if kind == 15:
        return AnyNode(Shaper(sample_rate))
    if kind == 16:
        return AnyNode(Reverb(sample_rate))
    if kind == 17:
        return AnyNode(Chorus(sample_rate))
    raise Error("unknown node kind")


comptime FIRST_WIDE_NAME = 18  # position of "pan" in `kind_names`


def kind_index(name: String) raises -> Int:
    """Kind number for a name; channel-aware kinds sit at `WIDE_BASE` and up."""
    var names = kind_names()
    for i in range(len(names)):
        if names[i] == name:
            return i if i < FIRST_WIDE_NAME else WIDE_BASE + i - FIRST_WIDE_NAME
    raise Error("unknown node kind: " + name)


def _inputs[*Ts: Processor](node: Variant[*Ts]) -> List[String]:
    comptime for i in range(len(Ts)):
        if node.isa[Ts[i]]():
            return Ts[i].input_names()
    return []


def _params[*Ts: Processor](node: Variant[*Ts]) -> List[String]:
    comptime for i in range(len(Ts)):
        if node.isa[Ts[i]]():
            return Ts[i].param_names()
    return []


def _set[*Ts: Processor](mut node: Variant[*Ts], param: Int, value: Float64):
    comptime for i in range(len(Ts)):
        if node.isa[Ts[i]]():
            node[Ts[i]].set(param, value)
            return


def _reset[*Ts: Processor](mut node: Variant[*Ts]):
    comptime for i in range(len(Ts)):
        if node.isa[Ts[i]]():
            node[Ts[i]].reset()
            return


def _process[*Ts: Processor](mut node: Variant[*Ts], ins: Ports, dst: SamplePtr, n: Int):
    comptime for i in range(len(Ts)):
        if node.isa[Ts[i]]():
            node[Ts[i]].process(ins, dst, n)
            return


struct Graph(Movable):
    """Graph of kernels rendering `channels` channels.

    Every node owns `channels` buffers of `block` frames, laid out one channel
    after another so a channel-aware node sees them as one planar block. Mono
    kernels get one instance per channel; channel-aware ones get a single
    instance that writes every channel.

    `render` splits any request into `block`-sized chunks, so one call renders a
    whole file offline or one audio callback in real time. Nothing is allocated
    after `add`.
    """

    var sample_rate: Float64
    var block: Int
    var channels: Int
    var nodes: List[AnyNode]  # `channels` instances per mono node
    var wide_nodes: List[AnyWideNode]  # one instance per channel-aware node
    var kinds: List[Int]
    var wide_of: List[Int]  # index into `wide_nodes`, or -1 for a mono node
    var wiring: List[Int]  # MAX_INPUTS source node indices per node, -1 unconnected
    var feedback_of: List[Int]  # index into `feedback` per port, -1 when direct
    var feedback: List[FeedbackLine]
    var arena: List[Float32]
    var output: Int

    def __init__(out self, sample_rate: Float64, block: Int, channels: Int):
        self.sample_rate = sample_rate
        self.block = block
        self.channels = channels
        self.nodes = List[AnyNode]()
        self.wide_nodes = List[AnyWideNode]()
        self.kinds = List[Int]()
        self.wide_of = List[Int]()
        self.wiring = List[Int]()
        self.feedback_of = List[Int]()
        self.feedback = List[FeedbackLine]()
        self.arena = List[Float32]()
        self.output = -1

    def add(mut self, kind: Int) raises -> Int:
        if kind >= WIDE_BASE:
            self.wide_nodes.append(_wide_node(kind, self.sample_rate))
            self.wide_of.append(len(self.wide_nodes) - 1)
            for _ in range(self.channels):  # keep the per-channel stride uniform
                self.nodes.append(_node(0, self.sample_rate))
        else:
            self.wide_of.append(-1)
            for _ in range(self.channels):
                self.nodes.append(_node(kind, self.sample_rate))
        self.kinds.append(kind)
        for _ in range(MAX_INPUTS):
            self.wiring.append(-1)
            self.feedback_of.append(-1)
        self.arena.resize(len(self.kinds) * self.channels * self.block, 0.0)
        self.output = len(self.kinds) - 1
        return len(self.kinds) - 1

    def _check_node(self, node: Int) raises:
        if not (0 <= node and node < len(self.kinds)):
            raise Error("no node with index " + String(node))

    def num_inputs(self, node: Int) raises -> Int:
        return len(self.input_names(node))

    def input_names(self, node: Int) raises -> List[String]:
        self._check_node(node)
        var wide = self.wide_of[node]
        if wide >= 0:
            return _wide_inputs(self.wide_nodes[wide])
        return _inputs(self.nodes[node * self.channels])

    def param_names(self, node: Int) raises -> List[String]:
        self._check_node(node)
        var wide = self.wide_of[node]
        if wide >= 0:
            return _wide_params(self.wide_nodes[wide])
        return _params(self.nodes[node * self.channels])

    def connect(mut self, src: Int, dst: Int, port: Int, delay: Int) raises:
        """Wire `src` into `dst`; `delay` above 0 makes it a feedback edge."""
        self._check_node(src)
        self._check_node(dst)
        if not (0 <= port and port < self.num_inputs(dst)):
            raise Error("no input port " + String(port) + " on that node")
        if delay <= 0:
            if src >= dst:
                raise Error(
                    "a node can only read from nodes added before it; give a"
                    " delay to feed a later node back"
                )
            self.wiring[dst * MAX_INPUTS + port] = src
            self.feedback_of[dst * MAX_INPUTS + port] = -1
            return
        if delay < 1:
            raise Error("a feedback delay must be at least one sample")
        self.feedback.append(
            FeedbackLine(src, delay, self.block, self.channels)
        )
        self.wiring[dst * MAX_INPUTS + port] = src
        self.feedback_of[dst * MAX_INPUTS + port] = len(self.feedback) - 1

    def set(mut self, node: Int, param: Int, value: Float64) raises:
        self._check_node(node)
        var wide = self.wide_of[node]
        if wide >= 0:
            _wide_set(self.wide_nodes[wide], param, value)
            return
        for c in range(self.channels):
            _set(self.nodes[node * self.channels + c], param, value)

    def set_output(mut self, node: Int) raises:
        self._check_node(node)
        self.output = node

    def chunk(self) -> Int:
        """Frames rendered at once: the shortest feedback loop caps it.

        Every node writes its whole chunk before the next reads it, so a loop
        shorter than the chunk could not be read before being overwritten.
        """
        var size = self.block
        for f in range(len(self.feedback)):
            size = min(size, self.feedback[f].delay)
        return size

    def remove(mut self, node: Int) raises:
        """Silence a node and drop every connection touching it.

        Handles stay valid: removing renumbers nothing, it empties the slot.
        Feedback lines that read the node are cleared with it.
        """
        self._check_node(node)
        if node == self.output:
            raise Error("cannot remove the output node; set another output first")
        self.kinds[node] = -1
        self.wide_of[node] = -1
        for c in range(self.channels):
            self.nodes[node * self.channels + c] = _node(0, self.sample_rate)
        for port in range(MAX_INPUTS):
            self.wiring[node * MAX_INPUTS + port] = -1
            self.feedback_of[node * MAX_INPUTS + port] = -1
        for other in range(len(self.kinds)):
            for port in range(MAX_INPUTS):
                var slot = other * MAX_INPUTS + port
                if self.wiring[slot] == node:
                    self.wiring[slot] = -1
                    self.feedback_of[slot] = -1
        for f in range(len(self.feedback)):
            if self.feedback[f].source == node:
                self.feedback[f].clear()
        var base = self.arena.unsafe_ptr()
        for i in range(self.channels * self.block):
            base[unsafe_offset=node * self.channels * self.block + i] = 0.0

    def removed(self, node: Int) raises -> Bool:
        self._check_node(node)
        return self.kinds[node] == -1

    def reset_node(mut self, node: Int) raises:
        """Clear one node's state and end its parameter ramps."""
        self._check_node(node)
        var wide = self.wide_of[node]
        if wide >= 0:
            _wide_reset(self.wide_nodes[wide])
            return
        for c in range(self.channels):
            _reset(self.nodes[node * self.channels + c])

    def reset(mut self):
        for f in range(len(self.feedback)):
            self.feedback[f].clear()
        for ref node in self.nodes:
            _reset(node)
        for ref node in self.wide_nodes:
            _wide_reset(node)
        for ref sample in self.arena:
            sample = 0.0

    def render(mut self, src: Int, dst: SamplePtr, frames: Int) raises:
        """Write `frames` samples of every channel of the output node to `dst`.

        `src` addresses the graph's planar input, or is 0 for silence; input
        nodes read from it. `dst` receives the same planar layout.
        """
        if len(self.kinds) == 0:
            raise Error("the graph has no nodes")
        var ports = InlineArray[Port, MAX_INPUTS](fill=Port())
        var ports_ptr = Ports(unsafe_from_address=Int(Pointer(to=ports)))
        var base = Int(self.arena.unsafe_ptr())
        var node_stride = self.channels * self.block * 4
        var channel_stride = self.block * 4
        var chunk = self.chunk()
        var done = 0
        while done < frames:
            var n = min(chunk, frames - done)
            for f in range(len(self.feedback)):
                self.feedback[f].read_into_scratch(n, self.channels)
            for j in range(len(self.kinds)):
                if self.kinds[j] == -1:  # removed: its buffer stays silent
                    continue
                var wide = self.wide_of[j]
                var output_base = base + j * node_stride
                if wide >= 0:
                    self._fill_ports(ports, j, 0, base, node_stride, 0, src, done, frames)
                    _wide_process(
                        self.wide_nodes[wide],
                        ports_ptr,
                        SamplePtr(unsafe_from_address=output_base),
                        n,
                        self.channels,
                    )
                    continue
                for c in range(self.channels):
                    self._fill_ports(
                        ports, j, c, base, node_stride, channel_stride, src, done, frames
                    )
                    _process(
                        self.nodes[j * self.channels + c],
                        ports_ptr,
                        SamplePtr(unsafe_from_address=output_base + c * channel_stride),
                        n,
                    )
            for f in range(len(self.feedback)):
                var source = self.feedback[f].source
                self.feedback[f].append(
                    SamplePtr(unsafe_from_address=base + source * node_stride),
                    n,
                    self.channels,
                    self.block,
                )
            var result = base + self.output * node_stride
            for c in range(self.channels):
                var channel = SamplePtr(
                    unsafe_from_address=result + c * channel_stride
                )
                for i in range(n):
                    dst[unsafe_offset=c * frames + done + i] = channel[unsafe_offset=i]
            done += n

    @always_inline
    def _fill_ports(
        mut self,
        mut ports: InlineArray[Port, MAX_INPUTS],
        node: Int,
        channel: Int,
        base: Int,
        node_stride: Int,
        channel_stride: Int,
        src: Int,
        done: Int,
        frames: Int,
    ):
        """Point each input port at its source for this node and channel.

        A channel-aware node passes `channel_stride` 0 so its ports address a
        source's first channel, with the rest following contiguously.
        """
        for k in range(MAX_INPUTS):
            var source = self.wiring[node * MAX_INPUTS + k]
            var loop = self.feedback_of[node * MAX_INPUTS + k]
            if loop >= 0:
                ports[k] = Port(
                    Int(self.feedback[loop].scratch.unsafe_ptr())
                    + channel * self.block * 4,
                    self.channels,
                )
            elif source >= 0:
                ports[k] = Port(
                    base + source * node_stride + channel * channel_stride,
                    self.channels,
                )
            else:
                ports[k] = Port(0, self.channels)
        if self.kinds[node] == 0 and src != 0:  # input node
            ports[0] = Port(src + (channel * frames + done) * 4, self.channels)
