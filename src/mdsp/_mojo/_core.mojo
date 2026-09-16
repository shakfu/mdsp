"""Python extension module `mdsp._core`.

Each kernel is exposed as `Bank[Kernel]`: one kernel instance per channel.
Buffers cross as addresses of planar `[channels, frames]` float32 arrays:
the audio input, the output, and one optional buffer per modulation input.
`mdsp._base` validates dtype, layout and shape before passing an address;
nothing here can check them.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import size_of

from stream import Stream, audio_devices as portaudio_devices
from dsp import (
    MAX_INPUTS,
    Chorus,
    Pan,
    Width,
    WideProcessor,
    Adsr,
    Compressor,
    Graph,
    Mix,
    Noise,
    Port,
    Reverb,
    Biquad,
    Delay,
    Gain,
    OnePole,
    Phasor,
    Ports,
    Processor,
    SamplePtr,
    Saw,
    Scale,
    Shaper,
    Sine,
    Square,
    Svf,
    kind_index,
    kind_names,
)


struct Bank[P: Processor](Movable, Writable):
    var units: List[Self.P]

    def __init__(out self, sample_rate: Float64, channels: Int):
        self.units = List[Self.P](capacity=channels)
        for _ in range(channels):
            self.units.append(Self.P(sample_rate))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Bank(channels=", len(self.units), ")")

    @staticmethod
    def py_init(out self: Self, args: PythonObject, kwargs: PythonObject) raises:
        if len(args) != 2:
            raise Error("expected (sample_rate, channels)")
        var sample_rate = Float64(py=args[0])
        var channels = Int(py=args[1])
        if not sample_rate > 0.0:
            raise Error("sample_rate must be positive")
        if channels < 1:
            raise Error("channels must be >= 1")
        self = Self(sample_rate, channels)

    @staticmethod
    def param_names(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var names = Python.list()
        for name in Self.P.param_names():
            names.append(PythonObject(name))
        return names

    @staticmethod
    def input_names(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var names = Python.list()
        for name in Self.P.input_names():
            names.append(PythonObject(name))
        return names

    @staticmethod
    def set(
        self_ptr: Pointer[Self, MutAnyOrigin], param: PythonObject, value: PythonObject
    ) raises -> PythonObject:
        var p = Int(py=param)
        var v = Float64(py=value)
        for ref unit in self_ptr[].units:
            unit.set(p, v)
        return PythonObject(None)

    @staticmethod
    def reset(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        for ref unit in self_ptr[].units:
            unit.reset()
        return PythonObject(None)

    @staticmethod
    def process(
        self_ptr: Pointer[Self, MutAnyOrigin],
        src: PythonObject,
        dst: PythonObject,
        frames: PythonObject,
        mods: PythonObject,
    ) raises -> PythonObject:
        """Process `len(units)` planar channels of `frames` samples each.

        `mods` holds one address per modulation input, in `input_names()` order
        after the audio input; 0 means unconnected.
        """
        var src_addr = Int(py=src)
        var dst_addr = Int(py=dst)
        var n = Int(py=frames)
        var stride = n * size_of[Float32]()
        var mod_addrs = InlineArray[Int, MAX_INPUTS](fill=0)
        var num_mods = Int(py=len(mods))
        if num_mods > MAX_INPUTS - 1:
            raise Error("too many modulation inputs")
        for k in range(num_mods):
            mod_addrs[k + 1] = Int(py=mods[k])
        var ports = InlineArray[Port, MAX_INPUTS](fill=Port())
        var ports_ptr = Ports(unsafe_from_address=Int(Pointer(to=ports)))
        ref units = self_ptr[].units
        # No PythonObject may be touched while the GIL is released.
        ref cpython = Python().cpython()
        var thread_state = cpython.PyEval_SaveThread()
        for c in range(len(units)):
            ports[0] = Port(src_addr + c * stride, 1)
            for k in range(1, num_mods + 1):
                var address = mod_addrs[k] + c * stride if mod_addrs[k] != 0 else 0
                ports[k] = Port(address, 1)
            units[c].process(
                ports_ptr, SamplePtr(unsafe_from_address=dst_addr + c * stride), n
            )
        cpython.PyEval_RestoreThread(thread_state)
        return PythonObject(None)


struct PyGraph(Movable, Writable):
    """One `Graph` per channel, as `Bank` holds one kernel per channel."""

    var graph: Graph
    var locked: Bool  # true while a stream renders it from the audio thread

    def __init__(out self, sample_rate: Float64, channels: Int, block: Int):
        self.graph = Graph(sample_rate, block, channels)
        self.locked = False

    def check_unlocked(self) raises:
        if self.locked:
            raise Error(
                "the graph is being rendered by a running stream; stop it first,"
                " or change parameters with Stream.set"
            )

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "Graph(nodes=",
            len(self.graph.kinds),
            ", channels=",
            self.graph.channels,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def py_init(out self: Self, args: PythonObject, kwargs: PythonObject) raises:
        if len(args) != 3:
            raise Error("expected (sample_rate, channels, block)")
        var sample_rate = Float64(py=args[0])
        var channels = Int(py=args[1])
        var block = Int(py=args[2])
        if not sample_rate > 0.0:
            raise Error("sample_rate must be positive")
        if channels < 1:
            raise Error("channels must be >= 1")
        if block < 1:
            raise Error("block must be >= 1")
        self = Self(sample_rate, channels, block)

    @staticmethod
    def add(
        self_ptr: Pointer[Self, MutAnyOrigin], kind: PythonObject
    ) raises -> PythonObject:
        self_ptr[].check_unlocked()
        var index = kind_index(String(py=kind))
        return PythonObject(self_ptr[].graph.add(index))

    @staticmethod
    def connect(
        self_ptr: Pointer[Self, MutAnyOrigin],
        src: PythonObject,
        dst: PythonObject,
        port: PythonObject,
        delay: PythonObject,
    ) raises -> PythonObject:
        self_ptr[].check_unlocked()
        var a = Int(py=src)
        var b = Int(py=dst)
        var p = Int(py=port)
        var d = Int(py=delay)
        self_ptr[].graph.connect(a, b, p, d)
        return PythonObject(None)

    @staticmethod
    def set(
        self_ptr: Pointer[Self, MutAnyOrigin],
        node: PythonObject,
        param: PythonObject,
        value: PythonObject,
    ) raises -> PythonObject:
        self_ptr[].check_unlocked()
        var n = Int(py=node)
        var p = Int(py=param)
        var v = Float64(py=value)
        self_ptr[].graph.set(n, p, v)
        return PythonObject(None)

    @staticmethod
    def remove(
        self_ptr: Pointer[Self, MutAnyOrigin], node: PythonObject
    ) raises -> PythonObject:
        self_ptr[].check_unlocked()
        self_ptr[].graph.remove(Int(py=node))
        return PythonObject(None)

    @staticmethod
    def set_output(
        self_ptr: Pointer[Self, MutAnyOrigin], node: PythonObject
    ) raises -> PythonObject:
        self_ptr[].check_unlocked()
        var n = Int(py=node)
        self_ptr[].graph.set_output(n)
        return PythonObject(None)

    @staticmethod
    def input_names(
        self_ptr: Pointer[Self, MutAnyOrigin], node: PythonObject
    ) raises -> PythonObject:
        var names = Python.list()
        for name in self_ptr[].graph.input_names(Int(py=node)):
            names.append(PythonObject(name))
        return names

    @staticmethod
    def param_names(
        self_ptr: Pointer[Self, MutAnyOrigin], node: PythonObject
    ) raises -> PythonObject:
        var names = Python.list()
        for name in self_ptr[].graph.param_names(Int(py=node)):
            names.append(PythonObject(name))
        return names

    @staticmethod
    def reset_node(
        self_ptr: Pointer[Self, MutAnyOrigin], node: PythonObject
    ) raises -> PythonObject:
        var n = Int(py=node)
        self_ptr[].graph.reset_node(n)
        return PythonObject(None)

    @staticmethod
    def reset(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        self_ptr[].graph.reset()
        return PythonObject(None)

    @staticmethod
    def render(
        self_ptr: Pointer[Self, MutAnyOrigin],
        src: PythonObject,
        dst: PythonObject,
        frames: PythonObject,
    ) raises -> PythonObject:
        """Render `frames` samples per channel; `src` is 0 when there is no input."""
        var src_addr = Int(py=src)
        var dst_addr = Int(py=dst)
        var n = Int(py=frames)
        ref cpython = Python().cpython()
        var thread_state = cpython.PyEval_SaveThread()
        var error = String("")
        try:
            self_ptr[].graph.render(
                src_addr, SamplePtr(unsafe_from_address=dst_addr), n
            )
        except e:
            error = String(e)
        cpython.PyEval_RestoreThread(thread_state)
        if error:
            raise Error(error)
        return PythonObject(None)


struct PyStream(Movable, Writable):
    """A PortAudio stream rendering a `PyGraph`.

    `mdsp.stream.Stream` keeps a reference to the graph object, so the graphs
    the callback renders outlive the stream.
    """

    var stream: Stream
    var graph: PythonObject  # keeps the graph alive while the stream runs

    def __init__(out self, var graph: PythonObject) raises:
        var graph_ptr = graph.downcast_value_ptr[PyGraph]()
        self.stream = Stream(
            Pointer[Graph, MutUntrackedOrigin](
                unsafe_from_address=Int(Pointer(to=graph_ptr[].graph))
            )
        )
        self.graph = graph^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Stream(running=", self.stream.handle != 0, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def py_init(out self: Self, args: PythonObject, kwargs: PythonObject) raises:
        if len(args) != 1:
            raise Error("expected (graph)")
        self = Self(args[0])

    @staticmethod
    def start(
        self_ptr: Pointer[Self, MutAnyOrigin],
        device: PythonObject,
        channels: PythonObject,
        sample_rate: PythonObject,
        block: PythonObject,
        input_device: PythonObject,
    ) raises -> PythonObject:
        ref graph_ptr = self_ptr[].graph.downcast_value_ptr[PyGraph]()[]
        graph_ptr.check_unlocked()
        self_ptr[].stream.open(
            Int(py=device),
            Int(py=channels),
            Float64(py=sample_rate),
            Int(py=block),
            Int(py=input_device),
        )
        graph_ptr.locked = True
        return PythonObject(None)

    @staticmethod
    def stop(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        self_ptr[].stream.close()
        self_ptr[].graph.downcast_value_ptr[PyGraph]()[].locked = False
        return PythonObject(None)

    @staticmethod
    def running(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].stream.handle != 0)

    @staticmethod
    def set(
        self_ptr: Pointer[Self, MutAnyOrigin],
        node: PythonObject,
        param: PythonObject,
        value: PythonObject,
    ) raises -> PythonObject:
        """Queue a change for the audio thread. Never blocks; may drop."""
        return PythonObject(
            self_ptr[].stream.queue[].push(
                Int32(Int(py=node)), Int32(Int(py=param)), Float32(Float64(py=value))
            )
        )

    @staticmethod
    def stats(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        ref stream = self_ptr[].stream
        return Python.dict(
            callbacks=PythonObject(Int(stream.callbacks.load())),
            underruns=PythonObject(Int(stream.underruns.load())),
            dropped=PythonObject(Int(stream.queue[].dropped[].load())),
            applied=PythonObject(Int(stream.applied.load())),
            worst_render_us=PythonObject(
                Float64(Int(stream.worst_render_ns.load())) / 1000.0
            ),
        )


def audio_devices() raises -> PythonObject:
    """Output devices PortAudio can see."""
    var out = Python.list()
    for device in portaudio_devices():
        out.append(
            Python.dict(
                index=PythonObject(device.index),
                name=PythonObject(device.name),
                max_input_channels=PythonObject(device.max_input_channels),
                max_output_channels=PythonObject(device.max_output_channels),
                default_sample_rate=PythonObject(device.default_sample_rate),
                default=PythonObject(device.is_default),
            )
        )
    return out


def node_kinds() raises -> PythonObject:
    """Names accepted by `Graph.add`."""
    var kinds = Python.list()
    for name in kind_names():
        kinds.append(PythonObject(name))
    return kinds


struct WideBank[P: WideProcessor](Movable, Writable):
    """A channel-aware kernel exposed to Python, with the same methods as `Bank`.

    One instance handles every channel, so unlike `Bank` there is nothing to
    replicate.
    """

    var unit: Self.P
    var channels: Int

    def __init__(out self, sample_rate: Float64, channels: Int):
        self.unit = Self.P(sample_rate)
        self.channels = channels

    def write_to(self, mut writer: Some[Writer]):
        writer.write("WideBank(channels=", self.channels, ")")

    @staticmethod
    def py_init(out self: Self, args: PythonObject, kwargs: PythonObject) raises:
        if len(args) != 2:
            raise Error("expected (sample_rate, channels)")
        var sample_rate = Float64(py=args[0])
        var channels = Int(py=args[1])
        if not sample_rate > 0.0:
            raise Error("sample_rate must be positive")
        if channels < 1:
            raise Error("channels must be >= 1")
        self = Self(sample_rate, channels)

    @staticmethod
    def param_names(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var names = Python.list()
        for name in Self.P.param_names():
            names.append(PythonObject(name))
        return names

    @staticmethod
    def input_names(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var names = Python.list()
        for name in Self.P.input_names():
            names.append(PythonObject(name))
        return names

    @staticmethod
    def set(
        self_ptr: Pointer[Self, MutAnyOrigin], param: PythonObject, value: PythonObject
    ) raises -> PythonObject:
        self_ptr[].unit.set(Int(py=param), Float64(py=value))
        return PythonObject(None)

    @staticmethod
    def reset(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        self_ptr[].unit.reset()
        return PythonObject(None)

    @staticmethod
    def process(
        self_ptr: Pointer[Self, MutAnyOrigin],
        src: PythonObject,
        dst: PythonObject,
        frames: PythonObject,
        mods: PythonObject,
    ) raises -> PythonObject:
        var src_addr = Int(py=src)
        var dst_addr = Int(py=dst)
        var n = Int(py=frames)
        var ports = InlineArray[Port, MAX_INPUTS](fill=Port())
        ref unit = self_ptr[].unit
        var channels = self_ptr[].channels
        ports[0] = Port(src_addr, channels)
        var num_mods = Int(py=len(mods))
        for k in range(num_mods):
            ports[k + 1] = Port(Int(py=mods[k]), channels)
        var ports_ptr = Ports(unsafe_from_address=Int(Pointer(to=ports)))
        ref cpython = Python().cpython()
        var thread_state = cpython.PyEval_SaveThread()
        unit.process(
            ports_ptr, SamplePtr(unsafe_from_address=dst_addr), n, channels
        )
        cpython.PyEval_RestoreThread(thread_state)
        return PythonObject(None)


def _add_wide_bank[P: WideProcessor](
    mut m: PythonModuleBuilder, name: StaticString
) raises:
    _ = (
        m.add_type[WideBank[P]](name)
        .def_py_init[WideBank[P].py_init]()
        .def_method[WideBank[P].param_names]("param_names")
        .def_method[WideBank[P].input_names]("input_names")
        .def_method[WideBank[P].set]("set")
        .def_method[WideBank[P].reset]("reset")
        .def_method[WideBank[P].process]("process")
    )


def _add_bank[P: Processor](mut m: PythonModuleBuilder, name: StaticString) raises:
    _ = (
        m.add_type[Bank[P]](name)
        .def_py_init[Bank[P].py_init]()
        .def_method[Bank[P].param_names]("param_names")
        .def_method[Bank[P].input_names]("input_names")
        .def_method[Bank[P].set]("set")
        .def_method[Bank[P].reset]("reset")
        .def_method[Bank[P].process]("process")
    )


def _short_type_names(module: PythonObject) raises:
    """Rebind `mdsp._core.X` attributes to `X`.

    Types are registered under dotted names so CPython sets `__module__`.
    Without it, importlib warns on import and doctest collection fails.
    """
    var builtins = Python.import_module("builtins")
    for name in builtins.list(builtins.vars(module)):
        var parts = name.rsplit(".", 1)
        if len(parts) == 2:
            builtins.setattr(module, parts[1], builtins.getattr(module, name))
            builtins.delattr(module, name)


@export
def PyInit__core() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("mdsp._core")
        _add_bank[Phasor](m, "mdsp._core.Phasor")
        _add_bank[Sine](m, "mdsp._core.Sine")
        _add_bank[Saw](m, "mdsp._core.Saw")
        _add_bank[Square](m, "mdsp._core.Square")
        _add_bank[OnePole](m, "mdsp._core.OnePole")
        _add_bank[Biquad](m, "mdsp._core.Biquad")
        _add_bank[Svf](m, "mdsp._core.Svf")
        _add_bank[Gain](m, "mdsp._core.Gain")
        _add_bank[Scale](m, "mdsp._core.Scale")
        _add_bank[Mix](m, "mdsp._core.Mix")
        _add_bank[Adsr](m, "mdsp._core.Adsr")
        _add_bank[Noise](m, "mdsp._core.Noise")
        _add_bank[Compressor](m, "mdsp._core.Compressor")
        _add_bank[Shaper](m, "mdsp._core.Shaper")
        _add_bank[Reverb](m, "mdsp._core.Reverb")
        _add_bank[Chorus](m, "mdsp._core.Chorus")
        _add_wide_bank[Pan](m, "mdsp._core.Pan")
        _add_wide_bank[Width](m, "mdsp._core.Width")
        _add_bank[Delay](m, "mdsp._core.Delay")
        m.def_function[node_kinds]("node_kinds")
        m.def_function[audio_devices]("audio_devices")
        _ = (
            m.add_type[PyStream]("mdsp._core.Stream")
            .def_py_init[PyStream.py_init]()
            .def_method[PyStream.start]("start")
            .def_method[PyStream.stop]("stop")
            .def_method[PyStream.running]("running")
            .def_method[PyStream.set]("set")
            .def_method[PyStream.stats]("stats")
        )
        _ = (
            m.add_type[PyGraph]("mdsp._core.Graph")
            .def_py_init[PyGraph.py_init]()
            .def_method[PyGraph.add]("add")
            .def_method[PyGraph.connect]("connect")
            .def_method[PyGraph.set]("set")
            .def_method[PyGraph.remove]("remove")
            .def_method[PyGraph.set_output]("set_output")
            .def_method[PyGraph.input_names]("input_names")
            .def_method[PyGraph.param_names]("param_names")
            .def_method[PyGraph.reset_node]("reset_node")
            .def_method[PyGraph.reset]("reset")
            .def_method[PyGraph.render]("render")
        )
        var module = m.finalize()
        _short_type_names(module)
        return module
    except e:
        abort(String("failed to create module mdsp._core: ", e))
