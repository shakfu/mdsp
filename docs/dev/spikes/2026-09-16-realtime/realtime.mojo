"""Real-time audio from a Mojo callback that never enters Python.

PortAudio is loaded with dlopen and given an `abi("C")` Mojo function as its
stream callback. That callback drains a lock-free queue of parameter changes
and renders the graph, so no Python code and no GIL are involved while audio
runs. Python only starts and stops the stream and pushes messages.

Build:
    mojo build --emit shared-lib --fp-mode contract=off \
        -I ../../../../src/mdsp/_mojo -I . realtime.mojo -o realtime.so
"""

from std.atomic import Atomic
from std.ffi import OwnedDLHandle, c_double, c_int
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.time import perf_counter_ns

from dsp import Graph, SamplePtr, kind_index

comptime QUEUE_CAPACITY = 1024  # power of two
comptime PA_CONTINUE = 0
comptime PA_OUTPUT_UNDERFLOW = 4
comptime PA_FLOAT32 = 1


struct Queue(Movable):
    """Single-producer, single-consumer ring of parameter changes.

    Python pushes, the audio callback drains. Neither blocks nor allocates.
    """

    var nodes: List[Int32]
    var params: List[Int32]
    var values: List[Float32]
    var head: Atomic[DType.int64]  # next slot to read, written by the consumer
    var tail: Atomic[DType.int64]  # next slot to write, written by the producer
    var dropped: Atomic[DType.int64]

    def __init__(out self):
        self.nodes = List[Int32](length=QUEUE_CAPACITY, fill=0)
        self.params = List[Int32](length=QUEUE_CAPACITY, fill=0)
        self.values = List[Float32](length=QUEUE_CAPACITY, fill=0.0)
        self.head = Atomic[DType.int64](0)
        self.tail = Atomic[DType.int64](0)
        self.dropped = Atomic[DType.int64](0)

    def push(mut self, node: Int32, param: Int32, value: Float32) -> Bool:
        var tail = self.tail.load()
        if tail - self.head.load() >= Int64(QUEUE_CAPACITY):
            _ = self.dropped.fetch_add(1)
            return False
        var slot = Int(tail) & (QUEUE_CAPACITY - 1)
        self.nodes[slot] = node
        self.params[slot] = param
        self.values[slot] = value
        self.tail.store(tail + 1)  # publish after the payload is written
        return True

    def drain(mut self, mut graph: Graph) raises:
        var head = self.head.load()
        var tail = self.tail.load()
        while head < tail:
            var slot = Int(head) & (QUEUE_CAPACITY - 1)
            graph.set(Int(self.nodes[slot]), Int(self.params[slot]), Float64(self.values[slot]))
            head += 1
        self.head.store(head)


struct Engine(Movable):
    """A graph rendered by PortAudio. Its address is the callback's user data."""

    var graph: Graph
    var queue: Queue
    var library: OwnedDLHandle
    var stream: Int
    var channels: Int
    var callbacks: Atomic[DType.int64]
    var underruns: Atomic[DType.int64]
    var worst_render_ns: Atomic[DType.int64]
    var peak_milli: Atomic[DType.int64]  # loudest sample seen, x1000

    def __init__(out self, sample_rate: Float64, block: Int) raises:
        self.graph = Graph(sample_rate, block)
        self.queue = Queue()
        self.library = OwnedDLHandle("libportaudio.so.2")
        self.stream = 0
        self.channels = 1
        self.callbacks = Atomic[DType.int64](0)
        self.underruns = Atomic[DType.int64](0)
        self.worst_render_ns = Atomic[DType.int64](0)
        self.peak_milli = Atomic[DType.int64](0)
        _ = self.library.get_function[c_int]("Pa_Initialize")()

    def build_demo(mut self) raises:
        """saw -> svf (cutoff from an LFO through an exponential scale) -> gain."""
        var saw = self.graph.add(kind_index("saw"))
        var lfo = self.graph.add(kind_index("sine"))
        var scale = self.graph.add(kind_index("scale"))
        var filt = self.graph.add(kind_index("svf"))
        var out = self.graph.add(kind_index("gain"))
        self.graph.set(saw, 0, 110.0)
        self.graph.set(lfo, 0, 0.5)
        self.graph.set(scale, 0, 300.0)
        self.graph.set(scale, 1, 5000.0)
        self.graph.set(scale, 2, 1.0)  # exponential
        self.graph.set(filt, 1, 800.0)
        self.graph.set(filt, 2, 4.0)
        self.graph.set(out, 0, 0.2)
        self.graph.connect(lfo, scale, 0)
        self.graph.connect(saw, filt, 0)
        self.graph.connect(scale, filt, 1)
        self.graph.connect(filt, out, 0)
        self.graph.set_output(out)
        for node in range(5):
            self.graph.reset_node(node)

    def render(mut self, output: Int, frames: Int):
        """Called from the audio thread only."""
        var started = perf_counter_ns()
        try:
            self.queue.drain(self.graph)
            self.graph.render(0, SamplePtr(unsafe_from_address=output), frames)
        except:
            for i in range(frames):
                SamplePtr(unsafe_from_address=output)[unsafe_offset=i] = 0.0
        var peak: Float32 = 0.0
        for i in range(frames):
            peak = max(peak, abs(SamplePtr(unsafe_from_address=output)[unsafe_offset=i]))
        var peak_milli = Int64(Int(peak * 1000.0))
        if peak_milli > self.peak_milli.load():
            self.peak_milli.store(peak_milli)
        var elapsed = Int64(Int(perf_counter_ns()) - Int(started))
        if elapsed > self.worst_render_ns.load():
            self.worst_render_ns.store(elapsed)
        _ = self.callbacks.fetch_add(1)


@export
def audio_callback(
    input: Int,
    output: Int,
    frames: UInt64,
    time_info: Int,
    status_flags: UInt64,
    user_data: Int,
) abi("C") -> c_int:
    """PortAudio stream callback. No Python, no GIL, no allocation."""
    var engine = Pointer[Engine, MutAnyOrigin](unsafe_from_address=user_data)
    if status_flags & UInt64(PA_OUTPUT_UNDERFLOW) != 0:
        _ = engine[].underruns.fetch_add(1)
    engine[].render(output, Int(frames))
    return c_int(PA_CONTINUE)


struct PyEngine(Movable, Writable):
    var engine: Engine

    def __init__(out self, sample_rate: Float64, block: Int) raises:
        self.engine = Engine(sample_rate, block)
        self.engine.build_demo()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Engine(running=", self.engine.stream != 0, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def py_init(out self: Self, args: PythonObject, kwargs: PythonObject) raises:
        self = Self(Float64(py=args[0]), Int(py=args[1]))

    @staticmethod
    def start(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        ref engine = self_ptr[].engine
        if engine.stream != 0:
            raise Error("already running")
        var open_default = engine.library.get_function[c_int]("Pa_OpenDefaultStream")
        var error = open_default(
            Pointer(to=engine.stream),
            c_int(0),
            c_int(engine.channels),
            UInt64(PA_FLOAT32),
            c_double(engine.graph.sample_rate),
            UInt64(engine.graph.block),
            audio_callback,
            Int(Pointer(to=engine)),
        )
        if error != 0:
            raise Error("Pa_OpenDefaultStream failed: " + String(Int(error)))
        error = engine.library.get_function[c_int]("Pa_StartStream")(engine.stream)
        if error != 0:
            raise Error("Pa_StartStream failed: " + String(Int(error)))
        return PythonObject(None)

    @staticmethod
    def stop(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        ref engine = self_ptr[].engine
        if engine.stream != 0:
            _ = engine.library.get_function[c_int]("Pa_StopStream")(engine.stream)
            _ = engine.library.get_function[c_int]("Pa_CloseStream")(engine.stream)
            engine.stream = 0
        return PythonObject(None)

    @staticmethod
    def set(
        self_ptr: Pointer[Self, MutAnyOrigin],
        node: PythonObject,
        param: PythonObject,
        value: PythonObject,
    ) raises -> PythonObject:
        """Queue a parameter change for the audio thread; never blocks."""
        var pushed = self_ptr[].engine.queue.push(
            Int32(Int(py=node)), Int32(Int(py=param)), Float32(Float64(py=value))
        )
        return PythonObject(pushed)

    @staticmethod
    def stats(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        ref engine = self_ptr[].engine
        return Python.dict(
            callbacks=PythonObject(Int(engine.callbacks.load())),
            underruns=PythonObject(Int(engine.underruns.load())),
            worst_render_us=PythonObject(Float64(Int(engine.worst_render_ns.load())) / 1000.0),
            dropped_messages=PythonObject(Int(engine.queue.dropped.load())),
            peak=PythonObject(Float64(Int(engine.peak_milli.load())) / 1000.0),
        )


@export
def PyInit_realtime() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("realtime")
        _ = (
            m.add_type[PyEngine]("Engine")
            .def_py_init[PyEngine.py_init]()
            .def_method[PyEngine.start]("start")
            .def_method[PyEngine.stop]("stop")
            .def_method[PyEngine.set]("set")
            .def_method[PyEngine.stats]("stats")
        )
        return m.finalize()
    except e:
        abort(String("failed to create module realtime: ", e))
