"""Real-time output through PortAudio, rendered without touching Python.

PortAudio is loaded with `dlopen` at `open`, so it stays an optional
dependency. The stream callback is an `abi("C")` Mojo function: it drains a
lock-free queue of parameter changes and renders each channel's graph straight
into the device buffer, so the audio thread never takes the GIL or allocates.

Measurements behind this design: docs/dev/spikes/2026-09-16-realtime.
"""

from std.atomic import Atomic
from std.ffi import OwnedDLHandle, c_double, c_int
from std.sys import CompilationTarget
from std.memory import OwnedPointer
from std.time import perf_counter_ns

from dsp import Graph, SamplePtr

comptime QUEUE_CAPACITY = 1024  # power of two
comptime PA_CONTINUE = 0
comptime PA_FLOAT32 = 1
comptime PA_OUTPUT_UNDERFLOW = 4
comptime PA_NO_DEVICE = -1


def _portaudio_names() -> List[String]:
    """Where PortAudio might be, most standard first.

    Homebrew on Apple silicon installs under /opt/homebrew, which dyld does
    not search by default, so the bare name alone is not enough there.
    """
    if CompilationTarget.is_macos():
        return [
            "libportaudio.2.dylib",
            "/opt/homebrew/lib/libportaudio.2.dylib",
            "/usr/local/lib/libportaudio.2.dylib",
        ]
    return ["libportaudio.so.2", "/usr/local/lib/libportaudio.so.2"]


def _open_portaudio() raises -> OwnedDLHandle:
    var names = _portaudio_names()
    for i in range(len(names)):
        try:
            return OwnedDLHandle(names[i])
        except:
            continue
    raise Error(
        "could not load PortAudio; install it (apt install libportaudio2, or"
        " brew install portaudio)"
    )


struct ParamQueue(Movable):
    """Single-producer, single-consumer ring of parameter changes.

    One Python thread pushes; the audio thread drains. Neither blocks nor
    allocates. A full queue drops the message and counts it.
    """

    # head, tail and dropped live in separate allocations. As fields of one
    # struct their writes clobbered each other across threads, losing messages;
    # Mojo does not promise a field layout, so padding could not fix it.
    var nodes: List[Int32]
    var params: List[Int32]
    var values: List[Float32]
    var head: OwnedPointer[Atomic[DType.int64]]  # advanced by the consumer
    var tail: OwnedPointer[Atomic[DType.int64]]  # advanced by the producer
    var dropped: OwnedPointer[Atomic[DType.int64]]

    def __init__(out self):
        self.nodes = List[Int32](length=QUEUE_CAPACITY, fill=0)
        self.params = List[Int32](length=QUEUE_CAPACITY, fill=0)
        self.values = List[Float32](length=QUEUE_CAPACITY, fill=0.0)
        self.head = OwnedPointer(Atomic[DType.int64](0))
        self.tail = OwnedPointer(Atomic[DType.int64](0))
        self.dropped = OwnedPointer(Atomic[DType.int64](0))

    def push(mut self, node: Int32, param: Int32, value: Float32) -> Bool:
        var tail = self.tail[].load()
        if tail - self.head[].load() >= Int64(QUEUE_CAPACITY):
            _ = self.dropped[].fetch_add(1)
            return False
        var slot = Int(tail) & (QUEUE_CAPACITY - 1)
        self.nodes[slot] = node
        self.params[slot] = param
        self.values[slot] = value
        self.tail[].store(tail + 1)  # publish only after the payload is written
        return True

    def drain(mut self, mut graph: Graph) raises -> Int:
        var start = self.head[].load()
        var head = start
        var tail = self.tail[].load()
        while head < tail:
            var slot = Int(head) & (QUEUE_CAPACITY - 1)
            var node = Int(self.nodes[slot])
            var param = Int(self.params[slot])
            var value = Float64(self.values[slot])
            graph.set(node, param, value)
            head += 1
        self.head[].store(head)
        return Int(head - start)


@fieldwise_init
struct StreamParameters(Copyable, Movable):
    """PortAudio's `PaStreamParameters`."""

    var device: Int32
    var channel_count: Int32
    var sample_format: UInt64
    var suggested_latency: Float64
    var host_api_specific: Int


struct Stream(Movable):
    """Renders a graph into an audio device.

    The callback reads this struct through PortAudio's user data, so it must
    not move once the stream is open. `mdsp.stream` keeps it inside a Python
    object that outlives the stream.
    """

    var graph: Pointer[Graph, MutUntrackedOrigin]
    # The queue sits in its own allocation. Embedded in this struct, the
    # callback's writes to `Stream` also wrote back stale queue fields, losing
    # messages the producer had queued meanwhile.
    var queue: OwnedPointer[ParamQueue]
    var library: OwnedDLHandle
    var handle: Int
    var scratch: List[Float32]  # one block per channel, planar
    var capture: List[Float32]  # planar input for the graph's Input nodes
    var input_channels: Int
    var callbacks: Atomic[DType.int64]
    var underruns: Atomic[DType.int64]
    var worst_render_ns: Atomic[DType.int64]
    var applied: Atomic[DType.int64]  # messages the audio thread has applied

    def __init__(out self, graph: Pointer[Graph, MutUntrackedOrigin]) raises:
        self.graph = graph
        self.queue = OwnedPointer(ParamQueue())
        self.library = _open_portaudio()
        self.handle = 0
        self.scratch = List[Float32]()
        self.capture = List[Float32]()
        self.input_channels = 0
        self.callbacks = Atomic[DType.int64](0)
        self.underruns = Atomic[DType.int64](0)
        self.worst_render_ns = Atomic[DType.int64](0)
        self.applied = Atomic[DType.int64](0)
        var error = self.library.get_function[c_int]("Pa_Initialize")()
        if error != 0:
            raise Error("Pa_Initialize failed: " + self.error_text(error))

    def error_text(self, code: c_int) -> String:
        var text = String("")
        try:
            var address = self.library.get_function[Int]("Pa_GetErrorText")(code)
            var chars = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=address)
            var i = 0
            while chars[unsafe_offset=i] != 0 and i < 256:
                text += String(chr(Int(chars[unsafe_offset=i])))
                i += 1
        except:
            text = String("error ", Int(code))
        return text

    def default_output_device(self) raises -> Int:
        return Int(self.library.get_function[c_int]("Pa_GetDefaultOutputDevice")())

    def device_count(self) raises -> Int:
        return Int(self.library.get_function[c_int]("Pa_GetDeviceCount")())

    def device_info(self, index: Int) raises -> Int:
        """Address of PortAudio's `PaDeviceInfo`, or 0 when there is none."""
        return self.library.get_function[Int]("Pa_GetDeviceInfo")(c_int(index))

    def open(
        mut self,
        device: Int,
        channels: Int,
        sample_rate: Float64,
        block: Int,
        input_device: Int,
    ) raises:
        """Open an output stream, and an input one when `input_device` is set.

        `input_device` of -2 means no input; -1 means the default device.
        """
        if self.handle != 0:
            raise Error("the stream is already open")
        self.scratch = List[Float32](length=channels * block, fill=0.0)
        self.input_channels = 0
        var input_parameters = StreamParameters(0, 0, 0, 0.0, 0)
        var input_pointer = Int(0)
        if input_device != -2:
            var input_index = (
                Int(self.library.get_function[c_int]("Pa_GetDefaultInputDevice")())
                if input_device < 0
                else input_device
            )
            if input_index < 0:
                raise Error("no default input device")
            var input_info = self.device_info(input_index)
            if input_info == 0:
                raise Error("no device with index " + String(input_index))
            var available = Int(
                Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=input_info + 20)[]
            )
            if available < 1:
                raise Error("device " + String(input_index) + " has no input channels")
            self.input_channels = min(channels, available)
            self.capture = List[Float32](
                length=self.input_channels * block, fill=0.0
            )
            input_parameters = StreamParameters(
                Int32(input_index),
                Int32(self.input_channels),
                UInt64(PA_FLOAT32),
                Pointer[Float64, ImmutAnyOrigin](unsafe_from_address=input_info + 32)[],
                0,
            )
            input_pointer = Int(Pointer(to=input_parameters))
        var index = self.default_output_device() if device < 0 else device
        if index < 0:
            raise Error("no default output device")
        var info = self.device_info(index)
        if info == 0:
            raise Error("no device with index " + String(index))
        # PaDeviceInfo: int, const char*, int hostApi, int maxIn, int maxOut,
        # then four PaTime doubles from offset 32; low output latency is at 40.
        var latency = Pointer[Float64, ImmutAnyOrigin](unsafe_from_address=info + 40)[]
        var parameters = StreamParameters(
            Int32(index),
            Int32(channels),
            UInt64(PA_FLOAT32),
            latency,
            0,
        )
        var error = self.library.get_function[c_int]("Pa_OpenStream")(
            Pointer(to=self.handle),
            input_pointer,
            Pointer(to=parameters),
            c_double(sample_rate),
            UInt64(block),
            UInt64(0),  # paNoFlag
            render_callback,
            Int(Pointer(to=self)),
        )
        if error != 0:
            self.handle = 0
            raise Error("Pa_OpenStream failed: " + self.error_text(error))
        error = self.library.get_function[c_int]("Pa_StartStream")(self.handle)
        if error != 0:
            _ = self.library.get_function[c_int]("Pa_CloseStream")(self.handle)
            self.handle = 0
            raise Error("Pa_StartStream failed: " + self.error_text(error))

    def close(mut self) raises:
        if self.handle != 0:
            var handle = self.handle
            self.handle = 0  # the callback may still be draining
            _ = self.library.get_function[c_int]("Pa_StopStream")(handle)
            _ = self.library.get_function[c_int]("Pa_CloseStream")(handle)

    def render(mut self, captured_input: Int, output: Int, frames: Int):
        """Audio thread only: apply queued changes, then fill the device buffer.

        The graph renders its channels planar into `scratch`; PortAudio wants
        them interleaved.
        """
        var started = Int(perf_counter_ns())
        ref graph = self.graph[]
        var channels = graph.channels
        var buffer = SamplePtr(unsafe_from_address=output)
        var planar = SamplePtr(unsafe_from_address=Int(self.scratch.unsafe_ptr()))
        var source = 0
        if self.input_channels > 0 and captured_input != 0:
            # De-interleave what the device captured; the graph wants planar.
            var captured = SamplePtr(unsafe_from_address=Int(self.capture.unsafe_ptr()))
            for c in range(self.input_channels):
                for i in range(frames):
                    captured[unsafe_offset=c * frames + i] = SamplePtr(
                        unsafe_from_address=captured_input
                    )[unsafe_offset=i * self.input_channels + c]
            source = Int(captured)
        try:
            _ = self.applied.fetch_add(Int64(self.queue[].drain(graph)))
            graph.render(source, planar, frames)
            for c in range(channels):
                for i in range(frames):
                    buffer[unsafe_offset=i * channels + c] = planar[
                        unsafe_offset=c * frames + i
                    ]
        except:
            for i in range(frames * channels):
                buffer[unsafe_offset=i] = 0.0
        var elapsed = Int64(Int(perf_counter_ns()) - started)
        if elapsed > self.worst_render_ns.load():
            self.worst_render_ns.store(elapsed)
        _ = self.callbacks.fetch_add(1)


@export
def render_callback(
    input: Int,
    output: Int,
    frames: UInt64,
    time_info: Int,
    status_flags: UInt64,
    user_data: Int,
) abi("C") -> c_int:
    """PortAudio stream callback: no Python, no GIL, no allocation."""
    var stream = Pointer[Stream, MutAnyOrigin](unsafe_from_address=user_data)
    if status_flags & UInt64(PA_OUTPUT_UNDERFLOW) != 0:
        _ = stream[].underruns.fetch_add(1)
    stream[].render(input, output, Int(frames))
    return c_int(PA_CONTINUE)


@fieldwise_init
struct DeviceInfo(Copyable, Movable):
    var index: Int
    var name: String
    var max_input_channels: Int
    var max_output_channels: Int
    var default_sample_rate: Float64
    var is_default: Bool


def _c_string(address: Int) -> String:
    var chars = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=address)
    var text = String("")
    var i = 0
    while chars[unsafe_offset=i] != 0 and i < 256:
        text += String(chr(Int(chars[unsafe_offset=i])))
        i += 1
    return text


def audio_devices() raises -> List[DeviceInfo]:
    """Every device PortAudio can see. Opens and closes PortAudio itself."""
    var library = _open_portaudio()
    var error = library.get_function[c_int]("Pa_Initialize")()
    if error != 0:
        raise Error("Pa_Initialize failed")
    var default = Int(library.get_function[c_int]("Pa_GetDefaultOutputDevice")())
    var default_input = Int(library.get_function[c_int]("Pa_GetDefaultInputDevice")())
    var count = Int(library.get_function[c_int]("Pa_GetDeviceCount")())
    var devices = List[DeviceInfo]()
    for index in range(count):
        # PaDeviceInfo: name at 8, maxInputChannels at 20, maxOutputChannels at
        # 24, defaultSampleRate at 64.
        var info = library.get_function[Int]("Pa_GetDeviceInfo")(c_int(index))
        if info == 0:
            continue
        var max_input = Int(
            Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=info + 20)[]
        )
        var max_output = Int(
            Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=info + 24)[]
        )
        devices.append(
            DeviceInfo(
                index,
                _c_string(Pointer[Int, ImmutAnyOrigin](unsafe_from_address=info + 8)[]),
                max_input,
                max_output,
                Pointer[Float64, ImmutAnyOrigin](unsafe_from_address=info + 64)[],
                index == default if max_output > 0 else index == default_input,
            )
        )
    _ = library.get_function[c_int]("Pa_Terminate")()
    return devices^
