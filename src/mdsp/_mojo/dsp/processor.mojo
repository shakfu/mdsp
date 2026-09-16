"""The contract every DSP kernel implements."""

comptime SamplePtr = Pointer[Float32, MutAnyOrigin]

comptime MAX_INPUTS = 4


@fieldwise_init
struct Port(Copyable, Movable, Writable):
    """One input of a node: where its samples are, and how many channels.

    Every kernel here is mono and ignores `channels`; it is carried so that
    channel-aware nodes can be added without changing these kernels.
    """

    var address: Int  # 0 means unconnected
    var channels: Int

    def __init__(out self):
        self.address = 0
        self.channels = 1


#: One entry per name in `input_names()`.
comptime Ports = Pointer[Port, MutAnyOrigin]


@always_inline
def input_address(ins: Ports, index: Int) -> Int:
    return ins[unsafe_offset=index].address


@always_inline
def input_channels(ins: Ports, index: Int) -> Int:
    return ins[unsafe_offset=index].channels


@always_inline
def audio_input(ins: Ports, dst: SamplePtr) -> SamplePtr:
    """Input port 0, or *dst* when unconnected: generators ignore it."""
    var address = input_address(ins, 0)
    return SamplePtr(unsafe_from_address=address) if address != 0 else dst


trait Processor(Copyable, Deinitable):
    """A mono unit: one audio input, optional modulation inputs, one output.

    Contract:
    - With only port 0 connected, `process(ins, dst, n)` produces exactly the
      output of `n` calls to `tick`.
    - `src == dst` (in-place) is allowed. Partial overlap is not.
    - A connected modulation input replaces the parameter of the same name for
      every sample; the parameter's own value and smoothing are left untouched.
    - `set` ramps over 10 ms; `reset` clears state and ends ramps.
    - `set` with an unknown parameter index is a no-op. Out-of-range values are
      clamped, not rejected: kernels cannot raise.
    """

    def __init__(out self, sample_rate: Float64): ...

    @staticmethod
    def param_names() -> List[String]:
        """Parameter names, indexed by the `param` argument of `set`."""
        ...

    @staticmethod
    def input_names() -> List[String]:
        """Input port names. Port 0 is audio; the rest modulate the parameter
        of the same name."""
        ...

    def set(mut self, param: Int, value: Float64): ...

    def reset(mut self):
        """Clear state and end parameter ramps. Parameter values are kept."""
        ...

    def tick(mut self, x: Float32) -> Float32: ...

    def process(mut self, ins: Ports, dst: SamplePtr, n: Int): ...


trait WideProcessor(Copyable, Deinitable):
    """A unit that sees every channel at once, such as a panner.

    `process` receives planar buffers: channel `c` of a port starts at
    `c * frames` samples into it. Mono kernels stay on `Processor`; these exist
    for the few units that must mix channels together.
    """

    def __init__(out self, sample_rate: Float64): ...

    @staticmethod
    def param_names() -> List[String]: ...

    @staticmethod
    def input_names() -> List[String]: ...

    def set(mut self, param: Int, value: Float64): ...

    def reset(mut self): ...

    def process(mut self, ins: Ports, dst: SamplePtr, frames: Int, channels: Int): ...
