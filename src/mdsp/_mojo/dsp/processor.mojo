"""The contract every DSP kernel implements."""

comptime SamplePtr = Pointer[Float32, MutAnyOrigin]

# Input buffer addresses, one per entry of `input_names()`; 0 means unconnected.
comptime Ports = Pointer[Int, MutAnyOrigin]
comptime MAX_INPUTS = 4


@always_inline
def input_address(ins: Ports, index: Int) -> Int:
    return ins[unsafe_offset=index]


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
