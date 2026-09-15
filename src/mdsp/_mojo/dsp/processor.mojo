"""The contract every DSP kernel implements."""

comptime SamplePtr = Pointer[Float32, MutAnyOrigin]


trait Processor(Copyable, Deinitable):
    """A mono, single-input, single-output unit.

    Contract:
    - `process(src, dst, n)` produces exactly the output of `n` calls to `tick`.
    - `src == dst` (in-place) is allowed. Partial overlap is not.
    - Generators ignore their input.
    - `set` with an unknown parameter index is a no-op. Out-of-range values
      are clamped, not rejected: kernels cannot raise.
    """

    def __init__(out self, sample_rate: Float64): ...

    @staticmethod
    def param_names() -> List[String]:
        """Parameter names, indexed by the `param` argument of `set`."""
        ...

    def set(mut self, param: Int, value: Float64): ...

    def reset(mut self):
        """Clear state. Parameters are kept."""
        ...

    def tick(mut self, x: Float32) -> Float32: ...

    def process(mut self, src: SamplePtr, dst: SamplePtr, n: Int): ...
