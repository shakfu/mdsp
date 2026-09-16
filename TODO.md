# TODO

## Critical

## High

- Licence: review `LicenseRef-MAX-Platform-Software-License` redistribution terms for the bundled Mojo runtime libraries before publishing wheels.
- Run `wheels.yml`: Linux aarch64 and macOS arm64 wheels are untested (target CPU names, macOS rpath rewrite and signing, delocate tag).

## Medium

- `Delay` smooths its delay time; `feedback` and `mix` still step. Kernels are capped at `MAX_INPUTS` (4) ports.
- Bundled runtime libraries keep their SONAMEs. Another extension built with a different Mojo version, loaded in the same process, would share whichever copy loads first.
- Denormals: recursive filters decaying towards zero may hit subnormal slowdowns on x86. Measure before adding a fix.
- `process(buf, out=...)` to avoid one allocation per call for small blocks.
- Address lookup (`ndarray.ctypes.data`) costs ~1.1 us per array, more than the 0.4 us Mojo call. Consider the buffer protocol.
- Explicit `fma` in kernels where it measured faster (OnePole, Delay), now that contraction is off.
- Graph engine in `_core` with a Python API, usable offline first.
- Real-time spike: Mojo `abi("C")` audio callback via a C audio library, no GIL; parameter changes over a lock-free queue.
- Graph feedback, and multichannel nodes.

## Low

- `repr` of `_core` objects prints every kernel field for every channel.
