# TODO

## Critical

## High

- Wheels:
  - `_core.so` links `libKGENCompilerRTShared.so`, `libMSupportGlobals.so`, `libAsyncRTRuntimeGlobals.so` (~2 MB) from `mojo-compiler` via an absolute RUNPATH into the build venv. Bundle them with an `$ORIGIN` RUNPATH, or depend on `mojo` at runtime (665 MB).
  - The runtime libraries are under `LicenseRef-MAX-Platform-Software-License`; check redistribution terms before bundling.
  - `mojo build` targets the host CPU by default; a distributable build needs an explicit `--target-cpu`.
  - `uv build` does not run `mojo build`.
- Verify Python 3.10 and 3.13 in CI; only 3.12 has been run locally.

## Medium

- Denormals: recursive filters decaying towards zero may hit subnormal slowdowns on x86. Measure before adding a fix.
- `process(buf, out=...)` to avoid one allocation per call for small blocks.
- Address lookup (`ndarray.ctypes.data`) costs ~1.1 us per array, more than the 0.4 us Mojo call. Consider the buffer protocol.
- `Delay` time changes jump; add parameter smoothing.
- Explicit `fma` in kernels where it measured faster (OnePole, Delay), now that contraction is off.
- Graph engine in Mojo (`Variant` over kernels). Open question: per-sample vs per-block scheduling; see `docs/dev/spikes/2026-09-15-dispatch-gil/`.

## Low

- `repr` of `_core` objects prints every kernel field for every channel.
