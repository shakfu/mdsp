# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Mojo kernels behind a `Processor` trait (`tick` and block `process`): `Phasor`, `Sine`, `OnePole`, `Biquad`, `Gain`. Python classes wrap them with `AudioBuffer` in/out, plus `Chain`. `make build` compiles `src/mdsp/_core.so` with `mojo build`.
- Kernels run with the GIL released, so separate instances scale across threads (3.25x on 4 threads in the spike).
- Types are registered under dotted names (`mdsp._core.Phasor`) and rebound to short names. Mojo 1.0 otherwise creates types without `__module__`: importlib warns on import, which aborts the interpreter under `-W error`, and doctest collection fails.
- `Delay`: linear-interpolated delay line with feedback and dry/wet mix. The line is a `List` inside the kernel, so kernels with heap state stay `Copyable` and fit `Bank` and the contract test unchanged.
- `Saw` and `Square`: PolyBLEP band-limited oscillators, 14-30 dB less aliasing than naive waveforms. All oscillators share one generic `Osc[S: Shape]` kernel.
- Mojo code is compiled with `--fp-mode contract=off`. The default `contract=fast` fused multiply-adds differently in `Delay.tick` and `Delay.process`, so outputs differed by 1 ulp; fusion also varies with target CPU. Cost is mixed: OnePole -23%, Delay -14%, Biquad +17%.
- Design spike on composition models, `Variant` dispatch and GIL release: `docs/dev/spikes/2026-09-15-dispatch-gil/`.

### Removed

- Template `add` / `greet` functions and the no-runtime-dependencies test. numpy and mojo are now runtime dependencies.

## [0.1.0] - 2026-09-15

### Added

- Initial project structure
- Core module with example functions
- Test suite with pytest
- Build system using uv_build
