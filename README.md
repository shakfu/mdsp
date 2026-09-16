# mdsp

Python audio DSP primitives with kernels written in [Mojo](https://mojolang.org).

All processing uses planar `[channels, frames]` float32 buffers. Each primitive processes a whole buffer in one Mojo call, with the GIL released. Python chains primitives.

```python
import mdsp

tone = mdsp.Phasor(freq=110.0).generate(48000)
chain = mdsp.Chain(mdsp.Biquad("lowpass", cutoff=800.0, q=4.0), mdsp.Gain(0.5))
out = chain.process(tone)  # AudioBuffer, 1 channel, 48000 frames
```

## Primitives

| Kind | Classes |
|-|-|
| Generators | `Phasor`, `Sine`, `Saw`, `Square` (PolyBLEP band-limited) |
| Filters | `OnePole`, `Biquad`, `Svf` (lowpass, highpass, bandpass, notch) |
| Delays | `Delay` (fractional, feedback, mix) |
| Ops | `Gain` |
| Composition | `Chain` |

`Svf` is a topology-preserving state-variable filter; modulate it rather than `Biquad`, which overshoots when its coefficients change fast.

## Modulation

A unit's `inputs` lists the parameters that accept a buffer instead of a fixed value. Pass one by name; it must match the audio in sample rate, channels and frames.

```python
lfo = mdsp.Sine(0.5).generate(48000)
cutoff = mdsp.AudioBuffer(2000 + 1500 * lfo.data, 48000)
swept = mdsp.Svf("lowpass", q=4.0).process(tone, cutoff=cutoff)
```

Parameter changes ramp over 10 ms, so they do not click. `reset()` applies the target at once.

Units are stateful: state carries across `process` / `generate` calls until `reset()`. An instance is not thread-safe; separate instances run in parallel threads.

Importing `mdsp` starts the Mojo runtime's idle worker pool: one thread per CPU in the process affinity mask. Start Python under `taskset` to limit it.

## Install

```bash
pip install mdsp
```

Wheels: Linux x86_64 and aarch64 (glibc 2.35+), macOS 13+ arm64, CPython 3.10-3.14. Free-threaded builds are not supported. Wheels bundle the Mojo runtime libraries; the Mojo compiler is not needed.

## Build

Requires `uv`, `make`, and a C linker (`gcc`). The Mojo compiler is installed into `.venv` as a dev dependency.

```bash
make build   # uv sync, then compile src/mdsp/_core.so in place
make test    # Python tests, doctests, Mojo kernel tests
make qa      # lint, format check, mypy, tests
make wheel   # platform wheel with bundled runtime, repaired into dist/
```

## Layout

| Path | Contents |
|-|-|
| `src/mdsp/_mojo/dsp/` | Mojo kernels implementing the `Processor` trait |
| `src/mdsp/_mojo/_core.mojo` | Python bindings: `Bank[P]`, one kernel per channel |
| `src/mdsp/_base.py` | Python base classes; validates buffers before they reach Mojo |
| `tests/mojo/` | Kernel contract tests, run by pytest via `mojo run` |
| `hatch_build.py` | Wheel build hook: compiles and bundles the extension |
| `docs/dev/` | Design spikes and decision records |

## Direction

The Python block chain is the first layer. The planned next layer is a graph engine in Mojo that runs a whole graph per call, reusing the same kernels. See `docs/dev/spikes/2026-09-15-dispatch-gil/`.
