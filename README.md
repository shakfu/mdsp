# mdsp

Python audio DSP primitives with kernels written in [Mojo](https://mojolang.org).

All processing uses planar `[channels, frames]` float32 buffers. Each primitive processes a whole buffer in one Mojo call, with the GIL released. Python chains primitives.

```python
import mdsp

buf = mdsp.read_wav("drums.wav")
chain = mdsp.Chain(
    mdsp.Svf("lowpass", cutoff=800.0, q=4.0, channels=buf.channels, sample_rate=buf.sample_rate),
    mdsp.Gain(0.5, channels=buf.channels, sample_rate=buf.sample_rate),
)
mdsp.write_wav("out.wav", chain.process(buf), fmt="int24")
```

## Primitives

| Kind | Classes |
|-|-|
| Generators | `Phasor`, `Sine`, `Saw`, `Square` (PolyBLEP band-limited), `Noise` (white or pink) |
| Envelopes | `Adsr` |
| Filters | `OnePole`, `Svf`, `Biquad` (lowpass, highpass, bandpass, notch, low/high shelf, peaking) |
| Delays | `Delay` (fractional, modulated, feedback, mix), `Chorus` (flanging and chorus) |
| Dynamics | `Compressor` (sidechain, soft knee), `Limiter` |
| Effects | `Shaper` (tanh, soft, hard), `Reverb` (Freeverb-style) |
| Stereo | `Pan` (constant power), `Width` |
| Ops | `Gain` |
| Ops | `Scale` (map [-1, 1] onto a parameter range), `Mix` (sum up to four inputs) |
| Composition | `Chain`, `Graph` |
| Files | `read_wav`, `write_wav` |
| Real time | `Stream`, `output_devices` |

`read_wav` and `write_wav` handle PCM 8/16/24/32 and float 32/64 WAV files with no dependencies. For other formats, read with [soundfile](https://pypi.org/project/soundfile/) and wrap the samples in an `AudioBuffer`.

`Svf` is a topology-preserving state-variable filter; modulate it rather than `Biquad`, which overshoots when its coefficients change fast.

## Modulation

A unit's `inputs` lists the parameters that accept a buffer instead of a fixed value. Pass one by name; it must match the audio in sample rate, channels and frames.

```python
lfo = mdsp.Sine(0.5).generate(48000)
cutoff = mdsp.AudioBuffer(2000 + 1500 * lfo.data, 48000)
swept = mdsp.Svf("lowpass", q=4.0).process(tone, cutoff=cutoff)
```

Parameter changes ramp over 10 ms, so they do not click. `reset()` applies the target at once.

## Graphs

A `Graph` wires units together and renders them entirely in Mojo, one call per block instead of one per unit. Nodes are opaque handles. A plain connection must run forward, from a node added earlier to one added later; a connection with a `delay` may run any direction, including a node to itself, which is how feedback is built.

```python
g = mdsp.Graph(sample_rate=48000.0, channels=2)
src   = g.add(mdsp.Input)                                   # the buffer passed to process()
lfo   = g.add(mdsp.Sine, freq=0.5)
sweep = g.add(mdsp.Scale, lo=300.0, hi=4000.0, curve="exponential")
filt  = g.add(mdsp.Svf, mode="lowpass", cutoff=800.0, q=4.0)
g.connect(lfo, sweep)
g.connect(src, filt)
g.connect(sweep, filt, "cutoff")
g.output = filt

out = g.process(buf)          # or g.generate(frames) with no Input node
g.set(filt, "q", 8.0)         # parameters change through the graph
```

`process`, `generate` and `Chain.process` take `out=` to write into a buffer you already own, which avoids allocating per call. With it, a 64-frame block through a 4-node graph costs 2.2 us against a 1333 us real-time deadline.

Against the same three units in a `Chain`, a graph is 3.2x faster on 64-frame blocks and produces identical samples.

```python
echo = g.add(mdsp.Mix, gain=1.0, gain2=0.6)
g.connect(tap, echo, "in2", delay=4800)   # feedback, 100 ms around the loop
```

A feedback delay counts samples, so a patch sounds the same whatever block size renders it. A loop shorter than `block` makes the graph render in smaller chunks, which costs speed but keeps the loop exact.

`remove(node)` empties a node's slot and drops its connections. Other handles keep working, because nothing is renumbered.

Most units are mono and run once per channel with their own state. `Pan` and `Width` see every channel at once, which is what panning and stereo width need.

Units are stateful: state carries across `process` / `generate` calls until `reset()`. An instance is not thread-safe; separate instances run in parallel threads.

Importing `mdsp` starts the Mojo runtime's idle worker pool: one thread per CPU in the process affinity mask. Start Python under `taskset` to limit it.

## Examples

`examples/offline_process.py` renders a file through a swept filter and delay, synthesising its own input if given none. `examples/realtime_synth.py` plays a two-oscillator voice and changes notes while it runs.

```bash
uv run python examples/offline_process.py out.wav
uv run python examples/realtime_synth.py 6
```

## Real-time output

`Stream` plays a `Graph` through an audio device. The audio thread runs a Mojo callback that never enters Python, so a busy interpreter cannot make it miss a deadline: measured 0 underruns down to 16-frame blocks (0.33 ms) while Python ran flat out.

```python
with mdsp.Stream(g) as stream:       # needs PortAudio; input_device=True to capture
    stream.set(filt, "cutoff", 1500.0)   # queued for the audio thread
    time.sleep(1.0)
print(stream.stats)  # callbacks, underruns, applied, dropped, worst_render_us
```

While a stream runs its graph is locked: build it or change parameters through `Stream.set`, which never blocks and reports a full queue rather than waiting. `output_devices()` and `input_devices()` list the devices to pick from. Passing `input_device` makes the graph's `Input` nodes read live audio. PortAudio is loaded only when a stream is created, under the Linux or macOS name for the library; the rest of mdsp works without it.

## Install

```bash
pip install mdsp
```

Wheels: Linux x86_64 and aarch64 (glibc 2.35+), macOS 13+ arm64, CPython 3.10-3.14. Free-threaded builds are not supported. Wheels bundle the Mojo runtime libraries; the Mojo compiler is not needed. `Stream` additionally needs PortAudio (`apt install libportaudio2`, `brew install portaudio`).

## Build

Requires `uv`, `make`, and a C linker: `gcc` on Linux, the Xcode command line tools on macOS. The Mojo compiler is installed into `.venv` as a dev dependency. Targets follow Mojo's own: Linux x86_64 and aarch64 with glibc 2.35+, and macOS 13+ on Apple silicon. There is no Intel macOS or native Windows build, because Modular ships none.

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
| `scripts/hatch_build.py` | Wheel build hook: compiles and bundles the extension |
| `docs/dev/` | Design spikes and decision records |

## Direction

The Python block chain is the first layer. The planned next layer is a graph engine in Mojo that runs a whole graph per call, reusing the same kernels. See `docs/dev/spikes/2026-09-15-dispatch-gil/`.
