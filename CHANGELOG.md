# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Mojo 1.1.0. `InlineArray[T, N]` is now `Array[T, N]`, and `Atomic` takes a type rather than a DType (`Atomic[DType.int64]` becomes `Atomic[Int64]`); both old spellings are parse errors. The `mojo-compiler` floor in `[build-system]` moved with it, having been left at `<1.1` where a source build would have compiled 1.1 sources with the 1.0 compiler.

### Fixed

- `hatchling>=1.32.3` in both `[build-system] requires` and the dev group. `uv sync` resolves the build backend separately from `uv.lock`, so the isolated build env took 1.32.3 while the lock held 1.32.0. 1.32.3 gave `BuildHookInterface` a second type parameter, and no single subscript in `MojoBuildHook` satisfies both arities. A shared floor over an unsubscripted base: the latter compiles on either version, but `disallow_any_generics` rejects it under strict mypy.

## [0.1.0]

### Added

- Mojo kernels behind a `Processor` trait (`tick` and block `process`): `Phasor`, `Sine`, `OnePole`, `Biquad`, `Gain`. Python classes wrap them with `AudioBuffer` in/out, plus `Chain`. `make build` compiles `src/mdsp/_core.so` with `mojo build`.

- Kernels run with the GIL released, so separate instances scale across threads (3.25x on 4 threads in the spike).

- Types are registered under dotted names (`mdsp._core.Phasor`) and rebound to short names. Mojo 1.0 otherwise creates types without `__module__`: importlib warns on import, which aborts the interpreter under `-W error`, and doctest collection fails.

- `Delay`: linear-interpolated delay line with feedback and dry/wet mix. The line is a `List` inside the kernel, so kernels with heap state stay `Copyable` and fit `Bank` and the contract test unchanged.

- `Saw` and `Square`: PolyBLEP band-limited oscillators, 14-30 dB less aliasing than naive waveforms. All oscillators share one generic `Osc[S: Shape]` kernel.

- Mojo code is compiled with `--fp-mode contract=off`. The default `contract=fast` fused multiply-adds differently in `Delay.tick` and `Delay.process`, so outputs differed by 1 ulp; fusion also varies with target CPU. Cost is mixed: OnePole -23%, Delay -14%, Biquad +17%.

- Binary wheels: hatchling replaces uv_build, which supports only pure Python and had packed the in-tree `_core.so` into a `py3-none-any` wheel pointing at the build venv. `scripts/hatch_build.py` compiles the extension, bundles the three Mojo runtime libraries into `mdsp/_libs/`, and tags one `py3-none` wheel per platform; `make wheel` repairs it to `manylinux_2_35` (the runtime libraries need glibc 2.35) or macOS. The Mojo compiler is no longer a runtime dependency. Details: `docs/dev/packaging.md`.

- macOS wheels target macOS 13 (`MACOSX_DEPLOYMENT_TARGET` overrides). `mojo build` targets the host OS, so a wheel built locally on macOS 26 was tagged `macosx_26_0`.

- Importing under free-threaded CPython raises `ImportError`; the extension segfaulted on import.

- Design spike on composition models, `Variant` dispatch and GIL release: `docs/dev/spikes/2026-09-15-dispatch-gil/`.

- Kernels take input ports instead of one buffer: port 0 is audio, the rest modulate the parameter of the same name. `Processor.process(buf, **mods)` and `Generator.generate(frames, **mods)` accept them; `inputs` lists what a unit takes. Available: `freq` on every oscillator, `gain`, `cutoff` on `OnePole` and `Svf`.

- `Svf`: topology-preserving state-variable filter. Under cutoff jumps of 200 Hz to 12 kHz every 32 samples it peaks at 5.8 where `Biquad` reaches 180, and it is faster when modulated. `Biquad` keeps no modulation input for that reason.

- Parameter changes ramp over 10 ms per sample, so output no longer depends on how callers split blocks, and `Delay` time changes glide instead of jumping. `reset()` ends the ramp. Constructor values apply at once.

- `read_wav` and `write_wav`: PCM 8/16/24/32 and IEEE float 32/64, including WAVE_FORMAT_EXTENSIBLE, with no dependencies. Reading validates each chunk header against the bytes present, so a file claiming a 4 GB data chunk fails instead of asking for the memory. Writing defaults to float32; integer formats clip to [-1, 1). A 3-minute stereo file writes in 130 ms and reads in 51 ms as int16.

- `Graph`: build a patch at runtime and render it in Mojo, one call per block rather than one per unit. Nodes are added by class (`g.add(mdsp.Svf, cutoff=800)`) or name, connected by port name, and changed through `g.set(node, ...)`; handles stay opaque so an audio thread can later apply changes from a queue. Output matches the equivalent `Chain` sample for sample, and is 2.9x faster on 64-frame blocks. One graph runs per channel.

- `Scale`: maps [-1, 1] onto a parameter range, linearly or exponentially, which is what turns an oscillator into a modulation source.

- Kernel input ports now carry a channel count. Every kernel is mono and ignores it; it is there so channel-aware nodes can arrive without touching existing kernels.

- `out=` on `Processor.process`, `Generator.generate`, `Chain.process` and both `Graph` render methods writes into a buffer the caller owns. A chain alternates between `out` and one reused scratch buffer, so a steady stream of same-sized blocks allocates nothing after the first call.

- `AudioBuffer.data` now returns a view and `AudioBuffer.address` caches the storage address. numpy refuses to reallocate a view, so the address cannot go stale; fetching it per call cost 1.1 us against 0.1 us cached. A 64-frame block through a 4-node graph fell from 7.4 us to 2.2 us, of which 1.1 us is the Mojo call.

- `Stream`: plays a `Graph` through an audio device. The PortAudio callback is a Mojo `abi("C")` function that drains a lock-free queue and renders without entering Python, so no underruns down to 16-frame blocks while the interpreter is busy; a callback that took the GIL missed 2999 of 3000 deadlines. While a stream runs its graph is locked, and `Stream.set` queues changes for the audio thread. Parameters that would allocate, such as `max_delay`, are refused. PortAudio is loaded only when a stream is created.

- The queue's `head`, `tail` and `dropped` sit in separate allocations. As fields of one struct their cross-thread writes clobbered each other: 194 of 2884 parameter changes vanished silently, and Mojo does not promise a field layout, so padding could not fix it. A test now asserts every message is either applied or counted as dropped.

- `Mix` sums up to four inputs with per-input gains, so a graph can combine voices or blend dry and wet. Every node had one audio input before, which limited graphs to a single chain.

- `Adsr` envelope, gated by a parameter or by another node at audio rate. Segment lengths are counted in samples rather than accumulated, so a segment ends exactly where and when it should; adding a step repeatedly drifted (a 10-sample attack reached 0.9999999999999999).

- `Noise`: white noise from an xorshift generator, seeded and repeatable.

- `examples/`: an offline file renderer and a live synth, both covered by tests.

- `Compressor`: feed-forward peak compressor in decibels with attack and release on the gain reduction, plus a `sidechain` input for ducking. `Limiter` is the same kernel with a high ratio and fast attack.

- `Shaper`: tanh, cubic soft-clip and hard-clip waveshaping. The output is normalised so a full-scale input stays full-scale, which keeps `drive` from doubling as a volume control.

- `Reverb`: eight damped comb filters into four allpasses, tuned after Jezar's public-domain Freeverb. Delay lengths scale with the sample rate, so the tail lasts the same time at any rate.

- Feedback in `Graph`: `connect(src, dst, port, delay=samples)` may run in any direction, including a node to itself. The delay counts samples rather than blocks, so a patch renders identically at any block size, verified from block 1 to 4800; it must cover one block, because every node writes once per block.
- Channel-aware nodes: `Pan` (constant power) and `Width` (side-signal scaling). The engine now holds one graph whose node buffers carry every channel, rather than an independent graph per channel. Mono kernels still get one instance per channel, so their state stays independent; channel-aware kernels get one instance that writes every channel. The port channel count added earlier for this is now used.

- `Graph.remove(node)` empties a node's slot and drops every connection touching it. Handles are not renumbered, so the ones callers hold keep working; a removed node renders silence and is refused by `set`, `connect` and `output`.
- Feedback delays may now be shorter than a block. The graph renders in chunks no longer than its shortest loop, which costs speed but keeps the loop exact; a 16-sample loop under a 64-sample block gives the expected decay.
- `Stream` captures audio: `input_device=True` (or an index from the new `input_devices()`) feeds the graph's `Input` nodes from a device. PortAudio is loaded as `libportaudio.2.dylib` on macOS and `libportaudio.so.2` elsewhere.
- `Compressor` takes a `knee` in dB, easing compression in quadratically around the threshold instead of switching at it.
- `Biquad` adds `lowshelf`, `highshelf` and `peaking` modes with a `gain` in dB, the RBJ cookbook shapes for EQ.
- `Noise` takes `color="pink"`, falling about 3 dB per octave (Paul Kellet's economy filter), scaled to stay inside full scale.
- `Chorus`: a delay swept by its own LFO, for chorus and flanging. `Delay` also gained a `delay` modulation input, so a sweep can come from any node.

- PortAudio is looked for under Homebrew's prefix as well as the linker's own search path, since dyld does not search `/opt/homebrew/lib` by default on Apple silicon.
- CI: QA runs on macOS as well as Linux, so `make build` and the Mojo kernel tests are exercised there; PortAudio is installed on every runner, and the tests that need a device now decide by opening a stream rather than by trusting the device list, which a headless runner can report wrongly.

- Initial project structure

- Core module with example functions

- Test suite with pytest

- Build system using uv_build

### Removed

- Template `add` / `greet` functions and the no-runtime-dependencies test. numpy is now a runtime dependency.

### Added
