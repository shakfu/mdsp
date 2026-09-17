# Is Mojo a good language for DSP?

Date: 2026-09-17. Toolchain: Mojo 1.0.0 stable. Verdict: good for the audio-thread kernel layer of a Python DSP library, which is what mdsp is. Weak as a general DSP language today.

Evidence is from this repo unless marked otherwise. Claims about the wider Mojo ecosystem are dated and should be re-checked against current docs.

## What Mojo bought mdsp

| Win | Evidence |
|-|-|
| Real-time callback with no GIL | `render_callback` is `@export ... abi("C")`, passed straight to `Pa_OpenStream` (`src/mdsp/_mojo/stream.mojo:305`). 25.7 us worst render of a 1333 us deadline, 0 underruns, main thread busy (`spikes/2026-09-16-realtime`). The GIL-taking version missed 2999 of 3000 deadlines. |
| One language for kernel and binding | `PythonModuleBuilder` and `Int(py=...)` sit in the same file as the DSP (`src/mdsp/_mojo/_core.mojo`). No pybind, no cffi, no second build system. |
| Cheap Python boundary | 1.9 us per call, 202 M samples/s zero-copy over numpy (`spikes/2026-09-15-dispatch-gil`). |
| Compile-time generics that collapse code | `Osc[S: Shape]` gives four oscillators from one kernel (`dsp/oscillators.mojo:145-148`). `Chain[A, B]` reached 575 M samples/s. |
| Runtime graph is affordable | `Variant` dispatch costs 2-7% over static block processing, which is what made `Graph` viable. |

## What it cost

1. **Undefined struct layout broke threading.** `ParamQueue`'s `head`, `tail` and `dropped` clobbered each other across threads as plain fields. All three moved into separate `OwnedPointer` allocations; padding could not fix it because Mojo does not promise a field layout (`stream.mojo:61-63`). The same fix was needed again for the queue inside `Stream` (`stream.mojo:126-129`). A language with no layout guarantee and no false-sharing control is a hard fit for lock-free audio.

2. **C FFI is hand-rolled.** `PaDeviceInfo` is read at literal byte offsets: `info + 20`, `+ 24`, `+ 40`, `+ 64` (`stream.mojo:209`, `:233`, `:359-370`). Null pointers pass as `Int` because `Pointer` is non-nullable. There is no header import. Every audio API worth targeting is a C or C++ ABI.

3. **The FP default cost correctness, then speed.** `contract=fast` fused differently in `Delay.tick` and `Delay.process`, giving 1-ulp divergence that also varied by target CPU. `--fp-mode contract=off` (`scripts/hatch_build.py:32-34`) costs OnePole -23% and Delay -14%.

4. **Runtime shipping.** Three Mojo runtime libraries are bundled per wheel and keep their SONAMEs; two Mojo extensions built against different toolchain versions in one process share whichever loads first (`TODO.md`). Free-threaded CPython segfaulted on import, so `_base.py` raises `ImportError` instead.

5. **No ecosystem.** `src/mdsp/io.py` is a hand-written WAV parser. `thirdparty/dsplib-mojo` offers `compute_dft_raw`, an O(n^2) DFT, not an FFT. No resampler, no filter design, no VST3 or CLAP path: Mojo had no C++ interop as of 2026-05.

## The finding that cuts against the usual pitch

`grep -r SIMD src/mdsp/_mojo/` returns nothing. Mojo's headline feature is unused across the whole DSP layer.

That follows from the workload. The dispatch spike measured per-sample interleaved `tick` at 181 M samples/s against 66 M for block `process`, because chained recursive filters are loop-carried dependencies that block mode serialises. Freeverb (`dsp/reverb.mojo`), `Svf`, `Biquad` and `Delay` are all scalar state machines. Audio DSP at block 64 is a latency problem, not a throughput problem.

The case for Mojo here is not SIMD and autotuning. It is a systems language whose Python interop is native. Those are different claims with different competitors.

## Alternative framings

**Rust plus PyO3.** It gives the same GIL release (`Python::allow_threads`), `#[repr(C)]` with defined layout, `std::sync::atomic` with a specified memory model, static linking, mature PortAudio and cpal bindings, and `bindgen` instead of `info + 40`. Costs 1, 2 and 4 above largely vanish. It does not give the kernel and its Python binding in one file, or `mojo build` as the entire toolchain. Inference, not measured: the 1.9 us call overhead and 202 M samples/s are not Mojo-specific, and PyO3 over the same numpy buffer should land in the same range.

**Where Mojo would dominate, and mdsp is not yet.** Partitioned convolution reverb, FFT vocoders, oversampled waveshaping, spectral processing, offline batch rendering on GPU. Those are SIMD-wide and throughput-bound. Freeverb will never pay for the language choice; a convolution reverb might.

## Open questions

- Price a Rust plus PyO3 spike against the existing `Graph` benchmark.
- Write a SIMD partitioned-convolution kernel to test whether Mojo's differentiator fires on this build.
