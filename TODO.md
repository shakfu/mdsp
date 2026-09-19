# TODO

## Critical

## High

## Medium

- [ ] `Stream` gaps: one producer thread for the queue (a second needs a lock), and no CI coverage of playback since runners have no audio device.

- [ ] Graph gaps: `Pan`, `Width`, `Chorus` and `Player` are the only channel-aware nodes.

- [ ] `Shift` splices at fixed window boundaries, which costs sidebands at the grain rate. The phase vocoder that `timestretch` now uses would remove them, but it needs the real-time framing layer below first.

- [ ] Real-time STFT framing: a ring buffer, analysis window, hop and overlap-add that a `Processor` can drive, so spectral units are not offline-only. Two problems it has to answer. Latency: a 1024 window is 21 ms, and nothing reports it, so a parallel path in a `Graph` drifts (the same gap `Compressor` lookahead opened). Cost spikes: the graph renders 64-frame blocks while a transform lands every hop, so `worst_render_us`, measured at 44 us, would spike every fourth block unless the work is spread across the hop.

- [ ] `dsp/fft.mojo` is scalar radix-2 and 6x slower than FFTW at 1024 points (6.05 us against 0.96 us). Radix-4 and a SIMD butterfly are where that goes; `ComplexSIMD[DType.float32, 4]` compiles and is the obvious unit. Partitioned convolution is the consumer that would need it, and it is the SIMD measurement `docs/dev/mojo-for-dsp.md` asks for.

- [ ] The vocoder's first and last window are approximate: edge frames reach past the source and read mirrored samples, which leaves the first and last 1024-sample block up to 6 dB low; reading zeros there put the tail 14 dB low at factor 4. Padding the source and trimming the output would fix it properly.

- [ ] `Compressor` lookahead delays its output and nothing reports the latency, so a parallel path in a `Graph` drifts out of alignment by the lookahead. Either report latency per node and compensate, or leave it to callers.

- [ ] Audio I/O gaps: no FLAC or other formats (point users at soundfile), no RF64/W64, and files are read whole rather than streamed.

- [ ] `Delay` smooths its delay time; `feedback` and `mix` still step. Kernels are capped at `MAX_INPUTS` (4) ports.

- [ ] Bundled runtime libraries keep their SONAMEs. Another extension built with a different Mojo version, loaded in the same process, would share whichever copy loads first.

- [ ] Denormals: recursive filters decaying towards zero may hit subnormal slowdowns on x86. Measure before adding a fix.

- [ ] Explicit `fma` in kernels where it measured faster (OnePole, Delay), now that contraction is off.

- [ ] Open question from `docs/dev/mojo-for-dsp.md`: price a Rust plus PyO3 spike against the `Graph` benchmark. The other one there, whether SIMD is Mojo's differentiator for this work, is now the FFT item above: no kernel uses SIMD today.

## Low

- [ ] `repr` of `_core` objects prints every kernel field for every channel.

- [ ] Explore making numpy optional. Kernels would accept any contiguous float32 buffer (PEP 3118), with numpy needed only for the `NDArray` API and `io`. Gain: a smaller install for users without numpy. Costs: two input paths, and a buffer numpy doesn't own can reallocate under a cached address (`AudioBuffer.address` relies on numpy refusing to resize viewed storage).
