# Spike: kernel interface for modulation, multi-input nodes and a graph engine

Date: 2026-09-16. Toolchain: Mojo 1.0.0, `--fp-mode contract=off`, AMD Ryzen 9 PRO 6950H, host CPU target.

Context: mdsp starts with offline/batch processing (option A, Python chains of kernels). This spike decides the kernel interface before more kernels are written, and checks whether real-time synthesis (option C, graph engine) is within reach.

## Status

Adopted in `src/mdsp/_mojo/dsp/`: per-sample smoothing (`smooth.mojo`), input ports on every kernel, and `Svf`. Kernels now take ports directly, so `engine.mojo`'s `Mono[P]` adapter and the benchmarks here no longer compile against the current sources; they are kept as the record of the measurements below.

## Files

| File | Experiment |
|-|-|
| `modulation.mojo` | E1 smoothing vs block splitting; E2 modulation interface cost; E3 filters under hard modulation |
| `engine.mojo` | E4 port-based `Node` trait, `Mono[P]` adapter for existing kernels, `Variant` graph engine |
| `graph_bench.mojo` | E4 benchmark: engine vs the same nodes called directly |
| `graph_ext.mojo`, `realtime_proxy.py` | E5 per-call latency from Python; E6 callback thread under GIL contention |

Build commands are in each file's docstring.

## Results

E1, smoothing a 0 -> 1 gain step; max difference between one 4096-sample block and irregular blocks:

| Smoother | Difference | Settled cost |
|-|-|-|
| Ramp across the processed block | 0.98 | - |
| Per-sample one-pole | 0.0 | 2005 M samples/s |
| Per-sample fixed-length linear ramp | 0.0 | 2524 M samples/s |

E2, SVF lowpass cutoff, M samples/s:

| Interface | Constant cutoff | Audio-rate cutoff |
|-|-|-|
| A: parameter is always a buffer | 101 | 119 |
| B: constant, plus optional modulation buffer | 230 | 119 |
| RBJ biquad, per-sample coefficients | - | 75 |

E3, cutoff jumping 200 Hz <-> 12 kHz every 32 samples, Q = 8, noise input in [-1, 1]: SVF peak 5.8 (RMS 1.4); transposed direct form II biquad peak 180 (RMS 24). SVF: A. Simper, [Linear Trapezoidal Integrated SVF](https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf).

E4, patch `saw + saw -> mix -> SVF (cutoff <- sine LFO -> exp scale) -> gain`, 7 nodes, 20 s:

| Block | Direct calls | Variant engine | Engine per block |
|-|-|-|-|
| 64 | 39.7 M samples/s | 38.4 M samples/s | 1.67 us |
| 512 | 37.8 | 40.3 | 12.7 us |
| 4096 | 39.0 | 40.7 | 101 us |

Engine and direct calls are bit-identical; the difference is within run-to-run noise. Existing kernels (`Saw`, `Sine`, `Gain`) ran unchanged through `Mono[P]`.

E5, 75,000 calls of 64 frames from a Python loop, GC enabled (deadline 1333 us):

| Path | p50 | p99.9 | max | Missed |
|-|-|-|-|-|
| C: one graph call | 1.96 us | 5.8 us | 23 us | 0 |
| A: Python chain of 5 units, no modulation | 22.7 us | 50 us | 2610 us | 1 |

C rendered 60 s offline in one call in 75 ms (798x real time).

E6, callback thread paced at 64 frames, rendering the graph:

| Switch interval | Main thread | Missed |
|-|-|-|
| 5 ms (default) | idle | 11 / 3000 |
| 5 ms | busy in Python | 2999 / 3000 |
| 1 ms | busy | 1737 / 3000 |
| 0.2 ms | busy | 0 / 3000 |

The idle misses come from `time.sleep` wake-up jitter in the pacing thread.

## Conclusions

1. Smoothing must advance per sample. Use a fixed-length linear ramp: exact endpoint, cheapest once settled, block-size independent.
2. Modulatable parameters: interface B. A parameter is a smoothed constant plus an optional modulation input; kernels select the constant loop once per block. Interface A costs 2.3x on unmodulated parameters.
3. Add a TPT SVF as the modulatable filter. Keep the RBJ biquad for static filtering; it overshoots 30x under fast modulation.
4. Node interface: `process(inputs, dst, frames)` with per-type `num_inputs`. The current `Processor` kernels need no rewrite; `Mono[P]` adapts them. `Variant` dispatch adds no measurable cost.
5. The graph engine also serves offline work: one call renders a file, and it avoids A's per-stage Python overhead (22.7 us vs 1.96 us per 64-frame block).
6. Real time: compute headroom is about 700x for this patch. The obstacle is the GIL. A callback that enters Python misses almost every deadline while other Python code runs. A reliable path needs an audio callback that never takes the GIL: a Mojo `abi("C")` callback driven by a C audio library, with Python sending parameter changes through a lock-free queue.

## Not tested

- C callbacks from Mojo into an audio API (PortAudio, miniaudio), and loading such a library from Mojo 1.0.
- Feedback in the graph (needs a delay of at least one block, or per-sample scheduling).
- Multichannel nodes; the engine is mono.
- Real audio-device timing; E6 simulates the callback with a sleeping Python thread.
