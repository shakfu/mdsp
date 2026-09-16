# TODO

## Critical

## High

## Medium

- [ ] Primitives still missing: pink noise, EQ shelves, chorus/flanger, and a soft-knee option for `Compressor` (the knee is hard).

- [ ] `Stream` gaps: output only (no device input), one producer thread for the queue (a second needs a lock), Linux-only soname (`libportaudio.so.2`; macOS needs `libportaudio.2.dylib`), and no CI coverage since runners have no audio device.

- [ ] Graph gaps: no cross-channel nodes (the port channel count is carried but unused), no feedback, and no way to remove a node.

- [ ] Audio I/O gaps: no FLAC or other formats (point users at soundfile), no RF64/W64, and files are read whole rather than streamed.

- [ ] `Delay` smooths its delay time; `feedback` and `mix` still step. Kernels are capped at `MAX_INPUTS` (4) ports.

- [ ] Bundled runtime libraries keep their SONAMEs. Another extension built with a different Mojo version, loaded in the same process, would share whichever copy loads first.

- [ ] Denormals: recursive filters decaying towards zero may hit subnormal slowdowns on x86. Measure before adding a fix.

- [ ] Explicit `fma` in kernels where it measured faster (OnePole, Delay), now that contraction is off.

- [ ] Graph feedback, and multichannel nodes.

## Low

- [ ] `repr` of `_core` objects prints every kernel field for every channel.
