# Spike: real-time audio from a Mojo callback

Date: 2026-09-16. Toolchain: Mojo 1.0.0, PortAudio 19.7 (`libportaudio.so.2`), Linux x86_64, default ALSA device.

Question: can mdsp render audio in real time without the GIL? The interface spike showed a callback that has to take the GIL misses almost every deadline while other Python code runs. The answer here is yes, and the margin is large.

## Files

| File | What it does |
|-|-|
| `realtime.mojo` | PortAudio loaded with `dlopen`; an `abi("C")` Mojo callback drains a lock-free queue and renders a `Graph` |
| `realtime_test.py` | Underruns with the main thread idle, busy in Python, and busy while sending 1000 parameter changes a second |
| `latency_sweep.py` | Block sizes from 16 to 512 frames |

Build and run commands are in each file's docstring.

## What the toolchain turned out to support

| Question | Answer |
|-|-|
| Load a C library | `OwnedDLHandle("libportaudio.so.2")`, then `get_function[ReturnType]("name")`. `external_call` needs the symbol at link time, so it does not suit an optional dependency. |
| Pass a Mojo function to C as a callback | Yes. `@export def f(...) abi("C")` passed straight as an argument; verified first with `qsort`. |
| Atomics | `std.atomic.Atomic[DType.int64]` with `load`, `store`, `fetch_add`. |
| Null pointers to C | `Pointer` is non-nullable; pass addresses as `Int`. |

## Results

Patch: `saw 110 Hz -> SVF lowpass (cutoff from a 0.5 Hz LFO through an exponential scale) -> gain`. 5 seconds per run, 64-frame blocks, 1333 us deadline.

| Main thread | Callbacks | Underruns | Worst render | Dropped messages |
|-|-|-|-|-|
| idle | 3752 / 3750 | 0 | 43.4 us | 0 |
| busy in Python | 3752 / 3750 | 0 | 28.6 us | 0 |
| busy, 1000 parameter changes/s | 3752 / 3750 | 0 | 25.7 us | 0 |

Block size sweep, 3 seconds each, main thread busy in Python:

| Block | Deadline | Underruns | Worst render | Load |
|-|-|-|-|-|
| 16 | 333 us | 0 | 19.9 us | 6.0% |
| 32 | 667 us | 0 | 21.8 us | 3.3% |
| 64 | 1333 us | 0 | 25.2 us | 1.9% |
| 128 | 2667 us | 0 | 28.7 us | 1.1% |
| 256 | 5333 us | 0 | 35.6 us | 0.7% |
| 512 | 10667 us | 0 | 45.3 us | 0.4% |

A tracked output peak of 0.46 confirms the callback renders signal rather than silence. Compare the interface spike: a callback that took the GIL missed 2999 of 3000 deadlines under the same busy main thread.

## Design that produced it

- **The audio thread never touches Python.** The callback reads the engine through its `user_data` address, drains the queue, and calls `Graph.render`. No `PythonObject`, no GIL, no allocation.
- **Parameter changes cross on a lock-free queue.** A single-producer single-consumer ring of 1024 `(node, param, value)` messages. The producer publishes by storing `tail` after writing the payload; the consumer advances `head` after applying. A full queue drops and counts rather than blocking.
- **PortAudio stays optional.** It is loaded by soname at start, so mdsp keeps working without it.

## What a production version must add

- **Lifetime.** The callback holds the engine's address. If the Python object is collected while the stream runs, that address dangles. Stop the stream from the type's destructor, and keep a reference while running.
- **Reject allocating parameters from the audio thread.** `Delay`'s `max_delay` reallocates its line. Applied from the callback it would allocate, which a real-time thread must not do. Either reject it while running or preallocate.
- **Graph edits while running.** `add` and `connect` are not safe against a running callback. They need a prepare-then-swap, or must be refused while the stream is open.
- **One producer only.** The queue assumes a single Python thread pushes. Several would need a lock on the Python side or a multi-producer design.
- **Scope of what was tested.** Mono, output only, default device, Linux/ALSA. No input, device selection, sample-rate negotiation, or macOS (`libportaudio.2.dylib`), and memory-ordering guarantees of Mojo's atomics were not examined beyond the defaults.
