"""E5: per-call latency of rendering 64-frame blocks from a Python loop.

No audio device: this measures what an audio callback would spend in Python and
Mojo, against the 1333 us deadline of 64 frames at 48 kHz. GC stays enabled and
the loop allocates Python objects between calls to provoke collections.

Run from this directory after building graph_ext.so:
    uv run --project ../../../.. python realtime_proxy.py
"""

import gc
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import graph_ext  # noqa: E402

import mdsp  # noqa: E402

SR, BLOCK, CALLS = 48000.0, 64, 75_000  # 100 s of audio
DEADLINE_US = BLOCK / SR * 1e6


def demo_graph():
    g = graph_ext.Graph(SR, BLOCK)
    s1, s2, mix, lfo, scale, svf, out = (
        g.add(k) for k in ("saw", "saw", "mix2", "sine", "expscale", "svf", "gain")
    )
    for node, param, value in [
        (s1, 0, 110.0), (s2, 0, 110.7), (lfo, 0, 0.5), (scale, 0, 200.0),
        (scale, 1, 4000.0), (svf, 1, 4.0), (out, 0, 0.5),
    ]:
        g.set(node, param, value)
    for src, dst, port in [(s1, mix, 0), (s2, mix, 1), (lfo, scale, 0), (mix, svf, 0), (scale, svf, 1), (svf, out, 0)]:
        g.connect(src, dst, port)
    return g


def report(name, samples_ns):
    us = np.array(samples_ns) / 1e3
    p = np.percentile(us, [50, 99, 99.9])
    print(
        f"  {name:34s} p50 {p[0]:6.2f}  p99 {p[1]:6.2f}  p99.9 {p[2]:6.2f}  max {us.max():7.2f} us"
        f"  | over deadline: {(us > DEADLINE_US).sum()}"
    )


def time_calls(fn):
    samples, churn = [], []
    for i in range(CALLS):
        t = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t)
        churn.append([i] * 8)  # garbage to trigger GC
        if len(churn) > 1000:
            churn.clear()
    return samples


def main():
    print(f"{CALLS} calls x {BLOCK} frames; deadline {DEADLINE_US:.0f} us; gc enabled: {gc.isenabled()}")

    g = demo_graph()
    out = np.zeros(BLOCK, np.float32)
    address = out.ctypes.data  # cached: the buffer is reused
    report("C: one graph call, reused buffer", time_calls(lambda: g.process(address, BLOCK)))

    # A: the same patch without cutoff modulation, as a Python chain of mdsp units.
    saw1, saw2 = mdsp.Saw(110.0), mdsp.Saw(110.7)
    chain = mdsp.Chain(mdsp.Gain(0.5), mdsp.Biquad("lowpass", 1000.0, 4.0), mdsp.Gain(0.5))

    def block_a():
        a = saw1.generate(BLOCK)
        b = saw2.generate(BLOCK)
        chain.process(mdsp.AudioBuffer(a.data + b.data, SR, copy=False))

    report("A: Python chain of 5 units", time_calls(block_a))

    # Whole-file offline render through the same graph, one call.
    total = int(SR * 60)
    big = np.zeros(total, np.float32)
    t = time.perf_counter_ns()
    demo_graph().process(big.ctypes.data, total)
    dt = (time.perf_counter_ns() - t) / 1e9
    print(f"  C offline: 60 s rendered in one call: {dt * 1e3:.1f} ms ({60 / dt:.0f}x real time)")


def callback_under_gil_contention(switch_interval_s, busy_main):
    """E6: a paced audio-callback thread while the main thread runs Python code.

    Lateness = callback end - its deadline. A real PortAudio callback also has to
    take the GIL, so it waits the same way.
    """
    import threading

    sys.setswitchinterval(switch_interval_s)
    g = demo_graph()
    out = np.zeros(BLOCK, np.float32)
    address = out.ctypes.data
    period_ns = int(BLOCK / SR * 1e9)
    n_callbacks = 3000  # 4 s
    late_us = []
    stop = threading.Event()

    def audio_thread():
        start = time.perf_counter_ns()
        for k in range(n_callbacks):
            deadline = start + (k + 1) * period_ns
            wait = deadline - period_ns - time.perf_counter_ns()
            if wait > 0:
                time.sleep(wait / 1e9)
            g.process(address, BLOCK)
            late_us.append((time.perf_counter_ns() - deadline) / 1e3)
        stop.set()

    t = threading.Thread(target=audio_thread)
    t.start()
    x = 0
    while busy_main and not stop.is_set():
        x = (x * 31 + 7) % 1000003  # pure-Python CPU work holding the GIL
    t.join()
    late = np.array(late_us)
    print(
        f"  switch {switch_interval_s * 1e3:4.1f} ms, main {'busy' if busy_main else 'idle'}:"
        f" missed {(late > 0).sum():4d}/{n_callbacks}  worst lateness {late.max():8.1f} us"
    )


if __name__ == "__main__":
    main()
    print("E6 paced callback thread (graph, 64 frames) vs main-thread Python work:")
    for interval, busy in [(0.005, False), (0.005, True), (0.001, True), (0.0002, True)]:
        callback_under_gil_contention(interval, busy)
