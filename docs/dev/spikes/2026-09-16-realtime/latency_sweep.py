"""Smallest block size the callback sustains, and proof it renders signal."""

import sys
import threading
import time

sys.path.insert(0, ".")
import realtime

SR, SECONDS = 48000.0, 3.0

print(f"{SECONDS:.0f} s per block size, main thread busy in Python")
for block in (16, 32, 64, 128, 256, 512):
    engine = realtime.Engine(SR, block)
    engine.start()
    deadline = time.perf_counter() + SECONDS
    spins = 0
    while time.perf_counter() < deadline:
        spins = (spins * 31 + 7) % 1000003
    stats = engine.stats()
    engine.stop()
    expected = SR * SECONDS / block
    load = stats["worst_render_us"] / (block / SR * 1e6) * 100
    print(
        f"  block {block:4d} ({block / SR * 1e6:6.0f} us deadline):"
        f" callbacks {stats['callbacks']:6d}/{expected:.0f}"
        f"  underruns {stats['underruns']:4d}"
        f"  worst render {stats['worst_render_us']:6.1f} us ({load:4.1f}% of deadline)"
        f"  peak {stats['peak']:.2f}"
    )
