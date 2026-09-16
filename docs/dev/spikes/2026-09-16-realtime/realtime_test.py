"""Does a Mojo audio callback survive a busy Python main thread?

Compare with docs/dev/spikes/2026-09-16-interface, where a callback that had to
take the GIL missed 2999 of 3000 deadlines under the same conditions.

Run: uv run --project ../../../.. python realtime_test.py
"""

import sys
import threading
import time

sys.path.insert(0, ".")
import realtime  # noqa: E402

SR, BLOCK, SECONDS = 48000.0, 64, 5.0
EXPECTED = SR * SECONDS / BLOCK


def run(label, busy, messages_per_second=0):
    engine = realtime.Engine(SR, BLOCK)
    engine.start()
    stop = threading.Event()

    def hammer():
        # Node 3 is the filter; parameter 1 is its cutoff.
        period = 1.0 / messages_per_second
        while not stop.is_set():
            engine.set(3, 1, 400.0 + 3000.0 * (time.perf_counter() % 1.0))
            time.sleep(period)

    sender = None
    if messages_per_second:
        sender = threading.Thread(target=hammer)
        sender.start()

    deadline = time.perf_counter() + SECONDS
    spins = 0
    while time.perf_counter() < deadline:
        if busy:
            spins = (spins * 31 + 7) % 1000003  # pure Python, holds the GIL
        else:
            time.sleep(0.05)

    stop.set()
    if sender:
        sender.join()
    stats = engine.stats()
    engine.stop()
    print(
        f"  {label:38s} callbacks {stats['callbacks']:6d}/{EXPECTED:.0f}"
        f"  underruns {stats['underruns']:4d}"
        f"  worst render {stats['worst_render_us']:6.1f} us"
        f"  dropped {stats['dropped_messages']}"
    )


if __name__ == "__main__":
    print(f"{SECONDS:.0f} s per run, {BLOCK} frames at {SR:.0f} Hz "
          f"(deadline {BLOCK / SR * 1e6:.0f} us per callback)")
    run("main thread idle", busy=False)
    run("main thread busy in Python", busy=True)
    run("busy + 1000 parameter changes/s", busy=True, messages_per_second=1000)
