"""Render a file offline: read a WAV, run it through a graph, write the result.

    uv run python examples/offline_process.py [input.wav] output.wav

Given one path it synthesises its own input, so it runs anywhere.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import mdsp


def build_effect(sample_rate: float, channels: int) -> mdsp.Graph:
    """Input -> lowpass swept by an LFO -> delay -> mixed back with the dry signal."""
    g = mdsp.Graph(sample_rate, channels=channels, block=512)
    source = g.add(mdsp.Input)
    lfo = g.add(mdsp.Sine, freq=0.25)
    sweep = g.add(mdsp.Scale, lo=400.0, hi=5000.0, curve="exponential")
    filtered = g.add(mdsp.Svf, mode="lowpass", q=3.0)
    echo = g.add(mdsp.Delay, max_delay=0.5, delay=0.25, feedback=0.35, mix=1.0)
    out = g.add(mdsp.Mix, gain=0.8, gain2=0.4)

    g.connect(lfo, sweep)
    g.connect(source, filtered)
    g.connect(sweep, filtered, "cutoff")
    g.connect(filtered, echo)
    g.connect(filtered, out)  # dry
    g.connect(echo, out, "in2")  # wet
    g.output = out
    return g


def demo_input(sample_rate: float) -> mdsp.AudioBuffer:
    """Four seconds of plucked notes, so the example needs no input file."""
    g = mdsp.Graph(sample_rate, block=512)
    tone = g.add(mdsp.Saw, freq=110.0)
    env = g.add(mdsp.Adsr, attack=0.005, decay=0.25, sustain=0.0, release=0.1)
    amp = g.add(mdsp.Gain, gain=1.0)
    g.connect(tone, amp)
    g.connect(env, amp, "gain")
    g.output = amp

    notes = [110.0, 138.6, 164.8, 220.0]
    parts = []
    for index in range(8):
        g.set(tone, "freq", notes[index % len(notes)])
        g.set(env, "gate", 1.0)
        parts.append(g.generate(int(sample_rate * 0.4)).data)
        g.set(env, "gate", 0.0)
        parts.append(g.generate(int(sample_rate * 0.1)).data)
    return mdsp.AudioBuffer(np.concatenate(parts, axis=1), sample_rate, copy=False)


def main(argv: list[str]) -> int:
    if len(argv) == 3:
        source, target = Path(argv[1]), Path(argv[2])
        buf = mdsp.read_wav(source)
        print(f"read {source}: {buf.channels} ch, {buf.duration:.2f} s")
    elif len(argv) == 2:
        target = Path(argv[1])
        buf = demo_input(48000.0)
        print(f"synthesised {buf.duration:.2f} s of input")
    else:
        print(__doc__)
        return 2

    graph = build_effect(buf.sample_rate, buf.channels)
    started = time.perf_counter()
    out = graph.process(buf)
    elapsed = time.perf_counter() - started

    mdsp.write_wav(target, out, fmt="int24")
    print(
        f"wrote {target}: {out.duration:.2f} s in {elapsed * 1e3:.0f} ms "
        f"({out.duration / elapsed:.0f}x real time)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
