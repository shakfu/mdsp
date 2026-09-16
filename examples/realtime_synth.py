"""Play a synth voice live, changing it while it runs.

    uv run python examples/realtime_synth.py [seconds]

Needs PortAudio and an output device. The audio thread runs Mojo only, so the
note pattern below is driven from Python without risking dropouts.
"""

from __future__ import annotations

import math
import sys
import time

import mdsp

SR, BLOCK = 48000.0, 64
NOTES = [220.0, 261.6, 329.6, 392.0, 329.6, 261.6]


def build_voice() -> tuple[mdsp.Graph, int, int, int]:
    """saw + detuned saw -> filter (cutoff from an LFO) -> envelope -> output."""
    g = mdsp.Graph(SR, block=BLOCK)
    a = g.add(mdsp.Saw, freq=NOTES[0])
    b = g.add(mdsp.Saw, freq=NOTES[0] * 1.005)  # detuned
    mix = g.add(mdsp.Mix, gain=0.5, gain2=0.5)
    lfo = g.add(mdsp.Sine, freq=0.3)
    sweep = g.add(mdsp.Scale, lo=500.0, hi=4000.0, curve="exponential")
    filt = g.add(mdsp.Svf, mode="lowpass", q=6.0)
    env = g.add(mdsp.Adsr, attack=0.01, decay=0.2, sustain=0.4, release=0.15)
    out = g.add(mdsp.Gain, gain=0.2)

    g.connect(a, mix)
    g.connect(b, mix, "in2")
    g.connect(lfo, sweep)
    g.connect(mix, filt)
    g.connect(sweep, filt, "cutoff")
    g.connect(filt, out)
    g.connect(env, out, "gain")
    g.output = out
    return g, a, b, env


def main(argv: list[str]) -> int:
    seconds = float(argv[1]) if len(argv) > 1 else 6.0
    devices = mdsp.output_devices()
    default = next((d for d in devices if d["default"]), devices[0])
    print(f"playing on {default['name']!r} for {seconds:.0f} s")

    graph, a, b, env = build_voice()
    with mdsp.Stream(graph) as stream:
        deadline = time.perf_counter() + seconds
        step = 0
        while time.perf_counter() < deadline:
            note = NOTES[step % len(NOTES)]
            stream.set(a, "freq", note)
            stream.set(b, "freq", note * 1.005)
            stream.set(env, "gate", 1.0)  # note on
            time.sleep(0.35)
            stream.set(env, "gate", 0.0)  # note off
            time.sleep(0.15)
            step += 1

            if step % 4 == 0:  # widen the filter every few notes
                stream.set(graph.output, "gain", 0.2 * (1 + 0.5 * math.sin(step)))

        stats = stream.stats
    print(
        f"callbacks {stats['callbacks']}, underruns {stats['underruns']}, "
        f"changes applied {stats['applied']}, dropped {stats['dropped']}, "
        f"worst render {stats['worst_render_us']:.1f} us of "
        f"{BLOCK / SR * 1e6:.0f} us available"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
