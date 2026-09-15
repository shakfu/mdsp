"""mdsp: audio DSP primitives with kernels written in Mojo.

>>> import mdsp
>>> tone = mdsp.Sine(freq=220.0).generate(48000)
>>> out = mdsp.Chain(mdsp.Biquad("lowpass", cutoff=1000.0), mdsp.Gain(0.5)).process(tone)
>>> out.channels, out.frames
(1, 48000)
"""

from mdsp._base import Chain, Generator, Processor
from mdsp.buffer import AudioBuffer
from mdsp.delay import Delay
from mdsp.filters import Biquad, OnePole
from mdsp.ops import Gain
from mdsp.oscillators import Phasor, Saw, Sine, Square

__all__ = [
    "AudioBuffer",
    "Biquad",
    "Chain",
    "Delay",
    "Gain",
    "Generator",
    "OnePole",
    "Phasor",
    "Processor",
    "Saw",
    "Sine",
    "Square",
]
__version__ = "0.1.0"
