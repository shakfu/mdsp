"""mdsp: audio DSP primitives with kernels written in Mojo.

>>> import mdsp
>>> tone = mdsp.Sine(freq=220.0).generate(48000)
>>> out = mdsp.Chain(mdsp.Biquad("lowpass", cutoff=1000.0), mdsp.Gain(0.5)).process(tone)
>>> out.channels, out.frames
(1, 48000)
"""

from mdsp._base import Chain, Generator, Processor
from mdsp.buffer import AudioBuffer
from mdsp.chorus import Chorus
from mdsp.delay import Delay
from mdsp.dynamics import Compressor, Limiter
from mdsp.envelope import Adsr
from mdsp.filters import Biquad, OnePole, Svf
from mdsp.graph import Graph, Input
from mdsp.io import read_wav, write_wav
from mdsp.ops import Gain, Mix, Scale, Shaper
from mdsp.oscillators import Noise, Phasor, Saw, Sine, Square
from mdsp.reverb import Reverb
from mdsp.stereo import Pan, Width
from mdsp.stream import Stream, input_devices, output_devices

__all__ = [
    "Adsr",
    "AudioBuffer",
    "Biquad",
    "Chain",
    "Chorus",
    "Compressor",
    "Delay",
    "Gain",
    "Generator",
    "Graph",
    "Input",
    "Limiter",
    "Mix",
    "Noise",
    "OnePole",
    "Pan",
    "Phasor",
    "Processor",
    "Reverb",
    "Saw",
    "Scale",
    "Shaper",
    "Sine",
    "Square",
    "Stream",
    "Svf",
    "Width",
    "input_devices",
    "output_devices",
    "read_wav",
    "write_wav",
]
__version__ = "0.1.0"
