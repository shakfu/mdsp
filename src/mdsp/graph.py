"""Runtime graphs of kernels, rendered entirely in Mojo.

A graph is built at runtime and rendered with one call per block, so a chain of
units costs one boundary crossing instead of one per unit. The same graph
renders a whole file offline or one audio callback.

Nodes are identified by opaque integer handles and changed through the graph
(`set`), never by writing into a node object. That keeps the door open for an
audio thread applying changes from a queue.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from mdsp import _core
from mdsp._base import check_planar
from mdsp.buffer import AudioBuffer
from mdsp.chorus import Chorus
from mdsp.delay import Delay
from mdsp.dynamics import Compressor, Limiter
from mdsp.envelope import Adsr
from mdsp.filters import _MODES, Biquad, OnePole, Svf
from mdsp.ops import CURVES, SHAPES, Gain, Mix, Scale, Shaper
from mdsp.oscillators import COLORS, Noise, Phasor, Saw, Sine, Square
from mdsp.reverb import Reverb
from mdsp.stereo import Pan, Width

__all__ = ["Graph", "Input", "NodeKind"]


class Input:
    """Marks the node that reads the buffer passed to `Graph.process`.

    A graph without one renders silence into its sources, which is what
    `Graph.generate` expects.
    """

    __slots__ = ()


NodeKind = (
    type[Input]
    | type[Phasor]
    | type[Sine]
    | type[Saw]
    | type[Square]
    | type[Gain]
    | type[OnePole]
    | type[Biquad]
    | type[Svf]
    | type[Delay]
    | type[Scale]
    | str
)

_KINDS: dict[type, str] = {
    Input: "input",
    Phasor: "phasor",
    Sine: "sine",
    Saw: "saw",
    Square: "square",
    Gain: "gain",
    OnePole: "onepole",
    Biquad: "biquad",
    Svf: "svf",
    Delay: "delay",
    Scale: "scale",
    Mix: "mix",
    Adsr: "adsr",
    Noise: "noise",
    Compressor: "compressor",
    Limiter: "compressor",
    Shaper: "shaper",
    Reverb: "reverb",
    Chorus: "chorus",
    Pan: "pan",
    Width: "width",
}

#: Parameters whose value is a name rather than a number.
_ENUMS: dict[str, tuple[str, ...]] = {
    "mode": _MODES,
    "curve": CURVES,
    "shape": SHAPES,
    "color": COLORS,
}


class Graph:
    """A runtime graph of kernels.

    Args:
        sample_rate: Shared by every node.
        channels: Independent copies of the graph, one per channel.
        block: Frames per internal chunk. Longer requests are split into
            chunks of this size, and nothing is allocated while rendering.

    A node may only read from nodes added before it, so graphs cannot contain
    cycles. Feedback needs a kernel that contains its own loop, such as `Delay`.

    >>> import numpy as np
    >>> g = Graph(sample_rate=48000.0)
    >>> lfo = g.add(Sine, freq=2.0)
    >>> tone = g.add(Saw, freq=220.0)
    >>> filt = g.add(Svf, mode="lowpass", cutoff=800.0, q=4.0)
    >>> g.connect(tone, filt)
    >>> g.connect(lfo, filt, "cutoff")
    >>> g.output = filt
    >>> g.generate(256).data.shape
    (1, 256)
    """

    KINDS: ClassVar[tuple[str, ...]] = tuple(_core.node_kinds())

    def __init__(
        self,
        sample_rate: float = 48000.0,
        channels: int = 1,
        block: int = 512,
    ) -> None:
        sr = float(sample_rate)
        if not (math.isfinite(sr) and sr > 0):
            raise ValueError(f"sample_rate must be positive and finite, got {sr}")
        for name, value in (("channels", channels), ("block", block)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        self._sample_rate = sr
        self._channels = channels
        self._block = block
        self._impl = _core.Graph(sr, channels, block)
        self._params: list[list[str]] = []
        self._inputs: list[list[str]] = []
        self._kinds: list[str] = []
        self._output = -1
        self._removed: set[int] = set()
        self._locked = False  # set while a Stream renders this graph

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def block(self) -> int:
        return self._block

    def __len__(self) -> int:
        return len(self._kinds)

    def add(self, kind: NodeKind, **params: float | str) -> int:
        """Add a node and return its handle.

        Args:
            kind: A unit class (`Sine`, `Svf`, ...), `Input`, or a name from
                `Graph.KINDS`.
            params: Starting parameter values, applied at once rather than
                ramped.
        """
        self._check_unlocked()
        name = _KINDS.get(kind) if isinstance(kind, type) else kind
        if not isinstance(name, str) or name not in self.KINDS:
            raise ValueError(
                f"unknown node kind {kind!r}; expected one of {self.KINDS}"
            )
        node = self._impl.add(name)
        self._kinds.append(name)
        self._params.append(self._impl.param_names(node))
        self._inputs.append(self._impl.input_names(node))
        self._output = node
        for param, value in params.items():
            self.set(node, param, value)
        self._impl.reset_node(node)
        return node

    def _check_unlocked(self) -> None:
        if self._locked:
            raise RuntimeError(
                "a running Stream is rendering this graph; stop it first, or use "
                "Stream.set, which queues the change for the audio thread"
            )

    def _check(self, node: int) -> None:
        if isinstance(node, bool) or not isinstance(node, int):
            raise TypeError(f"node handles are ints, got {node!r}")
        if not 0 <= node < len(self._kinds):
            raise ValueError(f"no node with handle {node}")

    def _check_live(self, node: int) -> None:
        self._check(node)
        if node in self._removed:
            raise ValueError(f"node {node} was removed")

    def kind(self, node: int) -> str:
        """The kind name of *node*."""
        self._check(node)
        return self._kinds[node]

    def params(self, node: int) -> tuple[str, ...]:
        """Parameter names *node* accepts."""
        self._check(node)
        return tuple(self._params[node])

    def inputs(self, node: int) -> tuple[str, ...]:
        """Input port names of *node*; port ``in`` is audio."""
        self._check(node)
        return tuple(self._inputs[node])

    def _encode(self, node: int, param: str, value: float | str) -> tuple[int, float]:
        """Validate a parameter change and return its index and number."""
        self._check(node)
        names = self._params[node]
        if param not in names:
            raise ValueError(
                f"{self._kinds[node]} has no parameter {param!r}; expected {names}"
            )
        choices = _ENUMS.get(param)
        if choices is not None:
            if value not in choices:
                raise ValueError(f"{param} must be one of {choices}, got {value!r}")
            return names.index(param), float(choices.index(str(value)))
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{param} must be finite, got {value!r}")
        return names.index(param), number

    def set(self, node: int, param: str, value: float | str) -> None:
        """Set a parameter. Changes ramp over 10 ms, as they do on a unit.

        Raises:
            RuntimeError: If a `Stream` is rendering this graph; use
                `Stream.set`, which queues the change for the audio thread.
        """
        self._check_unlocked()
        self._check_live(node)
        index, number = self._encode(node, param, value)
        self._impl.set(node, index, number)

    def connect(
        self, src: int, dst: int, port: str = "in", delay: int | None = None
    ) -> None:
        """Feed *src*'s output into *dst*'s input *port*.

        Args:
            delay: Samples of delay, which makes this a feedback edge. Any pair
                of nodes may be connected that way, including a node to itself.
                A loop shorter than `block` makes the graph render in smaller
                chunks, which costs speed but keeps the loop exact; counting in
                samples keeps the result independent of the block size.

        Raises:
            ValueError: For an unknown port, a delay below `block`, or a
                backwards connection without a delay.
        """
        self._check_unlocked()
        self._check_live(src)
        self._check_live(dst)
        names = self._inputs[dst]
        if port not in names:
            raise ValueError(
                f"{self._kinds[dst]} has no input {port!r}; expected {names}"
            )
        if delay is None:
            if src >= dst:
                raise ValueError(
                    f"node {dst} can only read from nodes added before it, not "
                    f"{src}; pass delay= to feed a later node back"
                )
            self._impl.connect(src, dst, names.index(port), 0)
            return
        if isinstance(delay, bool) or not isinstance(delay, int):
            raise TypeError(f"delay must be an int number of samples, got {delay!r}")
        if delay < 1:
            raise ValueError(f"a feedback delay must be at least 1 sample, got {delay}")
        self._impl.connect(src, dst, names.index(port), delay)

    def remove(self, node: int) -> None:
        """Silence a node and drop every connection to or from it.

        Handles stay valid: the slot is emptied rather than renumbered, so the
        handles of other nodes keep working. A removed node renders nothing.

        Raises:
            ValueError: If *node* is the graph's output; choose another first.
        """
        self._check_unlocked()
        self._check(node)
        if node == self._output:
            raise ValueError(
                f"node {node} is the output; set another output before removing it"
            )
        self._impl.remove(node)
        self._removed.add(node)

    def removed(self, node: int) -> bool:
        """Whether *node* has been removed."""
        self._check(node)
        return node in self._removed

    @property
    def output(self) -> int:
        """The node whose samples `process` and `generate` return."""
        return self._output

    @output.setter
    def output(self, node: int) -> None:
        self._check_unlocked()
        self._check_live(node)
        self._impl.set_output(node)
        self._output = node

    def reset(self, node: int | None = None) -> None:
        """Clear state and end parameter ramps, for one node or all of them."""
        if node is None:
            self._impl.reset()
        else:
            self._check(node)
            self._impl.reset_node(node)

    def _render(
        self, src: AudioBuffer | None, frames: int, out: AudioBuffer | None
    ) -> AudioBuffer:
        if not self._kinds:
            raise ValueError("the graph has no nodes")
        if out is None:
            result = AudioBuffer(
                np.empty((self._channels, frames), np.float32),
                self._sample_rate,
                copy=False,
            )
        else:
            if out.sample_rate != self._sample_rate:
                raise ValueError(
                    f"out sample_rate {out.sample_rate} != {self._sample_rate}"
                )
            if out.frames != frames:
                raise ValueError(f"out has {out.frames} frames, expected {frames}")
            result = out
        buffers = [result.data]
        address = 0
        if src is not None:
            buffers.append(src.data)
            address = src.address
        check_planar(buffers, (self._channels, frames))
        if not result.data.flags.writeable:
            raise ValueError("out is read-only")
        self._impl.render(address, result.address, frames)
        return result

    def process(self, buf: AudioBuffer, out: AudioBuffer | None = None) -> AudioBuffer:
        """Render the graph with *buf* feeding its `Input` nodes.

        Args:
            out: Write here instead of allocating, and return it. It must match
                *buf* in sample rate, channels and frames.
        """
        if buf.sample_rate != self._sample_rate:
            raise ValueError(
                f"buffer sample_rate {buf.sample_rate} != {self._sample_rate}"
            )
        if buf.channels != self._channels:
            raise ValueError(
                f"buffer has {buf.channels} channels, expected {self._channels}"
            )
        return self._render(buf, buf.frames, out)

    def generate(self, frames: int, out: AudioBuffer | None = None) -> AudioBuffer:
        """Render *frames* samples; `Input` nodes read silence.

        Args:
            out: Write here instead of allocating, and return it.
        """
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
            raise ValueError(f"frames must be an int >= 0, got {frames!r}")
        return self._render(None, frames, out)

    def __repr__(self) -> str:
        return (
            f"Graph(nodes={len(self._kinds)}, output={self._output}, "
            f"sample_rate={self._sample_rate}, channels={self._channels}, "
            f"block={self._block})"
        )
