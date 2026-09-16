"""Real-time audio output for a `Graph`.

The audio thread runs a Mojo callback: it applies queued parameter changes and
renders the graph without touching Python, so a busy interpreter cannot make it
miss a deadline. Measurements: `docs/dev/spikes/2026-09-16-realtime`.

Needs PortAudio (`libportaudio.so.2`), which mdsp loads only when a stream is
created. Everything else in mdsp works without it.
"""

from __future__ import annotations

import ctypes.util
import os
from types import TracebackType
from typing import Any, Self

from mdsp import _core
from mdsp.graph import Graph

__all__ = ["Stream", "input_devices", "output_devices"]

#: Parameters that allocate, so the audio thread must not apply them.
_ALLOCATING = frozenset({"max_delay"})


#: Checked when the linker's own search misses it, as it does for Homebrew on
#: Apple silicon, whose prefix dyld does not search by default.
_EXTRA_PATHS = (
    "/opt/homebrew/lib/libportaudio.2.dylib",
    "/usr/local/lib/libportaudio.2.dylib",
    "/usr/local/lib/libportaudio.so.2",
)


def _require_portaudio() -> None:
    if ctypes.util.find_library("portaudio") is not None:
        return
    if any(os.path.exists(path) for path in _EXTRA_PATHS):
        return
    raise RuntimeError(
        "PortAudio not found. Install it to use mdsp.Stream: "
        "apt install libportaudio2, or brew install portaudio."
    )


def output_devices() -> list[dict[str, Any]]:
    """Devices with output channels, each with ``index``, ``name``,
    ``max_output_channels``, ``max_input_channels``, ``default_sample_rate``
    and ``default``."""
    _require_portaudio()
    return [d for d in _core.audio_devices() if d["max_output_channels"] > 0]


def input_devices() -> list[dict[str, Any]]:
    """Devices with input channels, described as in `output_devices`."""
    _require_portaudio()
    return [d for d in _core.audio_devices() if d["max_input_channels"] > 0]


class Stream:
    """Plays a `Graph` through an audio device.

    Args:
        graph: Rendered on the audio thread. While the stream runs the graph is
            locked: building it further or setting parameters directly raises.
            Use `Stream.set` instead, which queues changes for the audio thread.
        device: Output device index from `output_devices`; the default device
            when omitted.
        input_device: Capture device index from `input_devices`, or ``True``
            for the default one. The graph's `Input` nodes then read live
            audio. Omitted means output only, and `Input` nodes read silence.

    The stream holds a reference to *graph*, so the kernels the callback renders
    cannot be collected while it runs. Stopping unlocks the graph, and closing
    the stream also stops it.

    Example, playing a filtered sawtooth for a second::

        g = mdsp.Graph(48000.0, block=64)
        tone = g.add(mdsp.Saw, freq=110.0)
        filt = g.add(mdsp.Svf, mode="lowpass", cutoff=800.0, q=4.0)
        g.connect(tone, filt)
        g.output = filt
        with mdsp.Stream(g) as stream:
            stream.set(filt, "cutoff", 1500.0)
            time.sleep(1.0)
    """

    def __init__(
        self,
        graph: Graph,
        *,
        device: int | None = None,
        input_device: int | bool | None = None,
    ) -> None:
        if not isinstance(graph, Graph):
            raise TypeError(f"expected a Graph, got {type(graph).__name__}")
        if len(graph) == 0:
            raise ValueError("the graph has no nodes")
        if device is not None and (
            isinstance(device, bool) or not isinstance(device, int)
        ):
            raise TypeError(f"device must be an int index or None, got {device!r}")
        _require_portaudio()
        self._graph = graph
        self._device = -1 if device is None else device
        if input_device is None or input_device is False:
            self._input_device = -2  # no capture
        elif input_device is True:
            self._input_device = -1  # default capture device
        elif isinstance(input_device, int):
            self._input_device = input_device
        else:
            raise TypeError(
                f"input_device must be an int index, True or None, got {input_device!r}"
            )
        self._impl = _core.Stream(graph._impl)

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def running(self) -> bool:
        return bool(self._impl.running())

    def start(self) -> Self:
        """Open the device and begin rendering. Locks the graph."""
        if self.running:
            raise RuntimeError("the stream is already running")
        self._impl.start(
            self._device,
            self._graph.channels,
            self._graph.sample_rate,
            self._graph.block,
            self._input_device,
        )
        self._graph._locked = True
        return self

    def stop(self) -> None:
        """Stop rendering and unlock the graph. Safe to call when stopped."""
        self._impl.stop()
        self._graph._locked = False

    def set(self, node: int, param: str, value: float | str) -> bool:
        """Change a parameter: queued for the audio thread while running,
        applied at once while stopped.

        Returns False if the queue was full and the change was dropped, which
        `stats` also counts. Never blocks.

        Raises:
            ValueError: For an unknown parameter, or one that would allocate on
                the audio thread; set those while the stream is stopped.
        """
        if param in _ALLOCATING:
            raise ValueError(
                f"{param!r} allocates, so it cannot change while the stream runs;"
                " stop the stream and use Graph.set"
            )
        index, number = self._graph._encode(node, param, value)
        if not self.running:
            self._graph._impl.set(node, index, number)
            return True
        return bool(self._impl.set(node, index, number))

    @property
    def stats(self) -> dict[str, Any]:
        """Counters from the audio thread: ``callbacks``, ``underruns``,
        ``dropped`` messages, and ``worst_render_us``."""
        return dict(self._impl.stats())

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def __del__(self) -> None:
        # The callback renders through this object's address; never outlive it.
        try:
            self.stop()
        except Exception:  # noqa: BLE001, S110  # pragma: no cover
            pass  # interpreter shutdown can leave _core unusable

    def __repr__(self) -> str:
        return (
            f"Stream(running={self.running}, channels={self._graph.channels}, "
            f"block={self._graph.block}, sample_rate={self._graph.sample_rate})"
        )
