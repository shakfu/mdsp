"""Real-time output.

Tests that need PortAudio or a device skip where they are missing, as on CI
runners. The ones that do run assert what the spike measured: no underruns
while the interpreter is busy.
"""

import ctypes.util
import threading
import time

import pytest

import mdsp
from mdsp import Gain, Graph, Saw, Stream, Svf, output_devices

SR = 48000.0

has_portaudio = ctypes.util.find_library("portaudio") is not None
needs_portaudio = pytest.mark.skipif(
    not has_portaudio, reason="PortAudio not installed"
)


def _can_open_a_stream(**kwargs) -> bool:
    """Open and close a stream for real.

    A machine can have PortAudio and even list devices while none of them
    opens, which is what CI runners look like; only an actual open decides.
    """
    if not has_portaudio:
        return False
    try:
        graph = Graph(SR, block=64)
        graph.add(Gain, gain=0.0)
        stream = Stream(graph, **kwargs)
        stream.start()
        stream.stop()
    except Exception:  # noqa: BLE001 - any failure means no usable device
        return False
    return True


def _has_output_device() -> bool:
    return _can_open_a_stream()


needs_device = pytest.mark.skipif(
    not _has_output_device(), reason="no default audio output device"
)


def _patch(block: int = 64, channels: int = 1) -> tuple[Graph, int, int]:
    g = Graph(SR, channels=channels, block=block)
    tone = g.add(Saw, freq=110.0)
    filt = g.add(Svf, mode="lowpass", cutoff=800.0, q=4.0)
    out = g.add(Gain, gain=0.05)  # quiet: these tests make sound
    g.connect(tone, filt)
    g.connect(filt, out)
    g.output = out
    return g, tone, filt


def test_construction_is_validated():
    with pytest.raises(TypeError, match="expected a Graph"):
        Stream(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no nodes"):
        Stream(Graph(SR))
    with pytest.raises(TypeError, match="device must be an int"):
        Stream(_patch()[0], device="default")  # type: ignore[arg-type]


def test_stream_is_exported():
    assert mdsp.Stream is Stream
    assert "Stream" in mdsp.__all__ and "output_devices" in mdsp.__all__


@needs_portaudio
def test_output_devices_reports_the_default():
    # A machine with PortAudio but no sound card lists nothing, which is fine:
    # what matters is that enumeration works and the fields are right.
    devices = output_devices()
    for device in devices:
        assert device["max_output_channels"] >= 1
        assert set(device) == {
            "index",
            "name",
            "max_input_channels",
            "max_output_channels",
            "default_sample_rate",
            "default",
        }
    assert sum(device["default"] for device in devices) <= 1


@needs_portaudio
def test_set_applies_immediately_while_stopped():
    g, _, filt = _patch()
    stream = Stream(g)
    assert not stream.running
    assert stream.set(filt, "cutoff", 1234.0) is True
    g.reset(filt)
    assert g.generate(64).frames == 64  # graph still usable when not running


@needs_portaudio
def test_allocating_parameters_are_refused():
    g = Graph(SR, block=64)
    echo = g.add(mdsp.Delay, max_delay=0.1, delay=0.01)
    with pytest.raises(ValueError, match="allocates"):
        Stream(g).set(echo, "max_delay", 0.5)


@needs_portaudio
def test_unknown_parameter_is_refused():
    g, _, filt = _patch()
    with pytest.raises(ValueError, match="no parameter"):
        Stream(g).set(filt, "resonance", 1.0)


@needs_device
def test_start_stop_and_stats():
    g, _, _ = _patch()
    stream = Stream(g)
    assert not stream.running
    stream.start()
    try:
        assert stream.running
        assert "running=True" in repr(stream)
        with pytest.raises(RuntimeError, match="already running"):
            stream.start()
        time.sleep(0.3)
        stats = stream.stats
    finally:
        stream.stop()
    assert not stream.running
    assert stats["callbacks"] > 0
    assert stats["dropped"] == 0
    assert stats["applied"] >= 0
    assert 0 < stats["worst_render_us"] < 64 / SR * 1e6  # inside one block period
    stream.stop()  # stopping twice is fine


@needs_device
def test_graph_is_locked_while_running():
    g, tone, filt = _patch()
    with Stream(g) as stream:
        assert stream.running
        for call in (
            lambda: g.set(filt, "cutoff", 900.0),
            lambda: g.add(Gain),
            lambda: g.connect(tone, filt),
            lambda: setattr(g, "output", tone),
        ):
            with pytest.raises(RuntimeError, match="running Stream"):
                call()
    assert not g._locked
    g.set(filt, "cutoff", 900.0)  # unlocked again


@needs_device
def test_no_underruns_while_python_is_busy():
    g, _, filt = _patch(block=64)
    with Stream(g) as stream:
        deadline = time.perf_counter() + 1.5
        spins = 0
        while time.perf_counter() < deadline:
            spins = (spins * 31 + 7) % 1000003  # holds the GIL
            stream.set(filt, "cutoff", 400.0 + (spins % 3000))
        stats = stream.stats
    assert stats["underruns"] == 0
    assert stats["callbacks"] > 1.0 * SR / 64  # kept up with the device clock
    assert stats["worst_render_us"] < 64 / SR * 1e6


@needs_device
def test_every_message_is_applied_or_counted_as_dropped():
    """Regression: queue indices sharing one struct lost messages across threads."""
    g, _, filt = _patch()
    with Stream(g) as stream:
        time.sleep(0.05)  # let the callback start draining
        results = [stream.set(filt, "cutoff", 500.0 + i) for i in range(20000)]
        time.sleep(0.2)  # let it drain what was queued
        stats = stream.stats
    assert stats["dropped"] == results.count(False)
    assert stats["applied"] == results.count(True)
    assert stats["underruns"] == 0


@needs_device
def test_stereo_stream():
    g, _, _ = _patch(channels=2)
    with Stream(g) as stream:
        time.sleep(0.2)
        assert stream.stats["underruns"] == 0


@needs_device
def test_parallel_streams_do_not_interfere():
    graphs = [_patch()[0] for _ in range(2)]
    streams = [Stream(g) for g in graphs]
    for stream in streams:
        stream.start()
    try:
        time.sleep(0.3)
        for stream in streams:
            assert stream.stats["callbacks"] > 0
    finally:
        for stream in streams:
            stream.stop()


@needs_device
def test_deleting_a_running_stream_stops_it():
    g, _, _ = _patch()
    stream = Stream(g)
    stream.start()
    stream.__del__()  # what garbage collection would do
    assert not stream.running
    assert not g._locked


@needs_device
def test_set_from_another_thread_reaches_the_audio_thread():
    g, _, filt = _patch()
    with Stream(g) as stream:
        sent = []

        def sender():
            for i in range(100):
                sent.append(stream.set(filt, "cutoff", 500.0 + i))
                time.sleep(0.001)

        thread = threading.Thread(target=sender)
        thread.start()
        thread.join()
        assert all(sent)
        assert stream.stats["dropped"] == 0
