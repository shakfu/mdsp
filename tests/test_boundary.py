"""The Python/Mojo boundary: validation, parameter indices, GIL release."""

import sys
import threading
import time
import warnings

import numpy as np
import pytest

from mdsp import (
    AudioBuffer,
    Biquad,
    Delay,
    Gain,
    OnePole,
    Phasor,
    Saw,
    Sine,
    Square,
    _core,
)
from mdsp._base import Param, _check_interpreter, _Unit

SR = 48000.0
UNITS = [Phasor, Sine, Saw, Square, OnePole, Biquad, Gain, Delay]


@pytest.mark.parametrize("cls", UNITS)
def test_python_params_match_mojo_param_names(cls):
    names = cls()._impl.param_names()
    params = {k: v for k, v in vars(cls).items() if isinstance(v, Param)}
    for name, param in params.items():
        assert names[param.index] == name
    # Every Mojo parameter is reachable from Python.
    assert all(hasattr(cls, name) for name in names)


@pytest.mark.parametrize(
    "args", [(0.0, 1), (float("nan"), 1), (SR, 0), (SR, 1.0), (SR, True)]
)
def test_unit_rejects_bad_construction(args):
    with pytest.raises(ValueError):
        Gain(1.0, sample_rate=args[0], channels=args[1])


@pytest.mark.parametrize("args", [(0.0, 1), (SR, 0), (SR,)])
def test_core_rejects_bad_construction(args):
    with pytest.raises(ValueError):
        _core.Gain(*args)


def test_sample_rate_mismatch():
    with pytest.raises(ValueError, match="sample_rate"):
        Gain(sample_rate=SR).process(AudioBuffer(np.ones(4), 44100))


def test_mutated_buffer_dtype_is_caught():
    buf = AudioBuffer(np.ones(8), SR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # NumPy >= 2.5
        buf.data.dtype = np.int32  # same itemsize: reinterprets in place
    with pytest.raises(ValueError, match="float32"):
        Gain().process(buf)


def _unit() -> _Unit:
    return Gain(channels=1)


def test_run_rejects_non_contiguous():
    arr = np.zeros((1, 8), np.float32)[:, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        _unit()._run(arr, np.zeros((1, 4), np.float32))


def test_run_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        _unit()._run(np.zeros((1, 8), np.float32), np.zeros((1, 4), np.float32))
    with pytest.raises(ValueError, match="shape"):
        _unit()._run(np.zeros(8, np.float32), np.zeros(8, np.float32))


def test_run_rejects_read_only_output():
    dst = np.zeros((1, 4), np.float32)
    dst.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        _unit()._run(np.zeros((1, 4), np.float32), dst)


def test_process_releases_gil():
    filt = Biquad(sample_rate=SR)
    # Size the call to take ~80 ms on this machine.
    probe = AudioBuffer(np.zeros(1_000_000), SR)
    start = time.perf_counter()
    filt.process(probe)
    frames = int(1_000_000 * 0.08 / (time.perf_counter() - start))
    x = AudioBuffer(np.zeros(min(frames, 100_000_000), np.float32), SR, copy=False)
    stop = threading.Event()
    max_gap = 0.0

    def spin():
        nonlocal max_gap
        last = time.perf_counter()
        while not stop.is_set():
            now = time.perf_counter()
            max_gap = max(max_gap, now - last)
            last = now

    interval = sys.getswitchinterval()
    sys.setswitchinterval(0.001)
    t = threading.Thread(target=spin)
    t.start()
    try:
        time.sleep(0.01)
        max_gap = 0.0
        start = time.perf_counter()
        filt.process(x)
        duration = time.perf_counter() - start
        time.sleep(0.01)
    finally:
        stop.set()
        t.join()
        sys.setswitchinterval(interval)
    # Holding the GIL stalls the spinning thread for the whole call.
    assert duration > 0.02, "call too short to distinguish"
    assert max_gap < duration / 2


def test_parallel_instances_match_sequential():
    xs = [
        AudioBuffer(np.random.default_rng(i).uniform(-1, 1, (2, 100_000)), SR)
        for i in range(4)
    ]
    expected = [Biquad("bandpass", 2000.0, 5.0, channels=2).process(x).data for x in xs]
    results: list[np.ndarray] = [np.empty(0)] * 4

    def work(i):
        results[i] = Biquad("bandpass", 2000.0, 5.0, channels=2).process(xs[i]).data

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for got, want in zip(results, expected):
        np.testing.assert_array_equal(got, want)


def test_free_threaded_interpreter_is_rejected():
    _check_interpreter(False)
    with pytest.raises(ImportError, match="free-threaded"):
        _check_interpreter(True)
