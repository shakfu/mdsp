"""The examples must keep working; they are the API's first impression."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
pytestmark = pytest.mark.skipif(not EXAMPLES.is_dir(), reason="examples/ not packaged")


def test_offline_example_renders_a_file(tmp_path):
    target = tmp_path / "out.wav"
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "offline_process.py"), str(target)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert target.exists()

    import mdsp

    rendered = mdsp.read_wav(target)
    assert rendered.duration > 1.0
    assert abs(rendered.data).max() > 0.01  # not silence


def test_offline_example_accepts_an_input_file(tmp_path):
    import numpy as np

    import mdsp

    source, target = tmp_path / "in.wav", tmp_path / "out.wav"
    mdsp.write_wav(
        source,
        mdsp.AudioBuffer(
            np.random.default_rng(0).uniform(-0.5, 0.5, (2, 4800)), 48000.0
        ),
    )
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "offline_process.py"),
            str(source),
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert mdsp.read_wav(target).channels == 2


def test_realtime_example_builds_its_voice():
    """Build the patch without opening a device, which CI does not have."""
    module = runpy.run_path(str(EXAMPLES / "realtime_synth.py"))
    graph, _a, _b, env = module["build_voice"]()
    assert len(graph) == 8
    assert graph.kind(env) == "adsr"
    assert abs(graph.generate(256).data).max() >= 0.0  # renders offline too
