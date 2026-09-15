"""The wheel build hook bundles the Mojo runtime so the extension loads without Mojo."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hatch_build

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="checks ELF runpaths; macOS is covered by wheel CI"
)


def test_bundled_extension_loads_its_own_runtime(tmp_path):
    ext = tmp_path / "_core.so"
    hatch_build.compile_extension(ext)
    copied = hatch_build.bundle_runtime(ext)

    names = sorted(p.name for p in copied)
    assert "libKGENCompilerRTShared.so" in names
    assert all(p.parent == tmp_path / "_libs" for p in copied)
    dynamic = subprocess.run(
        ["readelf", "-d", str(ext)], capture_output=True, text=True, check=True
    ).stdout
    assert "Library runpath: [$ORIGIN/_libs]" in dynamic

    # The environment running the tests has Mojo installed; the maps check
    # shows the copies in _libs are the ones loaded.
    code = f"""
import importlib.util
spec = importlib.util.spec_from_file_location("mdsp._core", {str(ext)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.Gain(48000.0, 1)
maps = open("/proc/self/maps").read()
print("\\n".join(sorted({{line.split()[-1] for line in maps.splitlines() if "libKGEN" in line}})))
"""
    loaded = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    assert loaded == [str(tmp_path / "_libs" / "libKGENCompilerRTShared.so")]
