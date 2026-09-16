"""Runs the Mojo-side kernel contract tests in tests/mojo."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hatch_build import MOJO_FLAGS

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "test_file",
    sorted((ROOT / "tests" / "mojo").glob("test_*.mojo")),
    ids=lambda p: p.name,
)
def test_mojo(test_file):
    mojo = shutil.which("mojo", path=str(Path(sys.executable).parent)) or shutil.which(
        "mojo"
    )
    assert mojo, "mojo compiler not found"
    result = subprocess.run(
        [
            mojo,
            "run",
            *MOJO_FLAGS,
            "-I",
            str(ROOT / "src" / "mdsp" / "_mojo"),
            str(test_file),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
