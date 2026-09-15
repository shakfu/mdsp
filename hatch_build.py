"""Compile the Mojo extension ``mdsp._core``.

As a hatchling build hook, it builds the wheel's ``mdsp/_core.so``, bundles the
Mojo runtime libraries it links into ``mdsp/_libs/``, and tags the wheel for the
build platform. Run ``auditwheel repair`` (Linux) or ``delocate-wheel`` (macOS)
afterwards for a PyPI platform tag. Editable installs are skipped: ``make build``
compiles ``src/mdsp/_core.so`` in place with this module's CLI.

The runtime libraries are bundled so the installed wheel does not depend on the
Mojo compiler packages.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

ROOT = Path(__file__).resolve().parent
MOJO_ROOT = ROOT / "src" / "mdsp" / "_mojo"
LIBS_DIR = "_libs"

# No FMA contraction: fused and unfused paths round differently, so `tick` and
# `process` diverged. tests/test_mojo_kernels.py imports these.
MOJO_FLAGS = ["--fp-mode", "contract=off"]

# Baselines that run on any CPU of the architecture. Override with the
# MDSP_TARGET_CPU environment variable; "host" optimises for the build machine.
DEFAULT_TARGET_CPU = {"x86_64": "x86-64-v2", "aarch64": "generic", "arm64": "apple-m1"}


def _mojo_env() -> dict[str, str]:
    from mojo.run import _mojo_env  # private; also used by mojo.importer

    env: dict[str, str] = _mojo_env()
    return env


def compile_extension(output: Path) -> None:
    """Build ``_core.mojo`` into the shared library *output*."""
    from mojo.run import subprocess_run_mojo

    cpu = os.environ.get("MDSP_TARGET_CPU") or DEFAULT_TARGET_CPU.get(
        platform.machine()
    )
    if cpu is None:
        raise RuntimeError(
            f"no default target CPU for {platform.machine()}; set MDSP_TARGET_CPU"
        )
    args = ["build", "--emit", "shared-lib", *MOJO_FLAGS]
    if cpu != "host":
        args += ["--target-cpu", cpu]
    args += ["-I", str(MOJO_ROOT), str(MOJO_ROOT / "_core.mojo"), "-o", str(output)]
    subprocess_run_mojo(args, check=True)


def _tool(name: str) -> str:
    # Build requirements install tools into the build environment's scripts dir,
    # which is not always on PATH.
    path = os.pathsep.join([sysconfig.get_path("scripts"), os.environ.get("PATH", "")])
    found = shutil.which(name, path=path)
    if found is None:
        raise RuntimeError(f"required tool not found: {name}")
    return found


def _run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def _needed(binary: Path) -> list[str]:
    """Names of the shared libraries *binary* links against."""
    if sys.platform == "darwin":
        lines = _run("otool", "-L", str(binary)).splitlines()[1:]
        return [Path(line.split()[0]).name for line in lines if line.strip()]
    return _run(_tool("patchelf"), "--print-needed", str(binary)).split()


def bundle_runtime(extension: Path) -> list[Path]:
    """Copy the Mojo runtime libraries *extension* needs into ``_libs/`` beside it.

    Returns the copied files. The extension's library search path is rewritten
    to that directory; the libraries already search their own directory.
    """
    lib_dir = Path(_mojo_env()["MODULAR_MOJO_MAX_PACKAGE_ROOT"]) / "lib"
    dest = extension.parent / LIBS_DIR
    dest.mkdir(exist_ok=True)
    copied: list[Path] = []
    pending = [extension]
    while pending:
        for name in _needed(pending.pop()):
            src, dst = lib_dir / name, dest / name
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
                copied.append(dst)
                pending.append(dst)
    if not copied:
        raise RuntimeError(f"{extension.name} links no libraries from {lib_dir}")

    if sys.platform == "darwin":
        load_commands = _run("otool", "-l", str(extension))
        for rpath in re.findall(
            r"cmd LC_RPATH\n.*\n\s+path (.+) \(offset", load_commands
        ):
            _run("install_name_tool", "-delete_rpath", rpath, str(extension))
        _run(
            "install_name_tool",
            "-add_rpath",
            f"@loader_path/{LIBS_DIR}",
            str(extension),
        )
        _run("codesign", "--force", "--sign", "-", str(extension))
    else:
        _run(_tool("patchelf"), "--set-rpath", f"$ORIGIN/{LIBS_DIR}", str(extension))
    return copied


class MojoBuildHook(BuildHookInterface[Any]):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel" or version == "editable":
            return
        self._tmp = Path(tempfile.mkdtemp(prefix="mdsp-build-"))
        extension = self._tmp / "_core.so"
        compile_extension(extension)
        for lib in bundle_runtime(extension):
            build_data["force_include"][str(lib)] = f"mdsp/{LIBS_DIR}/{lib.name}"
        build_data["force_include"][str(extension)] = "mdsp/_core.so"
        build_data["pure_python"] = False
        # One wheel serves every supported CPython: the extension resolves the
        # C API at load time. Tested on 3.10-3.14; free-threaded builds crash.
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        if sys.platform == "darwin" and target:
            plat = f"macosx_{target.replace('.', '_')}_{platform.machine()}"
        build_data["tag"] = f"py3-none-{plat}"

    def finalize(
        self, version: str, build_data: dict[str, Any], artifact_path: str
    ) -> None:
        if hasattr(self, "_tmp"):
            shutil.rmtree(self._tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python hatch_build.py OUTPUT")
    compile_extension(Path(sys.argv[1]))
