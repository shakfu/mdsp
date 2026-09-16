# Packaging: binary wheels

Date: 2026-09-16. Decision: hatchling with a build hook (`scripts/hatch_build.py`), one `py3-none` wheel per platform, Mojo runtime libraries bundled, compiler not a runtime dependency.

## Why not uv_build

uv_build "only supports pure Python code" ([docs](https://docs.astral.sh/uv/concepts/build-backend/)). It packed the in-tree `_core.so` into a `py3-none-any` wheel whose RUNPATH pointed at the build venv.

## Build flow

1. `scripts/hatch_build.py` compiles `_core.mojo` into a temp dir with `--fp-mode contract=off` and a baseline `--target-cpu`.
2. It copies the runtime libraries the extension links (found via `patchelf --print-needed` / `otool -L`) from the build environment's Mojo install into `mdsp/_libs/`, and sets the extension's search path to `$ORIGIN/_libs` (Linux) or `@loader_path/_libs` (macOS, then ad hoc re-signed).
3. `auditwheel repair` / `delocate-wheel` retags for PyPI. Both treat the libraries in `_libs` as part of the wheel and graft nothing.

Bundling in the hook, not in the repair step, keeps isolated builds working: the RUNPATH `mojo build` writes points into the build environment, which is deleted before repair or install.

## Bundled libraries

| Library | Size (Linux) | Role |
|-|-|-|
| `libKGENCompilerRTShared` | 1.2 MB | Mojo compiler runtime; `_core.so` imports 11 `KGEN_CompilerRT_*` functions (allocation, globals, AsyncRT CPU device, stack traces) |
| `libAsyncRTRuntimeGlobals` | 0.7 MB | Async runtime globals and an embedded TCMalloc (`TCMallocInternal*` names only; does not replace `malloc`) |
| `libMSupportGlobals` | 0.05 MB | Modular support-library globals |

Licence: `LicenseRef-MAX-Platform-Software-License` (from the `mojo-compiler` metadata). Redistribution terms need review before publishing.

## Measured constraints

| Constraint | Evidence | Consequence |
|-|-|-|
| Platforms | `mojo-compiler` 1.0.0 ships manylinux x86_64, manylinux aarch64, macOS 13 arm64 | Three wheels |
| glibc | `libAsyncRTRuntimeGlobals` needs `GLIBC_2.35`, `libKGENCompilerRTShared` needs `GLIBCXX_3.4.30`; `_core.so` alone needs 2.34 | `manylinux_2_35`, although Mojo's own wheels are tagged `2_34` |
| Python versions | One `_core.so` passed the suite on 3.10-3.14 | `py3-none` tag |
| Free-threaded | 3.14t segfaults on import (exit 139) | Import guard in `mdsp._base` raises `ImportError` |
| macOS minimum | `mojo build` targets the host macOS: `_core.so` built on 26 has `minos 26.0`, so `delocate-wheel` tagged the wheel `macosx_26_0`. The bundled runtime libraries have `minos 11.0`; `_core.so` imports only libSystem symbols available since 10.12. `--target-triple` alone changes the objects' minimum, not the link's; `MACOSX_DEPLOYMENT_TARGET` alone makes `ld` warn that objects target 26.0 | Hook sets both, default 13.0 |
| CPU baseline | `x86-64-v2` vs host (AVX2): Sine -21%, Delay -10%, others unchanged | Default `x86-64-v2`; `MDSP_TARGET_CPU` overrides |
| Threads | Mojo runtime init starts one idle worker per CPU in the affinity mask (16 here; any Mojo extension does this). 0 CPU idle, +7 MiB RSS. Fork, forkserver, spawn all work; the threads trigger Python's multi-threaded-fork `DeprecationWarning` | No pool-size setting found; `taskset` limits it. Default left unchanged |

## Verification status

- Linux x86_64: built, repaired, installed into clean venvs without Mojo on 3.10-3.14; sdist install from source also works.
- Linux aarch64 and macOS arm64: verified by `wheels.yml` on 2026-09-16 (run 35059670150). All three platforms built and all 15 wheel tests passed (3 platforms x Python 3.10-3.14), which covers the `generic` and `apple-m1` CPU targets and the macOS rpath rewrite and re-signing.
