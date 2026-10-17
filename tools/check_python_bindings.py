"""Build an sdist/wheel and verify the installed CPU bindings in isolation."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile


def run(arguments, directory, timeout=240):
    subprocess.run(arguments, cwd=directory, check=True, timeout=timeout)


def main():
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux":
        raise SystemExit("The initial binding check requires Linux CPython 3.12.")
    root = Path(__file__).resolve().parents[1]
    for package in ("build", "cmake", "ninja", "pybind11", "scikit-build-core"):
        print(f"{package}=={importlib.metadata.version(package)}", flush=True)
    with tempfile.TemporaryDirectory(prefix="meshtrain-bindings-") as temporary:
        scratch = Path(temporary)
        dist = scratch / "dist"
        run([sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist), str(root)], scratch)
        wheels, archives = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
        if len(wheels) != 1 or len(archives) != 1:
            raise RuntimeError("Expected exactly one sdist and its wheel.")
        with zipfile.ZipFile(wheels[0]) as wheel:
            names = wheel.namelist()
            if "meshtrain/__init__.py" not in names or not any(
                name.startswith("meshtrain/_native.") and name.endswith(".so") for name in names
            ):
                raise RuntimeError("Wheel is missing its package or compiled extension.")
            if any(name.endswith((".a", ".hpp", ".h")) or name.startswith("tests/") for name in names):
                raise RuntimeError("Wheel contains native SDK or test artifacts.")
            print(f"Wheel: {wheels[0].name}\nContents: {names}", flush=True)
        runtime = scratch / "runtime"
        venv.EnvBuilder(with_pip=True).create(runtime)
        python = str(runtime / "bin/python")
        run([python, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--require-hashes",
             "-r", str(root / "requirements/python-runtime.lock")], scratch)
        run([python, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-deps",
             str(wheels[0])], scratch)
        checks = scratch / "checks"
        shutil.copytree(root / "tests/python", checks, ignore=shutil.ignore_patterns("__pycache__"))
        run([python, "-I", "-X", "faulthandler", str(checks / "checks.py")], scratch, timeout=90)
    print("PASS: sdist-derived wheel, isolated install/import, buffer contracts, real TCP SUM, lifetime/GIL/failure checks", flush=True)


if __name__ == "__main__":
    main()
