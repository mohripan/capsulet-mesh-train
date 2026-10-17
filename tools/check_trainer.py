"""Build/install the real wheel and retain focused CPU training evidence."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import venv


def run(arguments, directory, log, timeout=240, env=None):
    print("Running:", " ".join(map(str, arguments)), flush=True)
    with log.open("w") as stream:
        result = subprocess.run(arguments, cwd=directory, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout, env=env)
    if result.returncode:
        print(log.read_text()[-16000:], flush=True)
        raise RuntimeError(f"command exited {result.returncode}; see {log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise SystemExit("The trainer check requires Linux CPython 3.12.")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Use a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    toolchain = {package: importlib.metadata.version(package)
                 for package in ("build", "cmake", "ninja", "pybind11", "scikit-build-core")}
    (output / "toolchain.json").write_text(json.dumps(toolchain, indent=2) + "\n")
    run([sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output / "dist"), str(root)],
        output, output / "build.log")
    wheels = list((output / "dist").glob("*.whl"))
    if len(wheels) != 1 or len(list((output / "dist").glob("*.tar.gz"))) != 1:
        raise RuntimeError("expected one sdist and its derived wheel")
    runtime = output / "runtime"
    venv.EnvBuilder(with_pip=True).create(runtime)
    python = str(runtime / "bin/python")
    lock = (root / "requirements/training-cpu.lock").read_text()
    offline_lock = output / "training-cpu-offline.lock"
    offline_lock.write_text(re.sub(r"torch @ https://\S+", "torch==2.14.1+cpu", lock))
    cached = os.environ.get("MESHTRAIN_TRAINING_WHEELHOUSE")
    wheelhouse = Path(cached).resolve() if cached else output / "cpu-wheels"
    if not cached:
        run([python, "-I", "-m", "pip", "download", "--require-hashes", "--only-binary=:all:", "--no-deps",
             "-r", str(root / "requirements/training-cpu.lock"), "-d", str(wheelhouse)],
            output, output / "download.log", timeout=600)
    run([python, "-I", "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse),
         "--require-hashes", "-r", str(offline_lock)], output, output / "dependencies.log", timeout=600)
    run([python, "-I", "-m", "pip", "install", "--no-index", "--no-deps", str(wheels[0])],
        output, output / "install.log")
    run([python, "-I", "-m", "pip", "freeze"], output, output / "runtime.txt")
    run([python, "-I", "-m", "meshtrain.training", "run", "--output", str(output / "run"),
         "--world-size", "4" if args.mode == "full" else "2", "--steps", "100" if args.mode == "full" else "3"],
        output, output / "training.log", timeout=150)
    checks = output / "checks"
    shutil.copytree(root / "tests/training", checks, ignore=shutil.ignore_patterns("__pycache__"))
    run([python, "-I", "-X", "faulthandler", str(checks / "checks.py"), str(output / "run")],
        output, output / "checks.log", timeout=180)
    report = json.loads((output / "run/verification.json").read_text())
    if report["status"] != "passed" or report["full_acceptance"] != (args.mode == "full"):
        raise RuntimeError("incorrect acceptance profile")
    print(f"PASS: installed CPU trainer {args.mode}; {len(report['checks'])} comparisons; evidence={output}", flush=True)


if __name__ == "__main__":
    main()
