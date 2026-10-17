"""Build the installed CUDA adapter and retain focused hardware evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import venv

from validation_support import digest, fresh, signal_handlers, write_json


def source_identity(root):
    files = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root).decode().split("\0")
    selected = {name: digest(root / name) for name in sorted(set(files)) if name and
                (name.startswith(("src/", "include/", "bindings/", "python/", "tests/", "tools/", "requirements/", "cmake/", ".github/"))
                 or name in ("CMakeLists.txt", "Dockerfile", "pyproject.toml", ".dockerignore")) and (root / name).is_file()}
    return dict(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                files=selected, sha256=hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest())


def run(arguments, output, name, timeout=240):
    started = time.monotonic()
    print(f"Running {name}; log={output / (name + '.log')}", flush=True)
    with (output / (name + ".log")).open("wb") as stream:
        process = subprocess.Popen(list(map(str, arguments)), cwd=output, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=os.environ | {"TMPDIR": str(output / "setup-tmp")})
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            raise
    if code:
        raise RuntimeError(f"{name} exited {code}; inspect {output / (name + '.log')}")
    return dict(arguments=list(map(str, arguments)), exit_code=code, elapsed_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise SystemExit("Use the locked Linux CPython 3.12 build toolchain with GPU access.")
    root = Path(__file__).resolve().parents[1]
    output = fresh(args.output)
    if shutil.disk_usage(output).free < 12 * 1024 ** 3:
        raise RuntimeError("CUDA setup needs 12 GiB free on its output filesystem; select a disk-backed Linux path")
    (output / "setup-tmp").mkdir()
    source = source_identity(root)
    report = dict(status="failed", source=source, commands={},
                  toolchain={package: importlib.metadata.version(package) for package in
                             ("build", "cmake", "ninja", "pybind11", "scikit-build-core")})
    write_json(output / "validation.json", report)
    try:
        report["commands"]["build"] = run([sys.executable, "-m", "build", "--no-isolation", "--outdir", output / "dist", root], output, "build")
        wheels = list((output / "dist").glob("*.whl"))
        if len(wheels) != 1 or len(list((output / "dist").glob("*.tar.gz"))) != 1:
            raise ValueError("expected one sdist and its derived wheel")
        venv.EnvBuilder(with_pip=True).create(output / "runtime")
        python = output / "runtime/bin/python"
        lock = root / "requirements/training-cuda.lock"
        offline = output / "training-cuda-offline.lock"
        offline.write_text(re.sub(r"torch @ https://\S+", "torch==2.14.1+cu132", lock.read_text()), encoding="utf-8")
        wheelhouse = args.wheelhouse.resolve() if args.wheelhouse else output / "cuda-wheels"
        if not args.wheelhouse:
            report["commands"]["download"] = run([python, "-I", "-m", "pip", "download", "--require-hashes", "--only-binary=:all:",
                "--no-deps", "-r", lock, "-d", wheelhouse], output, "download", timeout=1800)
        report["commands"]["dependencies"] = run([python, "-I", "-m", "pip", "install", "--no-index", "--find-links", wheelhouse,
            "--require-hashes", "-r", offline], output, "dependencies", timeout=600)
        report["commands"]["install"] = run([python, "-I", "-m", "pip", "install", "--no-index", "--no-deps", wheels[0]], output, "install")
        report["commands"]["freeze"] = run([python, "-I", "-m", "pip", "freeze"], output, "runtime")
        run(["/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else "nvidia-smi", "-q"], output, "driver", timeout=15)
        if source_identity(root)["sha256"] != source["sha256"]:
            raise ValueError("behavioral source changed during runtime setup; rerun fresh with the collected wheelhouse")
        shutil.copytree(root / "tests/cuda", output / "checks", ignore=shutil.ignore_patterns("__pycache__"))
        report["commands"]["controller-lifecycle"] = run([python, "-I", output / "checks/controller_checks.py",
            "--output", output / "controller-checks"], output, "controller-checks", timeout=10)
        report["commands"]["checks"] = run([python, "-I", "-X", "faulthandler", output / "checks/controller.py",
                                             "--output", output / "runs"], output, "checks", timeout=360)
        checks = json.loads((output / "runs/checks.json").read_text())
        if checks["status"] != "passed" or len(checks["cases"]) != 7:
            raise ValueError("incomplete GPU acceptance")
        samples = [sample | dict(world_size=case["world_size"]) for case in checks["cases"]
                   for worker in case["results"] for sample in worker.get("samples", []) if not sample["warmup"]]
        if len(samples) != 120:
            raise ValueError("incomplete per-rank measured timing samples")
        write_json(output / "timing-samples.json", samples)
        summaries = []
        for world in (1, 2):
            for pinned in (False, True):
                for count in (1025, 1048576):
                    chosen = [s for s in samples if (s["world_size"], s["pin_memory"], s["count"]) == (world, pinned, count)]
                    summaries.append(dict(world_size=world, pin_memory=pinned, count=count, samples=len(chosen),
                        aggregation="pooled per-rank synchronized wall samples",
                        stages={field: dict(median=statistics.median(s[field] for s in chosen),
                                            minimum=min(s[field] for s in chosen), maximum=max(s[field] for s in chosen))
                                for field in ("ready_wait_ms", "d2h_ms", "native_ms", "h2d_ms", "total_ms")}))
        write_json(output / "timing-summary.json", summaries)
        if source_identity(root)["sha256"] != source["sha256"]:
            raise ValueError("behavioral source changed during validation")
        report.update(status="passed", wheel_sha256=digest(wheels[0]), lock_sha256=digest(lock),
                      evidence="single-GPU Linux/WSL functional integration", cases=7)
    except BaseException as error:
        report["primary_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        primary = sys.exception()
        try:
            write_json(output / "validation.json", report)
        except OSError as error:
            if primary is None:
                raise
            primary.add_note(f"Secondary evidence persistence failure: {error}")
    print(f"PASS installed CUDA adapter: {output}", flush=True)


if __name__ == "__main__":
    signal_handlers()
    main()
