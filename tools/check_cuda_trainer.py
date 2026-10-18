"""Build/install and retain bounded single-GPU training acceptance."""

from __future__ import annotations

import argparse
import csv
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
import venv

from check_cuda_adapter import run, source_identity
from validation_support import digest, fresh, signal_handlers, write_json


def controller(arguments, output):
    with (output / "checks.log").open("wb") as stream:
        process = subprocess.Popen(list(map(str, arguments)), cwd=output, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=2280)
        except BaseException as error:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
                    error.add_note("Controller forcibly killed; inspect owned process records for unknown cleanup")
            raise
    if code:
        raise RuntimeError(f"GPU trainer checks exited {code}; inspect {output / 'checks.log'}")


def summarize(checks, output, mode):
    samples, rank_steps, summaries = [], [], []
    for case in checks["cases"]:
        for rank, record in enumerate(case.get("cuda", [])):
            for step in record["steps"]:
                context = dict(case=case["name"], host_memory=case["host_memory"], world_size=case["world_size"],
                               rank=rank, step=step["step"], warmup=step["warmup"])
                if case["name"].startswith("parity-"):
                    rank_steps.append(context | {key: step[key] for key in ("step_ms", "gradient_sync_ms", "trace_ms")})
                    for operation in ("gradient", "loss"):
                        samples.append(context | dict(operation=operation) | step[operation])
    measured = [sample for sample in samples if not sample["warmup"]]
    measured_steps = [step for step in rank_steps if not step["warmup"]]
    if mode == "full" and (len(measured) != 760 or len(measured_steps) != 380):
        raise ValueError("incomplete measured rank/step/operation evidence")
    for memory in ("pinned", "pageable"):
        for rank in (0, 1):
            for operation in ("gradient", "loss"):
                selected = [sample for sample in measured if (sample["host_memory"], sample["rank"], sample["operation"])
                            == (memory, rank, operation)]
                if selected:
                    summaries.append(dict(host_memory=memory, rank=rank, operation=operation, samples=len(selected),
                        stages={key: dict(median=statistics.median(sample[key] for sample in selected),
                                          minimum=min(sample[key] for sample in selected), maximum=max(sample[key] for sample in selected))
                                for key in ("ready_wait_ms", "d2h_ms", "native_ms", "h2d_ms", "total_ms")}))
    maxima = [dict(host_memory=memory, step=step,
                   max_rank_step_ms=max(item["step_ms"] for item in measured_steps if item["host_memory"] == memory and item["step"] == step))
              for memory in ("pinned", "pageable") for step in sorted({item["step"] for item in measured_steps})]
    write_json(output / "timing-samples.json", samples)
    write_json(output / "timing-summary.json", dict(aggregation="per-rank instrumented synchronized wall intervals",
                                                   stages=summaries, rank_steps=rank_steps, max_rank_steps=maxima))
    with (output / "timing-samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]) if samples else [])
        writer.writeheader()
        writer.writerows(samples)
    return dict(measured_staged_calls=len(measured), measured_rank_steps=len(measured_steps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise SystemExit("Use the locked Linux CPython 3.12 toolchain with GPU access")
    root = Path(__file__).resolve().parents[1]
    output = fresh(args.output)
    if shutil.disk_usage(output).free < 12 * 1024 ** 3:
        raise RuntimeError("CUDA setup needs 12 GiB free on its output filesystem; select a disk-backed Linux path")
    (output / "setup-tmp").mkdir()
    source = source_identity(root)
    report = dict(status="failed", mode=args.mode, source=source, commands={},
                  cuda_training_acceptance=False,
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
            raise ValueError("behavioral source changed during setup; repeat from a fresh output with the wheelhouse")
        shutil.copytree(root / "tests/cuda_training", output / "checks", ignore=shutil.ignore_patterns("__pycache__"))
        report["commands"]["controller-lifecycle"] = run([python, "-I", output / "checks/controller_checks.py",
            "--output", output / "controller-checks"], output, "controller-checks", timeout=20)
        controller([python, "-I", "-X", "faulthandler", output / "checks/controller.py", "--mode", args.mode,
                    "--output", output / "runs"], output)
        checks = json.loads((output / "runs/checks.json").read_text())
        if checks["status"] != "passed" or len(checks["cases"]) != 10:
            raise ValueError("incomplete GPU training check set")
        if not all(case["status"] == "passed" and case["cleanup_complete"] for case in checks["cases"]):
            raise ValueError("GPU case failed or cleanup unknown")
        report.update(summarize(checks, output, args.mode))
        if source_identity(root)["sha256"] != source["sha256"]:
            raise ValueError("behavioral source changed during validation")
        report.update(status="passed", wheel_sha256=digest(wheels[0]), lock_sha256=digest(lock), cases=10,
                      cuda_training_acceptance=args.mode == "full", evidence_scope="single-GPU Linux/WSL functional training")
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
    print(f"PASS installed CUDA trainer {args.mode}; evidence={output}", flush=True)


if __name__ == "__main__":
    signal_handlers()
    main()
