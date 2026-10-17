"""Run the installed trainer across four independent Docker filesystems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from validation_support import Docker, Resources, fresh, monitor, read_json, remaining, signal_handlers, write_json

PYTHON = "/opt/meshtrain/runtime/bin/python"
RUN = "/tmp/meshtrain-run"


def worker_arguments(rank):
    return ["-I", "-u", "-m", "meshtrain.training", "worker", "--config", f"{RUN}/config.json",
            "--rank", str(rank), "--output", RUN]


def copy_into(docker, source, name, deadline):
    docker.call(["cp", f"{source}/.", f"{name}:{RUN}"], timeout=remaining(deadline))


def verify_collected(resources, bundle, output, suffix="verify"):
    docker = resources.docker
    deadline = time.monotonic() + 120
    verifier = resources.create(suffix, ["-I", "-u", "-m", "meshtrain.training", "verify", "--output", RUN],
                                entrypoint=PYTHON)
    copy_into(docker, bundle, verifier, deadline)
    docker.call(["start", verifier], timeout=remaining(deadline))
    monitor(docker, [verifier], deadline)
    docker.logs(verifier, Path(output) / "verifier", timeout=remaining(deadline))
    docker.call(["cp", f"{verifier}:{RUN}/verification.json", str(Path(output) / "verification.json")], timeout=remaining(deadline))
    docker.call(["cp", f"{verifier}:{RUN}/reference", str(Path(output) / "reference")], timeout=remaining(deadline))
    return read_json(Path(output) / "verification.json")


def run_training(image, mode, output, *, worker_timeout=120, observer=None):
    if mode not in ("smoke", "full") or worker_timeout <= 0:
        raise ValueError("expected smoke/full mode and a positive worker timeout")
    output = fresh(output)
    docker = Docker(output)
    report = dict(schema=1, status="failed", mode=mode, full_acceptance=False)
    write_json(output / "validation.json", report)
    resources = None
    try:
        resources = Resources(docker, image)
        report.update(owner=resources.owner, image=resources.image)
        if observer:
            observer("resources", resources, [])
        endpoints = [dict(host=f"worker-{rank}", port=29500) for rank in range(4)]
        preparation = resources.create("prepare", ["-I", "-u", "-m", "meshtrain.training", "prepare", "--output", RUN,
            "--world-size", "4", "--steps", "100" if mode == "full" else "3", "--local-batch-size", "8",
            "--endpoints", json.dumps(endpoints), "--run-id", resources.owner], entrypoint=PYTHON)
        deadline = time.monotonic() + 120
        docker.call(["start", preparation], timeout=remaining(deadline))
        monitor(docker, [preparation], deadline)
        bundle = output / "run"
        docker.call(["cp", f"{preparation}:{RUN}", str(bundle)], timeout=remaining(deadline))
        docker.call(["cp", f"{preparation}:/opt/meshtrain/installed.json", str(output / "installed.json")], timeout=remaining(deadline))
        config = read_json(bundle / "config.json")
        report.update(run_id=config["run_id"], context_fingerprint=config["context_fingerprint"], inputs=config["context"]["inputs"])
        network = resources.network()
        workers = []
        for rank in range(4):
            worker = resources.create(f"worker-{rank}", worker_arguments(rank), entrypoint=PYTHON,
                                      network=network, alias=f"worker-{rank}")
            workers.append(worker)
            copy_into(docker, bundle, worker, deadline)
            if observer:
                observer(f"created-worker-{rank}", resources, workers)
        states = docker.states(workers)
        if any(item["mounts"] or item["pid_mode"] for item in states):
            raise ValueError("worker must have independent storage and PID namespace")
        if any(item["image"] != resources.image for item in states):
            raise ValueError("worker image identity mismatch")
        write_json(output / "worker-configuration.json", states)
        deadline = time.monotonic() + worker_timeout
        for rank, worker in enumerate(workers):
            docker.call(["start", worker], timeout=remaining(deadline))
            if observer:
                observer(f"started-worker-{rank}", resources, workers)
        report["workers"] = monitor(docker, workers, deadline)
        deadline = time.monotonic() + 120
        (bundle / "ranks").mkdir()
        for rank, worker in enumerate(workers):
            docker.logs(worker, output / "ranks" / str(rank), timeout=remaining(deadline))
            docker.call(["cp", f"{worker}:{RUN}/ranks/{rank}", str(bundle / "ranks" / str(rank))], timeout=remaining(deadline))
        numerical = verify_collected(resources, bundle, bundle)
        if numerical["status"] != "passed" or numerical["full_acceptance"] != (mode == "full"):
            raise ValueError("unexpected numerical acceptance profile")
        report.update(numerical_status="passed", comparisons=len(numerical["checks"]),
                      full_acceptance=numerical["full_acceptance"], status="passed")
    except BaseException as error:
        report.update(status="failed", full_acceptance=False, primary_error=f"{type(error).__name__}: {error}")
        if hasattr(error, "states"):
            report["workers"] = error.states
        raise
    finally:
        if resources is not None:
            cleanup = resources.cleanup()
            report["cleanup"] = cleanup
            report["secondary_exits"] = [dict(name=item["name"], exit_code=item["state"]["ExitCode"])
                                         for item in cleanup["after_stop"] if item["state"]["ExitCode"]]
            if cleanup["status"] != "passed":
                report.update(status="failed", full_acceptance=False)
        write_json(output / "validation.json", report)
    if report["status"] != "passed":
        raise RuntimeError("container validation cleanup failed; see validation.json")
    print(f"PASS four-container {mode}: {report['comparisons']} comparisons; output={output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info < (3, 12):
        raise SystemExit("Use Linux/WSL host Python 3.12+ with a Linux Docker daemon.")
    signal_handlers()
    run_training(args.image, args.mode, args.output)


if __name__ == "__main__":
    main()
