"""Verify a native four-container ring and ownership-scoped failure cleanup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from validation_support import Docker, GroupFailure, Resources, fresh, monitor, remaining, signal_handlers, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=("healthy", "worker-exit"), default="healthy")
    args = parser.parse_args()
    output = fresh(args.output)
    signal_handlers()
    docker = Docker(output)
    resources = Resources(docker, args.image)
    report = dict(schema=1, scenario=args.scenario, status="failed", topology="four Linux containers on one Docker host")
    started = time.monotonic()
    deadline = started + 60
    try:
        network = resources.network()
        names = []
        for rank in range(4):
            arguments = ["--worker", "container", rank, 4, 45000, "meshtrain-ring-"]
            entrypoint = "/workspace/build/native/tests/meshtrain_ring_tests"
            if args.scenario == "worker-exit" and rank == 2:
                arguments, entrypoint = ["-c", "exit 23"], "/bin/sh"
            names.append(resources.create(f"rank-{rank}", arguments, entrypoint=entrypoint,
                                          network=network, alias=f"meshtrain-ring-{rank}", init=True))
        docker.call(["start", *names], timeout=remaining(deadline))
        try:
            states = monitor(docker, names, deadline)
        except GroupFailure as error:
            if args.scenario != "worker-exit":
                raise
            states = error.states
            if not any(item["state"]["ExitCode"] == 23 for item in states):
                raise RuntimeError("expected worker exit was not observed") from error
            report["expected_failure"] = str(error)
        else:
            if args.scenario != "healthy":
                raise RuntimeError("induced worker exit unexpectedly succeeded")
        if any(item["image"] != resources.image or item["mounts"] or item["pid_mode"] for item in states):
            raise RuntimeError("worker image or isolation differs")
        report.update(status="passed", workers=states)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["execution_seconds"] = time.monotonic() - started
        report["cleanup"] = resources.cleanup()
        if report["cleanup"]["status"] != "passed":
            report["status"] = "failed"
        write_json(output / "result.json", report)
    if report["status"] != "passed":
        raise RuntimeError("ring container cleanup did not pass")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
