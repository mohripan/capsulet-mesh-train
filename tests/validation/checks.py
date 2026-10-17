"""Exercise real Docker failure, isolation and ownership boundaries."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import check_training_containers as trainer
from validation_support import Docker, GroupFailure, Resources, digest, fresh, monitor, read_json, remaining, signal_handlers, write_json


def fixture(kind, image, output):
    holder = {}
    original_arguments = trainer.worker_arguments

    def worker_arguments(rank):
        arguments = original_arguments(rank)
        if kind == "bad-config" and rank == 1:
            arguments[arguments.index("--config") + 1] = "/tmp/missing-worker-config.json"
        return arguments

    trainer.worker_arguments = worker_arguments

    def observe(stage, resources, workers):
        holder["resources"] = resources
        if kind == "spawn" and stage == "created-worker-0":
            original_call = resources.docker.call

            def call(arguments, **kwargs):
                if arguments[0] == "create" and any(str(item).endswith("-worker-1") for item in arguments):
                    subprocess.run(["/meshtrain-deliberately-missing-docker"], check=True)
                return original_call(arguments, **kwargs)

            resources.docker.call = call
        if kind == "interrupt" and stage == "created-worker-1":
            os.kill(os.getpid(), signal.SIGINT)
        if kind in ("kill", "timeout") and stage == "started-worker-0":
            state = resources.docker.states([workers[0]])[0]["state"]
            assert state["Running"], state
            resources.docker.call(["kill", "--signal", "KILL" if kind == "kill" else "STOP", workers[0]])
        if kind == "cleanup" and stage == "resources":
            original_call = resources.docker.call
            injected = False

            def call(arguments, **kwargs):
                nonlocal injected
                if arguments[:2] == ["network", "rm"] and not injected:
                    injected = True
                    return original_call(["network", "rm", resources.prefix + "-missing"], **kwargs)
                return original_call(arguments, **kwargs)

            resources.docker.call = call

    try:
        trainer.run_training(image, "smoke", output, worker_timeout=1.5 if kind == "timeout" else 120,
                             observer=observe)
    except (RuntimeError, OSError, KeyboardInterrupt):
        pass
    else:
        raise AssertionError("negative container job unexpectedly succeeded")
    report = read_json(output / "validation.json")
    assert report["status"] == "failed" and not report["full_acceptance"], report
    if kind == "cleanup":
        assert report["numerical_status"] == "passed" and report["cleanup"]["status"] == "failed", report
        recovery = holder["resources"].cleanup()
        assert recovery["status"] == "passed", recovery
        write_json(output / "test-recovery.json", recovery)
    else:
        assert report["cleanup"]["status"] == "passed", report
        expected = dict(spawn="FileNotFoundError", interrupt="KeyboardInterrupt", kill="worker-0", timeout="deadline", **{"bad-config": "worker-1"})[kind]
        assert expected in report["primary_error"], report
    if kind == "bad-config":
        assert "missing-worker-config.json" in (output / "logs/worker-1/stderr.log").read_text()
    owner = holder["resources"]
    assert owner.owned(time.monotonic() + 5) == ([], [])
    write_json(output / "fixture-result.json", dict(status="passed", kind=kind))


def corrupt_trace(root):
    path = root / "ranks/0/trace.npz"
    with zipfile.ZipFile(path) as archive:
        arrays = {name: archive.read(name) for name in archive.namelist()}
    data = arrays["gradients.npy"]
    assert data[:6] == b"\x93NUMPY"
    width = 2 if data[6] == 1 else 4
    header_size = int.from_bytes(data[8:8 + width], "little")
    offset = 8 + width + header_size
    header = ast.literal_eval(data[8 + width:offset].decode("latin1"))
    assert header["descr"] == "<f4" and not header["fortran_order"]
    arrays["gradients.npy"] = data[:offset] + struct.pack("<f", 10) + data[offset + 4:]
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in arrays.items():
            archive.writestr(name, value)
    metadata = read_json(root / "ranks/0/complete.json")
    metadata["files"]["trace.npz"] = digest(path)
    write_json(root / "ranks/0/complete.json", metadata)


def suite(image, output):
    output = fresh(output)
    evidence = dict(schema=1, status="failed", fixtures=[])
    write_json(output / "checks.json", evidence)
    sentinel = Resources(Docker(output / "sentinel"), image)
    try:
        network = sentinel.network()
        name = sentinel.create("sentinel", ["-I", "-c", "import time; time.sleep(600)"],
                               entrypoint=trainer.PYTHON, network=network)
        sentinel.docker.call(["start", name])
        for iteration in (1, 2):
            trainer.run_training(image, "smoke", output / f"smoke-{iteration}")
        for kind in ("bad-config", "kill", "spawn", "interrupt", "timeout", "cleanup"):
            log = output / f"{kind}.log"
            with log.open("w") as stream:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--fixture", kind,
                                            "--image", image, "--output", str(output / kind)],
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=55)
                    assert code == 0, log.read_text()
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                        manifest = output / kind / "resources.json"
                        if manifest.exists():
                            Resources.restore(Docker(manifest.parent), read_json(manifest)).cleanup()
            evidence["fixtures"].append(read_json(output / kind / "fixture-result.json"))
            assert sentinel.docker.states([name])[0]["state"]["Running"]
            assert sentinel.docker.call(["network", "inspect", network]).returncode == 0
        tampered = output / "tampered"
        original_hash = digest(output / "smoke-1/run/verification.json")
        shutil.copytree(output / "smoke-1/run", tampered / "run")
        corrupt_trace(tampered / "run")
        resources = Resources(Docker(tampered), image)
        try:
            try:
                trainer.verify_collected(resources, tampered / "run", tampered / "run")
            except GroupFailure:
                verifier = resources.containers[-1]
                resources.docker.call(["cp", f"{verifier}:{trainer.RUN}/verification.json", str(tampered / "run/verification.json")])
            else:
                raise AssertionError("finite corrupted gradient was accepted")
            report = read_json(tampered / "run/verification.json")
            assert report["status"] == "failed" and report["failure_count"] > 0, report
        finally:
            assert resources.cleanup()["status"] == "passed"
        assert digest(output / "smoke-1/run/verification.json") == original_hash
        try:
            trainer.run_training(image, "smoke", output / "smoke-1")
        except ValueError:
            pass
        else:
            raise AssertionError("nonempty output was accepted")
        evidence.update(status="passed", sentinel_preserved=True, smoke_attempts=2, corruption_rejected=True)
    finally:
        cleanup = sentinel.cleanup()
        evidence["sentinel_cleanup"] = cleanup
        if cleanup["status"] != "passed":
            evidence["status"] = "failed"
        write_json(output / "checks.json", evidence)
    assert evidence["status"] == "passed", evidence
    print(f"PASS: real Docker controller checks; evidence={output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", choices=("bad-config", "kill", "spawn", "interrupt", "timeout", "cleanup"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    signal_handlers()
    if arguments.fixture:
        fixture(arguments.fixture, arguments.image, arguments.output)
    else:
        suite(arguments.image, arguments.output)
