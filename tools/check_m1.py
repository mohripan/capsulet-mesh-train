"""Run fresh native/package/training and four-container M1 acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import os
import signal
import subprocess
import sys
import time
import uuid

from check_training_containers import run_training
from validation_support import Docker, Resources, digest, fresh, monitor, read_json, remaining, signal_handlers, write_json


def source_identity(root):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    files = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root).decode().split("\0")
    areas = ("src/", "include/", "bindings/", "python/", "tests/", "tools/", "requirements/", "cmake/", ".github/")
    selected = {name: digest(root / name) for name in sorted(set(files)) if name and
                (name.startswith(areas) or name in ("CMakeLists.txt", "Dockerfile", "pyproject.toml", ".dockerignore"))
                and (root / name).is_file()}
    changed = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True)
    dirty = any(line[3:] in selected for line in changed.splitlines())
    return dict(revision=revision, dirty=dirty, files=selected,
                sha256=hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest())


def command(arguments, log, *, cwd, timeout, env=None):
    started = time.monotonic()
    print(f"Running: {' '.join(map(str, arguments))}; log={log}", flush=True)
    with Path(log).open("wb") as stream:
        process = subprocess.Popen(arguments, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
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
        raise RuntimeError(f"command exited {code}; see {log}")
    return dict(arguments=list(map(str, arguments)), elapsed_seconds=time.monotonic() - started, exit_code=0)


def native_script(root, output, script, image):
    output = fresh(output)
    resources = Resources(Docker(output), image)
    report = dict(status="failed")
    try:
        environment = os.environ.copy()
        environment["MESHTRAIN_VALIDATION_OWNER"] = resources.owner
        report.update(command(["sh", str(root / "tools" / script), image], output / "script.log",
                              cwd=root, timeout=60, env=environment), status="passed")
    finally:
        report["cleanup"] = resources.cleanup()
        if report["cleanup"]["status"] != "passed":
            report["status"] = "failed"
        write_json(output / "validation.json", report)
    if report["status"] != "passed":
        raise RuntimeError("native container script cleanup failed")
    return report


def run_helper(image, output, arguments, *, entrypoint, timeout=120, copies=(), expected_exit=0, init=False):
    output = fresh(output)
    docker = Docker(output)
    resources = Resources(docker, image)
    report = dict(schema=1, status="failed", image=resources.image, expected_exit=expected_exit)
    write_json(output / "validation.json", report)
    try:
        helper = resources.create("program", arguments, entrypoint=entrypoint, init=init)
        deadline = time.monotonic() + timeout
        docker.call(["start", helper], timeout=remaining(deadline))
        states = monitor(docker, [helper], deadline, require_success=False)
        report["states"] = states
        docker.logs(helper, output / "logs/program", timeout=remaining(deadline))
        for source, destination in copies:
            path = output / destination
            path.parent.mkdir(parents=True, exist_ok=True)
            docker.call(["cp", f"{helper}:{source}", str(path)], timeout=remaining(deadline))
        if states[0]["state"]["ExitCode"] != expected_exit:
            raise RuntimeError(f"helper exited {states[0]['state']['ExitCode']}, expected {expected_exit}")
        report["status"] = "passed"
    except BaseException as error:
        report["primary_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["cleanup"] = resources.cleanup()
        if report["cleanup"]["status"] != "passed":
            report["status"] = "failed"
        write_json(output / "validation.json", report)
    if report["status"] != "passed":
        raise RuntimeError(f"helper cleanup failed: {output}")
    return report


def verify_selection(selected, label):
    names = {item["name"] for item in selected["tests"]}
    regular = {"core_contracts", "wire_protocol", "communicator_contracts", "centralized_collectives",
               "tcp_processes", "resolver_deadline", "installed_consumer"}
    if (label == "broad" and len(names) != 9) or (label == "existing" and not regular <= names):
        raise ValueError(f"incorrect {label} CTest selection")


def run(output, no_cache):
    root = Path(__file__).resolve().parents[1]
    output = fresh(output)
    source = source_identity(root)
    report = dict(schema=1, status="failed", source=source, no_build_layer_cache=no_cache,
                  host=dict(python=sys.version, platform=platform.platform()), components={})
    write_json(output / "validation.json", report)
    write_json(output / "source.json", source)
    try:
        docker = Docker(output / "preflight")
        report["docker"] = json.loads(docker.call(["version", "--format", "{{json .}}"], timeout=15).stdout)
        if report["docker"]["Server"]["Os"] != "linux":
            raise ValueError("M1 requires a Linux Docker daemon")
        suffix = uuid.uuid4().hex[:12]
        images = {}
        for target in ("native-test", "native-portable-test", "python-bindings-test", "trainer-smoke-test", "trainer-container"):
            image = f"meshtrain-{target}:validation-{suffix}"
            arguments = ["docker", "build", "--target", target, "-t", image, "--progress", "plain"]
            if no_cache:
                arguments.append("--no-cache")
            arguments.append(str(root))
            report["components"][target] = command(arguments, output / f"build-{target}.log", cwd=root, timeout=1200)
            images[target] = image
            write_json(output / "validation.json", report)
        native = images["native-test"]
        for tool, arguments in (("c++", ["--version"]), ("cmake", ["--version"]), ("ninja", ["--version"])):
            run_helper(native, output / f"version-{tool.replace('+', 'p')}", arguments, entrypoint=tool)
        ctest = ["--test-dir", "/workspace/build/native", "-L", "m1_integration", "-V", "--output-on-failure",
                 "--output-log", "/tmp/native-faults.log", "--output-junit", "/tmp/native-faults.xml"]
        report["components"]["native-faults"] = run_helper(native, output / "native-faults", ctest,
            entrypoint="ctest", timeout=150, copies=[("/workspace/build/native/tests/fault-evidence", "evidence"),
            ("/tmp/native-faults.log", "ctest.log"), ("/tmp/native-faults.xml", "junit.xml")])
        fault_results = [read_json(path) for path in (output / "native-faults/evidence").rglob("result.json")]
        if len(fault_results) != 28 or any(item["status"] != "passed" or item["case_elapsed_ms"] >= 5000 for item in fault_results):
            raise ValueError("native matrix/repeated evidence incomplete or outside five-second bound")
        write_json(output / "native-faults/results.json", fault_results)
        for label, option in (("broad", "-L"), ("existing", "-LE")):
            folder = output / f"selection-{label}"
            run_helper(native, folder, ["--test-dir", "/workspace/build/native", option, "m1_integration", "--show-only=json-v1"], entrypoint="ctest")
            selected = json.loads((folder / "logs/program/stdout.log").read_text())
            verify_selection(selected, label)
        executable = "/workspace/build/native/tests/meshtrain_failure_tests"
        run_helper(native, output / "missing-phase", ["--case", "startup-root-stop", "--omit-phase"], entrypoint=executable,
                   expected_exit=2, init=True, copies=[("/workspace/fault-evidence", "evidence")])
        if "required native phase marker missing" not in (output / "missing-phase/logs/program/stderr.log").read_text():
            raise ValueError("missing-phase harness did not fail as expected")
        run_helper(native, output / "watchdog-expiry", ["--watchdog-expiry"], entrypoint=executable, expected_exit=137, init=True, timeout=10)
        if "independent watchdog expired active=1" not in (output / "watchdog-expiry/logs/program/stderr.log").read_text():
            raise ValueError("independent watchdog did not demonstrate expiry")
        for script in ("check_tcp_containers.sh", "check_collective_containers.sh"):
            report["components"][script] = native_script(root, output / script.removesuffix(".sh"), script, native)
        local = "/tmp/meshtrain-localhost"
        copies = [(f"{local}/{name}", name) for name in ("run", "build.log", "dependencies.log", "install.log", "runtime.txt", "toolchain.json", "training.log", "checks.log", "dist")]
        report["components"]["localhost-full"] = run_helper(images["trainer-container"], output / "localhost-full",
            ["tools/check_trainer.py", "--mode", "full", "--output", local], entrypoint="python", timeout=300, copies=copies)
        local_report = read_json(output / "localhost-full/run/verification.json")
        if not local_report["full_acceptance"]:
            raise ValueError("localhost full trainer acceptance missing")
        report["components"]["controller-checks"] = command([sys.executable, str(root / "tests/validation/checks.py"),
            "--image", images["trainer-container"], "--output", str(output / "controller-checks")],
            output / "controller-checks.log", cwd=root, timeout=300)
        report["components"]["four-container-full"] = run_training(images["trainer-container"], "full", output / "four-container-full")
        if source_identity(root)["sha256"] != source["sha256"]:
            raise ValueError("behavioral source changed during validation")
        report.update(status="passed", native_cases=len(fault_results), full_acceptance=True)
    except BaseException as error:
        report.update(status="failed", full_acceptance=False, primary_error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "validation.json", report)
    print(f"PASS M1 acceptance: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info < (3, 12):
        raise SystemExit("Use Linux/WSL host Python 3.12+ and a Linux Docker daemon.")
    signal_handlers()
    run(args.output, args.no_cache)
