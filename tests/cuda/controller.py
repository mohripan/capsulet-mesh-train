"""Bound fresh GPU worker cases and retain owned-process evidence."""

import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import sys
import time
import uuid

_interrupted = None


def install_signal_handlers():
    def record(number, _):
        global _interrupted
        _interrupted = number
    signal.signal(signal.SIGINT, record)
    signal.signal(signal.SIGTERM, record)


def check_interrupted():
    if _interrupted is not None:
        raise InterruptedError(f"GPU controller received signal {_interrupted}")


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def case(root, mode, world, *, spawn=subprocess.Popen, execution_budget=40):
    output = root / f"{mode}-{world}"
    output.mkdir()
    report = dict(status="failed", mode=mode, world_size=world, cleanup_complete=False)
    processes, channels, logs, reserved = [], [], [], []
    selector = selectors.DefaultSelector()
    started = time.monotonic()
    try:
        check_interrupted()
        for _ in range(world):
            reservation = socket.socket()
            reservation.bind(("127.0.0.1", 0))
            reserved.append(reservation)
        config = dict(mode=mode, world_size=world, run_id=uuid.uuid4().hex,
                      endpoints=[["127.0.0.1", item.getsockname()[1]] for item in reserved])
        write(output / "config.json", config)
        for reservation in reserved:
            reservation.close()
        for rank in range(world):
            parent, child = socket.socketpair()
            channels.append(parent)
            stdout, stderr = (output / f"rank-{rank}.out").open("wb"), (output / f"rank-{rank}.err").open("wb")
            logs.extend((stdout, stderr))
            try:
                processes.append(spawn([sys.executable, "-I", "-u", str(Path(__file__).with_name("worker.py")),
                    "--config", str(output / "config.json"), "--rank", str(rank), "--control", str(child.fileno()),
                    "--output", str(output)], stdout=stdout, stderr=stderr, pass_fds=(child.fileno(),), start_new_session=True))
                check_interrupted()
            finally:
                child.close()
            selector.register(parent, selectors.EVENT_READ, (rank, bytearray()))
        write(output / "processes.json", dict(pids=[p.pid for p in processes]))
        markers, warm, released, action_sent = [], set(), False, False
        while True:
            check_interrupted()
            if time.monotonic() - started >= execution_budget:
                raise TimeoutError("GPU case exceeded worker execution budget")
            for key, _ in selector.select(0.02):
                rank, buffered = key.data
                chunk = key.fileobj.recv(4096)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffered.extend(chunk)
                while b"\n" in buffered:
                    line, _, rest = buffered.partition(b"\n")
                    buffered[:] = rest
                    marker = line.decode()
                    markers.append(dict(rank=rank, phase=marker))
                    if marker == "warm":
                        warm.add(rank)
            if len(warm) == world and not released:
                for channel in channels:
                    channel.sendall(b"G")
                released = True
            phases = {(entry["rank"], entry["phase"]) for entry in markers}
            trigger = "waiting" if mode == "lifetime" else "native-ready"
            if world == 2 and (1, "held") in phases and (0, trigger) in phases and not action_sent:
                channels[1].sendall(b"C" if mode == "closed" else b"G")
                action_sent = True
            if mode == "stall" and (0, "failed") in phases:
                channels[1].sendall(b"R")
                mode = "stall-released"
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"GPU worker failed: {codes}; inspect rank stderr")
            if all(code == 0 for code in codes):
                break
        results = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(world)]
        assert all(result["status"] == "passed" and result["cleanup_complete"] for result in results)
        report.update(status="passed", markers=markers, results=results)
    except BaseException as error:
        report["primary_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        cleanup_errors = []
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = min(started + 42, time.monotonic() + 0.5)
        for process in processes:
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=max(0.01, started + 44.5 - time.monotonic()))
            except subprocess.TimeoutExpired:
                cleanup_errors.append(f"owned PID {process.pid} not reaped; cleanup unknown")
        for resource in [*reserved, *channels, *logs]:
            resource.close()
        selector.close()
        report.update(exits=[process.returncode for process in processes], pids=[process.pid for process in processes],
                      cleanup_errors=cleanup_errors,
                      cleanup_complete=not cleanup_errors and all(process.poll() is not None for process in processes),
                      elapsed_seconds=time.monotonic() - started)
        if report["elapsed_seconds"] >= 45 or not report["cleanup_complete"]:
            report["status"] = "failed"
        write(output / "case.json", report)
    if report["status"] != "passed":
        raise RuntimeError("GPU case cleanup/bound failed")
    print(f"PASS {output.name}: {report['elapsed_seconds']:.3f}s; exits={report['exits']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", choices=("unit", "matrix", "lifetime", "closed", "stall", "policy"))
    parser.add_argument("--world-size", type=int, choices=(1, 2))
    args = parser.parse_args()
    args.output.mkdir()
    configurations = [("unit", 1), ("matrix", 1), ("matrix", 2), ("lifetime", 2), ("closed", 2), ("stall", 2), ("policy", 1)]
    if args.only:
        configurations = [(args.only, args.world_size or 1)]
    result = dict(status="failed", cases=[])
    try:
        for mode, world in configurations:
            result["cases"].append(case(args.output, mode, world))
        check_interrupted()
        result["status"] = "passed"
    finally:
        write(args.output / "checks.json", result)


if __name__ == "__main__":
    install_signal_handlers()
    main()
