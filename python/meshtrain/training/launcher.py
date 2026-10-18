from __future__ import annotations

import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

from .artifacts import write_json


class WorkerGroupError(RuntimeError):
    pass


def stop_workers(processes, grace):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + grace
    for process in processes:
        remaining = max(0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        if process.poll() is None:
            process.kill()
    for process in processes:
        process.wait(timeout=2)


def supervise(commands, output, timeout=120, grace=5, spawn=subprocess.Popen, *, check_interrupted=None):
    if not commands or timeout <= 0 or grace < 0:
        raise ValueError("expected commands, positive timeout and nonnegative termination grace")
    root = Path(output)
    processes, streams, exits = [], [], {}
    selector = selectors.DefaultSelector()
    primary = None
    primary_rank = None
    started = time.monotonic()

    def drain(wait):
        for key, _ in selector.select(wait):
            rank, name, log = key.data
            chunk = os.read(key.fileobj.fileno(), 65536)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            log.write(chunk)
            log.flush()
            console = sys.stderr if name == "stderr" else sys.stdout
            for line in chunk.decode("utf-8", errors="replace").splitlines():
                print(f"[rank {rank} {name}] {line}", file=console, flush=True)

    try:
        for rank, command in enumerate(commands):
            directory = root / "ranks" / str(rank)
            directory.mkdir(parents=True, exist_ok=True)
            process = spawn(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            processes.append(process)
            streams.extend((process.stdout, process.stderr))
            if check_interrupted is not None:
                write_json(root / "processes.json", dict(pids=[p.pid for p in processes]))
                check_interrupted()
            for name in ("stdout", "stderr"):
                pipe = getattr(process, name)
                log = (directory / f"{name}.log").open("wb")
                streams.append(log)
                selector.register(pipe, selectors.EVENT_READ, (rank, name, log))
        write_json(root / "processes.json", dict(pids=[process.pid for process in processes]))
        while len(exits) < len(processes) or selector.get_map():
            if check_interrupted is not None:
                check_interrupted()
            drain(0.02)
            for rank, process in enumerate(processes):
                code = process.poll()
                if code is not None and rank not in exits:
                    exits[rank] = code
                    if code and primary is None:
                        primary = f"rank {rank} exited with status {code}"
                        primary_rank = rank
            if primary:
                raise WorkerGroupError(primary)
            if time.monotonic() - started > timeout:
                raise WorkerGroupError(f"worker group exceeded {timeout:g}s wall timeout")
        result = dict(status="passed", exits=exits, pids=[process.pid for process in processes])
        if check_interrupted is not None:
            check_interrupted()
            result["reaped"] = [process.wait(timeout=0) == 0 for process in processes]
        return result
    except BaseException as error:
        primary = primary or f"{type(error).__name__}: {error}"
        stop_workers(processes, grace)
        deadline = time.monotonic() + 1
        while selector.get_map() and time.monotonic() < deadline:
            drain(0.02)
        exits = {rank: process.returncode for rank, process in enumerate(processes)}
        write_json(root / "group.json", dict(status="failed", primary_error=primary,
                                            exits=exits, pids=[process.pid for process in processes],
                                            secondary_errors=[f"rank {rank} exited with status {code}"
                                                              for rank, code in exits.items() if code and rank != primary_rank]))
        print(f"worker group failed: {primary}; exits={exits}", file=sys.stderr, flush=True)
        raise
    finally:
        if any(process.poll() is None for process in processes):
            stop_workers(processes, grace)
        selector.close()
        for stream in streams:
            stream.close()


def worker_commands(output, config):
    root = Path(output).resolve()
    return [[sys.executable, "-I", "-u", "-m", "meshtrain.training", "worker",
             "--config", str(root / "config.json"), "--rank", str(rank), "--output", str(root)]
            for rank in range(config["workload"]["world_size"])]
