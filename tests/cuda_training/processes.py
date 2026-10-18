"""Own direct child groups through completion, interruption and teardown."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

_interrupted = None


def install_signals():
    def record(number, _):
        global _interrupted
        _interrupted = number
    signal.signal(signal.SIGINT, record)
    signal.signal(signal.SIGTERM, record)


def check_interrupted():
    if _interrupted is not None:
        raise InterruptedError(f"controller received signal {_interrupted}")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class OwnedProcesses:
    def __init__(self, output, *, budget=180, spawn=subprocess.Popen):
        self.output = Path(output)
        self.output.mkdir(parents=True)
        self.started, self.deadline = time.monotonic(), time.monotonic() + budget
        self._spawn = spawn
        self.processes, self.logs, self.commands = [], [], []
        self.report = dict(status="failed", cleanup_complete=False)

    def start(self, arguments, label, *, pass_fds=(), env=None):
        check_interrupted()
        logs = []
        for suffix in ("out", "err"):
            log = (self.output / f"{label}.{suffix}").open("wb")
            self.logs.append(log)
            logs.append(log)
        stdout, stderr = logs
        process = self._spawn(list(map(str, arguments)), stdout=stdout, stderr=stderr,
                              start_new_session=True, pass_fds=pass_fds, env=env)
        self.processes.append(process)
        self.commands.append(dict(arguments=list(map(str, arguments)), label=label, pid=process.pid))
        write(self.output / "processes.json", dict(commands=self.commands))
        check_interrupted()
        return process

    def wait(self, processes, *, cap=None, progress=None):
        deadline = min(self.deadline, time.monotonic() + cap) if cap else self.deadline
        while True:
            check_interrupted()
            if progress:
                progress()
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"owned command failed with exit codes {codes}; inspect logs")
            if all(code == 0 for code in codes):
                for process in processes:
                    process.wait(timeout=0)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("owned profile execution deadline expired")
            time.sleep(0.01)

    def close(self, error=None):
        cleanup_errors = []
        finished_by = min(self.deadline + 10, time.monotonic() + 10)
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except OSError as secondary:
                    cleanup_errors.append(f"TERM/{process.pid}: {secondary}")
        grace = min(finished_by - 5, time.monotonic() + 5)
        for process in self.processes:
            try:
                process.wait(timeout=max(0, grace - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as secondary:
                    cleanup_errors.append(f"KILL/{process.pid}: {secondary}")
        for process in self.processes:
            try:
                process.wait(timeout=max(0, finished_by - time.monotonic()))
            except subprocess.TimeoutExpired:
                cleanup_errors.append(f"PID {process.pid} not reaped; cleanup unknown")
        for log in self.logs:
            log.close()
        self.report.update(elapsed_seconds=time.monotonic() - self.started,
                           commands=[entry | dict(exit_code=p.returncode, reaped=p.returncode is not None)
                                     for entry, p in zip(self.commands, self.processes)],
                           cleanup_complete=not cleanup_errors and all(p.returncode is not None for p in self.processes),
                           cleanup_errors=cleanup_errors)
        if error is not None:
            self.report["primary_error"] = f"{type(error).__name__}: {error}"
        if not self.report["cleanup_complete"] or time.monotonic() > self.deadline + 10:
            self.report["status"] = "failed"
        try:
            write(self.output / "case.json", self.report)
        except OSError as secondary:
            if error is None:
                raise
            error.add_note(f"Secondary controller evidence failure: {secondary}")
        if error is None and self.report["status"] != "passed":
            raise RuntimeError("owned profile failed or cleanup is unknown")


def absent(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False
