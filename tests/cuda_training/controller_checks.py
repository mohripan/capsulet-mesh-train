"""Standard-library negative process tests with no GPU execution."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import processes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    processes.install_signals()
    records = []
    for kind in ("timeout", "exit", "spawn", "interrupt"):
        processes._interrupted = None
        owned = []

        def spawn(arguments, **kwargs):
            if kind == "spawn" and owned:
                raise OSError("injected later spawn failure")
            code = "raise SystemExit(7)" if kind == "exit" and owned else "import time; time.sleep(60)"
            process = subprocess.Popen([sys.executable, "-I", "-c", code], **kwargs)
            owned.append(process)
            if kind == "interrupt" and len(owned) == 2:
                os.kill(os.getpid(), signal.SIGINT)
            return process

        group = processes.OwnedProcesses(args.output / kind, budget=0.2 if kind == "timeout" else 10, spawn=spawn)
        error = None
        try:
            children = [group.start(["fixture"], f"rank-{rank}") for rank in range(2)]
            group.wait(children)
        except (OSError, RuntimeError, TimeoutError, InterruptedError) as failure:
            error = failure
        finally:
            group.close(error)
        assert error is not None
        assert group.report["cleanup_complete"] and all(processes.absent(p.pid) for p in owned)
        expected = dict(timeout=TimeoutError, exit=RuntimeError, spawn=OSError, interrupt=InterruptedError)
        assert type(error) is expected[kind], error
        if kind == "exit":
            assert 7 in [p.returncode for p in owned]
        records.append(dict(kind=kind, status="passed", original=group.report))
    processes.write(args.output / "checks.json", dict(status="passed", gpu_execution=False, cases=records))


if __name__ == "__main__":
    main()
