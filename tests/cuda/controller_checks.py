"""Exercise controller cleanup with actual owned processes, without GPU work."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from controller import case, install_signal_handlers, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    install_signal_handlers()
    evidence = []
    for kind in ("timeout", "exit", "spawn", "interrupt"):
        owned = []

        def spawn(arguments, **kwargs):
            if kind == "spawn" and owned:
                raise OSError("injected later spawn failure")
            code = "import sys; sys.exit(7)" if kind == "exit" and owned else "import time; time.sleep(60)"
            process = subprocess.Popen([sys.executable, "-I", "-c", code], **kwargs)
            owned.append(process)
            if kind == "interrupt" and len(owned) == 2:
                os.kill(os.getpid(), signal.SIGINT)
            return process

        try:
            try:
                case(args.output, kind, 2, spawn=spawn, execution_budget=0.2 if kind == "timeout" else 40)
            except (TimeoutError, RuntimeError, OSError, KeyboardInterrupt):
                pass
            else:
                raise AssertionError("negative controller case unexpectedly passed")
            report = json.loads((args.output / f"{kind}-2/case.json").read_text())
            assert report["status"] == "failed" and report["cleanup_complete"]
            expected = dict(timeout="TimeoutError:", exit="RuntimeError:", spawn="OSError:", interrupt="InterruptedError:")
            assert report["primary_error"].startswith(expected[kind])
            if kind == "exit":
                assert 7 in report["exits"]
            for process in owned:
                assert process.poll() is not None
                try:
                    os.kill(process.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise AssertionError("owned process remains after controller cleanup")
            evidence.append(dict(kind=kind, status="passed", original=report))
        finally:
            for process in owned:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
    write(args.output / "checks.json", dict(status="passed", cases=evidence, gpu_execution=False))


if __name__ == "__main__":
    main()
