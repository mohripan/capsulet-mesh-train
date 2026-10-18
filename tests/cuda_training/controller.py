"""Run installed GPU trainer phases as directly owned processes."""

import argparse
import json
from pathlib import Path
import selectors
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from processes import OwnedProcesses, absent, check_interrupted, install_signals, write


def cli(*arguments):
    return [sys.executable, "-I", "-u", "-m", "meshtrain.training", *map(str, arguments)]


def fixture(mode, output, rank=0, control=None):
    command = [sys.executable, "-I", "-u", str(Path(__file__).with_name("fixtures.py")), mode,
               "--output", str(output), "--rank", str(rank)]
    return command + (["--control", str(control)] if control is not None else [])


def profile(root, name, world, steps, memory, *, mode="training", run=False):
    group = OwnedProcesses(root / name)
    output = group.output / "run"
    error, channels, selector = None, [], None
    try:
        prepare = cli("run" if run else "prepare", "--output", output, "--world-size", world, "--steps", steps,
                      "--device", "cuda:0", "--host-memory", memory)
        group.wait([group.start(prepare, "run" if run else "prepare")])
        if run:
            nested = json.loads((output / "group.json").read_text())
            assert nested["status"] == "passed" and all(nested["reaped"])
            assert all(absent(pid) for pid in nested["pids"])
        else:
            children, observed, buffered = [], set(), {}
            selector = selectors.DefaultSelector()
            for rank in range(world):
                if mode == "known":
                    parent, child = socket.socketpair()
                    parent.setblocking(False)
                    channels.extend((parent, child))
                    selector.register(parent, selectors.EVENT_READ, rank)
                    buffered[rank] = bytearray()
                    children.append(group.start(fixture(mode, output, rank, child.fileno()), f"rank-{rank}",
                                                pass_fds=(child.fileno(),)))
                    child.close()
                elif mode == "training":
                    children.append(group.start(cli("worker", "--output", output, "--config", output / "config.json",
                                                    "--rank", rank), f"rank-{rank}"))
                else:
                    children.append(group.start(fixture(mode, output, rank), f"rank-{rank}"))

            def progress():
                for key, _ in selector.select(0):
                    payload = key.fileobj.recv(4096)
                    if not payload:
                        selector.unregister(key.fileobj)
                    buffered[key.data].extend(payload)
                    while b"\n" in buffered[key.data]:
                        line, _, rest = buffered[key.data].partition(b"\n")
                        buffered[key.data][:] = rest
                        observed.add((key.data, line.decode()))
                if (0, "waiting") in observed and (1, "held") in observed and (1, "released") not in observed:
                    channels[2].sendall(b"G")
                    observed.add((1, "released"))

            try:
                group.wait(children, cap=120, progress=progress if mode == "known" else None)
            except RuntimeError as failure:
                if mode != "failure":
                    raise
                group.report["expected_failure"] = str(failure)
            finally:
                selector.close()
            if mode == "failure":
                for process in children:
                    if process.poll() is None:
                        try:
                            process.wait(timeout=min(5, max(0, group.deadline - time.monotonic())))
                        except subprocess.TimeoutExpired:
                            raise AssertionError("peer failure did not stop both actual workers")
                for rank in range(world):
                    state = json.loads((output / f"rank-{rank}-updates.json").read_text())
                    assert state == [dict(step=0, completed=True)], state
                    failure = json.loads((output / f"rank-{rank}-failure.json").read_text())
                    assert failure["code"] and not (output / f"ranks/{rank}/complete.json").exists()
                assert not (output / "verification.json").exists()
            elif mode in ("unit", "known", "overflow"):
                group.report["checks"] = [json.loads((output / f"rank-{rank}-checks.json").read_text()) for rank in range(world)]
                assert all(check["status"] == "passed" for check in group.report["checks"])
            else:
                write(output / "group.json", dict(status="passed", pids=[p.pid for p in children],
                                                   exits=[p.returncode for p in children], reaped=[True] * world))
                group.wait([group.start(cli("verify", "--output", output), "verify")])
        if mode == "training" or mode == "stream" or run:
            report = json.loads((output / "verification.json").read_text())
            assert report["status"] == "passed" and not report["full_acceptance"]
            assert report["cuda_training_acceptance"] == (world == 2 and steps == 100)
            group.report["verification"] = report
            group.report["cuda"] = [json.loads((output / f"ranks/{rank}/cuda.json").read_text()) for rank in range(world)]
            if mode == "stream":
                assert all(not item["preflight"]["default_stream"] for item in group.report["cuda"])
        group.report.update(status="passed", name=name, world_size=world, steps=steps, host_memory=memory)
        check_interrupted()
    except BaseException as failure:
        error = failure
        raise
    finally:
        if selector is not None:
            selector.close()
        for channel in channels:
            channel.close()
        group.close(error)
    return group.report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    install_signals()
    result = dict(status="failed", cases=[])
    specifications = [("unit", 1, 3, "pinned", "unit"), ("known", 2, 3, "pinned", "known"),
                      ("overflow", 2, 3, "pinned", "overflow"), ("failure", 2, 3, "pinned", "failure"),
                      ("identity-pinned", 1, 3, "pinned", "training"),
                      ("identity-pageable", 1, 3, "pageable", "training"),
                      ("stream", 2, 3, "pinned", "stream")]
    specifications += [(f"parity-{memory}", 2, 100 if args.mode == "full" else 3, memory, "training")
                       for memory in ("pinned", "pageable")]
    try:
        for name, world, steps, memory, mode in specifications:
            result["cases"].append(profile(args.output, name, world, steps, memory, mode=mode))
        result["cases"].append(profile(args.output, "convenience-run", 2, 3, "pinned", run=True))
        first = args.output / "identity-pinned/run"
        group = OwnedProcesses(args.output / "corruption")
        error = None
        try:
            group.wait([group.start(fixture("corrupt", first), "corruption")])
            group.report["status"] = "passed"
            result["corruption"] = json.loads((first / "corruption.json").read_text())
        except BaseException as failure:
            error = failure
            raise
        finally:
            group.close(error)
        group = OwnedProcesses(args.output / "launcher-checks")
        error = None
        try:
            group.wait([group.start(fixture("launcher", first), "launcher")])
            group.report["status"] = "passed"
            result["launcher"] = json.loads((first / "launcher.json").read_text())
        except BaseException as failure:
            error = failure
            raise
        finally:
            group.close(error)
        result["status"] = "passed"
    finally:
        write(args.output / "checks.json", result)


if __name__ == "__main__":
    main()
