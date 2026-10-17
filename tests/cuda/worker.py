"""Real GPU workers for numerical, lifetime and failure evidence."""

from dataclasses import asdict
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import socket
import sys
import threading
import time
from unittest.mock import patch
import weakref

import numpy as np
import torch

import meshtrain
from meshtrain import Communicator, Endpoint, EngineError, RunConfig
import meshtrain.cuda_adapter as cuda


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def send(channel, marker):
    channel.sendall((marker + "\n").encode())


def receive(channel):
    value = channel.recv(1)
    if value not in (b"G", b"C", b"R"):
        raise RuntimeError(f"invalid controller release: {value!r}")
    return value


def preflight():
    if torch.__version__ != "2.14.1+cu132" or torch.version.cuda is None:
        raise RuntimeError(f"Expected CUDA Torch 2.14.1+cu132, got {torch.__version__} / {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime cannot access a GPU; check the supported driver and WSL GPU visibility")
    assert np.__version__ == "2.5.4"
    for path in (meshtrain.__file__, meshtrain._native.__file__):
        assert Path(path).resolve().is_relative_to(Path(sys.prefix).resolve()), path
    torch.cuda.set_device(0)
    value = torch.arange(1025, device="cuda:0", dtype=torch.float32)
    result = value.square().add_(1).cpu()
    assert torch.equal(result, torch.arange(1025, dtype=torch.float32).square().add_(1))
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    return dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                python=sys.version, package=meshtrain.__file__, extension=meshtrain._native.__file__,
                gpu_name=properties.name, gpu_uuid=str(properties.uuid), total_memory=properties.total_memory,
                capability=list(torch.cuda.get_device_capability(0)), architectures=torch.cuda.get_arch_list(),
                device=0, components={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                                      if d.metadata["Name"].startswith(("cuda-", "nvidia-", "triton"))})


def native(config, rank):
    return Communicator(RunConfig(rank=rank, world_size=config["world_size"], run_id=config["run_id"],
        context_fingerprint="cuda-adapter-v1", endpoints=[Endpoint(*endpoint) for endpoint in config["endpoints"]],
        startup_timeout_ms=10000, collective_timeout_ms=1000 if config["mode"] in ("closed", "stall") else 10000))


def timing_record(timing, rank, warmup):
    record = asdict(timing) | dict(rank=rank, warmup=warmup)
    phases = [record[k] for k in ("ready_wait_ms", "d2h_ms", "native_ms", "h2d_ms")]
    assert all(np.isfinite(v) and v >= 0 for v in phases)
    assert np.isfinite(timing.total_ms) and sum(phases) <= timing.total_ms + 1e-6
    return record


def matrix(communicator, rank, world):
    samples, numerical = [], []
    for pinned in (False, True):
        for count in (0, 1, 1025, 1048576):
            workspace = cuda.CudaStagingAdapter(count, device="cuda:0", pin_memory=pinned)
            value = torch.empty(count, device="cuda:0")
            for random in (False, True):
                arrays = [np.random.default_rng(710 + r).uniform(-0.25, 0.25, count).astype(np.float32)
                          if random else np.full(count, r + 1, dtype=np.float32) for r in range(world)]
                expected = np.sum(np.stack(arrays).astype(np.float64), axis=0)
                value.copy_(torch.from_numpy(arrays[rank]))
                sequence = communicator.next_sequence
                record = workspace.all_reduce_sum_(value, communicator)
                assert communicator.next_sequence == sequence + 1 and torch.cuda.current_stream().query()
                actual = value.cpu().numpy()
                error = np.abs(actual.astype(np.float64) - expected)
                assert np.all(error <= 1e-6 + 1e-5 * np.abs(expected)) if random else np.array_equal(actual, expected)
                numerical.append(dict(count=count, pin_memory=pinned, random=random,
                                      max_absolute_error=float(error.max(initial=0)), sequence=communicator.next_sequence,
                                      timing=timing_record(record, rank, True)))
        for count in (1025, 1048576):
            workspace = cuda.CudaStagingAdapter(count, device="cuda:0", pin_memory=pinned)
            value = torch.empty(count, device="cuda:0")
            for iteration in range(15):
                value.fill_(rank + 1)
                samples.append(timing_record(workspace.all_reduce_sum_(value, communicator), rank, iteration < 5)
                               | dict(iteration=iteration))
                assert torch.all(value == world * (world + 1) / 2).item()
    return dict(samples=samples, numerical=numerical, next_sequence=communicator.next_sequence)


def lifetime(communicator, rank, channel):
    if rank == 1:
        send(channel, "held")
        receive(channel)
        cuda.CudaStagingAdapter(1025, device="cuda:0").all_reduce_sum_(torch.full((1025,), 2.0, device="cuda:0"), communicator)
        return dict(participated=True)
    workspace = cuda.CudaStagingAdapter(1025, device="cuda:0")
    tensor = torch.ones(1025, device="cuda:0")
    ready = threading.Event()
    original = cuda._copy
    results, errors = [], []

    def copy(destination, source, stream, pinned):
        result = original(destination, source, stream, pinned)
        if destination.device.type == "cpu":
            ready.set()
        return result

    def perform(value, adapter, comm):
        try:
            timing = adapter.all_reduce_sum_(value, comm)
            assert torch.all(value == 3).item()
            results.append(asdict(timing))
        except BaseException as error:
            errors.append(error)

    references = [weakref.ref(obj) for obj in (tensor, workspace, workspace._staging, workspace._view)]
    with patch.object(cuda, "_copy", copy):
        thread = threading.Thread(target=perform, args=(tensor, workspace, communicator))
        thread.start()
        assert ready.wait(5)
        try:
            workspace.all_reduce_sum_(tensor, communicator)
        except RuntimeError as error:
            assert "already active" in str(error)
        else:
            raise AssertionError("concurrent workspace call was accepted")
        del tensor, workspace
        gc.collect()
        progress, until = 0, time.monotonic() + 0.05
        while time.monotonic() < until:
            progress += 1
        assert thread.is_alive() and all(reference() is not None for reference in references)
        assert communicator.next_sequence == 0
        send(channel, "waiting")
        thread.join(timeout=10)
        assert not thread.is_alive() and not errors, errors
    gc.collect()
    assert len(results) == 1 and all(reference() is None for reference in references)
    return dict(python_progress=progress, owners_alive_while_waiting=True, owners_released_after_return=True,
                concurrent_rejected=True, timing=results[0], next_sequence=communicator.next_sequence)


def fault(communicator, rank, channel, mode):
    if rank == 1:
        send(channel, "held")
        action = receive(channel)
        if action == b"C":
            communicator.close()
        else:
            receive(channel)
        return dict(action=action.decode())
    workspace = cuda.CudaStagingAdapter(1025, device="cuda:0")
    value = torch.full((1025,), 7.0, device="cuda:0")
    original = cuda._copy

    def copy(destination, source, stream, pinned):
        result = original(destination, source, stream, pinned)
        assert destination.device.type == "cpu", "H2D occurred after native failure"
        send(channel, "native-ready")
        return result

    started = time.monotonic()
    with patch.object(cuda, "_copy", copy):
        try:
            workspace.all_reduce_sum_(value, communicator)
        except EngineError as error:
            context = {name: getattr(error, name) for name in
                       ("code", "rank", "run_id", "operation", "sequence", "peer", "peer_endpoint")}
            assert error.code == ("timeout" if mode == "stall" else "peer_closed"), context
            assert error.rank == 0 and error.sequence == 0 and error.peer == 1 and error.peer_endpoint
        else:
            raise AssertionError("faulted native collective succeeded")
    assert communicator.state == "failed" and communicator.next_sequence == 0
    assert torch.all(value == 7).item() and workspace._failure is None
    send(channel, "failed")
    return dict(error=context, input_unchanged=True, elapsed_ms=(time.monotonic() - started) * 1000)


def policy(communicator):
    workspace = cuda.CudaStagingAdapter(3, device="cuda:0")
    value = torch.ones(3, device="cuda:0")
    original = cuda._copy
    primary = RuntimeError("injected H2D copy failure")

    def copy(destination, source, stream, pinned):
        if destination.device.type == "cuda":
            raise primary
        return original(destination, source, stream, pinned)

    with patch.object(cuda, "_copy", copy), patch.object(Communicator, "close", side_effect=RuntimeError("injected close failure")):
        try:
            workspace.all_reduce_sum_(value, communicator)
        except RuntimeError as error:
            assert error is primary and error.__notes__
        else:
            raise AssertionError("injected copy failure was accepted")
    assert communicator.next_sequence == 1
    assert workspace._failed_owners[0] is value and workspace._failure["stream_settled"]
    assert not workspace._failure["cleanup_complete"] and workspace._failure["secondary_errors"]
    try:
        workspace.all_reduce_sum_(value, communicator)
    except RuntimeError as error:
        assert "poisoned" in str(error)
    else:
        raise AssertionError("poisoned workspace was reused")
    communicator.close()
    torch.cuda.synchronize()
    return dict(injected_policy_only=True, primary_preserved=True, native_sequence=1,
                original_failure=workspace._failure, recovery_closed=True, actual_stream_settled=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--control", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    channel = socket.socket(fileno=args.control)
    channel.settimeout(35)
    info = preflight()
    write(args.output / f"preflight-{args.rank}.json", info)
    send(channel, "warm")
    receive(channel)
    mode = config["mode"]
    with native(config, args.rank) as communicator:
        if mode == "unit":
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from unit import run
            result = dict(checks=run(communicator), next_sequence=communicator.next_sequence)
        elif mode == "matrix":
            result = matrix(communicator, args.rank, config["world_size"])
        elif mode == "lifetime":
            result = lifetime(communicator, args.rank, channel)
        elif mode in ("closed", "stall"):
            result = fault(communicator, args.rank, channel, mode)
        else:
            assert mode == "policy"
            result = policy(communicator)
    torch.cuda.synchronize()
    write(args.output / f"rank-{args.rank}.json", dict(status="passed", mode=mode, rank=args.rank,
                                                     cleanup_complete=True, **result))
    channel.close()


if __name__ == "__main__":
    main()
