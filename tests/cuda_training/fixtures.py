"""Focused installed CUDA gradient, failure and artifact checks."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
from unittest.mock import patch

from meshtrain.training.selection import configure_environment, deferred_signals
configure_environment("cuda:0")

import numpy as np
import torch

import meshtrain
from meshtrain import Communicator, EngineError
from meshtrain.cuda_adapter import CudaStagingAdapter
from meshtrain.cuda_gradient import CudaGradientSynchronizer
from meshtrain.training.artifacts import digest, read_json, write_json, write_npz
from meshtrain.training.cuda_execution import configure
from meshtrain.training.inputs import native_config
from meshtrain.training.launcher import supervise
from meshtrain.training.verification import verify
from meshtrain.training.worker import train_worker
from meshtrain.training.workload import configure_execution, flat_gradients, model


def network_with_gradients(value=1):
    network = model().to("cuda:0")
    for parameter in network.parameters():
        parameter.grad = torch.full_like(parameter, value)
    return network


def snapshots(network):
    values = []
    for parameter in network.parameters():
        gradient = parameter.grad
        if gradient is None:
            values.append(None)
        else:
            value = gradient.detach().cpu()
            if value.layout != torch.strided:
                value = value.to_dense()
            values.append(value.resolve_conj().resolve_neg().contiguous().view(torch.uint8).clone())
    return values


def unit(root):
    config = read_json(root / "config.json")
    evidence = []
    with Communicator(native_config(config, 0)) as communicator:
        for kind in ("missing", "nonfinite", "dtype", "device", "sparse", "autograd", "inference", "negative", "subclass", "identity", "shape"):
            network = network_with_gradients()
            synchronizer = CudaGradientSynchronizer(network, 1, device="cuda:0")
            parameter = network[0].weight
            if kind == "missing":
                parameter.grad = None
            elif kind == "nonfinite":
                parameter.grad[0, 0] = float("nan")
            elif kind == "dtype":
                parameter.grad.data = parameter.grad.double()
            elif kind == "device":
                network.cpu()
            elif kind == "sparse":
                parameter.grad = parameter.grad.to_sparse()
            elif kind == "autograd":
                parameter.grad.requires_grad_(True)
            elif kind == "inference":
                with torch.inference_mode():
                    parameter.grad = torch.ones_like(parameter)
            elif kind == "negative":
                parameter.grad = torch._neg_view(parameter.grad)
            elif kind == "subclass":
                class GradientSubclass(torch.Tensor):
                    pass
                parameter.grad = parameter.grad.as_subclass(GradientSubclass)
            elif kind == "identity":
                network[0].weight = torch.nn.Parameter(parameter.detach().clone())
            else:
                parameter.data = parameter.data.T
            pointers = [p.grad for p in network.parameters()]
            before = snapshots(network)
            try:
                synchronizer.synchronize(communicator)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError(f"accepted invalid {kind}")
            assert communicator.next_sequence == 0
            assert all(a is b for a, b in zip(pointers, [p.grad for p in network.parameters()]))
            assert all(a is None and b is None or a is not None and b is not None and torch.equal(a, b)
                       for a, b in zip(before, snapshots(network)))
            evidence.append(kind)
        for inference in (False, True):
            network = network_with_gradients()
            synchronizer = CudaGradientSynchronizer(network, 1, device="cuda:0")
            with torch.autograd.forward_ad.dual_level():
                primal = torch.ones_like(network[0].weight)
                tangent = torch.full_like(primal, 2)
                network[0].weight.grad = torch.autograd.forward_ad.make_dual(primal, tangent)
                with torch.inference_mode(inference):
                    try:
                        synchronizer.synchronize(communicator)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("forward AD gradient was admitted")
                    assert torch.is_inference_mode_enabled() is inference
                pair = torch.autograd.forward_ad.unpack_dual(network[0].weight.grad)
                assert torch.equal(pair.primal, primal) and torch.equal(pair.tangent, tangent)
            assert communicator.next_sequence == 0
            evidence.append(f"dual-inference-{inference}")
        network = network_with_gradients()
        with torch.inference_mode():
            synchronizer = CudaGradientSynchronizer(network, 1, device="cuda:0")
        assert not torch.is_inference(synchronizer._flat)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                synchronizer.synchronize(communicator)
        except ValueError as error:
            assert "capture" in str(error)
        else:
            raise AssertionError("graph capture was accepted")
        assert communicator.next_sequence == 0
        network[0].weight.grad = torch.ones((16, 32), device="cuda:0").T
        pointer = network[0].weight.grad.data_ptr()
        timing = synchronizer.synchronize(communicator)
        assert network[0].weight.grad.data_ptr() == pointer and timing.count == 676
        assert communicator.next_sequence == 1 and torch.equal(flat_gradients_tensor(network), torch.ones(676))
        evidence.extend(("capture", "ordinary-workspace", "noncontiguous-identity"))
    for kwargs in (dict(device="cuda:256"), dict(device="cuda"), dict(device="cuda:0", max_staging_bytes=4),
                   dict(device="cuda:0", pin_memory="yes")):
        try:
            CudaGradientSynchronizer(network_with_gradients(), 1, **kwargs)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"invalid constructor admitted {kwargs}")
    for world in (0, True):
        try:
            CudaGradientSynchronizer(network_with_gradients(), world, device="cuda:0")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid world size admitted")
    return dict(status="passed", groups=evidence, constructor_rejections=6, next_sequence=1)


def flat_gradients_tensor(network):
    return torch.from_numpy(flat_gradients(network))


def known(root, rank, channel):
    config = read_json(root / "config.json")
    network = network_with_gradients(rank + 1)
    network[0].weight.grad = torch.full((16, 32), rank + 1.0, device="cuda:0").T
    pointers = [parameter.grad.data_ptr() for parameter in network.parameters()]
    synchronizer = CudaGradientSynchronizer(network, 2, device="cuda:0")
    with Communicator(native_config(config, rank)) as communicator:
        if rank == 1:
            channel.sendall(b"held\n")
            assert channel.recv(1) == b"G"
            synchronizer.synchronize(communicator)
        else:
            entered, failures = threading.Event(), []
            original = CudaStagingAdapter.all_reduce_sum_

            def observe(self, tensor, comm):
                entered.set()
                return original(self, tensor, comm)

            def synchronize():
                try:
                    synchronizer.synchronize(communicator)
                except BaseException as error:
                    failures.append(error)

            with patch.object(CudaStagingAdapter, "all_reduce_sum_", observe):
                thread = threading.Thread(target=synchronize, daemon=True)
                thread.start()
                assert entered.wait(10)
                try:
                    synchronizer.synchronize(communicator)
                except RuntimeError as error:
                    assert "already active" in str(error)
                else:
                    raise AssertionError("concurrent gradient operation admitted")
                channel.sendall(b"waiting\n")
                thread.join(35)
                assert not thread.is_alive() and not failures, failures
        assert communicator.next_sequence == 1
        assert pointers == [parameter.grad.data_ptr() for parameter in network.parameters()]
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            actual = flat_gradients_tensor(network)
        assert torch.equal(actual, torch.full((676,), 1.5)) and stream.query()
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            for parameter in network.parameters():
                parameter.grad.fill_(rank + 2)
        consumer.wait_stream(producer)
        with torch.cuda.stream(consumer):
            synchronizer.synchronize(communicator)
        assert consumer.query() and communicator.next_sequence == 2
        assert torch.equal(flat_gradients_tensor(network), torch.full((676,), 2.5))
    return dict(status="passed", next_sequence=2, exact_average=1.5, dependent_stream_average=2.5, retained_storage=True,
                concurrent_rejection=rank == 0, completed_cross_stream=True)


def overflow(root, rank):
    config = read_json(root / "config.json")
    network = network_with_gradients(torch.finfo(torch.float32).max)
    synchronizer = CudaGradientSynchronizer(network, 2, device="cuda:0")
    expected = flat_gradients_tensor(network)
    with Communicator(native_config(config, rank)) as communicator:
        try:
            synchronizer.synchronize(communicator)
        except FloatingPointError:
            assert communicator.next_sequence == 1
            assert torch.equal(flat_gradients_tensor(network), expected)
            assert synchronizer._failure["cleanup_complete"]
        else:
            raise AssertionError("nonfinite reduced gradient accepted")
        try:
            synchronizer.synchronize(communicator)
        except RuntimeError as error:
            assert "poisoned" in str(error)
        else:
            raise AssertionError("poisoned gradient adapter reused")
    return dict(status="passed", next_sequence=1, unchanged_gradients=True)


def failure(root, rank):
    updates = []
    original_step, original_sync = torch.optim.SGD.step, CudaGradientSynchronizer.synchronize

    def step(self, *args, **kwargs):
        updates.append(dict(step=len(updates), completed=False))
        write_json(root / f"rank-{rank}-updates.json", updates)
        result = original_step(self, *args, **kwargs)
        torch.cuda.current_stream().synchronize()
        updates[-1]["completed"] = True
        write_json(root / f"rank-{rank}-updates.json", updates)
        return result

    def synchronize(self, communicator):
        if rank == 1 and communicator.next_sequence == 2:
            communicator.close()
        return original_sync(self, communicator)

    write_json(root / f"rank-{rank}-updates.json", updates)
    try:
        with patch.object(torch.optim.SGD, "step", step), patch.object(CudaGradientSynchronizer, "synchronize", synchronize):
            train_worker(root, root / "config.json", rank)
    except BaseException as error:
        write_json(root / f"rank-{rank}-failure.json", dict(type=type(error).__name__, message=str(error),
                   code=getattr(error, "code", None), sequence=getattr(error, "sequence", None),
                   peer=getattr(error, "peer", None), peer_endpoint=getattr(error, "peer_endpoint", None),
                   notes=getattr(error, "__notes__", [])))
        raise
    raise AssertionError("failed peer allowed successful training")


def corrupt(root):
    evidence = []
    for kind in ("missing", "stale", "digest", "device", "precision", "memory-mode", "components", "sequence", "shape", "nonfinite",
                 "gradient", "parameter", "early-schema", "early-environment"):
        output = root.parent / f"corrupt-{kind}"
        shutil.copytree(root, output)
        rank = output / "ranks/0"
        metadata = read_json(rank / "complete.json")
        if kind == "missing":
            (rank / "complete.json").unlink()
        elif kind == "stale":
            metadata["context_fingerprint"] = "stale"
        elif kind == "digest":
            with (rank / "trace.npz").open("ab") as stream:
                stream.write(b"tampered")
        elif kind == "sequence":
            metadata["next_sequence"] = 0
        elif kind in ("device", "precision", "memory-mode", "components"):
            metrics = read_json(rank / "cuda.json")
            if kind == "device":
                metrics["preflight"]["device"] = "cuda:1"
            elif kind == "components":
                metrics["preflight"]["components"]["nvidia-cublas"] = "wrong"
            else:
                metrics["preflight"]["execution"]["fp32_precision" if kind == "precision" else "host_memory"] = (
                    "tf32" if kind == "precision" else "pageable")
            write_json(rank / "cuda.json", metrics)
            metadata["files"]["cuda.json"] = digest(rank / "cuda.json")
        elif kind.startswith("early-"):
            if kind == "early-schema":
                config = read_json(output / "config.json")
                config["schema"] = 999
                write_json(output / "config.json", config)
        else:
            with np.load(rank / "trace.npz", allow_pickle=False) as archive:
                trace = {key: archive[key].copy() for key in archive.files}
            if kind == "shape":
                trace["gradients"] = trace["gradients"][:, :-1]
            else:
                trace["parameters" if kind == "parameter" else "gradients"][0, 0] = float("nan") if kind == "nonfinite" else 10
            write_npz(rank / "trace.npz", **trace)
            metadata["files"]["trace.npz"] = digest(rank / "trace.npz")
        if kind != "missing":
            write_json(rank / "complete.json", metadata)
        if kind.startswith("early-"):
            environment = os.environ | ({"CUBLAS_WORKSPACE_CONFIG": ":16:8"} if kind == "early-environment" else {})
            result = subprocess.run([sys.executable, "-I", "-m", "meshtrain.training", "verify", "--output", str(output)],
                                    capture_output=True, timeout=10, env=environment)
            assert result.returncode != 0
        else:
            try:
                verify(output)
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError(f"corruption accepted: {kind}")
        report = read_json(output / "verification.json")
        assert report["status"] == "failed" and not report["full_acceptance"] and not report.get("cuda_training_acceptance", False)
        if kind in ("gradient", "parameter"):
            assert report["failure_count"] > 0 and any(item["first_failure"] for item in report["checks"])
        evidence.append(dict(kind=kind, status="passed", rejection=report.get("error", "early CLI rejection")))
    write_json(root / "corruption.json", dict(status="passed", cases=evidence))


def launcher(root):
    records = []
    for number in (signal.SIGTERM, signal.SIGINT):
        output = root.parent / f"launcher-{number}"
        output.mkdir()
        owned = []

        def spawn(arguments, **kwargs):
            process = subprocess.Popen(arguments, **kwargs)
            owned.append(process)
            if len(owned) == 2:
                os.kill(os.getpid(), number)
            return process

        commands = [[sys.executable, "-I", "-c", "import time; time.sleep(60)"]] * 2
        try:
            with deferred_signals() as check:
                supervise(commands, output, timeout=10, grace=0.2, spawn=spawn, check_interrupted=check)
        except InterruptedError:
            pass
        else:
            raise AssertionError("deferred supervisor signal was ignored")
        assert all(process.returncode is not None for process in owned)
        records.append(dict(signal=number, pids=[p.pid for p in owned], exits=[p.returncode for p in owned], reaped=True))
    assert not torch.cuda.is_initialized()
    write_json(root / "launcher.json", dict(status="passed", gpu_execution=False, cases=records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("unit", "known", "overflow", "failure", "stream", "corrupt", "launcher"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--control", type=int)
    args = parser.parse_args()
    configure_execution()
    configure()
    for path in (meshtrain.__file__, meshtrain._native.__file__):
        assert Path(path).resolve().is_relative_to(Path(sys.prefix).resolve()), path
    if args.mode == "failure":
        failure(args.output, args.rank)
        return
    if args.mode == "stream":
        stream = torch.cuda.Stream(device="cuda:0")
        with torch.cuda.stream(stream):
            train_worker(args.output, args.output / "config.json", args.rank)
            assert stream.query()
        return
    if args.mode in ("corrupt", "launcher"):
        (corrupt if args.mode == "corrupt" else launcher)(args.output)
        return
    if args.mode == "unit":
        result = unit(args.output)
    elif args.mode == "overflow":
        result = overflow(args.output, args.rank)
    else:
        with socket.socket(fileno=args.control) as channel:
            result = known(args.output, args.rank, channel)
    write_json(args.output / f"rank-{args.rank}-checks.json", result)


if __name__ == "__main__":
    main()
