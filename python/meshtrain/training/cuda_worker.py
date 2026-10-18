"""Ordered blocking CUDA MLP steps with explicit SUM normalization."""

from dataclasses import asdict
import time

import torch

from meshtrain import Communicator
from meshtrain.cuda_adapter import CudaStagingAdapter
from meshtrain.cuda_gradient import CudaGradientSynchronizer
from .cuda_execution import activate, preflight
from .cuda_evidence import begin_memory, end_memory
from .results import finish
from .selection import from_config
from .workload import evaluate, flat_gradients, flat_parameters, optimizer


def _ms(started):
    return (time.perf_counter_ns() - started) / 1_000_000


def train_cuda_worker(root, config, native, network, features, labels, indices, rank):
    device, mode = from_config(config)
    network, features, labels = activate(config, network, features, labels)
    metrics = dict(preflight=preflight(config, network, features, labels), steps=[])
    optim = optimizer(network)
    synchronizer = CudaGradientSynchronizer(network, native.world_size, device=device, pin_memory=mode == "pinned")
    loss_adapter = CudaStagingAdapter(1, device=device, pin_memory=mode == "pinned")
    loss_buffer = torch.empty(1, device=device, dtype=torch.float32)
    metrics["memory"] = begin_memory(config, network, features, labels)
    before = evaluate(network, features, labels)
    losses, gradients, parameters = [], [], []
    with Communicator(native) as communicator:
        for step, global_indices in enumerate(indices):
            started = time.perf_counter_ns()
            optim.zero_grad(set_to_none=True)
            local = torch.from_numpy(global_indices[rank]).to(device)
            loss = torch.nn.functional.cross_entropy(network(features[local]), labels[local])
            if not torch.isfinite(loss).item():
                raise FloatingPointError("nonfinite local CUDA loss")
            loss.backward()
            if communicator.next_sequence != 2 * step:
                raise RuntimeError("unexpected CUDA gradient collective sequence")
            sync_started = time.perf_counter_ns()
            gradient_timing = synchronizer.synchronize(communicator)
            sync_ms = _ms(sync_started)
            loss_buffer.copy_(loss.detach().reshape(1))
            loss_timing = loss_adapter.all_reduce_sum_(loss_buffer, communicator)
            if not torch.isfinite(loss_buffer).all().item():
                raise FloatingPointError("nonfinite reduced CUDA loss")
            loss_buffer.div_(native.world_size)
            if not torch.isfinite(loss_buffer).all().item():
                raise FloatingPointError("nonfinite averaged CUDA loss")
            trace_started = time.perf_counter_ns()
            gradients.append(flat_gradients(network))
            losses.append(float(loss_buffer.item()))
            trace_ms = _ms(trace_started)
            optim.step()
            torch.cuda.current_stream(device).synchronize()
            if not all(torch.isfinite(parameter).all().item() for parameter in network.parameters()):
                raise FloatingPointError("nonfinite updated CUDA parameter")
            trace_started = time.perf_counter_ns()
            parameters.append(flat_parameters(network))
            trace_ms += _ms(trace_started)
            metrics["steps"].append(dict(step=step, warmup=step < 5, sequence_start=2 * step,
                                         gradient=asdict(gradient_timing), loss=asdict(loss_timing),
                                         gradient_sync_ms=sync_ms, step_ms=_ms(started), trace_ms=trace_ms))
            print(f"step={step} loss={losses[-1]:.8f} next_sequence={communicator.next_sequence}", flush=True)
        next_sequence = communicator.next_sequence
    after = evaluate(network, features, labels)
    end_memory(config, metrics["memory"])
    return finish(root / "ranks" / str(rank), config, network, losses, gradients, parameters, before, after, rank,
                  next_sequence, cuda_metrics=metrics)
