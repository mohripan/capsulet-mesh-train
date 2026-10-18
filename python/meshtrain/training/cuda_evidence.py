"""CUDA training observations and strict evidence validation."""

import math
import resource

import torch

from .selection import from_config

STAGES = ("ready_wait_ms", "d2h_ms", "native_ms", "h2d_ms", "total_ms")


def begin_memory(config, network, features, labels, *, reference=False):
    device, mode = from_config(config)
    torch.cuda.current_stream(device).synchronize()
    count = sum(parameter.numel() for parameter in network.parameters())
    free, total = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    return dict(boundary="after model/data/preflight and adapter allocation; before evaluation/training",
                baseline_allocated=torch.cuda.memory_allocated(device), baseline_reserved=torch.cuda.memory_reserved(device),
                free_before=free, total_device=total, reference=reference,
                requested_bytes=dict(parameters=4 * count, gradients=4 * count, features=features.numel() * features.element_size(),
                                     labels=labels.numel() * labels.element_size(),
                                     cuda_flat=0 if reference else 4 * count, cuda_loss=0 if reference else 4,
                                     host_staging=0 if reference else 4 * (count + 1),
                                     pinned=0 if reference or mode == "pageable" else 4 * (count + 1)))


def end_memory(config, record):
    device, _ = from_config(config)
    torch.cuda.current_stream(device).synchronize()
    free, total = torch.cuda.mem_get_info(device)
    record.update(allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device),
                  peak_allocated=torch.cuda.max_memory_allocated(device), peak_reserved=torch.cuda.max_memory_reserved(device),
                  free_after=free, total_device=total, rss_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  accounting="per-process Torch allocator and Linux RSS; excludes complete driver/context memory")


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid CUDA metric {label}")


def validate(config, metrics, rank):
    from .cuda_execution import components, execution_contract
    device, mode = from_config(config)
    preflight = metrics["preflight"]
    if preflight["execution"] != execution_contract(device, mode) or preflight["device"] != device:
        raise ValueError(f"{rank}: CUDA execution metadata differs from contract")
    if preflight["components"] != components() or preflight["cudnn_version"] != torch.backends.cudnn.version():
        raise ValueError(f"{rank}: CUDA runtime component identity differs")
    if not isinstance(preflight["gpu_uuid"], str) or not preflight["gpu_uuid"]:
        raise ValueError(f"{rank}: missing actual CUDA device identity")
    if type(preflight["pid"]) is not int or preflight["pid"] <= 0:
        raise ValueError(f"{rank}: invalid CUDA process identity")
    memory = metrics["memory"]
    reference = rank == "reference"
    count = sum(item["count"] for item in config["context"]["layout"])
    expected_bytes = dict(parameters=count * 4, gradients=count * 4, features=65536, labels=8192,
                          cuda_flat=0 if reference else count * 4, cuda_loss=0 if reference else 4,
                          host_staging=0 if reference else (count + 1) * 4,
                          pinned=0 if reference or mode == "pageable" else (count + 1) * 4)
    if memory["requested_bytes"] != expected_bytes or memory["reference"] is not reference:
        raise ValueError(f"{rank}: CUDA buffer accounting differs")
    for key in ("baseline_allocated", "baseline_reserved", "allocated", "reserved", "peak_allocated", "peak_reserved",
                "free_before", "free_after", "total_device", "rss_high_water_bytes"):
        _number(memory[key], key)
    if (memory["peak_allocated"] < max(memory["allocated"], memory["baseline_allocated"])
            or memory["peak_reserved"] < max(memory["reserved"], memory["baseline_reserved"], memory["peak_allocated"])
            or memory["total_device"] != preflight["total_memory"]):
        raise ValueError(f"{rank}: inconsistent CUDA memory observations")
    steps = metrics["steps"]
    if not isinstance(steps, list) or len(steps) != (0 if reference else config["workload"]["steps"]):
        raise ValueError(f"{rank}: incomplete CUDA timing trajectory")
    for step, record in enumerate(steps):
        if record["step"] != step or record["warmup"] is not (step < 5) or record["sequence_start"] != 2 * step:
            raise ValueError(f"{rank}: CUDA timing step/sequence mismatch")
        for field in ("gradient_sync_ms", "step_ms", "trace_ms"):
            _number(record[field], field)
        for kind, size in (("gradient", count), ("loss", 1)):
            timing = record[kind]
            if (timing["count"] != size or timing["bytes"] != 4 * size or timing["device"] != device
                    or timing["pin_memory"] is not (mode == "pinned")):
                raise ValueError(f"{rank}: CUDA staging timing identity differs")
            for field in STAGES:
                _number(timing[field], field)
            if sum(timing[field] for field in STAGES[:-1]) > timing["total_ms"] + 1e-6:
                raise ValueError(f"{rank}: CUDA phase intervals exceed staged call")
        if (record["gradient"]["total_ms"] > record["gradient_sync_ms"] + 1e-6
                or record["gradient_sync_ms"] + record["loss"]["total_ms"] + record["trace_ms"] > record["step_ms"] + 1e-6):
            raise ValueError(f"{rank}: CUDA nested intervals exceed step")
    return metrics
