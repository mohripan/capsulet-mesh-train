"""CLI-owned CUDA execution policy and real model preflight."""

import importlib.metadata
import os
import sys

import numpy as np
import torch

from meshtrain import _native
import meshtrain
from .selection import configure_environment, from_config, selection


def configure():
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False


def components():
    return {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
            if d.metadata["Name"].startswith(("cuda-", "nvidia-", "triton"))}


def execution_contract(device, host_memory):
    device, host_memory = selection(device, host_memory)
    configure_environment(device)
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the supported CUDA runtime and GPU driver")
    expected = dict(deterministic=True, deterministic_warn_only=False, intra_threads=1, inter_threads=1, autocast=False,
                    fp32_precision="ieee", matmul_precision="ieee", cudnn_precision="ieee", cudnn_benchmark=False)
    effective = dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                     autocast=torch.is_autocast_enabled("cuda"),
                     deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                     intra_threads=torch.get_num_threads(), inter_threads=torch.get_num_interop_threads(),
                     fp32_precision=torch.backends.fp32_precision,
                     matmul_precision=torch.backends.cuda.matmul.fp32_precision,
                     cudnn_precision=torch.backends.cudnn.fp32_precision, cudnn_benchmark=torch.backends.cudnn.benchmark)
    if effective != expected:
        raise ValueError(f"CUDA execution policy differs: {effective}")
    if int(device.split(":")[1]) >= torch.cuda.device_count():
        raise ValueError("CUDA device index is out of range")
    return dict(torch=torch.__version__, numpy=np.__version__, cuda_runtime=torch.version.cuda,
                device=device, host_memory=host_memory,
                cublas_workspace=":4096:8", **effective)


def activate(config, network, features, labels):
    device, host_memory = from_config(config)
    execution_contract(device, host_memory)
    torch.cuda.set_device(device)
    return network.to(device), features.to(device), labels.to(device)


def preflight(config, network, features, labels):
    device, mode = from_config(config)
    policy = execution_contract(device, mode)
    probe = torch.nn.functional.cross_entropy(network(features[:8]), labels[:8])
    probe.backward()
    if not torch.isfinite(probe).item() or not all(torch.isfinite(p.grad).all().item() for p in network.parameters()):
        raise FloatingPointError("nonfinite CUDA MLP preflight")
    network.zero_grad(set_to_none=True)
    torch.cuda.current_stream(device).synchronize()
    properties = torch.cuda.get_device_properties(device)
    return dict(execution=policy, python=sys.version, gpu_name=properties.name, gpu_uuid=str(properties.uuid),
                device=device, total_memory=properties.total_memory,
                capability=list(torch.cuda.get_device_capability(device)), architectures=torch.cuda.get_arch_list(),
                package=meshtrain.__file__, extension=_native.__file__, pid=os.getpid(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                environment={key: os.environ.get(key) for key in
                             ("CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")},
                components=components(),
                cudnn_version=torch.backends.cudnn.version(), default_stream=
                torch.cuda.current_stream(device) == torch.cuda.default_stream(device))
