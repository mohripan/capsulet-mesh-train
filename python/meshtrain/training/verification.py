from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from .artifacts import digest, identity, read_json, write_json
from .inputs import load_inputs
from .reference import train_reference
from .workload import evaluate, load_model

ATOL, RTOL = 1e-6, 1e-5


def load_result(root, config, rank):
    directory = root / ("reference" if rank == "reference" else f"ranks/{rank}")
    metadata = read_json(directory / "complete.json")
    expected_sequence = None if rank == "reference" else 2 * config["workload"]["steps"]
    if (any(metadata[key] != value for key, value in identity(config).items())
            or metadata["rank"] != rank or metadata["steps"] != config["workload"]["steps"]
            or metadata["next_sequence"] != expected_sequence):
        raise ValueError(f"{rank}: incomplete or mismatched artifact identity/sequence")
    execution = config["context"]["execution"]
    if any(metadata["environment"][key] != execution[key]
           for key in ("torch", "numpy", "intra_threads", "inter_threads", "deterministic")):
        raise ValueError(f"{rank}: execution metadata differs from prepared contract")
    if set(metadata["files"]) != {"trace.npz", "final.pt"}:
        raise ValueError(f"{rank}: incomplete artifact manifest")
    hashes = {name: digest(directory / name) for name in metadata["files"]}
    if hashes != metadata["files"]:
        raise ValueError(f"{rank}: artifact digest mismatch")
    with np.load(directory / "trace.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"losses", "gradients", "parameters"}:
            raise ValueError(f"{rank}: unexpected trace keys")
        trace = {key: archive[key].copy() for key in archive.files}
    steps = config["workload"]["steps"]
    count = sum(item["count"] for item in config["context"]["layout"])
    for key, value in trace.items():
        shape = (steps,) if key == "losses" else (steps, count)
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"{rank}: invalid {key} trace shape/dtype/values")
    state = torch.load(directory / "final.pt", map_location="cpu", weights_only=True)
    initial = load_model(root).state_dict()
    if set(state) != set(initial):
        raise ValueError(f"{rank}: incomplete final state keys")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != initial[name].shape
                or value.dtype != initial[name].dtype or not torch.isfinite(value).all().item()):
            raise ValueError(f"{rank}: invalid final state {name}")
    for key in ("evaluation_before", "evaluation_after"):
        if not np.isfinite(metadata[key]):
            raise ValueError(f"{rank}: nonfinite evaluation")
    hashes["complete.json"] = digest(directory / "complete.json")
    return dict(metadata=metadata, trace=trace, state=state, hashes=hashes)


def compare(actual, expected, label, quantity, step=None, layout=None):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"{label}/{quantity}: incompatible or nonfinite comparison")
    error = np.abs(actual - expected)
    failed = error > ATOL + RTOL * np.abs(expected)
    first = None
    if failed.any():
        index = int(np.flatnonzero(failed)[0])
        first = dict(step=step, flat_index=index)
        if layout is not None:
            item = next(item for item in layout if item["offset"] <= index < item["offset"] + item["count"])
            first.update(name=item["name"], parameter_flat_index=index - item["offset"])
    return dict(comparison=label, quantity=quantity, step=step,
                max_absolute_error=float(error.max()),
                max_relative_error=float((error / np.maximum(np.abs(expected), 1e-12)).max()),
                failure_count=int(failed.sum()), first_failure=first)


def verify(output):
    root = Path(output).resolve()
    report = dict(schema=1, status="failed", full_acceptance=False, atol=ATOL, rtol=RTOL)
    write_json(root / "verification.json", report)
    try:
        root, config, initial, features, labels, _ = load_inputs(root)
        report.update(identity(config))
        ranks = list(range(config["workload"]["world_size"]))
        results = {rank: load_result(root, config, rank) for rank in ranks}
        train_reference(root)
        results["reference"] = load_result(root, config, "reference")
        checks = []
        layout = config["context"]["layout"]
        pairs = [(rank, "reference") for rank in ranks] + list(combinations(ranks, 2))
        for actual_rank, expected_rank in pairs:
            actual, expected = results[actual_rank], results[expected_rank]
            label = f"{actual_rank} vs {expected_rank}"
            for quantity in ("losses", "gradients", "parameters"):
                for step in range(config["workload"]["steps"]):
                    checks.append(compare(actual["trace"][quantity][step], expected["trace"][quantity][step],
                                          label, quantity, step, None if quantity == "losses" else layout))
            for name in sorted(expected["state"]):
                checks.append(compare(actual["state"][name].numpy(), expected["state"][name].numpy(),
                                      label, f"final_state/{name}"))
            for key in ("evaluation_before", "evaluation_after"):
                checks.append(compare(actual["metadata"][key], expected["metadata"][key], label, key))
        full = tuple(config["workload"][key] for key in ("world_size", "steps", "local_batch_size")) == (4, 100, 8)
        evaluation = {}
        for rank, result in results.items():
            final = load_model(root)
            final.load_state_dict(result["state"], strict=True)
            before, after = evaluate(initial, features, labels), evaluate(final, features, labels)
            checks.append(compare(result["metadata"]["evaluation_before"], before, str(rank), "recorded_evaluation_before"))
            checks.append(compare(result["metadata"]["evaluation_after"], after, str(rank), "recorded_evaluation_after"))
            for item in layout:
                saved = result["state"][item["name"]].numpy().reshape(-1)
                traced = result["trace"]["parameters"][-1, item["offset"]:item["offset"] + item["count"]]
                checks.append(compare(traced, saved, str(rank), f"trace_final_state/{item['name']}"))
            evaluation[str(rank)] = dict(before=before, after=after, decreased=after < before)
        passed = not any(check["failure_count"] for check in checks)
        if full:
            passed = passed and all(item["decreased"] for item in evaluation.values())
        report.update(status="passed" if passed else "failed", profile="full" if full else "smoke_or_custom",
                      full_acceptance=full and passed, checks=checks, evaluation=evaluation,
                      failure_count=sum(check["failure_count"] for check in checks),
                      artifacts={str(rank): result["hashes"] for rank, result in results.items()},
                      config_sha256=digest(root / "config.json"), inputs=config["context"]["inputs"])
        write_json(root / "verification.json", report)
        if not passed:
            raise ValueError("numerical parity or evaluation decrease check failed; see verification.json")
        return report
    except BaseException as error:
        report.update(status="failed", full_acceptance=False, error=f"{type(error).__name__}: {error}")
        write_json(root / "verification.json", report)
        raise
