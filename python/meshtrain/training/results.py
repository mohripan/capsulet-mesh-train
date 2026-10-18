"""Versioned training completion records shared without worker arithmetic."""

import numpy as np

from .artifacts import digest, environment, identity, write_json, write_npz, write_state


def finish(directory, config, network, losses, gradients, parameters, before, after, rank, next_sequence,
           *, cuda_metrics=None):
    write_npz(directory / "trace.npz", losses=np.asarray(losses, dtype=np.float32),
              gradients=np.stack(gradients), parameters=np.stack(parameters))
    write_state(directory / "final.pt", network)
    files = ["trace.npz", "final.pt"]
    if config["schema"] == 2:
        if cuda_metrics is None:
            raise ValueError("CUDA completion requires device/memory/timing evidence")
        write_json(directory / "cuda.json", cuda_metrics)
        files.append("cuda.json")
    metadata = dict(**identity(config), rank=rank, steps=config["workload"]["steps"],
                    next_sequence=next_sequence, evaluation_before=before, evaluation_after=after,
                    environment=environment(), files={name: digest(directory / name) for name in files})
    write_json(directory / "complete.json", metadata)
    return metadata
