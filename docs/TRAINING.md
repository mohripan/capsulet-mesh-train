# CPU training through the C++ engine

The optional trainer uses our native centralized TCP SUM for gradients and loss. Four fresh Python processes train a CPU FP32 `16 -> 32 -> ReLU -> 4` MLP on different minibatches. An independent ordinary combined-batch reference checks every step. See [ADR 0003](adr/0003-replicated-sgd-training-contract.md) and the [task contract](openspecs/cpu-mlp-trainer/design.md).

## Install and run

Use Linux/WSL x86-64 and CPython 3.12. Build the wheel with the [pinned Python toolchain](PYTHON.md#build-and-verify), then install the CPU runtime explicitly:

```sh
build/toolchain-venv/bin/python -m build --no-isolation
uv venv --python 3.12 build/training-runtime
uv pip sync --require-hashes --python build/training-runtime/bin/python requirements/training-cpu.lock
uv pip install --no-deps --python build/training-runtime/bin/python dist/meshtrain-0.1.0-cp312-cp312-linux_x86_64.whl
build/training-runtime/bin/python -I -m meshtrain.training run --output build/my-training-run
```

Each output directory must be fresh and empty. Defaults are four workers, 100 steps, local batch eight, SGD learning rate 0.05, mean cross-entropy, one compute thread and deterministic algorithms. The run synchronizes 676 gradient elements and one loss element per step, explicitly dividing each native SUM by worker count. Sequence reaches 200. Both collectives and finiteness checks precede SGD.

`meshtrain[training]` declares the optional Torch version range; selecting the exact official CPU wheel through the lock avoids resolving a GPU distribution. The verified runtime pins Torch `2.14.1+cpu` and NumPy `2.5.4`; other versions/platforms are not established by a dependency range. The base package imports and runs without Torch. The standalone C++ build remains independent of Python/Torch.

For a short comparison:

```sh
build/training-runtime/bin/python -I -m meshtrain.training run \
  --world-size 2 --steps 3 --output build/my-smoke-run
```

The supervisor captures both streams with rank tags, treats any worker failure as a group failure, and terminates/reaps its remaining workers. `--worker-timeout` defaults to 120 seconds and `--termination-grace` to five seconds for the group. Native `--startup-timeout-ms` and `--collective-timeout-ms` each default to 30000. No automatic restart occurs. A loopback port reservation race can produce bounded startup failure; prepare a fresh attempt after failure.

## Independent worker commands

Prepare once, start every rank independently, wait for every successful exit, then verify:

```sh
python -I -m meshtrain.training prepare --output build/manual-run --world-size 2 --steps 3
python -I -m meshtrain.training worker --config build/manual-run/config.json --rank 0 --output build/manual-run
# Run rank 1 in another terminal while rank 0 waits for startup.
python -I -m meshtrain.training worker --config build/manual-run/config.json --rank 1 --output build/manual-run
python -I -m meshtrain.training verify --output build/manual-run
```

Use the installed runtime's `python` for all commands. `prepare` also accepts `--endpoints '[{"host":"worker-0","port":29500},{"host":"worker-1","port":29500}]'`, `--listen-address 0.0.0.0` and `--run-id ATTEMPT`. Copy the same complete prepared bundle to workers or mount it at a common run root; each rank writes only `ranks/<rank>`. Explicit endpoints support later container/host launching; four-container trainer validation belongs to #8. This local convenience supervisor is not a cluster scheduler.

## Evidence and numerical acceptance

`config.json` records workload, stable parameter layout, endpoints, run ID, execution settings and a SHA-256 context fingerprint. `inputs/` contains the exact saved initialization, features/teacher/labels and actual batch indices. Indices wrap modulo 1024 across steps; each step has disjoint rank slices. Workers validate input digests before joining.

Each `ranks/<rank>/` contains `trace.npz`, `final.pt`, `complete.json`, and launcher-captured stdout/stderr logs. Traces record global mean loss and averaged gradients before SGD, and parameters after SGD. Completion metadata binds file digests, identity, evaluation losses, environment and sequence. `reference/` contains the independent reference trajectory. These files are evidence, without resume/checkpoint semantics.

`verification.json` replaces any previous status before verification. It rejects missing/stale/incomplete/invalid artifacts, recomputes the reference from prepared inputs, and compares each rank with the reference and every rank pair. The elementwise rule is `abs(actual-reference) <= 1e-6 + 1e-5*abs(reference)`, calculated in FP64 on FP32 traces. All values must be finite; maximum absolute/relative errors, failure counts and first offending indices/names are reported. Complete final state keys/values and recomputed evaluation metrics receive separate checks.

Only the exact four-worker/100-step/local-eight workload with all comparisons passing and strictly decreased finite evaluation loss for every model sets `full_acceptance: true`. Evaluation uses the fixed training dataset and demonstrates functional learning, not generalization. Smoke/custom profiles are labeled separately. Inputs and seeds match exactly; FP32 batching/reduction can still produce small numerical differences.

## Adapter API

```python
from meshtrain.torch_adapter import GradientSynchronizer

synchronizer = GradientSynchronizer(model, world_size)
# After loss.backward(), before optimizer.step():
synchronizer.synchronize(communicator)
```

Use one adapter per worker and the communicator's world size. Trainable parameters are sorted by name and must retain identity, shape, dense CPU FP32 layout and device. Every gradient must be present, dense strided CPU FP32, matching shape and finite. Noncontiguous dense gradients are explicitly packed. A retained contiguous Torch buffer and its detached shared NumPy view live through the blocking C++ call; no implicit conversion, storage rebinding or third-party collective is used. Reduced overflow or native failure aborts without gradient copy-back. The example also aborts on nonfinite reduced loss before SGD.

The adapter covers this replicated MLP contract. Unused/sparse gradients, DDP buffer/hook semantics, tied/stateful models, GPU transfer, overlap and optimizer sharding need later designs.

## Reproduce focused verification

```sh
build/toolchain-venv/bin/python tools/check_trainer.py --mode smoke --output build/trainer-smoke
build/toolchain-venv/bin/python tools/check_trainer.py --mode full --output build/trainer-full
docker build --target trainer-smoke-test -t meshtrain-trainer-smoke-test:dev .
```

The tool builds an sdist-derived wheel, installs it in a fresh isolated CPU environment, runs the selected profile and focused boundary/failure checks, and retains logs, dependency/tool versions and `run/` artifacts. Dependencies are hash-checked, then installed offline from the prepared wheelhouse. Docker caches the official CPU wheel and transitive dependencies before source compilation. CI runs the short trainer target alongside native and binding jobs on push/PR. No nightly schedule is implemented. The full four-worker run is required task acceptance; broad fault/count matrices and four-container training remain #8.

Recorded commands, actual results and platform limits are in the [verification report](openspecs/cpu-mlp-trainer/verification.md).
