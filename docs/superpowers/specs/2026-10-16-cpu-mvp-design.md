# CPU MVP: standalone communication engine and four-process trainer

Date: 2026-10-16

Status: draft for review. The library-first direction and project boundaries are approved. The detailed defaults below are proposed for the first implementation milestone. This document specifies M1 only; `docs/ROADMAP.md` scopes later subsystems.

## 1. Outcome

Four independent CPU processes train a small MLP on different minibatches. They exchange and sum gradients through our own native C++ TCP engine, then explicitly average those gradients and apply SGD in Python. Their losses, averaged gradients, and updated parameters match a single-process reference using the same combined batches within the tolerances in section 7.

The engine is a standalone C++ library. It builds without Python, PyTorch, CUDA, MPI, or Kubernetes. An optional pybind11 extension exposes the first collective to Python. Native and Python tests establish real multi-process behavior, not only in-process arithmetic.

The principal demonstration uses four workers. The implementation supports a fixed positive worker count configured at startup; tests cover 1, 2, 3, 4, and 5. Worker count is not hardcoded to four or restricted to powers of two.

## 2. Boundaries

Included in M1:

- C++20 core with a reproducible CMake build and native tests.
- Linux TCP socket implementation, explicit message framing, and bounded I/O.
- Static worker configuration with cross-host-capable endpoints.
- Blocking, in-place centralized FP32 SUM all-reduce.
- Optional Python package and pybind11 buffer bindings.
- Local worker launcher and a per-rank entry point usable in separate containers/hosts.
- Deterministic MLP/reference comparison.
- Bounded startup and active-collective failure, actionable diagnostics, and process cleanup.
- A four-container Linux smoke test and documented repeatable checks.

Later milestones own ring/tree algorithms, CUDA, asynchronous work, backward overlap, public sharding collectives, Adam-state sharding, automatic checkpoint recovery, Kubernetes admission, external transports, custom PyTorch backends, and native Windows implementation. Their future extension points must not make M1 depend on those libraries.

## 3. Components and contracts

### 3.1 Configuration and identity

Each worker receives its rank, world size, common run identifier, ordered rank-to-endpoint list, and application context fingerprint. Each endpoint consists of a hostname or IPv4 address and port. The endpoint list must contain exactly one entry for each rank. Ports and listen interfaces are explicit; loopback is a local configuration choice rather than a transport assumption.

Rank is in `[0, world_size)`. World size is positive. A run identifier is a nonempty ASCII string of at most 128 bytes, unique to a launch attempt. The application fingerprint is a bounded string of at most 256 bytes used to catch incompatible training layouts or initial state. Native collective tests use a shared test-context fingerprint; the trainer supplies a hash of its parameter layout, initialization, and relevant training configuration.

Default startup and collective timeouts are separately configurable and both default to 30 seconds. The default maximum collective payload is 256 MiB per rank. Validate integer overflow and configured bounds before allocating or receiving payloads. A run may choose a smaller limit; all ranks must agree.

The MVP accepts little-endian IEEE-754 binary32 hosts. Headers have explicitly encoded fixed-width fields in network byte order; FP32 payloads have a documented little-endian representation. Reject an unsupported representation at startup. Never serialize a native C++ struct directly as the wire header.

### 3.2 Transport

The Linux transport owns sockets, exact-byte reads/writes, deadline enforcement, and resource cleanup. Socket handles use RAII. Handle partial reads/writes, interrupted system calls, peer closure, and writes to a closed peer without terminating the process through SIGPIPE.

Use one monotonic deadline for the entire startup operation or collective operation, including all retries and partial transfers. Slow progress must not reset the timeout indefinitely. Close/cancel must unblock an outstanding operation within the same bounded-failure contract.

The transport boundary exposes peer byte exchange and lifecycle/error information without exposing POSIX descriptors in public interfaces. The collective layer owns reduction order, collective metadata, and scratch-buffer policy. Future device transports may require richer capabilities and separate buffer types; the initial host-byte interface does not promise CUDA support.

M1 uses a star: rank 0 accepts one connection per other rank, and every nonzero rank connects to rank 0. A bounded startup retry handles the root listener not yet being ready. Duplicate ranks, inconsistent world size, mismatched run/context identifiers, protocol-version differences, and incompatible representation/limits fail startup. Rank 0 releases workers to begin only after the complete group has joined.

This configuration works across routable hosts when connectivity and endpoints are supplied. Local/container tests do not count as physical multi-host validation.

### 3.3 Protocol and communicator

The protocol distinguishes startup, collective requests/results, and errors. Its documented header includes magic/version, operation type, source/destination rank, monotonically increasing collective sequence, dtype/reduction identifier, element count, and payload byte length. Check count-to-byte consistency before reading data.

Communicator state proceeds through construction, joining, ready, failed, and closed. A failed communicator cannot be reused; create a fresh one for another attempt. Close is idempotent. A successful collective advances the sequence. At most one collective is in flight per communicator; concurrent calls are rejected. Every rank must issue the same collective sequence and compatible metadata.

The first public collective is blocking, in-place SUM all-reduce over a writable contiguous FP32 buffer. Every participating rank supplies the same count. An empty buffer is valid and still participates in sequence/metadata agreement. For world size one, the operation returns the buffer unchanged.

On any collective failure, the caller's buffer may be partially modified. No optimizer step may use it. The communicator enters failed state and closes its connections; error propagation to connected peers is best effort, with each peer's deadline providing the final bound.

### 3.4 Centralized algorithm

Rank 0 starts with its own buffer as the accumulator. It receives each other rank's buffer in ascending rank order and adds FP32 values in C++. It retains at most one additional full-sized receive buffer, giving root scratch space proportional to message size rather than worker count times message size. After every contribution has arrived, rank 0 sends the sum to all other ranks.

Other ranks send their input, then receive the result into their buffer. The algorithm performs SUM only. There is no hidden division, tensor serialization, autograd, or optimizer step inside the C++ collective.

The engine path must not call an existing all-reduce implementation. Third-party collectives appear only in separately named reference/benchmark code or future explicitly selected adapters.

### 3.5 Python boundary

Build the Python extension optionally, using pybind11 and a CMake-compatible wheel build. Python 3.12 on Linux is the first packaging target. The build issue must verify and record exact dependency/toolchain versions in a reproducible environment before claiming support.

The binding accepts a writable one-dimensional contiguous native FP32 NumPy view. Reject incompatible dtype, read-only buffers, noncontiguous layouts, implicit conversion, and invalid lengths. Do not silently create a temporary that receives the result while the caller's original buffer remains unchanged.

The Python training adapter accepts dense CPU FP32 tensors, explicitly constructs a flat gradient buffer in stable named-parameter order, and exposes its detached NumPy view to the binding. Copying gradients into and out of that flat buffer is intentional and documented. The binding retains the exporting Python object and buffer owner for the full blocking call.

Validate Python objects while holding the GIL, release it for native communication, and reacquire it before touching/destroying Python-owned objects or translating exceptions. The C++ core contains no Python object ownership. Do not accept arbitrary integer pointers as the public Python tensor API.

See the [pybind11 buffer documentation](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html) and [GIL guidance](https://pybind11.readthedocs.io/en/stable/advanced/misc.html).

### 3.6 Launcher and trainer

The local launcher creates independent processes, assigns ranks/endpoints, gives each attempt a fresh run identifier, captures rank-tagged output, monitors exit codes, and stops remaining children on failure. Default termination grace is five seconds before forced cleanup. It returns a nonzero status for any failed worker and preserves the first useful error. It does not automatically restart training in M1.

Provide a per-rank command accepting the same configuration for container/remote launches. A local subprocess launcher is not a cluster scheduler. Resource placement and remote launch automation are later integrations.

Use a deterministic FP32 MLP with input size 16, hidden size 32, ReLU, and four output classes. Use a fixed synthetic classification dataset with 1,024 examples: standard-normal features and a standard-normal teacher weight matrix of shape 16 by 4, with labels given by the argmax of features multiplied by the teacher matrix. Model initialization seed is 1234; data and teacher seeds are 5678 and 5679, using independent CPU generators. Disable stochastic layers, augmentation, and batch-dependent layers. Set CPU compute threads to one for the parity scenario.

The main parity run uses four workers, local batch size eight, 100 optimizer steps, mean cross-entropy loss, and SGD with learning rate 0.05, no momentum, and no weight decay. No scheduler or mixed precision is used.

At zero-based step `s`, workers consume nonoverlapping slices of the same ordered global batch of 32 samples. For rank `r` and local sample offset `j`, the dataset index is `(s * global_batch_size + r * local_batch_size + j) % 1024`. Record this schedule. The single-process reference consumes the concatenation of those slices in rank order, starts from exactly the same saved initial model, and applies one optimizer step per global batch.

Each worker computes a mean local loss and backpropagates it. Its flat gradients undergo our SUM all-reduce, then are divided by world size and copied to parameter gradients before SGD. Equal local batch sizes are required. A separate scalar all-reduce may aggregate loss reporting with the same explicit normalization. All model parameters must have dense gradients in this example; unsupported or missing gradients fail explicitly.

Save per-step comparison metrics and final parameters, along with the initial state, batch schedule, configuration, and environment. PyTorch distributed collectives may be used in separate baselines, not in the custom-engine training path.

## 4. Data flow

1. The launcher prepares the common configuration and initial model artifact.
2. Each worker loads identical parameters, selects its data slice, and joins the C++ communicator.
3. PyTorch performs forward and backward computation locally.
4. The Python adapter packs gradients in deterministic order.
5. The C++ engine transports, reduces, and distributes the sum.
6. Python averages the result and applies the same SGD update at every rank.
7. Verification compares the distributed run with the combined-batch reference.

Tensor payloads travel over our socket protocol, not a shared-memory gradient shortcut. Containers and remote workers use the same communication path as localhost.

## 5. Failure behavior

Errors include rank, run identifier, collective sequence where available, peer/endpoint, operation, and failure category. The category distinguishes configuration/metadata mismatch, transport closure, timeout, invalid buffer, and lifecycle misuse.

During startup or an active collective, a crashed, disconnected, or stalled peer must produce bounded failure. A group mismatch must fail with a useful error rather than run with incompatible state. If local validation prevents a collective from being issued, the launcher observes the failing process and stops its peers; independently launched peers remain bounded by their startup/collective timeout.

Idle failure outside a collective can be observed by the local launcher, an external monitor, or the next native operation. M1 does not promise background heartbeats or detection of an idle remote failure before the next operation.

Neither a broken collective nor a partially updated input is retried in place. M6 will introduce checkpoint recovery of the complete training group with a fresh session.

## 6. Build and repository shape

Use focused areas for public native headers, core implementation, platform networking, optional bindings, Python adapters/examples, native/Python integration tests, and benchmarks. Exact internal filenames can follow the implementation issue; the dependency direction is mandatory: integrations depend on the standalone core, not the reverse.

Provide a native-only configure/build/test path and an optional Python package path. Check in a reproducible Linux container configuration and executable validation instructions when implemented. A future CUDA build is optional and cannot become a prerequisite of the CPU core.

CI should build the standalone core, run native/process tests, build/install the Python package in an isolated environment, and run the deterministic trainer comparison. A documented four-container run is required for M1; it may be a separate integration job rather than every unit-test invocation.

## 7. Verification and acceptance

### Numerical correctness

- Test centralized all-reduce for 1, 2, 3, 4, and 5 ranks, including element counts 0, 1, 3, 1,024, 1,025, and 16,384.
- Use exactly representable fixtures for exact SUM checks and bounded random inputs for numerical checks. The random-value suite uses `atol=1e-6`, `rtol=1e-5` for the tested rank counts.
- For the 100-step trainer, compare averaged gradients, global losses, and parameters with `atol=1e-6`, `rtol=1e-5`; also compare all ranks with each other. Acceptance is elementwise `abs(actual - reference) <= atol + rtol * abs(reference)`. Report maximum absolute error and maximum relative error, defining the latter as `abs(actual - reference) / max(abs(reference), 1e-12)` for reporting rather than as a separate pass threshold.
- Compare loss on the same fixed evaluation set before and after training and require a finite, lower final loss. Do not require every minibatch loss to decrease.
- A failed tolerance check is investigated and documented. Do not silently widen tolerances to pass it.

### Protocol and lifetime

- Exercise fragmented I/O and payloads larger than a small socket buffer, not only one-write/one-read messages.
- Verify invalid framing, incompatible count/dtype/sequence, duplicate rank, wrong run/context identifier, and oversized lengths are rejected.
- Verify binding aliasing, rejected implicit casts/noncontiguous/read-only buffers, retained storage lifetime, and GIL release while communication waits.
- Repeat communicator construction/closure and failed attempts to expose leaked sockets, ports, child processes, or retained buffers.

### Failure and deployment

- Kill a worker during startup and during an active collective; stall one without closing its socket; verify useful errors and bounded process cleanup.
- With a one-second test operation deadline, the harness imposes a five-second overall failure bound in its controlled integration scenario. Record timing on failure to distinguish scheduling instability from an unbounded wait.
- Demonstrate the four-process trainer on localhost and the same trainer with one process in each of four Linux containers over TCP.
- Record exact commands, versions, numerical metrics, and limitations in the issue/report. The container result is not evidence of separate physical hosts or distinct GPUs.

M1 is complete only when these checks are reproducible from a clean environment and the required GitHub child issues are closed with evidence. Planning documents and passing isolated arithmetic tests do not satisfy that gate.

## 8. Design review checklist

- Standalone core and integration dependency directions are explicit.
- Run identity, collective ordering, buffer lifetime, and failed-communicator behavior are defined.
- SUM versus averaging and local versus global batch semantics are unambiguous.
- Startup and collective operations have a total deadline.
- CPU-only, container, future GPU, and physical multi-host claims remain separate.
- Later subsystems have their own scoped milestones instead of being hidden prerequisites of M1.

## 9. References

Implementation must verify APIs against the versions selected in the build environment. The first design references [PyTorch distributed semantics](https://docs.pytorch.org/docs/stable/distributed), [pybind11 buffers](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html), and [pybind11 GIL behavior](https://pybind11.readthedocs.io/en/stable/advanced/misc.html). See `docs/REFERENCES.md` for later GPU, sharding, recovery, and Kubernetes references.
