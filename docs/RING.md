# Native ring all-reduce

Issue [#13](https://github.com/mohripan/capsulet-mesh-train/issues/13) adds explicit ring FP32 SUM to the standalone C++ engine. The same blocking `Communicator::all_reduce_sum` API uses our C++ reduce-scatter and all-gather schedule over our TCP sockets. Centralized remains the default. Python configuration and the existing CPU/CUDA trainers continue to select centralized; ring selection is currently a native C++ feature.

## Select an algorithm

Populate a fixed `RunConfig` at every worker with the same ordered endpoint list, world size, run ID, application fingerprint, limits and algorithm. Assign that process its own rank. For example, on three localhost processes use distinct ports 45000–45002:

```cpp
#include <meshtrain/communicator.hpp>
#include <meshtrain/tcp.hpp>
#include <vector>

meshtrain::RunConfig config;
config.rank = rank;
config.world_size = 3;
config.run_id = "unique-launch-attempt";
config.context_fingerprint = "same-application-layout";
config.endpoints = {{"127.0.0.1", 45000}, {"127.0.0.1", 45001}, {"127.0.0.1", 45002}};
config.all_reduce_algorithm = meshtrain::AllReduceAlgorithm::ring;
meshtrain::Communicator communicator(config, meshtrain::connect_tcp(config));
std::vector<float> gradients{float(rank + 1), 2.0F};
communicator.all_reduce_sum(gradients);
```

This fragment assumes the application already parsed `rank`; launch all three processes independently. Every worker supplies the same full buffer count and collective order. Results are SUM without division; training explicitly averages. Use `AllReduceAlgorithm::centralized` for the original ascending-rank root algorithm. No automatic selection or fallback occurs.

Ring listens at each rank's configured port and uses direct neighbor data links plus separate root control connections. Supply resolvable/routable advertised names for containers/hosts and open the required ports. `listen_address` remains the local bind interface. Two ranks share one full-duplex data socket; one rank opens none. Root retains O(p) control descriptors. The [wire/transport document](TCP_PROTOCOL.md) defines startup, duplex ownership and exact v2 framing.

## Buffer, numerical and failure behavior

Counts need not be divisible by worker count. Empty input and empty chunks still participate; there is no padding. Each rank uses at most one additional largest-chunk receive buffer (`ceil(N/p)` floats). The full input must fit `max_payload_bytes`, independently of chunk size.

FP32 addition follows the fixed cyclic order for each chunk, which can differ from centralized rounding. Conformance uses exact fixtures and bounded finite random fixtures at `atol=1e-6`, `rtol=1e-5`, with an independently generated FP64 SUM and FP32 order oracle. Those are tested fixtures, not a universal error guarantee for arbitrary magnitudes or worker counts.

One collective deadline covers every prefix/header/payload and reduction step. There is one admitted operation at a time. `close()` cancels active I/O; any admitted failure leaves the communicator unusable and the input potentially partially modified. Do not apply an optimizer update from failed synchronization. Completion is local, without a transactional guarantee that every worker has returned. Join callers before destruction; recovery needs a fresh group/run ID.

Wire v2 is a coordinated upgrade: v1 and v2 workers cannot join. Existing native consumers recompile. Adapters without `exchange_bytes` remain source-buildable for centralized use; an admitted multi-rank ring call throws `invalid_configuration` rather than delegating or silently changing algorithms.

## Focused verification

Run these in Linux/WSL at the checkout root:

```sh
cmake -S . -B build/ring-dev -G Ninja -DCMAKE_BUILD_TYPE=Debug -DMESHTRAIN_WARNINGS_AS_ERRORS=ON
cmake --build build/ring-dev
ctest --test-dir build/ring-dev --output-on-failure -LE m1_integration
cmake -S . -B build/ring-portable -G Ninja -DMESHTRAIN_ENABLE_TCP=OFF -DMESHTRAIN_WARNINGS_AS_ERRORS=ON
cmake --build build/ring-portable
ctest --test-dir build/ring-portable --output-on-failure
```

The process matrix covers 1–5 ranks and deduplicated counts `0,1,p-1,p,p+1,2*p+1,1024,1025,131073,1048579`, for both ring and centralized. The ring path requests a 4 KiB send buffer for backpressure; the centralized comparison uses OS defaults. This is conformance, not a timing comparison. Dedicated duplex cases include 4 KiB send/receive requests, asymmetric/empty spans, EINTR and a shared two-rank socket. Native fault controllers enforce one-second operation deadlines and a separate five-second whole-group watchdog, with explicit phase synchronization, living-worker descriptor checks and reaping.

The native image has no Python or Torch. A Python standard-library host controller launches the small container boundary check and records resource ownership/cleanup:

```sh
docker build --target native-test -t meshtrain-native-test:ring .
python3 tools/check_ring_containers.py meshtrain-native-test:ring --output build/ring-containers
python3 tools/check_ring_containers.py meshtrain-native-test:ring --scenario worker-exit --output build/ring-container-failure
```

Use fresh output directories. The healthy check runs the same numerical worker in four isolated containers with advertised hostnames; the induced-exit check proves controller failure detection and scoped cleanup. A 60-second execution deadline and separate cleanup deadline bound the controller. Regular CI retains original native/binding/short-trainer regressions and adds these focused ring checks; broad M1 acceptance remains nightly/manual.

Actual commands, raw results, exact revisions and limits are recorded in [OpenSpec verification](openspecs/ring-all-reduce/verification.md). This implementation makes no ring training-parity, speedup, native Windows, physical multi-host or distinct-GPU claim. Tree reduction (#14) and comparable Gloo/model measurements (#15) remain separate work under M3.
