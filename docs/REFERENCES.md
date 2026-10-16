# Primary references

Consulted 2026-10-16. These are references for design and comparison, not a declaration of installed or supported package versions. Pin and verify actual versions in the relevant implementation issues.

| Topic | Primary source | Use in this project |
| --- | --- | --- |
| Collectives, backend capabilities, custom backends | [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed) | Collective semantics, CPU/GPU baseline choices, and later experimental backend integration. |
| Buffer validation and conversion | [pybind11 NumPy/buffer interface](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html) | Explicit layout/dtype validation; avoid silent temporary buffers. |
| Native waits and Python ownership | [pybind11 miscellaneous/GIL guidance](https://pybind11.readthedocs.io/en/stable/advanced/misc.html) | Release the GIL only while native work does not access Python state. |
| Fault-tolerant worker groups | [PyTorch fault-tolerant training tutorial](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html) | Group restart and restoration from a training snapshot. |
| Sharded optimizer states | [PyTorch ZeroRedundancyOptimizer recipe](https://docs.pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html) | Reference behavior for state ownership and memory accounting; our simplified optimizer is independently implemented. |
| FSDP2 collectives | [PyTorch fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) | Later study of all-gather/reduce-scatter placement and overlap. |
| Linux containers on Windows | [Docker WSL2 backend](https://docs.docker.com/desktop/features/wsl/) | Reproducible local Linux runtime. |
| CUDA in local containers | [Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/) | GPU passthrough prerequisites; actual compute support still requires a functional test. |
| Physical resource requests and limits | [Kubernetes resource management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/) | Placement/resource enforcement distinct from abstract job tokens. |
| Priority and interruption policies | [Kubernetes priority and preemption](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/) | Explain queue priority and the selected non-preemptive behavior. |
| Resource quota pools | [Kueue ClusterQueue](https://kueue.sigs.k8s.io/docs/concepts/cluster_queue/) | Evaluate resource quota, lending controls, and pool integration. |
| Whole-group admission | [Kueue all-or-nothing scheduling](https://kueue.sigs.k8s.io/v0.20/docs/concepts/all_or_nothing/) | Distributed jobs need their worker group to become ready; logical quota alone is not proof of physical placement. |
| External admission policy | [Kueue AdmissionCheck](https://kueue.sigs.k8s.io/docs/concepts/admission_check/) | Candidate extension seam for token policy; arbitrary label costs need an explicit integration design. |
| Weighted concurrency analogy | [Airflow pools](https://airflow.apache.org/docs/apache-airflow/2.11.0/administration-and-deployment/pools.html) | Explain per-job token costs; this does not add DAG execution to project scope. |

The ring latency/bandwidth expression in the roadmap was supplied in the project brief. It is a simplified model for an experiment, excluding reduction computation and other overhead. No benchmark results have been collected yet.
