"""Focused installed-package checks using real workers and subprocess cleanup."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import torch

import meshtrain
from meshtrain import Communicator
from meshtrain.torch_adapter import GradientSynchronizer
from meshtrain.training.artifacts import digest, read_json, write_json, write_npz
from meshtrain.training.inputs import load_inputs, native_config, prepare
from meshtrain.training.launcher import WorkerGroupError, supervise, worker_commands
from meshtrain.training.verification import verify
from meshtrain.training.workload import configure_execution, flat_gradients, model


def adapter_worker(root, rank, overflow):
    config = read_json(root / "config.json")
    network = model()
    value = torch.finfo(torch.float32).max if overflow else rank + 1.0
    for parameter in network.parameters():
        parameter.grad = torch.full_like(parameter, value)
    synchronizer = GradientSynchronizer(network, 2)
    with Communicator(native_config(config, rank)) as communicator:
        if overflow:
            try:
                synchronizer.synchronize(communicator)
            except FloatingPointError:
                assert communicator.next_sequence == 1
                for parameter in network.parameters():
                    assert torch.equal(parameter.grad, torch.full_like(parameter, value))
            else:
                raise AssertionError("overflow was accepted")
        else:
            synchronizer.synchronize(communicator)
            for parameter in network.parameters():
                torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 1.5), rtol=0, atol=0)
            assert communicator.next_sequence == 1


def fixture(kind, root):
    configure_execution()
    if kind == "failure":
        config = read_json(root / "config.json")
        commands = worker_commands(root, config)
        commands[1][commands[1].index("--config") + 1] = str(root / "missing.json")
    elif kind == "pipes":
        code = "import os; os.write(1, b'a'*262144); os.write(2, b'b'*262144)"
        commands = [[sys.executable, "-I", "-c", code]]
    else:
        commands = [[sys.executable, "-I", "-c", "import time; time.sleep(30)"]] * 2
    print("READY", flush=True)
    if sys.stdin.readline().strip() != "GO":
        raise ValueError("fixture requires GO")
    spawned = []

    def spawn(command, **kwargs):
        if kind == "spawn" and spawned:
            raise OSError("deliberate later spawn failure")
        process = subprocess.Popen(command, **kwargs)
        spawned.append(process)
        write_json(root / "fixture-pids.json", [item.pid for item in spawned])
        if kind == "interrupt" and len(spawned) == 2:
            signal.setitimer(signal.ITIMER_REAL, 0.1)
        return process

    def interrupt(*_):
        raise KeyboardInterrupt("deliberate fixture interruption")

    signal.signal(signal.SIGALRM, interrupt)
    try:
        with (root / "console.log").open("w") as console, contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
            try:
                supervise(commands, root, timeout=2, grace=0.1, spawn=spawn)
                assert kind == "pipes"
            except (WorkerGroupError, OSError, KeyboardInterrupt):
                assert kind != "pipes"
        assert all(process.poll() is not None for process in spawned)
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()
    if kind == "failure":
        report = read_json(root / "group.json")
        assert "rank 1" in report["primary_error"], report
        assert "missing.json" in (root / "ranks/1/stderr.log").read_text()
        assert all(code is not None for code in report["exits"].values())
    if kind == "pipes":
        assert (root / "ranks/0/stdout.log").stat().st_size == 262144
        assert (root / "ranks/0/stderr.log").stat().st_size == 262144


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix="meshtrain-trainer-checks-")
        cls.root = Path(cls.scratch.name)
        cls.source = Path(sys.argv[1]).resolve()

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def test_optional_cuda_adapter_rejects_cpu_only_runtime(self):
        from meshtrain.cuda_adapter import CudaStagingAdapter
        self.assertIsNone(torch.version.cuda)
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            CudaStagingAdapter(1, device="cuda:0")
        with self.assertRaises(ValueError):
            CudaStagingAdapter(1, device="cuda:256")

    def test_installed_runtime(self):
        self.assertTrue(Path(meshtrain.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()))
        self.assertEqual(torch.__version__, "2.14.1+cpu")
        self.assertEqual(np.__version__, "2.5.4")
        self.assertIsNone(torch.version.cuda)
        print(f"installed={meshtrain.__file__} torch={torch.__version__} numpy={np.__version__}")

    def test_adapter_validation_ownership_and_failed_call(self):
        network = model()
        sync = GradientSynchronizer(network, 1)

        class Unexpected:
            def all_reduce_sum(self, _):
                raise AssertionError("preflight admitted invalid input")

        with self.assertRaisesRegex(ValueError, "missing gradient"):
            sync.synchronize(Unexpected())
        for parameter in network.parameters():
            parameter.grad = torch.ones_like(parameter)
        first = next(network.parameters())
        first.grad = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.int64), [], first.shape, check_invariants=True)
        with self.assertRaisesRegex(ValueError, "dense CPU"):
            sync.synchronize(Unexpected())
        first.grad = torch.ones_like(first)
        first.grad[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            sync.synchronize(Unexpected())
        first.grad = torch.ones(first.shape[::-1]).T
        self.assertFalse(first.grad.is_contiguous())
        snapshot = flat_gradients(network)
        sync._view[0] = 17
        self.assertEqual(sync.staging[0].item(), 17)
        self.assertEqual(sync._view.ctypes.data, sync.staging.data_ptr())

        class Failure:
            def all_reduce_sum(self, array):
                array[:] = 123
                raise RuntimeError("native call failed")

        with self.assertRaisesRegex(RuntimeError, "native call failed"):
            sync.synchronize(Failure())
        np.testing.assert_array_equal(flat_gradients(network), snapshot)
        first.grad.zero_()
        self.assertTrue(np.all(snapshot == 1))
        network[0].weight = torch.nn.Parameter(first.detach().clone())
        with self.assertRaisesRegex(ValueError, "identity or layout"):
            sync.synchronize(Unexpected())
        for unsupported in (model().double(), model().to("meta")):
            with self.assertRaisesRegex(ValueError, "CPU float32"):
                GradientSynchronizer(unsupported, 1)
        changed = model()
        changed_sync = GradientSynchronizer(changed, 1)
        changed[0].weight = torch.nn.Parameter(torch.ones(31, 16))
        with self.assertRaisesRegex(ValueError, "layout"):
            changed_sync.synchronize(Unexpected())
        with self.assertRaises(ValueError):
            GradientSynchronizer(model(), True)

    def test_real_native_average_and_overflow(self):
        for overflow in (False, True):
            root = self.root / f"adapter-{overflow}"
            prepare(root, world_size=2, steps=1)
            commands = [[sys.executable, "-I", str(Path(__file__).resolve()), "--adapter", str(root), str(rank), str(int(overflow))]
                        for rank in range(2)]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                supervise(commands, root, timeout=30, grace=0.1)

    def test_prepared_schedule_and_input_identity(self):
        root, config, _, features, labels, indices = load_inputs(self.source)
        world, batch = config["workload"]["world_size"], config["workload"]["local_batch_size"]
        for step in range(len(indices)):
            expected = np.array([[(step * world * batch + rank * batch + item) % 1024
                                  for item in range(batch)] for rank in range(world)])
            np.testing.assert_array_equal(indices[step], expected)
            self.assertEqual(len(np.unique(indices[step])), world * batch)
        expected_features = torch.randn(1024, 16, generator=torch.Generator().manual_seed(5678))
        teacher = torch.randn(16, 4, generator=torch.Generator().manual_seed(5679))
        torch.testing.assert_close(features, expected_features, rtol=0, atol=0)
        torch.testing.assert_close(labels, (features @ teacher).argmax(1), rtol=0, atol=0)
        self.assertEqual(sum(item["count"] for item in config["context"]["layout"]), 676)
        self.assertEqual(set(torch.load(root / "inputs/initial.pt", weights_only=True)), set(model().state_dict()))
        with self.assertRaisesRegex(ValueError, "fresh empty"):
            prepare(root)

    def test_independent_worker_entrypoint(self):
        root = self.root / "independent"
        subprocess.run([sys.executable, "-I", "-m", "meshtrain.training", "prepare", "--output", str(root),
                        "--world-size", "2", "--steps", "3"], check=True, timeout=30, capture_output=True)
        config = read_json(root / "config.json")
        processes = [subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                     for command in worker_commands(root, config)]
        try:
            for process in processes:
                _, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, stderr.decode())
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=2)
        subprocess.run([sys.executable, "-I", "-m", "meshtrain.training", "verify", "--output", str(root)],
                       check=True, timeout=30, capture_output=True)
        report = read_json(root / "verification.json")
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["full_acceptance"])

    def test_artifact_rejection_invalidates_prior_success(self):
        for kind in ("missing", "identity", "shape", "nonfinite", "perturbed", "state_keys", "input_digest"):
            with self.subTest(kind=kind):
                root = self.root / kind
                shutil.copytree(self.source, root)
                rank = root / "ranks/0"
                if kind == "missing":
                    (rank / "complete.json").unlink()
                elif kind == "identity":
                    metadata = read_json(rank / "complete.json")
                    metadata["run_id"] = "stale"
                    write_json(rank / "complete.json", metadata)
                elif kind == "input_digest":
                    with (root / "inputs/data.npz").open("ab") as stream:
                        stream.write(b"changed")
                elif kind == "state_keys":
                    state = torch.load(rank / "final.pt", weights_only=True)
                    state.pop("0.bias")
                    torch.save(state, rank / "final.pt")
                else:
                    with np.load(rank / "trace.npz", allow_pickle=False) as archive:
                        trace = {key: archive[key].copy() for key in archive.files}
                    if kind == "shape":
                        trace["gradients"] = trace["gradients"][:, :-1]
                    else:
                        trace["gradients"][0, 0] = float("nan") if kind == "nonfinite" else 10
                    write_npz(rank / "trace.npz", **trace)
                if kind in ("shape", "nonfinite", "perturbed", "state_keys"):
                    metadata = read_json(rank / "complete.json")
                    metadata["files"] = {name: digest(rank / name) for name in metadata["files"]}
                    write_json(rank / "complete.json", metadata)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    verify(root)
                report = read_json(root / "verification.json")
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["full_acceptance"])
                if kind == "perturbed":
                    self.assertGreater(report["failure_count"], 0)
                    check = next(check for check in report["checks"] if check["first_failure"])
                    self.assertEqual(check["first_failure"]["name"], "0.bias")

    def test_supervisor_failure_pipes_spawn_and_interruption(self):
        for kind in ("failure", "pipes", "spawn", "interrupt"):
            with self.subTest(kind=kind):
                root = self.root / f"supervisor-{kind}"
                root.mkdir()
                if kind == "failure":
                    prepare(root, world_size=2, steps=1)
                process = subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), "--fixture", kind, str(root)],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True, start_new_session=True)
                try:
                    self.assertEqual(process.stdout.readline().strip(), "READY")
                    started = time.monotonic()
                    stdout, stderr = process.communicate("GO\n", timeout=5)
                    self.assertEqual(process.returncode, 0, stdout + stderr)
                    self.assertLess(time.monotonic() - started, 5)
                    for pid in read_json(root / "fixture-pids.json"):
                        with self.assertRaises(ProcessLookupError):
                            os.kill(pid, 0)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)


if __name__ == "__main__":
    if sys.argv[1] == "--adapter":
        configure_execution()
        adapter_worker(Path(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4])))
    elif sys.argv[1] == "--fixture":
        fixture(sys.argv[2], Path(sys.argv[3]))
    else:
        configure_execution()
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(not result.wasSuccessful())
