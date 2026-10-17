from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import multiprocessing
from pathlib import Path
import socket
import sys
import time
import unittest
import uuid
import weakref

import numpy as np

import meshtrain
from meshtrain import Communicator, Endpoint, EngineError, RunConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workers import worker


def single_config(**overrides):
    values = dict(
        rank=0, world_size=1, run_id="single", context_fingerprint="binding-check",
        endpoints=[Endpoint("127.0.0.1", 12345)],
    )
    values.update(overrides)
    return RunConfig(**values)


class ArrayLike:
    def __array__(self, *args, **kwargs):
        raise AssertionError("array conversion must not be invoked")


class ArraySubclass(np.ndarray):
    pass


class FailedExport(np.ndarray):
    def __buffer__(self, flags):
        raise BufferError("controlled export failure")


class DifferentStorageExport(np.ndarray):
    def __buffer__(self, flags):
        return memoryview(np.ones(3, dtype=np.float32))


class LocalContracts(unittest.TestCase):
    def test_installed_package(self):
        module = Path(meshtrain._native.__file__).resolve()
        self.assertTrue(module.is_relative_to(Path(sys.prefix).resolve()), module)
        self.assertEqual(meshtrain.__version__, importlib.metadata.version("meshtrain"))
        self.assertNotIn("torch", sys.modules)
        self.assertIsNone(importlib.util.find_spec("torch"))
        print(f"Installed extension: {module}\nPython: {sys.version.split()[0]}; NumPy: {np.__version__}", flush=True)

    def test_configuration(self):
        endpoints = [Endpoint("localhost", 12345)]
        config = single_config(endpoints=endpoints)
        endpoints.clear()
        config.endpoints.clear()
        self.assertEqual(len(config.endpoints), 1)
        self.assertEqual((config.startup_timeout_ms, config.collective_timeout_ms), (30000, 30000))
        self.assertEqual(config.max_payload_bytes, 268435456)
        for target, name, value in ((config, "rank", 1), (config.endpoints[0], "port", 8)):
            with self.assertRaises(AttributeError):
                setattr(target, name, value)
        for values in (dict(rank=True), dict(world_size=1.0), dict(max_payload_bytes=False)):
            with self.subTest(values=values), self.assertRaises(TypeError):
                single_config(**values)
        for values in (dict(rank=-1), dict(world_size=2**40), dict(startup_timeout_ms=2**70)):
            with self.subTest(values=values), self.assertRaises(OverflowError):
                single_config(**values)
        for port in (-1, 65536, 2**100):
            with self.subTest(port=port), self.assertRaises(OverflowError):
                Endpoint("localhost", port)
        with self.assertRaises(TypeError):
            Endpoint("localhost", True)
        for values in (dict(world_size=0), dict(run_id=""), dict(collective_timeout_ms=0)):
            with self.subTest(values=values), self.assertRaises(EngineError) as raised:
                single_config(**values)
            self.assertEqual(raised.exception.code, "invalid_configuration")
            self.assertEqual(raised.exception.operation, "validate_config")
            self.assertIsNone(raised.exception.sequence)

    def test_rejected_inputs_and_valid_reuse(self):
        readonly = np.ones(3, dtype=np.float32)
        readonly.flags.writeable = False
        misaligned = np.ndarray((3,), dtype=np.float32, buffer=bytearray(13), offset=1)
        bad = [
            ([1.0], TypeError), (ArrayLike(), TypeError), (1234, TypeError),
            (memoryview(np.ones(3, dtype=np.float32)), TypeError),
            (np.ones(3, dtype=np.float64), TypeError),
            (np.ones(3, dtype=np.int32), TypeError),
            (np.ones(3, dtype=">f4"), TypeError), (readonly, ValueError),
            (np.ones(6, dtype=np.float32)[::2], ValueError),
            (np.ones(3, dtype=np.float32)[::-1], ValueError),
            (misaligned, ValueError), (np.array(1, dtype=np.float32), ValueError),
            (np.ones((1, 3), dtype=np.float32), ValueError),
            (np.ones(3, dtype=np.float32).view(FailedExport), BufferError),
            (np.ones(3, dtype=np.float32).view(DifferentStorageExport), ValueError),
        ]
        with Communicator(single_config()) as communicator:
            for values, category in bad:
                before = values.tobytes() if isinstance(values, np.ndarray) else None
                with self.subTest(type=type(values), category=category), self.assertRaises(category):
                    communicator.all_reduce_sum(values)
                self.assertEqual((communicator.state, communicator.next_sequence), ("ready", 0))
                if before is not None:
                    self.assertEqual(values.tobytes(), before)
            for values in (
                np.empty(0, dtype=np.float32),
                np.ones(3, dtype=np.float32)[::3],
                np.ones(3, dtype=np.float32)[::-3],
                np.ndarray((1,), dtype=np.float32, buffer=bytearray(4), strides=(0,)),
                np.ones(3, dtype=np.float32).view(ArraySubclass),
            ):
                before = values.tobytes()
                self.assertIsNone(communicator.all_reduce_sum(values))
                self.assertEqual(values.tobytes(), before)
            self.assertEqual(communicator.next_sequence, 5)

    def test_identity_bits_payload_failure_and_context_manager(self):
        values = np.array([0x80000000, 0x7FC01234, 0x7F800000, 0x3F800000], dtype=np.uint32).view(np.float32)
        before = values.tobytes()
        with Communicator(single_config()) as communicator:
            communicator.all_reduce_sum(values)
            self.assertEqual(values.tobytes(), before)
        self.assertEqual(communicator.state, "closed")
        communicator.close()
        with self.assertRaisesRegex(RuntimeError, "body error"):
            with Communicator(single_config()) as failed_body:
                raise RuntimeError("body error")
        self.assertEqual(failed_body.state, "closed")
        with Communicator(single_config(max_payload_bytes=4)) as limited:
            with self.assertRaises(EngineError) as raised:
                limited.all_reduce_sum(values)
            error = raised.exception
            self.assertIsInstance(error, RuntimeError)
            self.assertEqual((error.code, error.rank, error.run_id, error.sequence), ("invalid_buffer", 0, "single", 0))
            self.assertEqual(limited.state, "failed")

    def test_finished_inputs_and_communicators_collect(self):
        for _ in range(12):
            communicator = Communicator(single_config())
            values = np.ones(3, dtype=np.float32)
            owner_ref, comm_ref = weakref.ref(values), weakref.ref(communicator)
            communicator.all_reduce_sum(values)
            del values, communicator
            gc.collect()
            self.assertIsNone(owner_ref())
            self.assertIsNone(comm_ref())


class ProcessContracts(unittest.TestCase):
    def setUp(self):
        self.context = multiprocessing.get_context("spawn")
        self.children = []
        sockets = [socket.socket(), socket.socket()]
        try:
            for listener in sockets:
                listener.bind(("127.0.0.1", 0))
            self.ports = [listener.getsockname()[1] for listener in sockets]
        finally:
            for listener in sockets:
                listener.close()
        self.run_id = uuid.uuid4().hex

    def tearDown(self):
        for process, connection in self.children:
            if process.is_alive():
                process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
            connection.close()
            process.close()

    def start(self, rank, scenario):
        parent, child = self.context.Pipe()
        settings = dict(rank=rank, ports=self.ports, run_id=self.run_id)
        process = self.context.Process(target=worker, args=(child, scenario, settings))
        process.start()
        child.close()
        self.children.append((process, parent))
        return process, parent

    def receive(self, child, expected, timeout=5):
        process, connection = child
        self.assertTrue(connection.poll(timeout), f"{expected} exceeded {timeout}s watchdog; exit={process.exitcode}")
        kind, payload = connection.recv()
        self.assertEqual(kind, expected, payload)
        return payload

    def finish(self, child, deadline=None):
        remaining = 5 if deadline is None else max(0, deadline - time.monotonic())
        payload = self.receive(child, "result", remaining)
        remaining = 2 if deadline is None else min(2, max(0, deadline - time.monotonic()))
        child[0].join(remaining)
        self.assertEqual(child[0].exitcode, 0)
        print(f"Process evidence: {payload}", flush=True)
        return payload

    def test_two_process_original_storage_sum(self):
        root, peer = self.start(0, "numerical"), self.start(1, "numerical")
        for child in (root, peer):
            self.assertEqual(self.finish(child)["max_absolute_error"], 0.0)

    def test_startup_releases_gil(self):
        root = self.start(0, "startup")
        elapsed = self.receive(root, "progress")
        peer = self.start(1, "peer")
        self.receive(peer, "ready")
        peer[1].send("stop")
        self.finish(root)
        self.finish(peer)
        print(f"Startup Python handshake: {elapsed:.6f}s", flush=True)

    def blocked_case(self, mode):
        root, peer = self.start(0, mode), self.start(1, "peer")
        self.receive(root, "ready")
        self.receive(peer, "ready")
        deadline = time.monotonic() + 5
        root[1].send("begin")
        elapsed = self.receive(root, "progress", max(0, deadline - time.monotonic()))
        if mode == "success":
            peer[1].send("sum")
        result = self.finish(root, deadline)
        if mode != "success":
            peer[1].send("stop")
        self.finish(peer)
        self.assertLess(result["elapsed_seconds"], 5)
        print(f"Collective Python handshake ({mode}): {elapsed:.6f}s", flush=True)

    def test_wait_lifetime_gil_and_overlapping_call(self):
        self.blocked_case("success")

    def test_close_unblocks_and_releases_input(self):
        self.blocked_case("close")

    def test_timeout_fields_terminal_state_and_released_input(self):
        self.blocked_case("timeout")


if __name__ == "__main__":
    unittest.main(verbosity=2)
