from __future__ import annotations

import gc
import threading
import time
import traceback
import weakref

import numpy as np

from meshtrain import Communicator, Endpoint, EngineError, RunConfig


class TrackedArray(np.ndarray):
    releases = []

    def __buffer__(self, flags):
        return memoryview(self.view(np.ndarray))

    def __release_buffer__(self, view):
        self.releases.append(threading.current_thread().name)


def config(rank, ports, run_id, **overrides):
    values = dict(
        rank=rank,
        world_size=2,
        run_id=run_id,
        context_fingerprint="python-binding-checks-v1",
        endpoints=[Endpoint("127.0.0.1", port) for port in ports],
        startup_timeout_ms=3000,
        collective_timeout_ms=1000,
    )
    values.update(overrides)
    return RunConfig(**values)


def background(function):
    started = threading.Event()
    finished = threading.Event()
    result = []
    errors = []

    def invoke():
        started.set()
        try:
            result.append(function())
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    assert started.wait(1), "call thread did not start"
    return thread, finished, result, errors


def numerical(connection, settings):
    rank = settings["rank"]
    with Communicator(config(**settings)) as communicator:
        for sequence, count in enumerate((0, 1, 1025, 1)):
            base = np.full(count + 2, -997, dtype=np.float32)
            view = base[1:-1]
            view[:] = np.arange(count, dtype=np.float32) + rank
            pointer = view.ctypes.data
            assert communicator.all_reduce_sum(view) is None
            np.testing.assert_array_equal(view, 2 * np.arange(count, dtype=np.float32) + 1)
            assert view.ctypes.data == pointer
            assert base[0] == base[-1] == -997
            assert communicator.next_sequence == sequence + 1
        connection.send(("result", {"calls": 4, "max_absolute_error": 0.0}))


def startup_progress(connection, settings):
    start = time.monotonic()
    thread, finished, results, errors = background(lambda: Communicator(config(**settings)))
    assert not finished.wait(0.05), "startup did not remain blocked for the withheld peer"
    connection.send(("progress", time.monotonic() - start))
    thread.join(4)
    assert not thread.is_alive(), "startup thread did not finish"
    assert not errors, repr(errors)
    assert results[0].state == "ready"
    results[0].close()
    connection.send(("result", "startup progress"))


def controlled_peer(connection, settings):
    with Communicator(config(**settings)) as communicator:
        connection.send(("ready", None))
        command = connection.recv()
        if command == "sum":
            values = np.full(1025, 2, dtype=np.float32)
            communicator.all_reduce_sum(values)
            np.testing.assert_array_equal(values, np.full(1025, 3, dtype=np.float32))
        elif command != "stop":
            raise AssertionError(f"unexpected peer command: {command}")
    connection.send(("result", "peer stopped"))


def collective_wait(connection, settings, mode):
    communicator = Communicator(config(**settings))
    communicator_ref = weakref.ref(communicator)
    connection.send(("ready", None))
    assert connection.recv() == "begin"
    owner = np.full(1027, -997, dtype=np.float32)
    view = owner[1:-1].view(TrackedArray)
    view[:] = 1
    owner_ref, view_ref = weakref.ref(owner), weakref.ref(view)
    finalized = []
    weakref.finalize(owner, finalized.append, "owner")
    weakref.finalize(view, finalized.append, "view")
    arguments = [view]
    del owner, view

    def reduce():
        communicator.all_reduce_sum(arguments.pop())

    start = time.monotonic()
    thread, finished, results, errors = background(reduce)
    assert not finished.wait(0.05), "collective did not remain blocked for the withheld peer"
    gc.collect()
    assert owner_ref() is not None and view_ref() is not None
    assert not finalized
    assert not TrackedArray.releases
    assert communicator.state == "ready" and communicator.next_sequence == 0
    if mode == "success":
        try:
            communicator.all_reduce_sum(np.ones(1025, dtype=np.float32))
        except EngineError as error:
            assert error.code == "invalid_state" and error.sequence == 0
        else:
            raise AssertionError("overlapping collective was accepted")
        assert communicator.state == "ready" and not finished.is_set()
        communicator = None
        gc.collect()
        assert communicator_ref() is not None, "active call lost its communicator owner"
    connection.send(("progress", time.monotonic() - start))
    if mode == "close":
        communicator.close()
    thread.join(2)
    assert not thread.is_alive(), "collective thread did not finish"
    elapsed = time.monotonic() - start
    if mode == "success":
        assert not errors, repr(errors)
        assert results == [None]
    else:
        assert len(errors) == 1 and isinstance(errors[0], EngineError), repr(errors)
        error = errors[0]
        expected_codes = {"timeout"} if mode == "timeout" else {"invalid_state", "peer_closed"}
        assert error.code in expected_codes, str(error)
        assert error.rank == 0 and error.run_id == settings["run_id"]
        assert error.sequence == 0 and error.operation == "all_reduce_sum.receive_header", str(error)
        assert communicator.state == ("failed" if mode == "timeout" else "closed")
        if mode == "timeout":
            assert error.peer == 1 and error.peer_endpoint.endswith(str(settings["ports"][1]))
            assert 0.8 <= elapsed < 5, elapsed
        try:
            communicator.all_reduce_sum(np.ones(1025, dtype=np.float32))
        except EngineError as refused:
            assert refused.code == "invalid_state" and refused.sequence == 0
        else:
            raise AssertionError("terminal communicator was reused")
    for error in errors:
        error.__traceback__ = None
    errors.clear()
    gc.collect()
    assert owner_ref() is None and view_ref() is None, "finished call retained its input"
    assert sorted(finalized) == ["owner", "view"]
    assert len(TrackedArray.releases) == 1, "export was not released exactly once"
    if communicator is not None:
        communicator.close()
        communicator.close()
    else:
        assert communicator_ref() is None, "finished call retained its communicator"
    connection.send(("result", {"mode": mode, "elapsed_seconds": elapsed}))


def worker(connection, scenario, settings):
    try:
        if scenario == "numerical":
            numerical(connection, settings)
        elif scenario == "startup":
            startup_progress(connection, settings)
        elif scenario == "peer":
            controlled_peer(connection, settings)
        else:
            collective_wait(connection, settings, scenario)
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        raise
    finally:
        connection.close()
