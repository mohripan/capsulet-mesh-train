"""Standard-library command evidence and ownership-scoped Docker resources."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

LABEL = "org.meshtrain.validation.run"


class GroupFailure(RuntimeError):
    def __init__(self, message, states):
        super().__init__(message)
        self.states = states


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fresh(path):
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"use a fresh empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def remaining(deadline, cap=5):
    value = min(cap, deadline - time.monotonic())
    if value <= 0:
        raise TimeoutError("validation phase deadline expired")
    return value


def signal_handlers():
    def terminate(number, _):
        raise InterruptedError(f"received signal {number}")
    signal.signal(signal.SIGTERM, terminate)


class Docker:
    def __init__(self, output):
        self.output = Path(output)
        (self.output / "commands").mkdir(parents=True, exist_ok=True)
        path = self.output / "commands.json"
        self.commands = read_json(path) if path.exists() else []

    def call(self, arguments, timeout=5, check=True):
        index = len(self.commands)
        started = time.monotonic()
        record = dict(arguments=["docker", *map(str, arguments)], status="running")
        self.commands.append(record)
        try:
            try:
                result = subprocess.run(record["arguments"], capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(f"Docker command deadline expired: {arguments[0]}") from error
            (self.output / f"commands/{index:04d}.out").write_text(result.stdout, encoding="utf-8")
            (self.output / f"commands/{index:04d}.err").write_text(result.stderr, encoding="utf-8")
            record.update(status="passed" if result.returncode == 0 else "failed", exit_code=result.returncode)
            if check and result.returncode:
                raise RuntimeError(f"docker {' '.join(map(str, arguments))}: {result.stderr.strip()}")
            return result
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            write_json(self.output / "commands.json", self.commands)

    def logs(self, name, directory, timeout=5):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            subprocess.run(["docker", "logs", name], stdout=stdout, stderr=stderr, check=True, timeout=timeout)

    def states(self, names, timeout=5):
        data = json.loads(self.call(["inspect", *names], timeout=timeout).stdout)
        return [dict(name=item["Name"].lstrip("/"), id=item["Id"], image=item["Image"], state=item["State"],
                     networks=list(item["NetworkSettings"]["Networks"]), mounts=item["Mounts"],
                     pid_mode=item["HostConfig"]["PidMode"], entrypoint=item["Config"]["Entrypoint"],
                     command=item["Config"]["Cmd"]) for item in data]


class Resources:
    def __init__(self, docker, image):
        self.docker = docker
        self.owner = uuid.uuid4().hex
        self.prefix = "meshtrain-validation-" + self.owner
        self.containers = []
        self.networks = []
        self.image = docker.call(["image", "inspect", image, "--format", "{{.Id}}"]).stdout.strip()
        self.ids = {}
        self.cleanup_result = None
        self.cleanup_history = []
        self.persist()

    def persist(self):
        write_json(self.docker.output / "resources.json", dict(owner=self.owner, image=self.image,
                   containers=self.containers, networks=self.networks, ids=self.ids, cleanup=self.cleanup_result,
                   cleanup_history=self.cleanup_history))

    @classmethod
    def restore(cls, docker, manifest):
        owner = manifest["owner"]
        if uuid.UUID(hex=owner).hex != owner:
            raise ValueError("invalid resource ownership token")
        resources = cls.__new__(cls)
        resources.docker, resources.owner = docker, owner
        resources.prefix = "meshtrain-validation-" + owner
        resources.image = manifest["image"]
        resources.containers, resources.networks = manifest["containers"], manifest["networks"]
        resources.ids, resources.cleanup_result = manifest["ids"], manifest.get("cleanup")
        resources.cleanup_history = manifest.get("cleanup_history", [])
        return resources

    def network(self):
        name = self.prefix + "-network"
        self.networks.append(name)
        self.persist()
        self.docker.call(["network", "create", "--label", f"{LABEL}={self.owner}", name])
        return name

    def create(self, suffix, arguments, *, entrypoint, network="none", alias=None, init=False):
        name = self.prefix + "-" + suffix
        self.containers.append(name)
        self.persist()
        command = ["create", "--name", name, "--label", f"{LABEL}={self.owner}", "--network", network]
        if alias:
            command += ["--network-alias", alias]
        if init:
            command += ["--init"]
        command += ["--entrypoint", entrypoint, self.image, *map(str, arguments)]
        self.ids[name] = self.docker.call(command).stdout.strip()
        self.persist()
        return name

    def owned(self, deadline):
        owned_filter = f"label={LABEL}={self.owner}"
        containers = self.docker.call(["ps", "-aq", "--no-trunc", "--filter", owned_filter], timeout=remaining(deadline)).stdout.split()
        networks = self.docker.call(["network", "ls", "-q", "--filter", owned_filter], timeout=remaining(deadline)).stdout.split()
        return containers, networks

    def cleanup(self, timeout=15):
        started = time.monotonic()
        deadline = started + timeout
        errors = []
        before_stop, after_stop = [], []
        try:
            containers, _ = self.owned(deadline)
            if containers:
                before_stop = self.docker.states(containers, timeout=remaining(deadline))
                running = [item["id"] for item in before_stop if item["state"]["Running"]]
                if running:
                    self.docker.call(["kill", *running], timeout=remaining(deadline), check=False)
                after_stop = self.docker.states(containers, timeout=remaining(deadline))
            log_deadline = min(deadline - 5, time.monotonic() + 4)
            for item in before_stop:
                name = item["name"]
                try:
                    self.docker.logs(name, self.docker.output / "logs" / name.removeprefix(self.prefix + "-"),
                                     timeout=remaining(log_deadline, 1))
                except Exception as error:
                    errors.append(f"logs/{name}: {error}")
            if containers:
                self.docker.call(["rm", "-f", *containers], timeout=remaining(deadline))
            _, networks = self.owned(deadline)
            if networks:
                self.docker.call(["network", "rm", *networks], timeout=remaining(deadline))
            containers, networks = self.owned(deadline)
            if containers or networks:
                errors.append(f"remaining owned resources: {containers}, {networks}")
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        self.cleanup_result = dict(status="passed" if not errors else "failed", errors=errors,
                                   elapsed_seconds=time.monotonic() - started,
                                   before_stop=before_stop, after_stop=after_stop)
        self.cleanup_history.append(self.cleanup_result)
        self.persist()
        return self.cleanup_result


def monitor(docker, names, deadline, *, require_success=True):
    states = []
    while True:
        try:
            states = docker.states(names, timeout=remaining(deadline))
        except TimeoutError as error:
            raise GroupFailure("worker group deadline expired", states) from error
        for rank, item in enumerate(states):
            state = item["state"]
            if state["Status"] in ("exited", "dead") and state["ExitCode"] and require_success:
                raise GroupFailure(f"rank {rank} ({item['name']}) exited with status {state['ExitCode']}: {state['Error']}", states)
        if all(item["state"]["Status"] in ("exited", "dead") for item in states):
            return states
        if time.monotonic() >= deadline:
            raise GroupFailure("worker group deadline expired", states)
        time.sleep(min(0.1, remaining(deadline)))
