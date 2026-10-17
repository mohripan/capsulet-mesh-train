"""Install an sdist-derived wheel and hash-checked offline CPU runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("runtime output must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output / "dist"), str(root)], check=True)
    wheels = list((output / "dist").glob("*.whl"))
    if len(wheels) != 1 or len(list((output / "dist").glob("*.tar.gz"))) != 1:
        raise ValueError("expected one sdist and derived wheel")
    venv.EnvBuilder(with_pip=True).create(output / "runtime")
    python = str(output / "runtime/bin/python")
    lock = (root / "requirements/training-cpu.lock").read_text()
    offline = output / "training-cpu-offline.lock"
    offline.write_text(re.sub(r"torch @ https://\S+", "torch==2.14.1+cpu", lock))
    subprocess.run([python, "-I", "-m", "pip", "install", "--no-index", "--find-links",
                    os.environ["MESHTRAIN_TRAINING_WHEELHOUSE"], "--require-hashes", "-r", str(offline)], check=True)
    subprocess.run([python, "-I", "-m", "pip", "install", "--no-index", "--no-deps", str(wheels[0])], check=True)
    probe = """import json, pathlib, sys, meshtrain, torch, numpy
assert pathlib.Path(meshtrain.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
assert pathlib.Path(meshtrain._native.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
assert torch.__version__ == '2.14.1+cpu' and torch.version.cuda is None
print(json.dumps(dict(package=meshtrain.__file__, extension=meshtrain._native.__file__, python=sys.version, torch=torch.__version__, numpy=numpy.__version__)))"""
    result = subprocess.run([python, "-I", "-c", probe], capture_output=True, text=True, check=True)
    metadata = json.loads(result.stdout)
    metadata["wheel_sha256"] = hashlib.sha256(wheels[0].read_bytes()).hexdigest()
    metadata["lock_sha256"] = hashlib.sha256((root / "requirements/training-cpu.lock").read_bytes()).hexdigest()
    (output / "installed.json").write_text(json.dumps(metadata, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
