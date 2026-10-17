"""Reclaim only Docker resources identified by recorded validation owners."""

import argparse
from pathlib import Path

from validation_support import Docker, Resources, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failed = False
    for path in args.output.rglob("resources.json"):
        resources = Resources.restore(Docker(path.parent), read_json(path))
        result = resources.cleanup()
        print(f"cleanup owner={resources.owner} status={result['status']}", flush=True)
        failed = failed or result["status"] != "passed"
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
