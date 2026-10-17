from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

def main():
    parser = argparse.ArgumentParser(description="Native-engine CPU MLP/reference demonstration")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "worker", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name in ("prepare", "run"):
            command.add_argument("--world-size", type=int, default=4)
            command.add_argument("--steps", type=int, default=100)
            command.add_argument("--local-batch-size", type=int, default=8)
            command.add_argument("--endpoints", type=json.loads, help='JSON list of {"host": ..., "port": ...}')
            command.add_argument("--run-id")
            command.add_argument("--listen-address", default="0.0.0.0")
            command.add_argument("--startup-timeout-ms", type=int, default=30000)
            command.add_argument("--collective-timeout-ms", type=int, default=30000)
        if name == "run":
            command.add_argument("--worker-timeout", type=float, default=120)
            command.add_argument("--termination-grace", type=float, default=5)
        if name == "worker":
            command.add_argument("--config", type=Path, required=True)
            command.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    if args.command == "worker" and not args.config.is_file():
        raise FileNotFoundError(f"worker configuration does not exist: {args.config}")
    from .artifacts import write_json
    from .inputs import prepare
    from .launcher import supervise, worker_commands
    from .verification import verify
    from .worker import train_worker
    from .workload import configure_execution

    configure_execution()
    if args.command in ("prepare", "run"):
        config = prepare(args.output, args.world_size, args.steps, args.local_batch_size,
                         args.endpoints, args.run_id, args.listen_address,
                         args.startup_timeout_ms, args.collective_timeout_ms)
        if args.command == "prepare":
            print(f"prepared {args.output / 'config.json'} run_id={config['run_id']}")
            return
        group = supervise(worker_commands(args.output, config), args.output,
                          args.worker_timeout, args.termination_grace)
        report = verify(args.output)
        write_json(args.output / "group.json", group)
    elif args.command == "worker":
        train_worker(args.output, args.config, args.rank)
        return
    else:
        report = verify(args.output)
    print(f"verification={report['status']} full_acceptance={report['full_acceptance']} output={args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        if hasattr(error, "code"):
            print({key: getattr(error, key) for key in ("code", "rank", "run_id", "operation", "sequence", "peer", "peer_endpoint")},
                  file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
