"""Command line entry point.

A cluster is a JSONL node list; a task set is a JSONL task list. Both are validated with line numbers,
so a bad request says exactly where it went wrong. Every subcommand writes one JSON document to stdout;
errors are one JSON document on stderr. Exit codes: 0 success, 2 input/usage error, 3 a report whose
verdict is negative (tasks left unplaced, or two runs that disagreed).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from . import __version__
from .errors import ParseError, SchedulerError, ValidationError
from .model import Node, Resources, Task
from .policies import POLICIES
from .simulator import Simulation, compare_policies, replay, simulate

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_NEGATIVE = 3


def canonical(document: dict[str, Any]) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _emit(document: dict[str, Any]) -> None:
    sys.stdout.write(canonical(document) + "\n")


def _read_lines(path: str) -> list[str]:
    if path == "-":
        return sys.stdin.read().splitlines()
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().splitlines()
    except OSError as error:
        raise ValidationError(f"cannot read input: {error.strerror or error}", value=path) from error


def _rows(path: str, what: str) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for number, text in enumerate(_read_lines(path), start=1):
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            document = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ParseError(f"{what} line {number}: invalid JSON: {error.msg}", line=number) from error
        if not isinstance(document, dict):
            raise ParseError(f"{what} line {number}: expected a JSON object", line=number)
        rows.append((number, document))
    return rows


def _pairs(document: dict[str, Any], key: str, number: int, what: str) -> tuple[tuple[str, str], ...]:
    value = document.get(key, {})
    if value in (None, {}):
        return ()
    if not isinstance(value, dict):
        raise ParseError(f"{what} line {number}: {key} must be an object", line=number)
    return tuple(sorted((str(item), str(target)) for item, target in value.items()))


def _strings(document: dict[str, Any], key: str, number: int, what: str) -> tuple[str, ...]:
    value = document.get(key, [])
    if value in (None, []):
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ParseError(f"{what} line {number}: {key} must be a list of strings", line=number)
    return tuple(sorted(str(item) for item in value))


def _resources(document: dict[str, Any], number: int, what: str) -> Resources:
    cpu = document.get("cpu", 0)
    memory = document.get("memory", 0)
    for name, value in (("cpu", cpu), ("memory", memory)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ParseError(f"{what} line {number}: {name} must be an integer", line=number)
    return Resources(cpu, memory)


def load_cluster(path: str) -> tuple[Node, ...]:
    nodes: list[Node] = []
    for number, document in _rows(path, "cluster"):
        unknown = sorted(set(document) - {"id", "cpu", "memory", "labels", "taints"})
        if unknown:
            raise ParseError(f"cluster line {number}: unknown field(s): {', '.join(unknown)}", line=number)
        if "id" not in document:
            raise ParseError(f"cluster line {number}: missing id", line=number)
        nodes.append(
            Node(
                id=str(document["id"]),
                capacity=_resources(document, number, "cluster"),
                labels=_pairs(document, "labels", number, "cluster"),
                taints=_strings(document, "taints", number, "cluster"),
            )
        )
    if not nodes:
        raise ValidationError("the cluster is empty")
    return tuple(nodes)


def load_tasks(path: str) -> tuple[Task, ...]:
    tasks: list[Task] = []
    known = {"id", "cpu", "memory", "priority", "queue", "arrival", "duration", "affinity", "antiAffinity", "tolerations"}
    for number, document in _rows(path, "task"):
        unknown = sorted(set(document) - known)
        if unknown:
            raise ParseError(f"task line {number}: unknown field(s): {', '.join(unknown)}", line=number)
        if "id" not in document:
            raise ParseError(f"task line {number}: missing id", line=number)
        for name in ("priority", "arrival", "duration"):
            value = document.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ParseError(f"task line {number}: {name} must be an integer", line=number)
        tasks.append(
            Task(
                id=str(document["id"]),
                request=_resources(document, number, "task"),
                priority=int(document.get("priority", 0)),
                queue=str(document.get("queue", "default")),
                arrival=int(document.get("arrival", 0)),
                duration=int(document.get("duration", 1)),
                affinity=_pairs(document, "affinity", number, "task"),
                anti_affinity=_strings(document, "antiAffinity", number, "task"),
                tolerations=_strings(document, "tolerations", number, "task"),
            )
        )
    if not tasks:
        raise ValidationError("the task set is empty")
    return tuple(tasks)


def load_queue_weights(path: str) -> dict[str, int]:
    """Parse the optional JSONL ``{"queue": ..., "weight": ...}`` file.

    Shape problems -- broken JSON, a non-object row, missing fields, wrong types, unknown fields --
    are parse errors naming the raw line. Semantic problems -- a non-positive weight or a queue
    declared twice -- are validation errors. A file with no valid rows is rejected rather than
    silently treated as "no config" (which would run the baseline instead of the requested mode).
    """
    weights: dict[str, int] = {}
    for number, document in _rows(path, "queue weights"):
        unknown = sorted(set(document) - {"queue", "weight"})
        if unknown:
            raise ParseError(f"queue weights line {number}: unknown field(s): {', '.join(unknown)}", line=number)
        if "queue" not in document:
            raise ParseError(f"queue weights line {number}: missing queue", line=number)
        if "weight" not in document:
            raise ParseError(f"queue weights line {number}: missing weight", line=number)
        queue = document["queue"]
        weight = document["weight"]
        if not isinstance(queue, str) or not queue:
            raise ParseError(f"queue weights line {number}: queue must be a non-empty string", line=number)
        if isinstance(weight, bool) or not isinstance(weight, int):
            raise ParseError(f"queue weights line {number}: weight must be an integer", line=number)
        if weight <= 0:
            raise ValidationError(
                f"queue weights line {number}: weight must be positive", line=number, queue=queue, value=weight
            )
        if queue in weights:
            raise ValidationError(
                f"queue weights line {number}: duplicate queue {queue!r}", line=number, queue=queue
            )
        weights[queue] = weight
    if not weights:
        raise ValidationError("the queue weights file is empty", value=path)
    return weights


def _options(args: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {
        "policy": args.policy,
        "allow_preemption": args.preemption,
        "backfill": not args.no_backfill,
    }
    path = getattr(args, "queue_weights", None)
    if path:
        options["queue_weights"] = load_queue_weights(path)
    return options


# -- commands ------------------------------------------------------------------------------------
def _command_describe(_: argparse.Namespace) -> int:
    _emit(
        {
            "name": "cluster-scheduler-simulator",
            "version": __version__,
            "subcommands": ["describe", "metrics", "policies", "replay", "simulate", "trace", "validate"],
            "policies": list(POLICIES),
            "clusterFields": ["id", "cpu", "memory", "labels", "taints"],
            "taskFields": [
                "id",
                "cpu",
                "memory",
                "priority",
                "queue",
                "arrival",
                "duration",
                "affinity",
                "antiAffinity",
                "tolerations",
            ],
            "deterministic": "no wall clock, no set iteration, ties always break on id",
            "exitCodes": {"ok": EXIT_OK, "error": EXIT_ERROR, "negativeVerdict": EXIT_NEGATIVE},
        }
    )
    return EXIT_OK


def _command_validate(args: argparse.Namespace) -> int:
    nodes = load_cluster(args.cluster)
    tasks = load_tasks(args.tasks)
    # Constructing the simulation *is* the validation: `Cluster` catches duplicate node ids and
    # `Simulation` catches duplicate task ids. It takes no options on purpose -- `validate` checks the
    # inputs, and the policy is already constrained by argparse. (Reading `_options(args)` here crashed:
    # the validate subparser has no policy flags, which the test suite caught immediately.)
    Simulation(nodes=nodes, tasks=tasks)
    _emit(
        {
            "nodes": len(nodes),
            "tasks": len(tasks),
            "totalCapacity": _total_capacity(nodes),
            "requested": _total_requested(tasks),
        }
    )
    return EXIT_OK


def _total_capacity(nodes: tuple[Node, ...]) -> dict[str, int]:
    cpu = sum(node.capacity.cpu for node in nodes)
    memory = sum(node.capacity.memory for node in nodes)
    return {"cpu": cpu, "memory": memory}


def _total_requested(tasks: tuple[Task, ...]) -> dict[str, int]:
    cpu = sum(task.request.cpu for task in tasks)
    memory = sum(task.request.memory for task in tasks)
    return {"cpu": cpu, "memory": memory}


def _command_simulate(args: argparse.Namespace) -> int:
    result = simulate(load_cluster(args.cluster), load_tasks(args.tasks), **_options(args))
    _emit(result.to_document())
    return EXIT_OK if not result.unplaced else EXIT_NEGATIVE


def _command_trace(args: argparse.Namespace) -> int:
    result = simulate(load_cluster(args.cluster), load_tasks(args.tasks), **_options(args))
    _emit(
        {
            "policy": result.policy,
            "makespan": result.makespan,
            "decisions": result.decisions,
            "unplaced": result.unplaced,
        }
    )
    return EXIT_OK if not result.unplaced else EXIT_NEGATIVE


def _command_metrics(args: argparse.Namespace) -> int:
    result = simulate(load_cluster(args.cluster), load_tasks(args.tasks), **_options(args))
    _emit({"policy": result.policy, "metrics": result.metrics})
    return EXIT_OK if not result.unplaced else EXIT_NEGATIVE


def _command_policies(args: argparse.Namespace) -> int:
    report = compare_policies(load_cluster(args.cluster), load_tasks(args.tasks), **{k: v for k, v in _options(args).items() if k != "policy"})
    _emit(report)
    best = min(report["policies"], key=lambda entry: (entry["unplaced"], entry["makespan"], entry["policy"]))
    return EXIT_OK if best["unplaced"] == 0 else EXIT_NEGATIVE


def _command_replay(args: argparse.Namespace) -> int:
    report = replay(load_cluster(args.cluster), load_tasks(args.tasks), **_options(args))
    _emit(report)
    return EXIT_OK if report["identical"] else EXIT_NEGATIVE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cluster-scheduler-simulator", description="deterministic cluster scheduling simulation")
    parser.add_argument("--version", action="version", version=f"cluster-scheduler-simulator {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("describe", help="print capabilities as JSON").set_defaults(handler=_command_describe)

    def with_inputs(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--cluster", required=True, help="JSONL nodes, or - for stdin")
        sub.add_argument("--tasks", required=True, help="JSONL tasks")
        return sub

    def with_options(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sub.add_argument("--policy", choices=list(POLICIES), default="first-fit")
        sub.add_argument("--preemption", action="store_true", help="allow higher priorities to evict lower ones")
        sub.add_argument("--no-backfill", action="store_true", help="disable conservative backfill")
        sub.add_argument(
            "--queue-weights",
            metavar="PATH",
            help="JSONL {queue, weight} rows; enables weighted fair-share scheduling",
        )
        return sub

    with_inputs("validate", "parse and validate both inputs").set_defaults(handler=_command_validate)
    with_options(with_inputs("simulate", "run the simulation")).set_defaults(handler=_command_simulate)
    with_options(with_inputs("trace", "per-decision scheduling trace")).set_defaults(handler=_command_trace)
    with_options(with_inputs("metrics", "metrics only")).set_defaults(handler=_command_metrics)
    with_options(with_inputs("replay", "run twice and check the traces match")).set_defaults(handler=_command_replay)
    with_options(with_inputs("policies", "compare first-fit and best-fit")).set_defaults(handler=_command_policies)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except SchedulerError as error:
        sys.stderr.write(canonical(error.to_document()) + "\n")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
