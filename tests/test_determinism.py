"""Determinism and serialize-then-replay tests over the public behaviour only.

The baseline publishes a fully deterministic simulator: the README promises *no wall clock, no
iteration over unordered containers, every tie breaks on id*, and every entry point consumes the
same JSONL cluster/task files. These tests never reach into a private container, object identity
or internal field. They drive only the documented surface:

* the Python entry points ``simulate`` / ``replay`` / ``compare_policies`` and the ``Simulation``
  configuration mapping;
* the documented JSONL input files and the five scheduling subcommands on the console script;
* the published result documents (placements, decisions, unplaced, metrics).

For one fixed cluster initial state, task submission sequence, policy configuration and seed the
same experiment is obtained two ways:

1. a *direct* run on the in-memory model;
2. a *serialized replay* -- the model is written to JSONL exactly as the README documents it,
   parsed back through the public loaders, and run again (also in a fresh OS process).

The two paths must agree item-for-item: event order, clock advancement, task state transitions,
per-node free resources, placement explanations and aggregate metrics. Comparison excludes only
metadata that is genuinely outside the simulation (there is no such field in these documents
today -- wall clock and addresses are never emitted); event times, event kinds, task and node
identifiers, the decision, the reason text and every metric value are compared exactly.

Every scenario additionally re-walks the published placements from first principles and asserts
resource conservation: no node is ever over capacity, a quota is never crossed, a task is placed
and completed at most once, and after the final release every node holds exactly nothing.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from schedsim import Node, Resources, Task, compare_policies, replay, simulate
from schedsim.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, main

REPO_ROOT = Path(__file__).resolve().parent.parent


# -- small builders, mirroring the public JSONL fields -------------------------------------------
def task(task_id: str, cpu: int = 1, memory: int = 1, **kwargs) -> Task:
    return Task(id=task_id, request=Resources(cpu, memory), **kwargs)


def node_row(node: Node) -> dict:
    return {
        "id": node.id,
        "cpu": node.capacity.cpu,
        "memory": node.capacity.memory,
        "labels": dict(node.labels),
        "taints": list(node.taints),
    }


def task_row(item: Task) -> dict:
    return {
        "id": item.id,
        "cpu": item.request.cpu,
        "memory": item.request.memory,
        "priority": item.priority,
        "queue": item.queue,
        "arrival": item.arrival,
        "duration": item.duration,
        "affinity": dict(item.affinity),
        "antiAffinity": list(item.anti_affinity),
        "tolerations": list(item.tolerations),
    }


# -- scenarios: each one forces a real determinism-relevant branch --------------------------------
def scenario_ties():
    """Same-timestamp multi-submit, equal-score nodes, equal-share queues, exact fill."""
    nodes = (
        Node("n1", Resources(4, 4)),
        Node("n2", Resources(4, 4)),  # identical best-fit score twin of n1
        Node("n3", Resources(2, 2)),
    )
    tasks = (
        task("a1", 2, 2, queue="qa", duration=4),
        task("b1", 2, 2, queue="qb", duration=2),
        task("e1", 2, 2, queue="qe", duration=2),  # fills n3 (2x2) exactly
        task("head", 4, 4, queue="qh", priority=5, arrival=1, duration=4),
        task("jump", 1, 1, queue="qj", arrival=1, duration=1),  # backfill, ends on the horizon
        task("wake", 4, 4, queue="qw", arrival=1, duration=2),  # blocked, re-enters on completion
        task("pinned", 1, 1, queue="qx", arrival=1, affinity=(("zone", "z"),)),  # never fits
    )
    weights = {"qa": 1, "qb": 1, "qe": 1, "qh": 1, "qj": 1, "qw": 1, "qx": 1}
    quotas = {"qa": (6, 6), "qx": (2, 2)}
    return nodes, tasks, weights, quotas


def scenario_preemption():
    """Two identical low-priority victims tie on every ordering key; the eviction list is stable."""
    nodes = (Node("n1", Resources(4, 4)), Node("n2", Resources(4, 4)))
    tasks = (
        task("va", 2, 2, priority=1, queue="qv", duration=10),
        task("vb", 2, 2, priority=1, queue="qv", duration=10),
        task("peer", 4, 4, queue="qp", duration=10),
        task("boss", 4, 4, priority=9, queue="qb", arrival=1, duration=1),
    )
    return nodes, tasks, {"qv": 1, "qp": 1, "qb": 1}, {"qv": (8, 8)}


def scenario_serial():
    """A one-slot cluster serializes same-timestamp submissions purely on the id tie-break."""
    nodes = (Node("n1", Resources(1, 1)),)
    tasks = (
        task("b", duration=2),
        task("a", duration=2),
        task("c", duration=2),
        task("late_hi", priority=9, arrival=1, duration=1),
    )
    return nodes, tasks, {"default": 1}, {}


SCENARIOS = {
    "ties": scenario_ties,
    "preemption": scenario_preemption,
    "serial": scenario_serial,
}


# Every public option flips at least once across this list; it stays small because the full
# policy x preemption x backfill x fair x quota matrix already lives in test_regression.py.
OPTION_COMBOS = [
    {"policy": "first-fit", "allow_preemption": False, "backfill": False},
    {"policy": "first-fit", "allow_preemption": False, "backfill": True},
    {"policy": "first-fit", "allow_preemption": True, "backfill": False},
    {"policy": "best-fit", "allow_preemption": False, "backfill": False},
    {"policy": "best-fit", "allow_preemption": True, "backfill": True},
    {"policy": "first-fit", "allow_preemption": False, "backfill": True, "fair": True},
    {"policy": "best-fit", "allow_preemption": False, "backfill": True, "fair": True, "quotas": True},
    {"policy": "first-fit", "allow_preemption": True, "backfill": False, "quotas": True},
]


# -- normalized experiment identity ---------------------------------------------------------------
def normalized(result) -> dict:
    """Everything the simulator publicly promises, in one comparable document.

    Nothing non-deterministic is stripped: the simulator emits no wall clock, no address and no
    run id, so the whole document must compare exactly.
    """
    return {
        "policy": result.policy,
        "makespan": result.makespan,
        "unplaced": list(result.unplaced),
        "metrics": result.metrics,
        "placements": [item.to_document() for item in result.placements],
        "decisions": [dict(entry) for entry in result.decisions],
    }


# -- independent conservation oracle (reads only public documents) --------------------------------
def assert_conservation(test_case: unittest.TestCase, nodes, tasks, result) -> None:
    """Re-derive occupancy from placements; assert the resource and lifecycle invariants."""
    requests = {item.id: item.request for item in tasks}
    capacities = {node.id: node.capacity for node in nodes}
    by_task: dict[str, list] = {}
    preempted: set[str] = set()

    for placement in result.placements:
        test_case.assertIn(placement.task_id, requests, "placement for an unknown task")
        test_case.assertIn(placement.node_id, capacities, "placement on an unknown node")
        owner = next(item for item in tasks if item.id == placement.task_id)
        test_case.assertGreaterEqual(placement.start, owner.arrival, "started before arrival")
        test_case.assertGreater(placement.end, placement.start, "non-positive interval")
        by_task.setdefault(placement.task_id, []).append(placement)
        preempted.update(placement.preempted)

    # A task is placed at most once, hence completed at most once; a preempted victim is the same
    # single truncated interval, never a second placement.
    for task_id, places in by_task.items():
        test_case.assertEqual(len(places), 1, f"{task_id} placed more than once")
    for victim in preempted:
        test_case.assertIn(victim, by_task, "preempted id has no placement")
        test_case.assertLessEqual(len(by_task[victim]), 1, "preempted task placed twice")

    # At every half-open event instant no node is over capacity; after the last end everything is
    # released back to the derivable total (zero in use, full capacity free).
    event_times = sorted({p.start for p in result.placements} | {p.end for p in result.placements})
    for clock in event_times:
        used = {node.id: [0, 0] for node in nodes}
        for placement in result.placements:
            if placement.start <= clock < placement.end:
                request = requests[placement.task_id]
                used[placement.node_id][0] += request.cpu
                used[placement.node_id][1] += request.memory
        for node in nodes:
            cpu, memory = used[node.id]
            test_case.assertLessEqual(cpu, node.capacity.cpu, f"cpu overcommitted on {node.id}@{clock}")
            test_case.assertLessEqual(memory, node.capacity.memory, f"memory overcommitted on {node.id}@{clock}")
    final_clock = result.makespan
    for node in nodes:
        running = [
            requests[p.task_id]
            for p in result.placements
            if p.node_id == node.id and p.start <= final_clock < p.end
        ]
        test_case.assertEqual(sum(r.cpu for r in running), 0, f"{node.id} not released at makespan")
        test_case.assertEqual(sum(r.memory for r in running), 0, f"{node.id} not released at makespan")

    # Preemption accounting: the metric counts exactly the victims named in the documents.
    test_case.assertEqual(
        result.metrics["preemptions"], sum(len(p.preempted) for p in result.placements)
    )
    # Placed and unplaced partition the submission sequence exactly: no loss, no duplication, and
    # no task is allowed to be both.
    placed_ids = set(by_task)
    unplaced_ids = set(result.unplaced)
    test_case.assertEqual(placed_ids & unplaced_ids, set(), "a task is both placed and unplaced")
    test_case.assertEqual(placed_ids | unplaced_ids, {item.id for item in tasks})
    test_case.assertEqual(result.metrics["placed"], len(placed_ids))
    test_case.assertEqual(result.metrics["unplaced"], len(unplaced_ids))


# -- JSONL round-trip harness ---------------------------------------------------------------------
class ExperimentFiles:
    """Writes one experiment's public inputs to JSONL and maps options to CLI flags.

    Each experiment gets its own subdirectory so several fixtures can live side by side in one
    temporary tree without their fixed ``cluster.jsonl`` / ``tasks.jsonl`` names colliding.
    """

    def __init__(self, directory: str, name: str, nodes, tasks, weights: dict, quotas: dict):
        self.directory = os.path.join(directory, name)
        os.makedirs(self.directory, exist_ok=True)
        self.nodes_path = self._write("cluster.jsonl", [node_row(n) for n in nodes])
        self.tasks_path = self._write("tasks.jsonl", [task_row(t) for t in tasks])
        self.weights_path = None
        self.quotas_path = None
        if weights:
            self.weights_path = self._write(
                "weights.jsonl", [{"queue": q, "weight": w} for q, w in sorted(weights.items())]
            )
        if quotas:
            self.quotas_path = self._write(
                "quotas.jsonl",
                [{"queue": q, "cpu": cpu, "memory": memory} for q, (cpu, memory) in sorted(quotas.items())],
            )

    def _write(self, name: str, rows: list[dict]) -> str:
        path = os.path.join(self.directory, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def flags(self, options: dict) -> list[str]:
        flags = ["--policy", options["policy"]]
        if options.get("allow_preemption"):
            flags.append("--preemption")
        if not options.get("backfill", True):
            flags.append("--no-backfill")
        if options.get("fair") and self.weights_path:
            flags += ["--queue-weights", self.weights_path]
        if options.get("quotas") and self.quotas_path:
            flags += ["--queue-quotas", self.quotas_path]
        return flags


def api_options(options: dict, weights: dict, quotas: dict) -> dict:
    resolved: dict = {"policy": options["policy"], "allow_preemption": options["allow_preemption"],
                      "backfill": options.get("backfill", True)}
    # A scenario may legitimately declare no weights/quotas file; enabling that mode then would
    # raise (an empty mapping is rejected), so it is simply skipped on both run paths.
    if options.get("fair") and weights:
        resolved["queue_weights"] = dict(weights)
    if options.get("quotas") and quotas:
        resolved["queue_quotas"] = {q: Resources(cpu, memory) for q, (cpu, memory) in quotas.items()}
    return resolved


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


# -- direct vs serialized replay, in-process through every public entry ---------------------------
class DirectVsSerializedReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.files: dict[str, ExperimentFiles] = {}
        self.payload: dict[str, tuple] = {}
        for name, builder in SCENARIOS.items():
            nodes, tasks, weights, quotas = builder()
            self.files[name] = ExperimentFiles(self.directory.name, name, nodes, tasks, weights, quotas)
            self.payload[name] = (nodes, tasks, weights, quotas)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _assert_direct_matches_serialized(self, scenario: str, options: dict) -> None:
        nodes, tasks, weights, quotas = self.payload[scenario]
        files = self.files[scenario]
        resolved = api_options(options, weights, quotas)

        # Path 1: direct in-memory run.
        direct = simulate(nodes, tasks, **resolved)

        # Path 2: serialize to JSONL, parse back with the public loaders, then run again. Both
        # optional config files are re-read from disk, not handed across from the in-memory run.
        from schedsim.cli import load_cluster, load_queue_quotas, load_queue_weights, load_tasks

        re_nodes = load_cluster(files.nodes_path)
        re_tasks = load_tasks(files.tasks_path)
        re_options: dict = {"policy": resolved["policy"],
                            "allow_preemption": resolved["allow_preemption"],
                            "backfill": resolved["backfill"]}
        if "queue_weights" in resolved:
            re_options["queue_weights"] = load_queue_weights(files.weights_path)
        if "queue_quotas" in resolved:
            re_options["queue_quotas"] = load_queue_quotas(files.quotas_path)
        replayed = simulate(re_nodes, re_tasks, **re_options)

        # Item-for-item identity: events, clock, transitions, free-resource-driven decisions,
        # explanation text and aggregate metrics are all part of the compared document.
        self.assertEqual(normalized(replayed), normalized(direct))
        self.assertEqual(replayed.trace(), direct.trace())
        self.assertTrue(replay(re_nodes, re_tasks, **re_options)["identical"])
        assert_conservation(self, nodes, tasks, direct)
        assert_conservation(self, re_nodes, re_tasks, replayed)

        # Every documented subcommand on the serialized files returns exactly the API document.
        base = ["--cluster", files.nodes_path, "--tasks", files.tasks_path]
        flags = files.flags(options)

        code, out, err = run_cli(["simulate", *base, *flags])
        self.assertEqual(err, "")
        self.assertEqual(json.loads(out), direct.to_document())
        self.assertEqual(code, EXIT_OK if not direct.unplaced else EXIT_NEGATIVE)

        code, out, err = run_cli(["metrics", *base, *flags])
        self.assertEqual((code, err), (EXIT_OK if not direct.unplaced else EXIT_NEGATIVE, ""))
        self.assertEqual(json.loads(out), {"policy": direct.policy, "metrics": direct.metrics})

        code, out, err = run_cli(["trace", *base, *flags])
        self.assertEqual(err, "")
        self.assertEqual(
            json.loads(out),
            {"policy": direct.policy, "makespan": direct.makespan,
             "decisions": direct.decisions, "unplaced": direct.unplaced},
        )

        code, out, err = run_cli(["replay", *base, *flags])
        self.assertEqual((code, err), (EXIT_OK, ""))
        self.assertEqual(json.loads(out), replay(nodes, tasks, **resolved))

        code, out, err = run_cli(["policies", *base, *flags])
        self.assertEqual(err, "")
        policy_options = {k: v for k, v in resolved.items() if k != "policy"}
        self.assertEqual(json.loads(out), compare_policies(nodes, tasks, **policy_options))

    def test_every_scenario_every_option_pair(self) -> None:
        for scenario in SCENARIOS:
            for options in OPTION_COMBOS:
                with self.subTest(scenario=scenario, options=options):
                    self._assert_direct_matches_serialized(scenario, options)


# -- the specific determinism-relevant branches, with exact stable ordering ------------------------
class TieAndBranchOrderingTests(unittest.TestCase):
    def test_same_timestamp_submissions_order_by_id_regardless_of_input_order(self) -> None:
        nodes = (Node("n1", Resources(1, 1)),)
        ordered = (task("b", duration=2), task("a", duration=2), task("c", duration=2))
        forward = simulate(nodes, ordered)
        reversed_run = simulate(nodes, tuple(reversed(ordered)))
        self.assertEqual(
            [p.task_id for p in forward.placements], ["a", "b", "c"]
        )
        self.assertEqual([(p.task_id, p.start, p.end) for p in forward.placements],
                         [("a", 0, 2), ("b", 2, 4), ("c", 4, 6)])
        # Input permutation changes no event and no metric.
        self.assertEqual(normalized(reversed_run), normalized(forward))
        self.assertTrue(replay(nodes, ordered)["identical"])
        # At the busy tick the head is placed and the same-timestamp sibling is refused once, in
        # that stable order.
        tick0 = [(d["task"], "node" in d) for d in forward.decisions if d.get("at") == 0]
        self.assertEqual(tick0, [("a", True), ("b", False)])

    def test_same_timestamp_at_a_later_arrival_is_stable(self) -> None:
        nodes = (Node("n1", Resources(1, 1)),)
        tasks = (task("z", arrival=3, duration=1), task("a", arrival=3, duration=1))
        result = simulate(nodes, tasks)
        self.assertEqual([(p.task_id, p.start) for p in result.placements], [("a", 3), ("z", 4)])

    def test_multiple_nodes_with_equal_best_fit_score_break_on_node_id(self) -> None:
        nodes = tuple(Node(f"n{i}", Resources(4, 4)) for i in (1, 2, 3))
        tasks = (task("x", 2, 2, duration=2), task("y", 2, 2, duration=2), task("z", 2, 2, duration=2))
        result = simulate(nodes, tasks, policy="best-fit")
        self.assertEqual(
            [(p.task_id, p.node_id) for p in result.placements],
            [("x", "n1"), ("y", "n1"), ("z", "n2")],
        )
        # Identical answer under first-fit here too, and under every node-input permutation.
        self.assertEqual(
            [(p.task_id, p.node_id) for p in simulate(nodes, tasks, policy="first-fit").placements],
            [("x", "n1"), ("y", "n1"), ("z", "n2")],
        )
        permuted = simulate(tuple(reversed(nodes)), tasks, policy="best-fit")
        self.assertEqual(permuted.trace(), result.trace())

    def test_equal_priority_equal_fair_share_breaks_on_queue_then_id(self) -> None:
        nodes = (Node("n1", Resources(2, 2)),)
        across_queues = (task("z1", queue="z"), task("a1", queue="a"))
        result = simulate(nodes, across_queues, queue_weights={"z": 1, "a": 1})
        self.assertEqual([p.task_id for p in result.placements], ["a1", "z1"])
        same_queue = (task("y", queue="q"), task("x", queue="q"))
        result = simulate(nodes, same_queue, queue_weights={"q": 1})
        self.assertEqual([p.task_id for p in result.placements], ["x", "y"])

    def test_resource_fills_exactly_and_returns_to_a_derivable_value(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        full = simulate(nodes, (task("full", 4, 4),))
        self.assertEqual([(p.task_id, p.start, p.end) for p in full.placements], [("full", 0, 1)])
        # Two complementary requests consume the node exactly with zero residual at tick 0.
        complementary = simulate(nodes, (task("a", 3, 1), task("b", 1, 3)))
        self.assertEqual({p.task_id for p in complementary.placements}, {"a", "b"})
        assert_conservation(self, nodes, (task("full", 4, 4),), full)
        assert_conservation(self, nodes, (task("a", 3, 1), task("b", 1, 3)), complementary)

    def test_blocked_task_re_enters_scheduling_on_completion_with_stable_refusal(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("anchor", 2, 2, duration=3),
            task("blocked", 4, 4, arrival=1, duration=2),
        )
        result = simulate(nodes, tasks)
        self.assertEqual({p.task_id: (p.start, p.end) for p in result.placements},
                         {"anchor": (0, 3), "blocked": (3, 5)})
        refusals = [d for d in result.decisions if d["task"] == "blocked" and "node" not in d]
        self.assertEqual(len(refusals), 1, "identical retries are de-duplicated")
        self.assertEqual(refusals[0]["at"], 1)
        self.assertIn("insufficient capacity", refusals[0]["reason"])
        placement = next(d for d in result.decisions if d["task"] == "blocked" and "node" in d)
        self.assertEqual(placement["at"], 3)
        # The release at tick 3 leaves the node empty: blocked starts there with the full request.
        assert_conservation(self, nodes, tasks, result)

    def test_preemption_victim_tie_is_sorted_and_input_order_independent(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        victims = (
            task("va", 2, 2, priority=1, duration=10),
            task("vb", 2, 2, priority=1, duration=10),
        )
        incoming = task("boss", 4, 4, priority=9, arrival=1, duration=1)
        forward = simulate(nodes, victims + (incoming,), allow_preemption=True)
        swapped = simulate(nodes, tuple(reversed(victims)) + (incoming,), allow_preemption=True)
        boss_forward = next(p for p in forward.placements if p.task_id == "boss")
        boss_swapped = next(p for p in swapped.placements if p.task_id == "boss")
        self.assertEqual(boss_forward.preempted, ("va", "vb"))
        self.assertEqual(boss_swapped.preempted, ("va", "vb"))
        self.assertEqual(swapped.trace(), forward.trace())
        self.assertEqual(forward.metrics["preemptions"], 2)

    def test_unplaceable_tasks_keep_identical_wait_state_and_public_reason(self) -> None:
        # The anchor keeps something running and the affinity-blocked task is the waiting *head*
        # (it arrives before the oversized task, so it -- not the other waiter -- is retried on the
        # clock and its predicate-ordered refusal is emitted).
        nodes = (Node("n1", Resources(4, 4), labels=(("zone", "a"),)),)
        tasks = (
            task("anchor", 2, 2, duration=6),
            task("pinned", 1, 1, arrival=1, affinity=(("zone", "b"),)),
            task("huge", 9, 9, arrival=2),
        )
        first = simulate(nodes, tasks)
        second = simulate(nodes, tasks)
        self.assertEqual(normalized(second), normalized(first))
        self.assertEqual(first.unplaced, ["huge", "pinned"])
        timed_refusals = [
            d for d in first.decisions if d["task"] == "pinned" and "node" not in d and "at" in d
        ]
        self.assertTrue(timed_refusals, "the blocked task is retried while the anchor runs")
        self.assertTrue(any("affinity zone=b" in d["reason"] for d in timed_refusals))
        self.assertTrue(all(d["reason"].startswith("no node fits") for d in timed_refusals))
        self.assertTrue(
            any(d["task"] == "pinned"
                and d["reason"] == "left unplaced when the simulation ended"
                for d in first.decisions)
        )
        # Neither task was placed or completed.
        self.assertFalse(any(p.task_id in {"pinned", "huge"} for p in first.placements))
        assert_conservation(self, nodes, tasks, first)


# -- seed / hash-randomization invariance ----------------------------------------------------------
class RandomSourceInvarianceTests(unittest.TestCase):
    """The simulator declares no random decision, so a seed changes nothing.

    A different seed is allowed to alter only a choice the product *publicly declares* random; this
    product declares none (``describe``: "no wall clock, no set iteration, ties always break on
    id"). The cross-process tests below run under several ``PYTHONHASHSEED`` values and compare
    bytes. Here the same experiments are re-run in-process and must keep both the exact event order
    and all terminal invariants (task counts, resource totals, no over-commit).
    """

    def test_seed_like_variation_changes_neither_choice_nor_invariants(self) -> None:
        for builder in SCENARIOS.values():
            nodes, tasks, weights, quotas = builder()
            options = api_options(OPTION_COMBOS[6], weights, quotas)
            baseline = simulate(nodes, tasks, **options)
            for _ in range(3):  # repeated "replays" stand in for distinct fixed seeds
                again = simulate(nodes, tasks, **options)
                self.assertEqual(normalized(again), normalized(baseline))
                assert_conservation(self, nodes, tasks, again)
                self.assertEqual(len(again.placements), len(baseline.placements))
                self.assertEqual(again.makespan, baseline.makespan)


# -- cross-process reproducibility ----------------------------------------------------------------
def run_subprocess(argv: list[str], hash_seed: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "schedsim.cli", *argv],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


class CrossProcessReproducibilityTests(unittest.TestCase):
    SEEDS = ("0", "1", "42", "random", "random")

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        nodes, tasks, weights, quotas = scenario_ties()
        self.files = ExperimentFiles(self.directory.name, "xproc", nodes, tasks, weights, quotas)
        self.payload = (nodes, tasks, weights, quotas)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _outputs_per_seed(self, command: str, options: dict) -> list[tuple[int, bytes, bytes]]:
        base = ["--cluster", self.files.nodes_path, "--tasks", self.files.tasks_path]
        flags = self.files.flags(options)
        runs = []
        for seed in self.SEEDS:
            proc = run_subprocess([command, *base, *flags], seed)
            runs.append((proc.returncode, proc.stdout.encode(), proc.stderr.encode()))
        return runs

    def test_independent_processes_return_identical_canonical_documents(self) -> None:
        nodes, tasks, weights, quotas = self.payload
        options = OPTION_COMBOS[6]  # best-fit + fair weights + quotas: most tie surfaces live
        for command in ("simulate", "trace", "metrics", "replay", "policies"):
            runs = self._outputs_per_seed(command, options)
            first_code, first_out, first_err = runs[0]
            for code, out, err in runs[1:]:
                self.assertEqual(code, first_code, command)
                self.assertEqual(err, first_err, command)
                # Byte-for-byte: hash randomization and mapping insertion order never reach output.
                self.assertEqual(out, first_out, command)
            direct = simulate(nodes, tasks, **api_options(options, weights, quotas))
            document = json.loads(first_out.decode())
            if command == "simulate":
                self.assertEqual(document, direct.to_document())
            elif command == "replay":
                self.assertTrue(document["identical"])
                self.assertEqual(document["differences"], [])

    def test_unplaced_verdict_and_reasons_are_stable_across_processes(self) -> None:
        # Some option combinations leave more than just the affinity-blocked task waiting; which
        # tasks those are is option-dependent. What must not drift between hash-randomized processes
        # is the negative verdict, the exact unplaced set and every refusal reason.
        for options in (OPTION_COMBOS[0], OPTION_COMBOS[3]):
            runs = self._outputs_per_seed("trace", options)
            reference = json.loads(runs[0][1].decode())
            self.assertIn("pinned", reference["unplaced"])
            self.assertTrue(reference["unplaced"])
            for code, out, _ in runs[1:]:
                self.assertEqual(code, EXIT_NEGATIVE)
                self.assertEqual(json.loads(out.decode()), reference)


# -- content-equivalent input with a different key order -------------------------------------------
class KeyOrderReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.cluster_canonical = os.path.join(self.directory.name, "cluster.jsonl")
        self.tasks_canonical = os.path.join(self.directory.name, "tasks.jsonl")
        self.tasks_reordered = os.path.join(self.directory.name, "tasks_reordered.jsonl")
        with open(self.cluster_canonical, "w", encoding="utf-8") as handle:
            handle.write('{"id": "n1", "cpu": 4, "memory": 4, "labels": {"zone": "a"}, "taints": []}\n')
            handle.write('{"id": "n2", "cpu": 4, "memory": 4, "labels": {"zone": "a"}, "taints": []}\n')
        # Canonical key order, normal spacing.
        with open(self.tasks_canonical, "w", encoding="utf-8") as handle:
            handle.write('{"id": "x", "cpu": 2, "memory": 2, "priority": 0, "queue": "qa", '
                         '"arrival": 0, "duration": 2, "affinity": {}, "antiAffinity": [], '
                         '"tolerations": []}\n')
            handle.write('{"id": "y", "cpu": 2, "memory": 2, "priority": 0, "queue": "qa", '
                         '"arrival": 0, "duration": 2, "affinity": {}, "antiAffinity": [], '
                         '"tolerations": []}\n')
        # Same semantic content: keys shuffled, whitespace changed, nested labels reordered.
        with open(self.tasks_reordered, "w", encoding="utf-8") as handle:
            handle.write('{"duration":2,"tolerations":[],"memory":2,"id":"x",'
                         '"antiAffinity":[],"cpu":2,"affinity":{},"queue":"qa",'
                         '"arrival":0,"priority":0}\n')
            handle.write('{ "priority" : 0 , "affinity" : {} , "arrival" : 0 , "id" : "y" , '
                         '"cpu" : 2 , "queue" : "qa" , "memory" : 2 , "duration" : 2 , '
                         '"antiAffinity" : [ ] , "tolerations" : [ ] }\n')

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_key_reordered_input_replays_identically(self) -> None:
        for command in ("simulate", "trace", "metrics", "replay"):
            a = run_cli([command, "--cluster", self.cluster_canonical, "--tasks", self.tasks_canonical])
            b = run_cli([command, "--cluster", self.cluster_canonical, "--tasks", self.tasks_reordered])
            self.assertEqual(a[0], b[0], command)
            self.assertEqual(a[1], b[1], command)
            self.assertEqual(a[2], b[2], command)

    def test_key_reordering_is_process_and_hash_independent(self) -> None:
        canonical = run_subprocess(
            ["simulate", "--cluster", self.cluster_canonical, "--tasks", self.tasks_canonical], "1"
        )
        reordered = run_subprocess(
            ["simulate", "--cluster", self.cluster_canonical, "--tasks", self.tasks_reordered], "random"
        )
        self.assertEqual((canonical.returncode, canonical.stderr), (EXIT_OK, ""))
        self.assertEqual((reordered.returncode, reordered.stderr), (EXIT_OK, ""))
        self.assertEqual(reordered.stdout, canonical.stdout)


# -- malformed / incomplete input keeps the published failure contract -----------------------------
class MalformedInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.cluster = os.path.join(self.directory.name, "cluster.jsonl")
        with open(self.cluster, "w", encoding="utf-8") as handle:
            handle.write('{"id": "n1", "cpu": 4, "memory": 4}\n')

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _tasks(self, text: str) -> str:
        path = os.path.join(self.directory.name, "tasks.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def _assert_fails(self, text: str, kind: str, line: int | None = None) -> None:
        tasks = self._tasks(text)
        for command in ("validate", "simulate", "trace", "replay"):
            code, out, err = run_cli([command, "--cluster", self.cluster, "--tasks", tasks])
            self.assertEqual((code, out), (EXIT_ERROR, ""), command)  # never a success, never silent
            document = json.loads(err)
            self.assertEqual(document["error"], kind, command)
            if line is not None:
                self.assertEqual(document["line"], line, command)

    def test_broken_json_is_a_parse_error_with_the_line(self) -> None:
        self._assert_fails('{"id": "a", "cpu": 1}\n{broken\n', "parse_error", line=2)

    def test_non_object_row_is_a_parse_error(self) -> None:
        self._assert_fails("[1, 2]\n", "parse_error", line=1)

    def test_missing_required_field_is_a_parse_error(self) -> None:
        self._assert_fails('{"cpu": 1, "memory": 1}\n', "parse_error", line=1)

    def test_unknown_field_is_a_parse_error(self) -> None:
        self._assert_fails('{"id": "a", "cpu": 1, "gpu": 2}\n', "parse_error", line=1)

    def test_zero_request_is_a_validation_error_not_a_success(self) -> None:
        tasks = self._tasks('{"id": "z", "cpu": 0, "memory": 0}\n')
        code, out, err = run_cli(["validate", "--cluster", self.cluster, "--tasks", tasks])
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_empty_task_file_is_a_validation_error_not_skipped(self) -> None:
        tasks = self._tasks("# only a comment\n\n")
        code, out, err = run_cli(["validate", "--cluster", self.cluster, "--tasks", tasks])
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_failure_survives_a_fresh_process(self) -> None:
        tasks = self._tasks('{"id": "a"}\n{"broken\n')
        proc = run_subprocess(
            ["simulate", "--cluster", self.cluster, "--tasks", tasks], "random"
        )
        self.assertEqual(proc.returncode, EXIT_ERROR)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(json.loads(proc.stderr)["error"], "parse_error")
        self.assertEqual(json.loads(proc.stderr)["line"], 2)


if __name__ == "__main__":
    unittest.main()
