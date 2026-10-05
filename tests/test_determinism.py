"""Deterministic scheduling and experiment-replay tests through the public entry points only.

Every check here drives the simulator the way the README documents it: the Python entry points
(``simulate`` / ``replay`` / ``Simulation``) and the JSONL command line (``simulate`` / ``trace``
/ ``metrics`` / ``policies`` / ``replay``). No test walks an internal container, relies on an
object address or wall-clock time, or invokes a private method -- so a failure here means the
published determinism contract moved, not the test.

The same fixed cluster, task sequence, policy configuration and random seed are run twice per
case: once directly from the model objects and once after serializing both inputs to JSONL and
loading them back through the published parsers. The two paths must agree item for item on event
order, clock advancement, task state transitions, per-node free resources, decision reasons and
summary metrics.

The branches that actually determine a tie are all covered:

* several tasks submitted at the same timestamp;
* several nodes with an identical best-fit score (ties break on node id, never hash order);
* equal-priority waiters from equal-share queues (queue name, then task id);
* resources filled *exactly* (zero free, nothing over);
* a task that is temporarily unplaceable and re-enters scheduling when a completion event frees
  room, including a silent backfill probe at a tick where nothing fits;
* hard queue-quota waits admitted on the release tick;
* preemption as the release event;
* the one kind of choice the README declares random -- none: the package promises "no wall clock,
  no iteration over unordered containers, every tie breaks on node id", so fixing or changing
  every Python-level random source (seeds, consumed ``random`` draws, mapping insertion order)
  must leave the whole result byte-identical. That empty "random decisions" set is asserted by
  making any ``random`` choice raise while a simulation runs.

Cross-process reproducibility is exercised with ``PYTHONHASHSEED`` fixed to several values and
left unset: independent interpreter processes emit the same canonical documents, so hash
randomization and set/map iteration order cannot change an equal-score decision. Inputs whose
content is equal but whose JSON keys are ordered differently must give identical results; corrupt
or incomplete inputs must keep producing the documented ``parse_error`` / ``validation_error``
failures -- never a success or a silent skip.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

from schedsim import Node, Resources, Task, simulate
from schedsim.cli import (
    EXIT_ERROR,
    EXIT_NEGATIVE,
    EXIT_OK,
    load_cluster,
    load_queue_quotas,
    load_queue_weights,
    load_tasks,
    main,
)
from schedsim.errors import SchedulerError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def task(task_id: str, cpu: int = 1, memory: int = 1, **kwargs) -> Task:
    return Task(id=task_id, request=Resources(cpu, memory), **kwargs)


# -- public input shapes (the same initial state, expressible as JSONL) ---------------------------
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


class Scenario:
    """One fixed experiment: initial cluster, submission sequence and policy configuration."""

    def __init__(
        self,
        name: str,
        nodes: tuple[Node, ...],
        tasks: tuple[Task, ...],
        weights: dict[str, int] | None = None,
        quotas: dict[str, Resources] | None = None,
        weight_rows: list[dict] | None = None,
        quota_rows: list[dict] | None = None,
    ) -> None:
        self.name = name
        self.nodes = nodes
        self.tasks = tasks
        self.weights = weights
        self.quotas = quotas
        self.weight_rows = weight_rows
        self.quota_rows = quota_rows

    def node_rows(self) -> list[dict]:
        return [node_row(node) for node in self.nodes]

    def task_rows(self) -> list[dict]:
        return [task_row(item) for item in self.tasks]


def scenario_same_timestamp_batch() -> Scenario:
    """Five tasks submitted at t=0 fill every node exactly; a sixth can never fit.

    n1 and n2 have identical capacity, so best-fit is a genuine equal-score decision; a 1-unit
    task arriving at t=1 is unplaceable then (the cluster is saturated) and lands only after the
    t=2 completion events. The impossible 5x5 task stays waiting for the whole run with reasons
    that must be reproduced verbatim.
    """
    return Scenario(
        "same-timestamp-batch",
        (
            Node("n1", Resources(4, 4)),
            Node("n2", Resources(4, 4)),
            Node("n3", Resources(2, 2)),
        ),
        (
            task("a", 4, 4, arrival=0, duration=4),
            task("b", 2, 2, arrival=0, duration=2),
            task("c", 2, 2, arrival=0, duration=2),
            task("d", 2, 2, arrival=0, duration=2),
            task("e", 1, 1, arrival=1, duration=1),
            task("x", 5, 5, arrival=0, duration=1),
        ),
    )


def scenario_blocked_then_retried() -> Scenario:
    """A task that cannot be placed at its arrival tick and is placed on a completion event."""
    return Scenario(
        "blocked-then-retried",
        (Node("n1", Resources(2, 2)),),
        (
            task("anchor", 2, 2, arrival=0, duration=3),
            task("waiter", 2, 2, arrival=1, duration=2),
        ),
    )


def scenario_equal_priority_ties() -> Scenario:
    """Equal-priority waiters: id order without fair share, queue alternation with it."""
    return Scenario(
        "equal-priority-ties",
        (Node("n1", Resources(4, 4)),),
        (
            task("z1", 1, 1, queue="z", arrival=0, duration=1),
            task("a1", 1, 1, queue="a", arrival=0, duration=1),
            task("z2", 1, 1, queue="z", arrival=0, duration=1),
            task("a2", 1, 1, queue="a", arrival=0, duration=1),
        ),
        weights={"a": 1, "z": 1},
        weight_rows=[{"queue": "a", "weight": 1}, {"queue": "z", "weight": 1}],
    )


def scenario_equal_score_nodes() -> Scenario:
    """Six identical requests over two identical nodes: three exact fills each, ties on node id."""
    return Scenario(
        "equal-score-nodes",
        (Node("n1", Resources(3, 3)), Node("n2", Resources(3, 3))),
        tuple(task(f"t{index:02d}", 1, 1, arrival=0, duration=2) for index in range(6)),
    )


def scenario_quota_wait() -> Scenario:
    """A queue-quota refusal at t=1; the same task is admitted on the t=2 completion release."""
    return Scenario(
        "quota-wait",
        (Node("n1", Resources(8, 8)),),
        (
            task("q1", 3, 3, queue="q", arrival=0, duration=2),
            task("q2", 2, 2, queue="q", arrival=0, duration=2),
            task("q3", 2, 2, queue="q", arrival=1, duration=1),
        ),
        quotas={"q": Resources(5, 5)},
        quota_rows=[{"queue": "q", "cpu": 5, "memory": 5}],
    )


def scenario_preemption_release() -> Scenario:
    """A completion supplied by preemption: the high-priority task enters on the eviction tick."""
    return Scenario(
        "preemption-release",
        (Node("n1", Resources(4, 4)),),
        (
            task("low", 4, 4, priority=1, arrival=0, duration=10),
            task("high", 4, 4, priority=9, arrival=1, duration=1),
        ),
    )


def scenario_rich_queues() -> Scenario:
    """Heterogeneous nodes, three weighted queues with hard quotas, one quota-impossible task.

    At t=0 the weighted order interleaves the queues (a placement re-ranks the waiters); f6 asks
    for more than its queue's ceiling and is quota-refused before any node is probed; f5 is
    quota-blocked at t=1 and admitted at t=2 once a completion releases the usage.
    """
    return Scenario(
        "rich-queues",
        (
            Node("n1", Resources(4, 4)),
            Node("n2", Resources(4, 4)),
            Node("n3", Resources(2, 2)),
        ),
        (
            task("f1", 2, 2, queue="qa", arrival=0, duration=2),
            task("f2", 2, 2, queue="qa", arrival=0, duration=2),
            task("f3", 4, 4, queue="qb", arrival=0, duration=4),
            task("f4", 2, 2, queue="qc", arrival=0, duration=2),
            task("f5", 1, 1, queue="qa", arrival=1, duration=1),
            task("f6", 5, 5, queue="qb", arrival=0, duration=1),
        ),
        weights={"qa": 2, "qb": 1, "qc": 1},
        quotas={"qa": Resources(4, 4), "qb": Resources(4, 4), "qc": Resources(2, 2)},
        weight_rows=[
            {"queue": "qa", "weight": 2},
            {"queue": "qb", "weight": 1},
            {"queue": "qc", "weight": 1},
        ],
        quota_rows=[
            {"queue": "qa", "cpu": 4, "memory": 4},
            {"queue": "qb", "cpu": 4, "memory": 4},
            {"queue": "qc", "cpu": 2, "memory": 2},
        ],
    )


SCENARIOS = (
    scenario_same_timestamp_batch(),
    scenario_blocked_then_retried(),
    scenario_equal_priority_ties(),
    scenario_equal_score_nodes(),
    scenario_quota_wait(),
    scenario_preemption_release(),
    scenario_rich_queues(),
)


def rich_option_matrix(scenario: Scenario):
    """Policy x preemption x backfill x fair-share x quota, enabling the configured files only."""
    for policy in ("first-fit", "best-fit"):
        for allow_preemption in (False, True):
            for backfill in (False, True):
                for fair in (False, True):
                    for with_quotas in (False, True):
                        options: dict[str, object] = {
                            "policy": policy,
                            "allow_preemption": allow_preemption,
                            "backfill": backfill,
                        }
                        if fair and scenario.weights is not None:
                            options["queue_weights"] = dict(scenario.weights)
                        if with_quotas and scenario.quotas is not None:
                            options["queue_quotas"] = dict(scenario.quotas)
                        yield options


# -- independent invariants, rebuilt only from the published placements and decisions -------------
def used_per_node(nodes: tuple[Node, ...], tasks_by_id: dict[str, Task], placements, clock: int):
    """Running [cpu, memory] on each node at ``clock`` under half-open [start, end) intervals."""
    used = {node.id: [0, 0] for node in nodes}
    for item in placements:
        if item.start <= clock < item.end:
            request = tasks_by_id[item.task_id].request
            used[item.node_id][0] += request.cpu
            used[item.node_id][1] += request.memory
    return used


def assert_schedule_invariants(
    tc: unittest.TestCase, nodes: tuple[Node, ...], tasks: tuple[Task, ...], result
) -> None:
    """Resource conservation and state-transition rules, derived from public results alone."""
    tasks_by_id = {item.id: item for item in tasks}
    nodes_by_id = {node.id: node for node in nodes}

    # -- no task is placed twice; intervals respect arrival, duration and preemption --------------
    placement_ids = [item.task_id for item in result.placements]
    tc.assertEqual(len(placement_ids), len(set(placement_ids)), "a task was placed more than once")
    # A victim is identified by appearing in ANOTHER placement's preempted tuple; its own record
    # still carries preempted=() (the record is truncated in place, never duplicated).
    evicted: dict[str, object] = {}
    for placement in result.placements:
        for victim in placement.preempted:
            tc.assertNotIn(victim, evicted, "a task was preempted twice")
            tc.assertIn(victim, tasks_by_id)
            tc.assertLess(
                tasks_by_id[victim].priority,
                tasks_by_id[placement.task_id].priority,
                "evicted task was not lower priority",
            )
            evicted[victim] = placement
    for placement in result.placements:
        tc.assertIn(placement.task_id, tasks_by_id)
        tc.assertIn(placement.node_id, nodes_by_id)
        owner = tasks_by_id[placement.task_id]
        tc.assertGreaterEqual(placement.start, owner.arrival, "task started before its arrival")
        tc.assertGreater(placement.end, placement.start, "non-positive run interval")
        if placement.task_id in evicted:
            tc.assertLessEqual(placement.end - placement.start, owner.duration)
        else:
            tc.assertEqual(placement.end - placement.start, owner.duration, "only preemption truncates")

    # a victim is really a running placement whose interval ends exactly at the eviction tick, and
    # the preemptions metric counts each real eviction exactly once (no task "completes twice")
    for victim, evictor in evicted.items():
        victim_placement = next(item for item in result.placements if item.task_id == victim)
        tc.assertEqual(victim_placement.end, evictor.start, "victim interval must end at the eviction tick")
        tc.assertLessEqual(victim_placement.start, evictor.start)
    tc.assertEqual(result.metrics["preemptions"], sum(len(p.preempted) for p in result.placements))

    # -- capacity is never over-committed and releases return to the derivable totals --------------
    event_clocks = sorted({p.start for p in result.placements} | {p.end for p in result.placements})
    previous_clock = None
    previous_used = None
    for clock in event_clocks:
        used = used_per_node(nodes, tasks_by_id, result.placements, clock)
        for node in nodes:
            tc.assertGreaterEqual(used[node.id][0], 0)
            tc.assertGreaterEqual(used[node.id][1], 0)
            tc.assertLessEqual(used[node.id][0], node.capacity.cpu, f"cpu over-committed on {node.id} at {clock}")
            tc.assertLessEqual(used[node.id][1], node.capacity.memory, f"memory over-committed at {clock}")
        if previous_clock is not None:
            # between two event clocks occupancy is constant; the delta at `clock` is exactly the
            # sum of starting requests minus the sum of ending requests at that node
            for node in nodes:
                starts = sum(
                    tasks_by_id[p.task_id].request.cpu
                    for p in result.placements
                    if p.node_id == node.id and p.start == clock
                )
                starts_mem = sum(
                    tasks_by_id[p.task_id].request.memory
                    for p in result.placements
                    if p.node_id == node.id and p.start == clock
                )
                ends = sum(
                    tasks_by_id[p.task_id].request.cpu
                    for p in result.placements
                    if p.node_id == node.id and p.end == clock
                )
                ends_mem = sum(
                    tasks_by_id[p.task_id].request.memory
                    for p in result.placements
                    if p.node_id == node.id and p.end == clock
                )
                tc.assertEqual(used[node.id][0], previous_used[node.id][0] + starts - ends)
                tc.assertEqual(used[node.id][1], previous_used[node.id][1] + starts_mem - ends_mem)
        previous_clock, previous_used = clock, used

    # after the last completion every node is fully released back to its capacity
    if event_clocks:
        final_used = used_per_node(nodes, tasks_by_id, result.placements, result.makespan)
        for node in nodes:
            tc.assertEqual(final_used[node.id], [0, 0], f"resources leaked on {node.id} after makespan")

    # cpu-time is conserved independently: the sum over intervals equals per-node tick accounting
    interval_cpu_time = sum(
        tasks_by_id[p.task_id].request.cpu * (p.end - p.start) for p in result.placements
    )
    walked_cpu_time = 0
    for clock in range(result.makespan):
        walked_cpu_time += sum(
            values[0] for values in used_per_node(nodes, tasks_by_id, result.placements, clock).values()
        )
    tc.assertEqual(interval_cpu_time, walked_cpu_time)

    # -- the decision walk keeps a stable clock and mirrors the placements one-for-one -------------
    placements_out = sorted(result.placements, key=lambda p: (p.start, p.task_id))
    tc.assertEqual(result.placements, placements_out, "placement output order must be (start, task)")
    last_clock = 0
    placement_decisions = []
    final_entries = []
    for entry in result.decisions:
        if "at" not in entry:
            final_entries.append(entry)
            continue
        tc.assertGreaterEqual(entry["at"], last_clock, "the event clock ran backwards")
        last_clock = entry["at"]
        if "node" in entry:
            placement_decisions.append(entry)
    # every successful decision corresponds to exactly one placement at that tick on that node...
    tc.assertEqual(
        sorted((d["task"], d["node"], d["at"]) for d in placement_decisions),
        sorted((p.task_id, p.node_id, p.start) for p in result.placements),
        "successful decisions and placements disagree",
    )
    # ...and no task gets both a placement and an unplaced verdict
    tc.assertEqual(
        {p.task_id for p in result.placements},
        {d["task"] for d in placement_decisions},
    )

    # -- placed / unplaced partition the task set exactly ------------------------------------------
    placed_ids = {p.task_id for p in result.placements}
    tc.assertEqual(
        sorted(result.unplaced), sorted(item.id for item in tasks if item.id not in placed_ids)
    )
    tc.assertEqual(len(result.placements) + len(result.unplaced), len(tasks))
    final_unplaced = [entry["task"] for entry in final_entries]
    tc.assertEqual(sorted(final_unplaced), sorted(result.unplaced))
    for entry in final_entries:
        tc.assertEqual(entry["reason"], "left unplaced when the simulation ended")

    tc.assertEqual(result.makespan, max((p.end for p in result.placements), default=0))
    tc.assertEqual(result.metrics["placed"], len(result.placements))
    tc.assertEqual(result.metrics["unplaced"], len(result.unplaced))


def refusal_entries(result) -> list[dict]:
    return [entry for entry in result.decisions if "at" in entry and "node" not in entry]


# -- direct simulation vs serialized replay --------------------------------------------------------
class DirectVsReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write(self, name: str, rows: list[dict], reorder_keys: bool = False) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                if reorder_keys:
                    row = deep_reorder_keys(row)
                handle.write(json.dumps(row) + "\n")
        return path

    def load_options(self, scenario: Scenario, options: dict[str, object]) -> dict[str, object]:
        """Reload every mapping option from its JSONL file, so the replay path uses only files."""
        replayed = {key: value for key, value in options.items()}
        if "queue_weights" in options:
            path = self.write("weights.jsonl", scenario.weight_rows or [])
            replayed["queue_weights"] = load_queue_weights(path)
        if "queue_quotas" in options:
            path = self.write("quotas.jsonl", scenario.quota_rows or [])
            replayed["queue_quotas"] = load_queue_quotas(path)
        return replayed

    def replay_from_files(self, scenario: Scenario, options: dict[str, object], *, reorder_keys: bool = False):
        cluster_path = self.write("cluster.jsonl", scenario.node_rows(), reorder_keys=reorder_keys)
        tasks_path = self.write("tasks.jsonl", scenario.task_rows(), reorder_keys=reorder_keys)
        replayed_nodes = load_cluster(cluster_path)
        replayed_tasks = load_tasks(tasks_path)
        # the serialized models are the same initial state as the in-memory objects
        self.assertEqual(replayed_nodes, scenario.nodes)
        self.assertEqual(replayed_tasks, scenario.tasks)
        replayed_options = self.load_options(scenario, options)
        return simulate(replayed_nodes, replayed_tasks, **replayed_options)

    def test_every_scenario_matches_its_serialized_replay_under_every_option(self) -> None:
        for scenario in SCENARIOS:
            for options in rich_option_matrix(scenario):
                label = f"{scenario.name}/{options['policy']}/preempt={options['allow_preemption']}/backfill={options['backfill']}/fair={'queue_weights' in options}/quotas={'queue_quotas' in options}"
                with self.subTest(label):
                    direct = simulate(scenario.nodes, scenario.tasks, **options)
                    replayed = self.replay_from_files(scenario, options)
                    # The published format declares NO non-deterministic metadata, so nothing is
                    # excluded: documents, decisions, reasons, clock values and metrics all match.
                    self.assertEqual(replayed.to_document(), direct.to_document())
                    self.assertEqual(replayed.decisions, direct.decisions)
                    self.assertEqual(replayed.trace(), direct.trace())
                    assert_schedule_invariants(self, scenario.nodes, scenario.tasks, replayed)

    def test_replay_entry_point_reports_identical_for_every_scenario(self) -> None:
        for scenario in SCENARIOS:
            options: dict[str, object] = {"policy": "best-fit", "allow_preemption": True}
            if scenario.weights is not None:
                options["queue_weights"] = dict(scenario.weights)
            if scenario.quotas is not None:
                options["queue_quotas"] = dict(scenario.quotas)
            with self.subTest(scenario.name):
                direct = simulate(scenario.nodes, scenario.tasks, **options)
                from schedsim import replay as replay_run

                report = replay_run(scenario.nodes, scenario.tasks, **options)
                self.assertTrue(report["identical"])
                self.assertEqual(report["differences"], [])
                self.assertEqual([list(item) for item in report["trace"]][:20],
                                 [list(item) for item in direct.trace()][:20])
                self.assertEqual(report["placements"], len(direct.placements))
                self.assertEqual(report["makespan"], direct.makespan)

    def test_cli_documents_are_identical_between_direct_and_replayed_inputs(self) -> None:
        scenario = scenario_rich_queues()
        weights_path = self.write("w.jsonl", scenario.weight_rows or [])
        quotas_path = self.write("q.jsonl", scenario.quota_rows or [])
        cluster_path = self.write("cluster.jsonl", scenario.node_rows())
        tasks_path = self.write("tasks.jsonl", scenario.task_rows())
        for extra in (
            [],
            ["--policy", "best-fit", "--preemption"],
            ["--queue-weights", weights_path],
            ["--queue-weights", weights_path, "--queue-quotas", quotas_path],
        ):
            with self.subTest(extra=extra):
                api_options = self._options_for(extra, scenario)
                api_document = simulate(scenario.nodes, scenario.tasks, **api_options).to_document()
                code, out, err = run_cli(
                    ["simulate", "--cluster", cluster_path, "--tasks", tasks_path, *extra]
                )
                self.assertIn(code, (EXIT_OK, EXIT_NEGATIVE))
                self.assertEqual(err, "")
                self.assertEqual(json.loads(out), api_document)

                code, trace_out, err = run_cli(
                    ["trace", "--cluster", cluster_path, "--tasks", tasks_path, *extra]
                )
                self.assertEqual(err, "")
                trace_document = json.loads(trace_out)
                direct = simulate(scenario.nodes, scenario.tasks, **api_options)
                self.assertEqual(trace_document["decisions"], direct.decisions)
                self.assertEqual(trace_document["makespan"], direct.makespan)
                self.assertEqual(trace_document["unplaced"], direct.unplaced)

                code, metrics_out, _ = run_cli(
                    ["metrics", "--cluster", cluster_path, "--tasks", tasks_path, *extra]
                )
                self.assertEqual(json.loads(metrics_out)["metrics"], api_document["metrics"])

                code, replay_out, _ = run_cli(
                    ["replay", "--cluster", cluster_path, "--tasks", tasks_path, *extra]
                )
                self.assertEqual(code, EXIT_OK)
                self.assertTrue(json.loads(replay_out)["identical"])

    def _options_for(self, extra: list[str], scenario: Scenario) -> dict[str, object]:
        options: dict[str, object] = {}
        if "--policy" in extra:
            options["policy"] = extra[extra.index("--policy") + 1]
        if "--preemption" in extra:
            options["allow_preemption"] = True
        if "--no-backfill" in extra:
            options["backfill"] = False
        if scenario.weights is not None and any(a == "--queue-weights" for a in extra):
            options["queue_weights"] = dict(scenario.weights)
        if scenario.quotas is not None and any(a == "--queue-quotas" for a in extra):
            options["queue_quotas"] = dict(scenario.quotas)
        return options


# -- the specific determinism-sensitive branches ---------------------------------------------------
class DeterministicBranchTests(unittest.TestCase):
    def test_same_timestamp_submissions_have_one_stable_order(self) -> None:
        scenario = scenario_same_timestamp_batch()
        result = simulate(scenario.nodes, scenario.tasks)
        at_zero = [d["task"] for d in result.decisions if d.get("at") == 0 and "node" in d]
        self.assertEqual(at_zero, ["a", "b", "c", "d"])
        placements = {p.task_id: (p.node_id, p.start, p.end) for p in result.placements}
        self.assertEqual(placements["a"], ("n1", 0, 4))
        self.assertEqual(placements["b"], ("n2", 0, 2))
        self.assertEqual(placements["c"], ("n2", 0, 2))
        self.assertEqual(placements["d"], ("n3", 0, 2))
        # t=0 saturates every node exactly: free capacity is zero but nothing is over-committed
        tasks_by_id = {t.id: t for t in scenario.tasks}
        used = used_per_node(scenario.nodes, tasks_by_id, result.placements, 0)
        self.assertEqual(used, {"n1": [4, 4], "n2": [4, 4], "n3": [2, 2]})
        assert_schedule_invariants(self, scenario.nodes, scenario.tasks, result)

    def test_equal_score_nodes_always_break_on_node_id_under_best_fit(self) -> None:
        scenario = scenario_equal_score_nodes()
        result = simulate(scenario.nodes, scenario.tasks, policy="best-fit")
        placements = {p.task_id: p.node_id for p in result.placements}
        self.assertEqual([placements[f"t{i:02d}"] for i in range(6)],
                         ["n1", "n1", "n1", "n2", "n2", "n2"])
        tasks_by_id = {t.id: t for t in scenario.tasks}
        used = used_per_node(scenario.nodes, tasks_by_id, result.placements, 0)
        self.assertEqual(used, {"n1": [3, 3], "n2": [3, 3]})  # both filled exactly
        first_fit = simulate(scenario.nodes, scenario.tasks)
        self.assertEqual(first_fit.trace(), result.trace())  # identical nodes: policy cannot matter
        assert_schedule_invariants(self, scenario.nodes, scenario.tasks, result)

    def test_equal_priority_ties_break_on_id_then_queue(self) -> None:
        scenario = scenario_equal_priority_ties()
        baseline = simulate(scenario.nodes, scenario.tasks)
        fair = simulate(scenario.nodes, scenario.tasks, queue_weights=scenario.weights)
        self.assertEqual(
            [d["task"] for d in baseline.decisions if "node" in d], ["a1", "a2", "z1", "z2"]
        )
        # equal priority, equal running share: the queue name tie-break alternates a, z, a, z
        self.assertEqual(
            [d["task"] for d in fair.decisions if "node" in d], ["a1", "z1", "a2", "z2"]
        )
        assert_schedule_invariants(self, scenario.nodes, scenario.tasks, fair)

    def test_temporarily_unplaceable_task_re_enters_on_the_completion_event(self) -> None:
        scenario = scenario_blocked_then_retried()
        result = simulate(scenario.nodes, scenario.tasks)
        decisions = [(d["task"], d.get("at"), d.get("node"), d["reason"]) for d in result.decisions]
        self.assertEqual(decisions, [
            ("anchor", 0, "n1", "placed by first-fit"),
            ("waiter", 1, None, "no node fits (n1: insufficient capacity: needs cpu=2,memory=2 has cpu=0,memory=0)"),
            ("waiter", 3, "n1", "placed by first-fit"),
        ])
        waiter = next(p for p in result.placements if p.task_id == "waiter")
        self.assertEqual((waiter.start, waiter.end), (3, 5))
        # waiting at t=1, released at t=3: per-node usage follows the completion exactly
        tasks_by_id = {t.id: t for t in scenario.tasks}
        self.assertEqual(used_per_node(scenario.nodes, tasks_by_id, result.placements, 1), {"n1": [2, 2]})
        self.assertEqual(used_per_node(scenario.nodes, tasks_by_id, result.placements, 3), {"n1": [2, 2]})
        assert_schedule_invariants(self, scenario.nodes, scenario.tasks, result)

    def test_silent_backfill_probe_then_placement_after_completion(self) -> None:
        scenario = scenario_same_timestamp_batch()
        result = simulate(scenario.nodes, scenario.tasks)
        # at t=1 `e` probes while the cluster is saturated and records nothing; at t=2 the b/c
        # completion admits it with the fixed `backfill` reason
        self.assertFalse(
            any(d["task"] == "e" and d.get("at") == 1 for d in result.decisions),
            "a failed backfill probe must leave no trace entry",
        )
        e_decision = next(d for d in result.decisions if d["task"] == "e")
        self.assertEqual(e_decision, {"task": "e", "node": "n2", "reason": "backfill", "at": 2})
        e_placement = next(p for p in result.placements if p.task_id == "e")
        self.assertEqual((e_placement.node_id, e_placement.start, e_placement.end), ("n2", 2, 3))

    def test_quota_wait_reasons_are_stable_and_admission_hits_the_release_tick(self) -> None:
        scenario = scenario_quota_wait()
        result = simulate(scenario.nodes, scenario.tasks, queue_quotas=scenario.quotas)
        refusals = refusal_entries(result)
        self.assertEqual(len(refusals), 1)
        self.assertEqual(
            refusals[0]["reason"],
            "queue quota exceeded for queue q: task requests cpu=2,memory=2, "
            "running use cpu=5,memory=5, limit cpu=5,memory=5",
        )
        self.assertEqual(refusals[0]["at"], 1)
        admitted = next(d for d in result.decisions if d["task"] == "q3" and "node" in d)
        self.assertEqual(admitted["at"], 2)
        assert_schedule_invariants(self, scenario.nodes, scenario.tasks, result)

    def test_preemption_is_the_release_event_and_is_replayed_verbatim(self) -> None:
        scenario = scenario_preemption_release()
        result = simulate(scenario.nodes, scenario.tasks, allow_preemption=True)
        high = next(p for p in result.placements if p.task_id == "high")
        low = next(p for p in result.placements if p.task_id == "low")
        self.assertEqual((high.node_id, high.start, high.end, high.preempted), ("n1", 1, 2, ("low",)))
        self.assertEqual((low.start, low.end), (0, 1))
        decision = next(d for d in result.decisions if d["task"] == "high")
        self.assertEqual(
            decision,
            {"task": "high", "node": "n1", "reason": "preempting 1 lower-priority task(s) on n1", "at": 1},
        )
        assert_schedule_invariants(self, scenario.nodes, scenario.tasks, result)

    def test_unplaceable_tasks_keep_waiting_state_and_public_reasons_across_replay(self) -> None:
        scenario = scenario_same_timestamp_batch()
        direct = simulate(scenario.nodes, scenario.tasks)
        replayed = simulate(scenario.nodes, scenario.tasks)  # fresh objects, same inputs
        self.assertEqual(direct.unplaced, ["x"])
        self.assertFalse(any(p.task_id == "x" for p in direct.placements))
        x_decisions = [d for d in direct.decisions if d["task"] == "x"]
        # one final waiting verdict plus dated capacity refusals, all reasons reproduced exactly
        self.assertEqual(x_decisions, [d for d in replayed.decisions if d["task"] == "x"])
        self.assertEqual(x_decisions[-1]["reason"], "left unplaced when the simulation ended")
        self.assertTrue(
            all(d["reason"].startswith("no node fits") for d in x_decisions if "at" in d)
        )
        self.assertEqual([d["at"] for d in x_decisions if "at" in d], [0, 2, 4])

        rich = scenario_rich_queues()
        rich_result = simulate(
            rich.nodes, rich.tasks, queue_weights=rich.weights, queue_quotas=rich.quotas
        )
        self.assertEqual(rich_result.unplaced, ["f6"])
        f6 = [d for d in rich_result.decisions if d["task"] == "f6"]
        self.assertTrue(
            any(d.get("reason", "").startswith("queue quota exceeded for queue qb") and d.get("at") == 0
                for d in f6)
        )
        # the dated quota refusal names the public numbers and is retained even after later retries
        dated = [d for d in f6 if "at" in d]
        self.assertEqual([d["at"] for d in dated], [0, 4])
        self.assertTrue(all("limit cpu=4,memory=4" in d["reason"] for d in dated))
        self.assertEqual(f6[-1]["reason"], "left unplaced when the simulation ended")


# -- random sources and insertion orders -----------------------------------------------------------
class RandomSourceTests(unittest.TestCase):
    SCENARIO = staticmethod(scenario_rich_queues)

    def fixed_inputs(self):
        scenario = self.SCENARIO()
        options = {
            "policy": "best-fit",
            "allow_preemption": True,
            "queue_weights": dict(scenario.weights),
            "queue_quotas": dict(scenario.quotas),
        }
        return scenario, options

    def test_changing_the_global_random_seed_changes_no_decision(self) -> None:
        scenario, options = self.fixed_inputs()
        random.seed(123456789)
        baseline = simulate(scenario.nodes, scenario.tasks, **options)
        baseline_document = canonical(baseline.to_document())
        for seed in (0, 1, 2, 42, 1337, 2**31 - 1):
            random.seed(seed)
            # consume randomness while building and running, the way a randomized policy might
            for _ in range(500):
                random.random()
                random.randint(0, 10**9)
            result = simulate(scenario.nodes, scenario.tasks, **options)
            self.assertEqual(canonical(result.to_document()), baseline_document, f"seed {seed} changed output")
            self.assertEqual(result.decisions, baseline.decisions)
        # The README declares no random decisions at all, so a different "seed" may change nothing:
        # task count, resource totals and the terminal statistics are part of the equal documents.
        self.assertEqual(len(baseline.placements) + len(baseline.unplaced), len(scenario.tasks))
        self.assertEqual(baseline.metrics["placed"], 5)
        self.assertEqual(baseline.metrics["unplaced"], 1)

    def test_simulation_never_calls_the_random_module(self) -> None:
        scenario, options = self.fixed_inputs()

        def refuse(*_args, **_kwargs):
            raise AssertionError("the scheduler made a random choice; the determinism contract forbids it")

        guards = {name: getattr(random, name) for name in ("random", "randint", "choice", "choices", "shuffle", "sample")}
        try:
            for name in guards:
                setattr(random, name, refuse)
            result = simulate(scenario.nodes, scenario.tasks, **options)
        finally:
            for name, original in guards.items():
                setattr(random, name, original)
        self.assertEqual(result.unplaced, ["f6"])
        assert_schedule_invariants(self, scenario.nodes, scenario.tasks, result)

    def test_mapping_insertion_order_changes_no_equal_score_decision(self) -> None:
        scenario, _ = self.fixed_inputs()
        weight_orders = [
            {"qa": 2, "qb": 1, "qc": 1},
            {"qc": 1, "qb": 1, "qa": 2},
            {"qb": 1, "qa": 2, "qc": 1},
        ]
        quota_orders = [
            {"qa": Resources(4, 4), "qb": Resources(4, 4), "qc": Resources(2, 2)},
            {"qc": Resources(2, 2), "qb": Resources(4, 4), "qa": Resources(4, 4)},
        ]
        documents = set()
        for weights in weight_orders:
            for quotas in quota_orders:
                result = simulate(
                    scenario.nodes,
                    scenario.tasks,
                    policy="best-fit",
                    queue_weights=weights,
                    queue_quotas=quotas,
                )
                documents.add(canonical(result.to_document()))
                documents.add(canonical(result.decisions))
        self.assertEqual(len(documents), 2, "weight/quota insertion order changed a decision")

    def test_input_container_order_is_irrelevant(self) -> None:
        scenario, options = self.fixed_inputs()
        baseline = simulate(scenario.nodes, scenario.tasks, **options)
        variants = (
            (tuple(reversed(scenario.nodes)), scenario.tasks),
            (scenario.nodes, tuple(reversed(scenario.tasks))),
            (tuple(reversed(scenario.nodes)), tuple(reversed(scenario.tasks))),
        )
        for nodes, tasks in variants:
            result = simulate(nodes, tasks, **options)
            self.assertEqual(result.trace(), baseline.trace())
            self.assertEqual(result.decisions, baseline.decisions)
            self.assertEqual(result.to_document(), baseline.to_document())


def canonical(document) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deep_reorder_keys(value):
    """The same JSON content with object keys emitted in the reverse order."""
    if isinstance(value, dict):
        return {key: deep_reorder_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [deep_reorder_keys(item) for item in value]
    return value


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


# -- key-order-equivalent inputs -------------------------------------------------------------------
class KeyOrderEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_text(self, name: str, text: str) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def write_rows(self, name: str, rows: list[dict], reorder: bool = False) -> str:
        lines = []
        for row in rows:
            lines.append(json.dumps(deep_reorder_keys(row) if reorder else row))
        return self.write_text(name, "\n".join(lines) + "\n")

    def test_content_equivalent_inputs_with_different_key_order_agree(self) -> None:
        scenario = scenario_rich_queues()
        normal_cluster = self.write_rows("cluster.jsonl", scenario.node_rows())
        normal_tasks = self.write_rows("tasks.jsonl", scenario.task_rows())
        reversed_cluster = self.write_rows("cluster_rev.jsonl", scenario.node_rows(), reorder=True)
        reversed_tasks = self.write_rows("tasks_rev.jsonl", scenario.task_rows(), reorder=True)
        normal_weights = self.write_rows("weights.jsonl", scenario.weight_rows or [])
        reversed_weights = self.write_rows("weights_rev.jsonl", scenario.weight_rows or [], reorder=True)
        normal_quotas = self.write_rows("quotas.jsonl", scenario.quota_rows or [])
        reversed_quotas = self.write_rows("quotas_rev.jsonl", scenario.quota_rows or [], reorder=True)

        # files differ as bytes on disk...
        with open(normal_tasks, encoding="utf-8") as handle:
            normal_bytes = handle.read()
        with open(reversed_tasks, encoding="utf-8") as handle:
            reversed_bytes = handle.read()
        self.assertNotEqual(normal_bytes, reversed_bytes)
        # ...but parse into the identical model objects
        self.assertEqual(load_cluster(normal_cluster), load_cluster(reversed_cluster))
        self.assertEqual(load_tasks(normal_tasks), load_tasks(reversed_tasks))
        self.assertEqual(load_queue_weights(normal_weights), load_queue_weights(reversed_weights))
        self.assertEqual(load_queue_quotas(normal_quotas), load_queue_quotas(reversed_quotas))

        for command in ("simulate", "trace", "metrics", "policies", "replay"):
            code_normal, out_normal, err_normal = run_cli([
                command, "--cluster", normal_cluster, "--tasks", normal_tasks,
                "--queue-weights", normal_weights, "--queue-quotas", normal_quotas,
                "--policy", "best-fit",
            ])
            code_reversed, out_reversed, err_reversed = run_cli([
                command, "--cluster", reversed_cluster, "--tasks", reversed_tasks,
                "--queue-weights", reversed_weights, "--queue-quotas", reversed_quotas,
                "--policy", "best-fit",
            ])
            self.assertIn(code_normal, (EXIT_OK, EXIT_NEGATIVE))
            self.assertEqual(err_normal, "")
            # event times, types, ids, nodes, reasons and metrics: the whole document, nothing
            # excluded (the format publishes no non-deterministic metadata)
            self.assertEqual((code_reversed, out_reversed, err_reversed),
                             (code_normal, out_normal, err_normal), command)

    def test_inner_object_key_order_is_irrelevant(self) -> None:
        # labels / affinity are objects whose key order must also be ignored
        normal = self.write_text("n.jsonl", '{"id": "n1", "cpu": 4, "memory": 4, "labels": {"zone": "a", "rack": "2"}}\n')
        reversed_order = self.write_text("nr.jsonl", '{"labels": {"rack": "2", "zone": "a"}, "memory": 4, "cpu": 4, "id": "n1"}\n')
        self.assertEqual(load_cluster(normal), load_cluster(reversed_order))
        normal_tasks = self.write_text("t.jsonl", '{"id": "a", "cpu": 1, "memory": 1, "affinity": {"zone": "a"}}\n')
        reversed_tasks = self.write_text("tr.jsonl", '{"affinity": {"zone": "a"}, "memory": 1, "cpu": 1, "id": "a"}\n')
        self.assertEqual(load_tasks(normal_tasks), load_tasks(reversed_tasks))
        code_a, out_a, _ = run_cli(["simulate", "--cluster", normal, "--tasks", normal_tasks])
        code_b, out_b, _ = run_cli(["simulate", "--cluster", reversed_order, "--tasks", reversed_tasks])
        self.assertEqual((code_a, out_a), (code_b, out_b))


# -- corrupt and incomplete inputs keep failing by the public contract -----------------------------
class CorruptInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.good_cluster = self.write("cluster.jsonl", [{"id": "n1", "cpu": 4, "memory": 4}])
        self.good_tasks = self.write("tasks.jsonl", [{"id": "a", "cpu": 1, "memory": 1}])

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write(self, name: str, rows: list[dict]) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def write_text(self, name: str, text: str) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def assert_input_fails(self, cluster: str, tasks: str, *, kind: str, line: int | None = None) -> None:
        for command in ("validate", "simulate", "trace", "metrics"):
            code, out, err = run_cli([command, "--cluster", cluster, "--tasks", tasks])
            self.assertEqual((code, out), (EXIT_ERROR, ""), command)
            document = json.loads(err)
            self.assertEqual(document["error"], kind, command)
            if line is not None:
                self.assertEqual(document["line"], line, command)

    def test_broken_json_is_a_parse_error_with_the_line(self) -> None:
        broken = self.write_text("broken.jsonl", '{"id": "a", "cpu": 1, "memory": 1}\n{not json\n')
        self.assert_input_fails(self.good_cluster, broken, kind="parse_error", line=2)
        with self.assertRaises(SchedulerError):
            load_tasks(broken)

    def test_non_object_row_is_a_parse_error(self) -> None:
        broken = self.write_text("nonobject.jsonl", "[1, 2, 3]\n")
        self.assert_input_fails(self.good_cluster, broken, kind="parse_error", line=1)

    def test_missing_required_field_is_a_parse_error(self) -> None:
        missing_task = self.write_text("noid.jsonl", '{"cpu": 1, "memory": 1}\n')
        self.assert_input_fails(self.good_cluster, missing_task, kind="parse_error", line=1)
        missing_node = self.write_text("noname.jsonl", '{"cpu": 1, "memory": 1}\n')
        self.assert_input_fails(missing_node, self.good_tasks, kind="parse_error", line=1)

    def test_unknown_field_is_a_parse_error_not_a_silent_skip(self) -> None:
        unknown = self.write_text("unknown.jsonl", '{"id": "a", "cpu": 1, "memory": 1, "surprise": 9}\n')
        self.assert_input_fails(self.good_cluster, unknown, kind="parse_error", line=1)

    def test_empty_inputs_are_validation_errors(self) -> None:
        empty = self.write_text("empty.jsonl", "")
        self.assert_input_fails(empty, self.good_tasks, kind="validation_error")
        self.assert_input_fails(self.good_cluster, empty, kind="validation_error")

    def test_corrupt_auxiliary_files_fail_without_success_output(self) -> None:
        cases = {
            "weights-badjson": ("weights", '{"queue": "a", "weight": 1}\n{x\n', "parse_error"),
            "weights-missing": ("weights", '{"queue": "a"}\n', "parse_error"),
            "weights-unknown": ("weights", '{"queue": "a", "weight": 1, "x": 2}\n', "parse_error"),
            "weights-zero": ("weights", '{"queue": "a", "weight": 0}\n', "validation_error"),
            "weights-dup": ("weights", '{"queue": "a", "weight": 1}\n{"queue": "a", "weight": 2}\n',
                            "validation_error"),
            "quotas-badjson": ("quotas", '{"queue": "a", "cpu": 1, "memory": 1}\n{x\n', "parse_error"),
            "quotas-missing": ("quotas", '{"queue": "a", "cpu": 1}\n', "parse_error"),
            "quotas-unknown": ("quotas", '{"queue": "a", "cpu": 1, "memory": 1, "x": 2}\n', "parse_error"),
            "quotas-zero": ("quotas", '{"queue": "a", "cpu": 0, "memory": 1}\n', "validation_error"),
            "quotas-dup": ("quotas", '{"queue": "a", "cpu": 1, "memory": 1}\n{"queue": "a", "cpu": 2, "memory": 2}\n',
                           "validation_error"),
        }
        for name, (kind_file, text, kind) in cases.items():
            path = self.write_text(f"{name}.jsonl", text)
            flag = "--queue-weights" if kind_file == "weights" else "--queue-quotas"
            with self.subTest(name):
                code, out, err = run_cli(
                    ["simulate", "--cluster", self.good_cluster, "--tasks", self.good_tasks, flag, path]
                )
                self.assertEqual((code, out), (EXIT_ERROR, ""))
                self.assertEqual(json.loads(err)["error"], kind)
                # a trailing valid row must not rescue a file whose earlier row is broken/duplicate
                trailing = (
                    '{"queue": "b", "weight": 1}\n'
                    if kind_file == "weights"
                    else '{"queue": "b", "cpu": 1, "memory": 1}\n'
                )
                mixed = self.write_text(f"{name}-mixed.jsonl", text.rstrip("\n") + "\n" + trailing)
                code, out, _ = run_cli(
                    ["simulate", "--cluster", self.good_cluster, "--tasks", self.good_tasks, flag, mixed]
                )
                self.assertEqual((code, out), (EXIT_ERROR, ""), "corrupt rows must not be skipped past")


# -- cross-process reproducibility under hash randomization ----------------------------------------
CHILD_SCRIPT = r"""
import json, sys
from schedsim.cli import load_cluster, load_tasks, load_queue_weights, load_queue_quotas
from schedsim.simulator import simulate

cluster_path, tasks_path, policy, preempt, backfill, weights_path, quotas_path = sys.argv[1:8]
options = {"policy": policy, "allow_preemption": preempt == "1", "backfill": backfill == "1"}
if weights_path != "-":
    options["queue_weights"] = load_queue_weights(weights_path)
if quotas_path != "-":
    options["queue_quotas"] = load_queue_quotas(quotas_path)
result = simulate(load_cluster(cluster_path), load_tasks(tasks_path), **options)
sys.stdout.write(json.dumps(result.to_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
sys.stdout.write(json.dumps(result.decisions, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
sys.stdout.write(json.dumps(result.trace(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
"""


class CrossProcessTests(unittest.TestCase):
    HASH_SEEDS = ("0", "1", "12345")

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        scenario = scenario_rich_queues()
        self.cluster_path = self.write_rows("cluster.jsonl", scenario.node_rows())
        self.tasks_path = self.write_rows("tasks.jsonl", scenario.task_rows())
        self.weights_path = self.write_rows("weights.jsonl", scenario.weight_rows or [])
        self.quotas_path = self.write_rows("quotas.jsonl", scenario.quota_rows or [])
        self.scenario = scenario

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_rows(self, name: str, rows: list[dict]) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def child_environment(self, seed: str | None) -> dict[str, str]:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = REPO_ROOT + os.pathsep + environment.get("PYTHONPATH", "")
        if seed is None:
            environment.pop("PYTHONHASHSEED", None)
        else:
            environment["PYTHONHASHSEED"] = seed
        return environment

    def run_api_child(self, options: dict[str, object], seed: str | None) -> tuple[str, str, str]:
        argv = [
            sys.executable,
            "-c",
            CHILD_SCRIPT,
            self.cluster_path,
            self.tasks_path,
            str(options.get("policy", "first-fit")),
            "1" if options.get("allow_preemption") else "0",
            "0" if options.get("backfill") is False else "1",
            self.weights_path if "queue_weights" in options else "-",
            self.quotas_path if "queue_quotas" in options else "-",
        ]
        completed = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            env=self.child_environment(seed),
            check=True,
            capture_output=True,
            text=True,
        )
        lines = completed.stdout.splitlines()
        self.assertEqual(len(lines), 3)
        return tuple(lines)  # type: ignore[return-value]

    def test_independent_processes_return_itemwise_identical_results(self) -> None:
        option_sets = [
            {"policy": "first-fit"},
            {"policy": "best-fit", "allow_preemption": True},
            {"policy": "first-fit", "backfill": False},
            {"policy": "best-fit", "queue_weights": True, "queue_quotas": True},
            {"policy": "first-fit", "allow_preemption": True, "backfill": False,
             "queue_weights": True, "queue_quotas": True},
        ]
        for options in option_sets:
            real_options = dict(options)
            if "queue_weights" in real_options:
                real_options["queue_weights"] = dict(self.scenario.weights)
            if "queue_quotas" in real_options:
                real_options["queue_quotas"] = dict(self.scenario.quotas)
            in_process = simulate(self.scenario.nodes, self.scenario.tasks, **real_options)
            expected = (
                canonical(in_process.to_document()),
                canonical(in_process.decisions),
                canonical([list(item) for item in in_process.trace()]),
            )
            with self.subTest(options=options):
                for seed in (*self.HASH_SEEDS, None):  # fixed seeds plus an unset hash seed
                    document, decisions, trace = self.run_api_child(options, seed)
                    self.assertEqual((document, decisions, trace), expected)

    def test_cli_output_is_byte_identical_across_hash_seeds(self) -> None:
        for command in ("simulate", "trace", "metrics", "policies", "replay"):
            argv = [
                sys.executable, "-m", "schedsim.cli", command,
                "--cluster", self.cluster_path, "--tasks", self.tasks_path,
                "--queue-weights", self.weights_path,
                "--queue-quotas", self.quotas_path,
                "--policy", "best-fit",
            ]
            seen = set()
            for seed in (*self.HASH_SEEDS, None):
                completed = subprocess.run(
                    argv, cwd=REPO_ROOT, env=self.child_environment(seed),
                    check=False, capture_output=True, text=True,
                )
                seen.add((completed.returncode, completed.stdout, completed.stderr))
            self.assertEqual(len(seen), 1, f"{command} output varies with PYTHONHASHSEED")
            code, out, err = seen.pop()
            self.assertEqual(err, "")
            # the rich scenario leaves f6 unplaced (its own request exceeds the quota): exit 3 for
            # the verdict-bearing commands, 0 for replay which only compares the two internal runs
            self.assertEqual(code, EXIT_OK if command == "replay" else EXIT_NEGATIVE)
            if command == "simulate":
                self.assertEqual(json.loads(out)["unplaced"], ["f6"])
            if command == "replay":
                self.assertTrue(json.loads(out)["identical"])

    def test_set_and_dict_iteration_order_in_process_matches_every_child(self) -> None:
        # The equal-score branch is exactly where hash-dependent iteration could leak in; feed the
        # simulation through set comprehensions in-process and through fresh interpreter children,
        # and show every path still picks the same lowest-id node for the six identical requests.
        scenario = scenario_equal_score_nodes()
        node_set = {node for node in scenario.nodes}  # iteration order is hash-dependent
        task_set = {item for item in scenario.tasks}
        in_process = simulate(tuple(node_set), tuple(sorted(task_set, key=lambda item: item.id)),
                              policy="best-fit")
        expected_nodes = {f"t{i:02d}": ("n1" if i < 3 else "n2") for i in range(6)}
        self.assertEqual({p.task_id: p.node_id for p in in_process.placements}, expected_nodes)

        equal_nodes = self.write_rows("eq_nodes.jsonl", [node_row(n) for n in scenario.nodes])
        equal_tasks = self.write_rows("eq_tasks.jsonl", [task_row(t) for t in scenario.tasks])
        argv = [
            sys.executable, "-c", CHILD_SCRIPT, equal_nodes, equal_tasks,
            "best-fit", "0", "1", "-", "-",
        ]
        for seed in (*self.HASH_SEEDS, None):
            completed = subprocess.run(
                argv, cwd=REPO_ROOT, env=self.child_environment(seed), check=True,
                capture_output=True, text=True,
            )
            document = json.loads(completed.stdout.splitlines()[0])
            self.assertEqual(
                {p["task"]: p["node"] for p in document["placements"]},
                expected_nodes,
                f"PYTHONHASHSEED={seed} changed the equal-score node decision",
            )


if __name__ == "__main__":
    unittest.main()
