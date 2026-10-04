"""Systematic regression for the scheduling state machine.

Every check here is an *independent oracle*: it rebuilds each half-open run interval
``[start, end)`` from the public results (placements, decisions, metrics) and re-derives
capacity usage, queue ordering, preemption, backfill and fair-share semantics from first
principles. Nothing in this file calls back into the simulator's bookkeeping, so a
disagreement means the product moved, not the test.

Covered, per scenario and per option combination (policy x preemption x backfill x weights):

* resource conservation: no node's occupancy ever exceeds capacity at any event time;
* time semantics: no task starts before its arrival, completions free capacity before the
  same tick's new decisions, non-preempted tasks are placed exactly once;
* preemption: the victim's interval ends at the eviction tick, the released amount equals
  the victim's request, and the preemption counter equals the number of victims;
* backfill: the jumped head genuinely could not be placed, the candidate fits without
  preemption, and its end does not pass the earliest running completion (equality ok);
* cross-entry consistency: simulate vs trace vs metrics vs policies vs replay;
* fair share: priority dominates, equal priority follows the pre-decision weighted
  dominant share, queue resource time stops at eviction, no fair fields without weights;
* determinism: permuting node/task input order changes no placement, reason or metric;
* unplaced tasks stay in ``unplaced`` with stable, predicate-ordered refusal reasons.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from typing import NamedTuple

from schedsim import POLICIES, Node, Resources, Task, compare_policies, replay, simulate
from schedsim.cli import EXIT_NEGATIVE, EXIT_OK, load_cluster, load_tasks, main


def task(task_id: str, cpu: int = 1, memory: int = 1, **kwargs) -> Task:
    return Task(id=task_id, request=Resources(cpu, memory), **kwargs)


# -- scenarios -----------------------------------------------------------------------------------
class Scenario(NamedTuple):
    name: str
    nodes: tuple[Node, ...]
    tasks: tuple[Task, ...]
    weights: dict[str, int]


def scenario_staggered() -> Scenario:
    """Heterogeneous nodes, staggered arrivals, CPU-bound tasks, one impossible request."""
    return Scenario(
        "staggered",
        (
            Node("n1", Resources(4, 8)),
            Node("n2", Resources(4, 8)),
            Node("n3", Resources(2, 16)),
        ),
        (
            task("a", cpu=2, memory=1, priority=1, queue="q1", arrival=0, duration=4),
            task("b", cpu=3, memory=1, priority=2, queue="q1", arrival=0, duration=3),
            task("c", cpu=1, memory=1, priority=0, queue="q2", arrival=1, duration=2),
            task("d", cpu=4, memory=2, priority=3, queue="q2", arrival=2, duration=2),
            task("e", cpu=2, memory=2, priority=1, queue="q3", arrival=3, duration=5),
            task("f", cpu=9, memory=1, priority=1, queue="q3", arrival=0, duration=1),
        ),
        {"q1": 2, "q2": 1, "q3": 3},
    )


def scenario_memory_bound() -> Scenario:
    """Memory is the bottleneck resource; one task needs more memory than any node has."""
    return Scenario(
        "memory-bound",
        (
            Node("n1", Resources(8, 4)),
            Node("n2", Resources(6, 6)),
            Node("n3", Resources(2, 8)),
        ),
        (
            task("m1", cpu=1, memory=3, priority=1, queue="q1", arrival=0, duration=5),
            task("m2", cpu=1, memory=2, priority=2, queue="q1", arrival=0, duration=3),
            task("m3", cpu=2, memory=4, priority=0, queue="q2", arrival=1, duration=2),
            task("m4", cpu=1, memory=6, priority=3, queue="q2", arrival=2, duration=2),
            task("m5", cpu=1, memory=1, priority=1, queue="q3", arrival=3, duration=4),
            task("m6", cpu=1, memory=9, priority=1, queue="q3", arrival=0, duration=1),
        ),
        {"q1": 1, "q2": 3, "q3": 2},
    )


def scenario_preemption() -> Scenario:
    """Low-priority long tasks fill the cluster; high-priority tasks arrive later."""
    return Scenario(
        "preemption",
        (Node("n1", Resources(4, 4)), Node("n2", Resources(4, 4))),
        (
            task("low1", cpu=2, memory=2, priority=1, queue="lo", arrival=0, duration=10),
            task("low2", cpu=2, memory=2, priority=1, queue="lo", arrival=0, duration=10),
            task("low3", cpu=2, memory=2, priority=2, queue="lo", arrival=0, duration=10),
            task("hi1", cpu=2, memory=2, priority=9, queue="hi", arrival=1, duration=2),
            task("hi2", cpu=4, memory=4, priority=8, queue="hi", arrival=2, duration=1),
            task("mid", cpu=1, memory=1, priority=5, queue="hi", arrival=1, duration=3),
        ),
        {"lo": 1, "hi": 2},
    )


def scenario_backfill() -> Scenario:
    """A blocked head, a candidate ending exactly at the horizon, and one one tick past it.

    The boundary candidate must jump first: every successful backfill shortens the horizon
    (the earliest running completion), so a same-tick jumper behind a shorter one never
    reaches the original boundary.
    """
    return Scenario(
        "backfill",
        (Node("n1", Resources(4, 4)), Node("n2", Resources(2, 2))),
        (
            task("long", cpu=2, memory=2, priority=1, queue="q1", arrival=0, duration=10),
            task("head", cpu=4, memory=4, priority=5, queue="q2", arrival=1, duration=6),
            task("boundary", cpu=1, memory=1, priority=1, queue="q2", arrival=1, duration=9),
            task("toolate", cpu=1, memory=1, priority=1, queue="q3", arrival=1, duration=10),
            task("zshort", cpu=1, memory=1, priority=1, queue="q3", arrival=1, duration=2),
        ),
        {"q1": 1, "q2": 2, "q3": 1},
    )


def scenario_fair() -> Scenario:
    """Three queues with distinct weights; fair share must re-rank equal-priority tasks."""
    return Scenario(
        "fair",
        (Node("n1", Resources(6, 6)), Node("n2", Resources(2, 2))),
        (
            task("A1", cpu=2, memory=2, priority=0, queue="A", arrival=0, duration=6),
            task("A2", cpu=2, memory=2, priority=0, queue="A", arrival=0, duration=4),
            task("B1", cpu=2, memory=2, priority=0, queue="B", arrival=0, duration=4),
            task("B2", cpu=1, memory=1, priority=0, queue="B", arrival=1, duration=3),
            task("C1", cpu=1, memory=3, priority=0, queue="C", arrival=0, duration=5),
            task("C2", cpu=3, memory=1, priority=0, queue="C", arrival=2, duration=2),
        ),
        {"A": 3, "B": 1, "C": 2},
    )


SCENARIOS = (
    scenario_staggered(),
    scenario_memory_bound(),
    scenario_preemption(),
    scenario_backfill(),
    scenario_fair(),
)


def option_matrix(weights: dict[str, int]):
    """Every combination of policy, preemption, backfill and queue weights."""
    for policy in POLICIES:
        for allow_preemption in (False, True):
            for backfill in (False, True):
                for fair in (False, True):
                    options: dict[str, object] = {
                        "policy": policy,
                        "allow_preemption": allow_preemption,
                        "backfill": backfill,
                    }
                    if fair:
                        options["queue_weights"] = dict(weights)
                    yield options


# -- independent oracle --------------------------------------------------------------------------
class Interval:
    """One decided run interval; the end is mutable because preemption truncates it."""

    __slots__ = ("task_id", "node_id", "start", "end")

    def __init__(self, task_id: str, node_id: str, start: int, end: int) -> None:
        self.task_id = task_id
        self.node_id = node_id
        self.start = start
        self.end = end


def oracle_free(nodes: tuple[Node, ...], requests: dict[str, Resources], intervals, clock: int) -> dict[str, list[int]]:
    """Free capacity per node at ``clock`` under half-open intervals (raw, unclamped)."""
    free = {node.id: [node.capacity.cpu, node.capacity.memory] for node in nodes}
    for item in intervals:
        if item.start <= clock < item.end:
            request = requests[item.task_id]
            free[item.node_id][0] -= request.cpu
            free[item.node_id][1] -= request.memory
    return free


def oracle_first_failure(task_: Task, node: Node, free_cpu: int, free_memory: int) -> str | None:
    """The first predicate failure, in the documented predicate order, or None."""
    if task_.request.cpu > free_cpu or task_.request.memory > free_memory:
        return (
            f"insufficient capacity: needs cpu={task_.request.cpu},memory={task_.request.memory} "
            f"has cpu={free_cpu},memory={free_memory}"
        )
    labels = node.label_map()
    for key, value in task_.affinity:
        if labels.get(key) != value:
            return f"affinity {key}={value} not satisfied (node has {key}={labels.get(key)!r})"
    for key in task_.anti_affinity:
        if key in labels:
            return f"anti-affinity: node carries label {key}"
    for taint in node.taints:
        if taint not in task_.tolerations:
            return f"untolerated taint {taint}"
    return None


def oracle_node_order(nodes: tuple[Node, ...], free: dict[str, list[int]], policy: str) -> list[Node]:
    if policy == "first-fit":
        return sorted(nodes, key=lambda node: node.id)
    return sorted(nodes, key=lambda node: (free[node.id][0], free[node.id][1], node.id))


def oracle_select(nodes, free, task_: Task, policy: str) -> tuple[str | None, str]:
    """Node choice and reason, rebuilt from the predicates alone."""
    reasons = []
    for node in oracle_node_order(nodes, free, policy):
        failure = oracle_first_failure(task_, node, *free[node.id])
        if failure is None:
            return node.id, f"placed by {policy}"
        reasons.append(f"{node.id}: {failure}")
    return None, f"no node fits ({'; '.join(reasons)})"


def oracle_shares(tasks_by_id, intervals, clock, total_cpu, total_memory, weights) -> dict[str, float]:
    """Weighted dominant share of every queue from the intervals running at ``clock``."""
    used: dict[str, list[int]] = {}
    for item in intervals:
        if item.start <= clock < item.end:
            queue = tasks_by_id[item.task_id].queue
            request = tasks_by_id[item.task_id].request
            cpu, memory = used.get(queue, [0, 0])
            used[queue] = [cpu + request.cpu, memory + request.memory]
    shares: dict[str, float] = {}
    for queue, (cpu, memory) in used.items():
        cpu_fraction = cpu / total_cpu if total_cpu else 0.0
        memory_fraction = memory / total_memory if total_memory else 0.0
        shares[queue] = max(cpu_fraction, memory_fraction) / weights.get(queue, 1)
    return shares


def oracle_order(pending, shares, fair: bool):
    if not fair:
        return sorted(pending, key=lambda t: (-t.priority, t.arrival, t.id))
    return sorted(
        pending,
        key=lambda t: (-t.priority, shares.get(t.queue, 0.0), t.arrival, t.queue, t.id),
    )


def oracle_victims(task_: Task, node_id: str, intervals, tasks_by_id, free, clock: int):
    """The smallest set of lower-priority running tasks whose eviction fits ``task_``."""
    candidates = [
        item
        for item in intervals
        if item.node_id == node_id
        and item.start <= clock < item.end
        and tasks_by_id[item.task_id].priority < task_.priority
    ]
    candidates.sort(
        key=lambda item: (
            tasks_by_id[item.task_id].priority,
            -tasks_by_id[item.task_id].request.cpu,
            -tasks_by_id[item.task_id].request.memory,
            item.task_id,
        )
    )
    cpu, memory = free[node_id]
    chosen: list[str] = []
    for item in candidates:
        chosen.append(item.task_id)
        request = tasks_by_id[item.task_id].request
        cpu += request.cpu
        memory += request.memory
        if task_.request.cpu <= cpu and task_.request.memory <= memory:
            return chosen
    return None


def check_run(tc: unittest.TestCase, scenario: Scenario, options: dict, result) -> None:
    """Every invariant, checked against one simulation result."""
    nodes, tasks = scenario.nodes, scenario.tasks
    policy = options["policy"]
    fair = "queue_weights" in options
    weights: dict[str, int] = options.get("queue_weights", {})
    allow_preemption = options["allow_preemption"]
    tasks_by_id = {t.id: t for t in tasks}
    requests = {t.id: t.request for t in tasks}
    node_ids = {node.id for node in nodes}
    total_cpu = sum(node.capacity.cpu for node in nodes)
    total_memory = sum(node.capacity.memory for node in nodes)

    tc.assertEqual(result.policy, policy)

    # -- interval algebra, rebuilt from the public placements ------------------------------------
    preempted_by = {}
    for placement in result.placements:
        for victim in placement.preempted:
            preempted_by[victim] = placement
    seen: dict[str, list] = {}
    for placement in result.placements:
        tc.assertIn(placement.task_id, tasks_by_id)
        tc.assertIn(placement.node_id, node_ids)
        owner = tasks_by_id[placement.task_id]
        tc.assertGreaterEqual(placement.start, owner.arrival, "task started before its arrival")
        tc.assertGreater(placement.end, placement.start)
        if placement.task_id in preempted_by:
            tc.assertLessEqual(placement.end - placement.start, owner.duration)
        else:
            tc.assertEqual(placement.end - placement.start, owner.duration, "only preemption truncates")
        seen.setdefault(placement.task_id, []).append(placement)
    for task_id, places in seen.items():
        if task_id not in preempted_by:
            tc.assertEqual(len(places), 1, "non-preempted task placed more than once")
    for victim, evictor in preempted_by.items():
        # the victim's interval ends exactly at the eviction tick, releasing its full request
        tc.assertTrue(
            any(p.end == evictor.start for p in seen[victim]),
            f"victim {victim} has no interval ending at the eviction tick {evictor.start}",
        )
        tc.assertLess(tasks_by_id[victim].priority, tasks_by_id[evictor.task_id].priority)
    # capacity is never exceeded at any event time; half-open intervals mean a completion at t
    # releases before the decisions at t (the new placement at t must fit alongside the rest)
    events = sorted({p.start for p in result.placements} | {p.end for p in result.placements})
    for clock in events:
        free = oracle_free(nodes, requests, result.placements, clock)
        for node in nodes:
            tc.assertGreaterEqual(free[node.id][0], 0, f"cpu overcommitted on {node.id} at {clock}")
            tc.assertGreaterEqual(free[node.id][1], 0, f"memory overcommitted on {node.id} at {clock}")
    tc.assertEqual(
        result.placements,
        sorted(result.placements, key=lambda p: (p.start, p.task_id)),
        "placement output order changed",
    )

    # -- decision walk: replay the trace against independently recomputed state -------------------
    decided: list[Interval] = []
    final_entries = []
    last_at = 0
    for entry in result.decisions:
        if "at" not in entry:
            final_entries.append(entry)
            continue
        clock = entry["at"]
        tc.assertGreaterEqual(clock, last_at, "the event clock ran backwards")
        last_at = clock
        current = tasks_by_id[entry["task"]]
        free = oracle_free(nodes, requests, decided, clock)
        placed_so_far = {item.task_id for item in decided}
        pending = [t for t in tasks if t.arrival <= clock and t.id not in placed_so_far]
        shares = (
            oracle_shares(tasks_by_id, decided, clock, total_cpu, total_memory, weights) if fair else {}
        )
        order = oracle_order(pending, shares, fair)
        if fair:
            tc.assertEqual(entry["queue"], current.queue)
            tc.assertEqual(entry["weight"], weights.get(current.queue, 1))
            tc.assertEqual(
                entry["weightedDominantShare"],
                round(shares.get(current.queue, 0.0), 6),
                "fair context must be the pre-decision share",
            )
        else:
            for key in ("queue", "weight", "weightedDominantShare"):
                tc.assertNotIn(key, entry, "fair-share field present without queue weights")

        if "node" not in entry:
            # a refusal: the reason must be the predicate-ordered failure, rebuilt independently
            _, expected_reason = oracle_select(nodes, free, current, policy)
            tc.assertEqual(entry["reason"], expected_reason)
            continue

        reason = entry["reason"]
        if reason == "backfill":
            head = order[0]
            tc.assertNotEqual(head.id, current.id, "a backfill candidate may not be the head")
            # the jumped head genuinely could not be placed at this tick
            for node in nodes:
                tc.assertIsNotNone(
                    oracle_first_failure(head, node, *free[node.id]),
                    f"jumped head {head.id} actually fits {node.id}",
                )
                if allow_preemption and head.priority > 0:
                    tc.assertIsNone(
                        oracle_victims(head, node.id, decided, tasks_by_id, free, clock),
                        f"jumped head {head.id} could have been placed by preemption",
                    )
            running_ends = [item.end for item in decided if item.start <= clock < item.end]
            tc.assertTrue(running_ends, "backfill happened with nothing running")
            horizon = min(running_ends)
            tc.assertLessEqual(
                clock + current.duration,
                horizon,
                "backfilled task ends past the earliest running completion",
            )
            # the candidate must be the FIRST qualifier behind the head in the declared order
            for earlier in order[1:]:
                if earlier.id == current.id:
                    break
                fits_some = any(
                    oracle_first_failure(earlier, node, *free[node.id]) is None
                    for node in oracle_node_order(nodes, free, policy)
                )
                tc.assertFalse(
                    fits_some and clock + earlier.duration <= horizon,
                    f"{earlier.id} qualified for backfill before {current.id}",
                )
            expected_node, _ = oracle_select(nodes, free, current, policy)
            tc.assertEqual(entry["node"], expected_node)
        elif reason.startswith("preempting "):
            tc.assertEqual(order[0].id, current.id, "a preempting task must be the queue head")
            tc.assertGreater(current.priority, 0)
            for node in nodes:
                tc.assertIsNotNone(oracle_first_failure(current, node, *free[node.id]))
            expected = None
            for node in sorted(nodes, key=lambda n: n.id):
                victims = oracle_victims(current, node.id, decided, tasks_by_id, free, clock)
                if victims:
                    expected = (node.id, victims)
                    break
            tc.assertIsNotNone(expected, "preemption recorded but no node can supply victims")
            tc.assertEqual(entry["node"], expected[0])
            tc.assertEqual(reason, f"preempting {len(expected[1])} lower-priority task(s) on {expected[0]}")
            placement = next(p for p in result.placements if p.task_id == current.id)
            tc.assertEqual(list(placement.preempted), sorted(expected[1]))
            for victim_id in expected[1]:
                interval = next(item for item in decided if item.task_id == victim_id)
                tc.assertLessEqual(interval.start, clock)
                tc.assertLess(clock, interval.end, "victim was not running at the eviction tick")
                interval.end = clock  # released exactly the victim's request at this tick
        else:
            tc.assertEqual(reason, f"placed by {policy}")
            tc.assertEqual(order[0].id, current.id, "the placed task must be the queue head")
            expected_node, _ = oracle_select(nodes, free, current, policy)
            tc.assertEqual(entry["node"], expected_node)
        decided.append(Interval(current.id, entry["node"], clock, clock + current.duration))

    # the walk's reconstruction must equal the published placements exactly
    rebuilt = sorted((item.task_id, item.node_id, item.start, item.end) for item in decided)
    published = sorted((p.task_id, p.node_id, p.start, p.end) for p in result.placements)
    tc.assertEqual(rebuilt, published, "trace decisions and simulate placements disagree")

    # -- unplaced tasks and their final trace entries ---------------------------------------------
    unplaced_expected = sorted(t.id for t in tasks if t.id not in {item.task_id for item in decided})
    tc.assertEqual(result.unplaced, unplaced_expected)
    final_order = oracle_order([tasks_by_id[i] for i in unplaced_expected], {}, fair)
    tc.assertEqual([entry["task"] for entry in final_entries], [t.id for t in final_order])
    for entry in final_entries:
        tc.assertEqual(entry["reason"], "left unplaced when the simulation ended")
        if fair:
            tc.assertEqual(entry["weight"], weights.get(tasks_by_id[entry["task"]].queue, 1))
            tc.assertEqual(entry["weightedDominantShare"], 0.0)
        else:
            tc.assertNotIn("weightedDominantShare", entry)

    # -- metrics, recomputed from the intervals alone ---------------------------------------------
    makespan = max((p.end for p in result.placements), default=0)
    tc.assertEqual(result.makespan, makespan)
    waits = [p.start - tasks_by_id[p.task_id].arrival for p in result.placements]
    used_cpu_time = sum(requests[p.task_id].cpu * (p.end - p.start) for p in result.placements)
    unplaced_tasks = [tasks_by_id[i] for i in unplaced_expected]
    if unplaced_tasks:
        smallest = min(unplaced_tasks, key=lambda t: (t.request.cpu, t.request.memory, t.id)).request
        blocked = sum(
            1
            for node in nodes
            if smallest.cpu > node.capacity.cpu or smallest.memory > node.capacity.memory
        )
        wasted = sum(
            node.capacity.cpu
            for node in nodes
            if smallest.cpu > node.capacity.cpu or smallest.memory > node.capacity.memory
        )
        fragmentation = {"nodesBlocked": blocked, "wastedCpu": wasted}
    else:
        fragmentation = {"nodesBlocked": 0, "wastedCpu": 0}
    expected_metrics: dict[str, object] = {
        "makespan": makespan,
        "placed": len(result.placements),
        "unplaced": len(unplaced_expected),
        "preemptions": sum(len(p.preempted) for p in result.placements),
        "averageWait": round(sum(waits) / len(waits), 3) if waits else 0.0,
        "maxWait": max(waits) if waits else 0,
        "utilization": round(used_cpu_time / (total_cpu * makespan), 6) if total_cpu and makespan else 0.0,
        "fragmentation": fragmentation,
    }
    if fair:
        queues: dict[str, object] = {}
        for name in sorted({t.queue for t in tasks}):
            owned = [p for p in result.placements if tasks_by_id[p.task_id].queue == name]
            cpu_time = sum(requests[p.task_id].cpu * (p.end - p.start) for p in owned)
            memory_time = sum(requests[p.task_id].memory * (p.end - p.start) for p in owned)
            queue_waits = [p.start - tasks_by_id[p.task_id].arrival for p in owned]
            cpu_share = cpu_time / (total_cpu * makespan) if total_cpu and makespan else 0.0
            memory_share = memory_time / (total_memory * makespan) if total_memory and makespan else 0.0
            queues[name] = {
                "weight": weights.get(name, 1),
                "placed": len(owned),
                "unplaced": sum(1 for t in tasks if t.queue == name and t.id in unplaced_expected),
                "averageWait": round(sum(queue_waits) / len(queue_waits), 3) if queue_waits else 0.0,
                "cpuTime": cpu_time,
                "memoryTime": memory_time,
                "dominantShare": round(max(cpu_share, memory_share) / weights.get(name, 1), 6),
            }
        expected_metrics["queues"] = queues
    tc.assertEqual(result.metrics, expected_metrics)


def check_cross_entry(tc: unittest.TestCase, scenario: Scenario, options: dict, result) -> None:
    """The same input through the other public entries must agree with ``simulate``."""
    fair = "queue_weights" in options
    policy_options = {key: value for key, value in options.items() if key != "policy"}
    report = compare_policies(scenario.nodes, scenario.tasks, **policy_options)
    tc.assertEqual([entry["policy"] for entry in report["policies"]], list(POLICIES))
    for entry in report["policies"]:
        solo = simulate(scenario.nodes, scenario.tasks, policy=entry["policy"], **policy_options)
        tc.assertEqual(entry["makespan"], solo.makespan)
        tc.assertEqual(entry["placed"], len(solo.placements))
        tc.assertEqual(entry["unplaced"], len(solo.unplaced))
        tc.assertEqual(entry["averageWait"], solo.metrics["averageWait"])
        tc.assertEqual(entry["utilization"], solo.metrics["utilization"])
        if fair:
            tc.assertEqual(entry["queues"], solo.metrics["queues"])
        else:
            tc.assertNotIn("queues", entry)
    check = replay(scenario.nodes, scenario.tasks, **options)
    tc.assertTrue(check["identical"])
    tc.assertEqual(check["differences"], [])
    tc.assertEqual(check["trace"], [list(item) for item in result.trace()][:20])
    tc.assertEqual(check["placements"], len(result.placements))
    tc.assertEqual(check["makespan"], result.makespan)


# -- the matrix -----------------------------------------------------------------------------------
class MatrixTests(unittest.TestCase):
    def test_every_scenario_every_option_combination(self) -> None:
        for scenario in SCENARIOS:
            for options in option_matrix(scenario.weights):
                label = (
                    f"{scenario.name}/{options['policy']}"
                    f"/preemption={options['allow_preemption']}"
                    f"/backfill={options['backfill']}"
                    f"/fair={'queue_weights' in options}"
                )
                with self.subTest(label):
                    result = simulate(scenario.nodes, scenario.tasks, **options)
                    check_run(self, scenario, options, result)
                    check_cross_entry(self, scenario, options, result)


# -- the scenarios actually exercise what the matrix checks ---------------------------------------
class ScenarioContentTests(unittest.TestCase):
    def test_preemption_fires_and_truncates_victims(self) -> None:
        scenario = scenario_preemption()
        with_preemption = simulate(scenario.nodes, scenario.tasks, allow_preemption=True)
        self.assertEqual(with_preemption.metrics["preemptions"], 3)
        intervals = {p.task_id: (p.start, p.end) for p in with_preemption.placements}
        self.assertEqual(intervals["low1"], (0, 1))  # evicted at tick 1
        self.assertEqual(intervals["low3"], (0, 2))  # evicted at tick 2
        self.assertEqual(intervals["mid"], (1, 2))  # evicted one tick after being placed
        evictor = next(p for p in with_preemption.placements if p.task_id == "hi2")
        self.assertEqual(evictor.preempted, ("low3", "mid"))
        without = simulate(scenario.nodes, scenario.tasks)
        self.assertEqual(without.metrics["preemptions"], 0)
        self.assertTrue(all(not p.preempted for p in without.placements))

    def test_queue_resource_time_stops_at_the_eviction_tick(self) -> None:
        scenario = scenario_preemption()
        result = simulate(
            scenario.nodes,
            scenario.tasks,
            allow_preemption=True,
            queue_weights=scenario.weights,
        )
        queues = result.metrics["queues"]
        # low1 ran 1 tick (2 cpu), low3 ran 2 ticks (2 cpu each), low2 ran all 10.
        self.assertEqual(queues["lo"]["cpuTime"], 2 * 1 + 2 * 2 + 2 * 10)
        self.assertEqual(queues["lo"]["memoryTime"], 2 * 1 + 2 * 2 + 2 * 10)
        self.assertEqual(queues["hi"]["cpuTime"], 2 * 2 + 1 * 1 + 4 * 1)

    def test_backfill_fires_and_respects_the_horizon(self) -> None:
        scenario = scenario_backfill()
        result = simulate(scenario.nodes, scenario.tasks)
        backfilled = {
            entry["task"]: entry["at"] for entry in result.decisions if entry.get("reason") == "backfill"
        }
        self.assertEqual(backfilled, {"boundary": 1, "zshort": 1})
        # boundary ends exactly at the horizon (long's completion at 10) and is accepted;
        # toolate would end one tick past it and is never backfilled.
        intervals = {p.task_id: (p.start, p.end) for p in result.placements}
        self.assertEqual(intervals["boundary"], (1, 10))
        self.assertNotIn("toolate", backfilled)
        self.assertGreaterEqual(intervals["toolate"][0], 10)
        without = simulate(scenario.nodes, scenario.tasks, backfill=False)
        self.assertFalse(any(entry.get("reason") == "backfill" for entry in without.decisions))

    def test_impossible_tasks_stay_unplaced_in_every_combination(self) -> None:
        for scenario, stuck in ((scenario_staggered(), "f"), (scenario_memory_bound(), "m6")):
            for options in option_matrix(scenario.weights):
                with self.subTest(f"{scenario.name}/{stuck}", options=options):
                    result = simulate(scenario.nodes, scenario.tasks, **options)
                    self.assertIn(stuck, result.unplaced)
                    self.assertFalse(any(p.task_id == stuck for p in result.placements))

    def test_fair_mode_reranks_equal_priority_tasks(self) -> None:
        scenario = scenario_fair()
        baseline = simulate(scenario.nodes, scenario.tasks)
        fair = simulate(scenario.nodes, scenario.tasks, queue_weights=scenario.weights)
        baseline_order = [d["task"] for d in baseline.decisions if "node" in d][:3]
        fair_order = [d["task"] for d in fair.decisions if "node" in d][:3]
        self.assertEqual(baseline_order, ["A1", "A2", "B1"])
        self.assertEqual(fair_order, ["A1", "B1", "A2"])

    def test_priority_dominates_share_in_every_fair_combination(self) -> None:
        scenario = scenario_preemption()
        for options in option_matrix(scenario.weights):
            if "queue_weights" not in options:
                continue
            with self.subTest(options=options):
                result = simulate(scenario.nodes, scenario.tasks, **options)
                starts = {p.task_id: p.start for p in result.placements}
                self.assertLess(starts["hi1"], starts["low2"] + 10)
                self.assertEqual(starts["hi1"], 1)


# -- time semantics -------------------------------------------------------------------------------
class TimeSemanticsTests(unittest.TestCase):
    def test_completion_frees_capacity_for_the_same_tick(self) -> None:
        nodes = (Node("n1", Resources(2, 2)),)
        tasks = (
            task("a", cpu=2, memory=2, duration=3),
            task("b", cpu=2, memory=2, duration=2, arrival=1),
        )
        result = simulate(nodes, tasks)
        starts = {p.task_id: p.start for p in result.placements}
        # b starts at the exact tick a completes -- not one tick later.
        self.assertEqual(starts, {"a": 0, "b": 3})

    def test_backfill_boundary_equality_accepted_one_tick_late_rejected(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)

        def run(candidate_duration: int):
            tasks = (
                task("long", cpu=2, memory=2, duration=10),
                task("a_head", cpu=4, memory=4, duration=10, arrival=1),
                task("z_candidate", cpu=1, memory=1, duration=candidate_duration, arrival=1),
            )
            return simulate(nodes, tasks)

        accepted = run(9)  # 1 + 9 == 10 == the horizon
        candidate = next(p for p in accepted.placements if p.task_id == "z_candidate")
        self.assertEqual((candidate.start, candidate.end), (1, 10))
        decision = next(d for d in accepted.decisions if d["task"] == "z_candidate" and "node" in d)
        self.assertEqual(decision["reason"], "backfill")

        rejected = run(10)  # 1 + 10 == 11 > the horizon
        self.assertFalse(
            any(d["task"] == "z_candidate" and d.get("reason") == "backfill" for d in rejected.decisions)
        )
        candidate = next(p for p in rejected.placements if p.task_id == "z_candidate")
        self.assertGreaterEqual(candidate.start, 10)


# -- determinism under input permutation -----------------------------------------------------------
def permutations(nodes, tasks):
    yield "original", nodes, tasks
    yield "nodes-reversed", tuple(reversed(nodes)), tasks
    yield "tasks-reversed", nodes, tuple(reversed(tasks))
    yield "both-reversed", tuple(reversed(nodes)), tuple(reversed(tasks))
    if len(tasks) > 2:
        yield "tasks-rotated", nodes, tasks[1:] + tasks[:1]


class PermutationTests(unittest.TestCase):
    def test_input_permutations_change_no_placement_reason_or_metric(self) -> None:
        combos = (
            {"policy": "first-fit", "allow_preemption": False, "backfill": True},
            {"policy": "best-fit", "allow_preemption": True, "backfill": True, "fair": True},
            {"policy": "first-fit", "allow_preemption": True, "backfill": False, "fair": True},
            {"policy": "best-fit", "allow_preemption": False, "backfill": False},
        )
        for scenario in SCENARIOS:
            for combo in combos:
                options = {key: value for key, value in combo.items() if key != "fair"}
                if combo.get("fair"):
                    options["queue_weights"] = dict(scenario.weights)
                base = simulate(scenario.nodes, scenario.tasks, **options)
                for label, nodes, tasks in permutations(scenario.nodes, scenario.tasks):
                    with self.subTest(f"{scenario.name}/{label}", options=options):
                        other = simulate(nodes, tasks, **options)
                        self.assertEqual(other.trace(), base.trace())
                        self.assertEqual(other.decisions, base.decisions)
                        self.assertEqual(other.metrics, base.metrics)
                        self.assertEqual(other.unplaced, base.unplaced)
                        self.assertTrue(replay(nodes, tasks, **options)["identical"])


# -- cross-entry consistency through the CLI -------------------------------------------------------
def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class CLICrossEntryTests(unittest.TestCase):
    """The same input through every public subcommand must tell one story."""

    CLUSTER = [
        {"id": "n1", "cpu": 4, "memory": 4},
        {"id": "n2", "cpu": 2, "memory": 2},
    ]
    TASKS = [
        {"id": "long", "cpu": 2, "memory": 2, "priority": 1, "queue": "q1", "duration": 10},
        {"id": "head", "cpu": 4, "memory": 4, "priority": 5, "queue": "q2", "arrival": 1, "duration": 6},
        {"id": "boundary", "cpu": 1, "memory": 1, "priority": 1, "queue": "q2", "arrival": 1, "duration": 9},
        {"id": "zshort", "cpu": 1, "memory": 1, "priority": 1, "queue": "q3", "arrival": 1, "duration": 2},
    ]
    WEIGHTS = [{"queue": "q1", "weight": 1}, {"queue": "q2", "weight": 2}, {"queue": "q3", "weight": 1}]

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.cluster = self.write("cluster.jsonl", self.CLUSTER)
        self.tasks = self.write("tasks.jsonl", self.TASKS)
        self.weights = self.write("weights.jsonl", self.WEIGHTS)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write(self, name: str, rows: list[dict]) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def documents(self, extra: list[str] | None = None):
        extra = extra or []
        documents = {}
        for command in ("simulate", "trace", "metrics", "replay", "policies"):
            code, out, err = run_cli([command, "--cluster", self.cluster, "--tasks", self.tasks, *extra])
            self.assertEqual((code, err), (EXIT_OK, ""), command)
            documents[command] = json.loads(out)
        return documents

    def test_entries_agree_with_each_other_and_the_api(self) -> None:
        for extra in ([], ["--queue-weights", self.weights], ["--policy", "best-fit", "--preemption"]):
            with self.subTest(extra=extra):
                documents = self.documents(extra)
                simulate_doc = documents["simulate"]
                placements = {
                    (p["task"], p["node"], p["start"]) for p in simulate_doc["placements"]
                }
                # trace's successful decisions are exactly simulate's placements
                traced = {
                    (d["task"], d["node"], d["at"])
                    for d in documents["trace"]["decisions"]
                    if "node" in d
                }
                self.assertEqual(placements, traced)
                self.assertEqual(documents["trace"]["makespan"], simulate_doc["makespan"])
                self.assertEqual(documents["trace"]["unplaced"], simulate_doc["unplaced"])
                # metrics recomputed by the metrics entry equal simulate's own
                self.assertEqual(documents["metrics"]["metrics"], simulate_doc["metrics"])
                # replay reports the same trajectory
                self.assertTrue(documents["replay"]["identical"])
                self.assertEqual(documents["replay"]["makespan"], simulate_doc["makespan"])
                self.assertEqual(documents["replay"]["placements"], len(simulate_doc["placements"]))
                # each policies entry equals a dedicated simulation under that policy
                solo_options = self._options_for(extra)
                solo_options["policy"] = "best-fit"
                solo = simulate(load_cluster(self.cluster), load_tasks(self.tasks), **solo_options)
                best = next(e for e in documents["policies"]["policies"] if e["policy"] == "best-fit")
                self.assertEqual(best["makespan"], solo.makespan)
                self.assertEqual(best["utilization"], solo.metrics["utilization"])
                # the CLI document is byte-for-byte the API document
                api = simulate(
                    load_cluster(self.cluster),
                    load_tasks(self.tasks),
                    **self._options_for(extra),
                )
                self.assertEqual(simulate_doc, api.to_document())

    def _options_for(self, extra: list[str]) -> dict[str, object]:
        options: dict[str, object] = {}
        if "--policy" in extra:
            options["policy"] = extra[extra.index("--policy") + 1]
        if "--preemption" in extra:
            options["allow_preemption"] = True
        if "--queue-weights" in extra:
            options["queue_weights"] = {"q1": 1, "q2": 2, "q3": 1}
        return options

    def test_fair_fields_appear_exactly_when_weights_are_given(self) -> None:
        with_weights = self.documents(["--queue-weights", self.weights])
        for decision in with_weights["trace"]["decisions"]:
            self.assertIn("weightedDominantShare", decision)
        self.assertIn("queues", with_weights["metrics"]["metrics"])
        for entry in with_weights["policies"]["policies"]:
            self.assertIn("queues", entry)

        without = self.documents()
        for decision in without["trace"]["decisions"]:
            self.assertNotIn("weightedDominantShare", decision)
        self.assertNotIn("queues", without["metrics"]["metrics"])
        for entry in without["policies"]["policies"]:
            self.assertNotIn("queues", entry)

    def test_unplaceable_task_is_a_negative_verdict_everywhere_but_replay(self) -> None:
        tasks = self.write("stuck.jsonl", self.TASKS + [{"id": "huge", "cpu": 9, "memory": 9}])
        for command in ("simulate", "trace", "metrics", "policies"):
            code, out, _ = run_cli([command, "--cluster", self.cluster, "--tasks", tasks])
            self.assertEqual(code, EXIT_NEGATIVE, command)
            document = json.loads(out)
            if command == "policies":
                self.assertTrue(all(entry["unplaced"] == 1 for entry in document["policies"]))
            elif command == "metrics":
                self.assertEqual(document["metrics"]["unplaced"], 1)
            else:
                self.assertEqual(document["unplaced"], ["huge"])
        code, out, _ = run_cli(["replay", "--cluster", self.cluster, "--tasks", tasks])
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(json.loads(out)["identical"])


if __name__ == "__main__":
    unittest.main()
