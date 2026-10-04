"""Systematic state-machine regression: independent oracles over public results.

Nothing in this file calls the scheduler's placement, predicate or fair-share
internals. Every property -- capacity conservation, half-open interval
semantics, backfill preconditions, preemption bookkeeping, fair-share ordering,
refusal reasons -- is re-derived from the public result (placements, decisions,
metrics) with a second, independent model of the documented behaviour, so a
production bug cannot hide behind shared code. The public entry points
(simulate, trace, metrics, policies, replay -- Python and CLI) are
cross-checked against each other on the same inputs.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from collections import defaultdict
from typing import NamedTuple

from schedsim import POLICIES, Node, Resources, Task, compare_policies, replay, simulate
from schedsim.cli import EXIT_NEGATIVE, EXIT_OK, main


# ---------------------------------------------------------------------------
# Independent model of the documented semantics (shares no code with schedsim)
# ---------------------------------------------------------------------------
def fits_reason(task: Task, node: Node, free: Resources) -> str | None:
    """None when the task fits, else the first failure in predicate order.

    Predicate order is the documented contract: capacity, then affinity /
    anti-affinity, then taints. The reason strings are part of the public
    output, so they are re-stated here verbatim rather than imported.
    """
    if task.request.cpu > free.cpu or task.request.memory > free.memory:
        return (
            f"insufficient capacity: needs cpu={task.request.cpu},memory={task.request.memory} "
            f"has cpu={free.cpu},memory={free.memory}"
        )
    labels = dict(node.labels)
    for key, value in task.affinity:
        if labels.get(key) != value:
            return f"affinity {key}={value} not satisfied (node has {key}={labels.get(key)!r})"
    for key in task.anti_affinity:
        if key in labels:
            return f"anti-affinity: node carries label {key}"
    for taint in node.taints:
        if taint not in task.tolerations:
            return f"untolerated taint {taint}"
    return None


def node_order(nodes: tuple[Node, ...], free: dict[str, Resources], policy: str) -> list[Node]:
    """Candidate order per policy; every tie breaks on node id."""
    ordered = sorted(nodes, key=lambda node: node.id)
    if policy == "first-fit":
        return ordered
    return sorted(ordered, key=lambda node: (free[node.id].cpu, free[node.id].memory, node.id))


def queue_shares(running: list[Task], weights: dict[str, int], capacity: Resources) -> dict[str, float]:
    """Weighted dominant share per queue: max(cpu fraction, memory fraction) / weight."""
    used: dict[str, Resources] = {}
    for task in running:
        used[task.queue] = used.get(task.queue, Resources(0, 0)).plus(task.request)
    shares: dict[str, float] = {}
    for queue, request in used.items():
        cpu_fraction = request.cpu / capacity.cpu if capacity.cpu else 0.0
        memory_fraction = request.memory / capacity.memory if capacity.memory else 0.0
        shares[queue] = max(cpu_fraction, memory_fraction) / weights.get(queue, 1)
    return shares


def pending_order(
    waiting: list[Task],
    running: list[Task],
    weights: dict[str, int] | None,
    capacity: Resources,
) -> list[Task]:
    """The documented waiting-list order: priority, then fair share (fair mode), then id ties."""
    if weights is None:
        return sorted(waiting, key=lambda task: (-task.priority, task.arrival, task.id))
    shares = queue_shares(running, weights, capacity)
    return sorted(
        waiting,
        key=lambda task: (-task.priority, shares.get(task.queue, 0.0), task.arrival, task.queue, task.id),
    )


def total_capacity(nodes: tuple[Node, ...]) -> Resources:
    return Resources(sum(n.capacity.cpu for n in nodes), sum(n.capacity.memory for n in nodes))


# ---------------------------------------------------------------------------
# Scenarios: several nodes, staggered arrivals/durations, CPU- and memory-bound
# ---------------------------------------------------------------------------
class Scenario(NamedTuple):
    name: str
    nodes: tuple[Node, ...]
    tasks: tuple[Task, ...]
    weights: dict[str, int]


def scenario_mixed_bottlenecks() -> Scenario:
    """Heterogeneous nodes; CPU-hungry and memory-hungry tasks compete."""
    nodes = (
        Node("n1", Resources(4, 16), labels=(("zone", "a"),)),
        Node("n2", Resources(8, 4), labels=(("zone", "b"),)),
        Node("n3", Resources(4, 4), labels=(("zone", "a"),), taints=("gpu",)),
    )
    tasks = (
        Task("cpu-hog-1", Resources(3, 1), priority=2, queue="batch", arrival=0, duration=4),
        Task("cpu-hog-2", Resources(2, 1), priority=3, queue="latency", arrival=2, duration=2),
        Task("mem-hog-1", Resources(1, 6), priority=1, queue="batch", arrival=0, duration=3),
        Task("mem-hog-2", Resources(1, 8), priority=1, queue="latency", arrival=3, duration=5),
        Task("gpu-1", Resources(2, 2), priority=2, queue="latency", arrival=1, duration=2, tolerations=("gpu",)),
        Task("pinned", Resources(1, 1), priority=0, queue="batch", arrival=4, duration=3, affinity=(("zone", "a"),)),
        Task("wide", Resources(6, 2), priority=4, queue="latency", arrival=5, duration=2),
    )
    return Scenario("mixed-bottlenecks", nodes, tasks, {"batch": 1, "latency": 2})


def scenario_preemption() -> Scenario:
    """Low-priority tasks fill the cluster; higher priorities arrive later."""
    nodes = (Node("n1", Resources(4, 4)), Node("n2", Resources(4, 4)))
    tasks = (
        Task("low-1", Resources(2, 2), priority=1, queue="batch", arrival=0, duration=10),
        Task("low-2", Resources(2, 2), priority=1, queue="batch", arrival=0, duration=10),
        Task("low-3", Resources(2, 2), priority=2, queue="batch", arrival=0, duration=8),
        Task("mid-1", Resources(2, 2), priority=3, queue="latency", arrival=1, duration=6),
        Task("high-1", Resources(3, 3), priority=9, queue="latency", arrival=3, duration=2),
        Task("high-2", Resources(2, 1), priority=7, queue="latency", arrival=4, duration=3),
    )
    return Scenario("preemption", nodes, tasks, {"batch": 1, "latency": 3})


def scenario_backfill() -> Scenario:
    """A blocked head, a candidate ending exactly at the horizon, one a tick too late.

    "long" only fits the big node, so the head is blocked under both policies
    until it completes; "edge" ends exactly at that horizon and may jump,
    "over" ends one tick past it and may not.
    """
    nodes = (Node("n1", Resources(6, 6)), Node("n2", Resources(2, 2)))
    tasks = (
        Task("long", Resources(3, 3), priority=1, queue="batch", arrival=0, duration=10),
        Task("head", Resources(6, 6), priority=5, queue="latency", arrival=1, duration=6),
        Task("edge", Resources(1, 1), priority=1, queue="batch", arrival=1, duration=9),
        Task("over", Resources(1, 1), priority=1, queue="batch", arrival=1, duration=10),
        Task("tiny", Resources(1, 1), priority=1, queue="latency", arrival=2, duration=2),
    )
    return Scenario("backfill", nodes, tasks, {"batch": 1, "latency": 1})


def scenario_unplaceable() -> Scenario:
    """One failing task per predicate, plus one failing several (capacity must win)."""
    nodes = (
        Node("n1", Resources(4, 8), labels=(("zone", "a"),), taints=("gpu",)),
        Node("n2", Resources(4, 8), labels=(("zone", "b"),), taints=("gpu",)),
    )
    tasks = (
        Task("ok-1", Resources(2, 2), priority=5, queue="batch", arrival=0, duration=3, tolerations=("gpu",)),
        Task("ok-2", Resources(2, 2), priority=5, queue="latency", arrival=1, duration=2, tolerations=("gpu",)),
        Task("x-affinity", Resources(1, 1), priority=2, queue="latency", arrival=0, duration=1,
             affinity=(("zone", "c"),), tolerations=("gpu",)),
        Task("x-cpu", Resources(9, 1), priority=2, queue="batch", arrival=0, duration=1, tolerations=("gpu",)),
        Task("x-mixed", Resources(9, 9), priority=2, queue="batch", arrival=0, duration=1,
             affinity=(("zone", "c"),)),
        Task("x-taint", Resources(1, 1), priority=2, queue="latency", arrival=0, duration=1),
    )
    return Scenario("unplaceable", nodes, tasks, {"batch": 1, "latency": 1})


SCENARIOS = (
    scenario_mixed_bottlenecks(),
    scenario_preemption(),
    scenario_backfill(),
    scenario_unplaceable(),
)


def option_grid(weights: dict[str, int]):
    for policy in POLICIES:
        for allow_preemption in (False, True):
            for backfill in (False, True):
                for queue_weights in (None, weights):
                    yield {
                        "policy": policy,
                        "allow_preemption": allow_preemption,
                        "backfill": backfill,
                        "queue_weights": queue_weights,
                    }


def option_label(options: dict[str, object]) -> str:
    return (
        f"policy={options['policy']},preemption={options['allow_preemption']},"
        f"backfill={options['backfill']},fair={options['queue_weights'] is not None}"
    )


# ---------------------------------------------------------------------------
# Oracle 1: half-open interval semantics and capacity conservation
# ---------------------------------------------------------------------------
def check_intervals(tc: unittest.TestCase, scenario: Scenario, result) -> None:
    task_by_id = {task.id: task for task in scenario.tasks}
    node_by_id = {node.id: node for node in scenario.nodes}
    placements = result.placements

    ids = [item.task_id for item in placements]
    tc.assertEqual(len(ids), len(set(ids)), "each placed task appears exactly once")

    evicted_by = {}
    for item in placements:
        for victim in item.preempted:
            tc.assertNotIn(victim, evicted_by, f"{victim} evicted more than once")
            tc.assertIn(victim, ids, f"{victim} evicted but never placed")
            evicted_by[victim] = item

    by_id = {item.task_id: item for item in placements}
    for item in placements:
        task = task_by_id[item.task_id]
        tc.assertIn(item.node_id, node_by_id)
        tc.assertGreaterEqual(item.start, task.arrival, f"{item.task_id} started before its arrival")
        tc.assertGreaterEqual(item.end, item.start)
        if item.task_id in evicted_by:
            # A preempted task's original interval ends exactly at the eviction tick.
            evictor = evicted_by[item.task_id]
            tc.assertLess(item.end - item.start, task.duration, "evicted interval must be truncated")
            tc.assertEqual(item.end, evictor.start, "evicted interval must end at the eviction tick")
            tc.assertEqual(item.node_id, evictor.node_id, "the evictor takes the victim's node")
        else:
            tc.assertEqual(item.end - item.start, task.duration, "unpreempted tasks run to completion")

    # The preemption counter is exactly the number of victims actually evicted.
    tc.assertEqual(result.metrics["preemptions"], sum(len(item.preempted) for item in placements))
    tc.assertEqual(result.metrics["preemptions"], len(evicted_by))

    # Capacity conservation at every event tick over half-open [start, end)
    # intervals: a task ending at t releases before any decision made at t.
    ticks = {task.arrival for task in scenario.tasks}
    for item in placements:
        ticks.add(item.start)
        ticks.add(item.end)
    for decision in result.decisions:
        if "at" in decision:
            ticks.add(decision["at"])
    for tick in sorted(ticks):
        used: dict[str, Resources] = defaultdict(lambda: Resources(0, 0))
        for item in placements:
            if item.start <= tick < item.end:
                used[item.node_id] = used[item.node_id].plus(task_by_id[item.task_id].request)
        for node_id, load in used.items():
            capacity = node_by_id[node_id].capacity
            tc.assertTrue(
                load.cpu <= capacity.cpu and load.memory <= capacity.memory,
                f"node {node_id} overcommitted at tick {tick}: {load} > {capacity}",
            )


# ---------------------------------------------------------------------------
# Oracle 2: replay every decision tick against an independent scheduler state
# ---------------------------------------------------------------------------
def audit_decisions(tc: unittest.TestCase, scenario: Scenario, options: dict[str, object], result) -> None:
    policy = str(options["policy"])
    weights = options["queue_weights"]
    allow_preemption = bool(options["allow_preemption"])
    task_by_id = {task.id: task for task in scenario.tasks}
    node_by_id = {node.id: node for node in scenario.nodes}
    capacity = total_capacity(scenario.nodes)
    placements = {item.task_id: item for item in result.placements}
    evicted_at = {victim: item.start for item in result.placements for victim in item.preempted}

    by_tick: dict[int, list[dict]] = defaultdict(list)
    final_entries = []
    for decision in result.decisions:
        if "at" in decision:
            by_tick[decision["at"]].append(decision)
        else:
            final_entries.append(decision)

    # Tasks never placed stay in `unplaced` and each gets one closing entry.
    tc.assertEqual(sorted(entry["task"] for entry in final_entries), result.unplaced)
    for entry in final_entries:
        tc.assertIn("unplaced", str(entry["reason"]))
        task = task_by_id[entry["task"]]
        if weights is None:
            for key in ("queue", "weight", "weightedDominantShare"):
                tc.assertNotIn(key, entry)
        else:
            tc.assertEqual(entry["queue"], task.queue)
            tc.assertEqual(entry["weight"], weights.get(task.queue, 1))
            tc.assertIn("weightedDominantShare", entry)

    for tick in sorted(by_tick):
        # Rebuild the state at the top of the tick from public intervals only:
        # a victim stays active until its eviction tick, with its original end.
        active = [
            item
            for item in result.placements
            if item.start < tick
            and item.start + task_by_id[item.task_id].duration > tick
            and evicted_at.get(item.task_id, tick) >= tick
        ]
        used = {node.id: Resources(0, 0) for node in scenario.nodes}
        for item in active:
            used[item.node_id] = used[item.node_id].plus(task_by_id[item.task_id].request)
        waiting = [
            task
            for task in scenario.tasks
            if task.arrival <= tick and (task.id not in placements or placements[task.id].start >= tick)
        ]
        share_sequences: dict[int, list[float]] = defaultdict(list)

        for decision in by_tick[tick]:
            task = task_by_id[decision["task"]]
            running = [task_by_id[item.task_id] for item in active]

            # Fair-share context: present exactly in fair mode, computed
            # pre-decision from the intervals running at this moment.
            if weights is None:
                for key in ("queue", "weight", "weightedDominantShare"):
                    tc.assertNotIn(key, decision)
            else:
                tc.assertEqual(decision["queue"], task.queue)
                tc.assertEqual(decision["weight"], weights.get(task.queue, 1))
                shares = queue_shares(running, weights, capacity)
                tc.assertEqual(decision["weightedDominantShare"], round(shares.get(task.queue, 0.0), 6))

            free = {node.id: node.capacity.minus(used[node.id]) for node in scenario.nodes}
            ordered = pending_order(waiting, running, weights, capacity)

            if "node" not in decision:
                # A refusal is always the current head, and the reason string
                # is the per-node first predicate failure in policy order.
                tc.assertTrue(ordered, f"refusal for {task.id} with an empty waiting list at {tick}")
                tc.assertEqual(ordered[0].id, task.id, f"refusal for non-head {task.id} at {tick}")
                expected = "no node fits (" + "; ".join(
                    f"{node.id}: {fits_reason(task, node, free[node.id])}"
                    for node in node_order(scenario.nodes, free, policy)
                ) + ")"
                tc.assertEqual(decision["reason"], expected)
                continue

            item = placements[task.id]
            tc.assertIn(task, waiting, f"{task.id} placed at {tick} without waiting")
            tc.assertEqual((item.node_id, item.start), (decision["node"], tick))
            head = ordered[0]

            if head.id == task.id:
                if item.preempted:
                    tc.assertEqual(
                        decision["reason"],
                        f"preempting {len(item.preempted)} lower-priority task(s) on {item.node_id}",
                    )
                    audit_preemption(tc, task, item, scenario, task_by_id, active, used)
                else:
                    tc.assertEqual(decision["reason"], f"placed by {policy}")
                    tc.assertIsNone(fits_reason(task, node_by_id[item.node_id], free[item.node_id]))
            else:
                # Backfill: the skipped head genuinely could not be placed,
                # the candidate fits with no preemption, and it ends no later
                # than the earliest running completion (equality accepted).
                tc.assertEqual(decision["reason"], "backfill")
                for node in scenario.nodes:
                    tc.assertIsNotNone(
                        fits_reason(head, node, free[node.id]),
                        f"skipped head {head.id} actually fit {node.id} at {tick}",
                    )
                tc.assertIsNone(
                    fits_reason(task, node_by_id[item.node_id], free[item.node_id]),
                    f"backfilled {task.id} needed preemption to fit at {tick}",
                )
                ends = [running_item.end for running_item in active]
                tc.assertTrue(ends, f"backfill of {task.id} at {tick} with nothing running")
                horizon = min(ends)
                tc.assertLessEqual(item.end, horizon, f"backfilled {task.id} ends past the horizon")
                # Every candidate ordered before this one was ineligible: past
                # the time boundary or fitting no node.
                for candidate in ordered[1:]:
                    if candidate.id == task.id:
                        break
                    past_boundary = tick + candidate.duration > horizon
                    fits_somewhere = any(
                        fits_reason(candidate, node, free[node.id]) is None for node in scenario.nodes
                    )
                    tc.assertTrue(
                        past_boundary or not fits_somewhere,
                        f"{candidate.id} was eligible before {task.id} at {tick}",
                    )

            if weights is not None:
                share_sequences[task.priority].append(decision["weightedDominantShare"])

            # Apply the decision to the independent state.
            for victim_id in item.preempted:
                victim = placements[victim_id]
                active.remove(victim)
                used[victim.node_id] = used[victim.node_id].minus(task_by_id[victim_id].request)
            used[item.node_id] = used[item.node_id].plus(task.request)
            active.append(item)
            waiting.remove(task)

        if weights is not None and not allow_preemption:
            # Same tick, same priority: picks happen in non-decreasing
            # pre-decision weighted dominant share (shares only grow within a
            # tick once nothing can be evicted).
            for priority, sequence in share_sequences.items():
                tc.assertEqual(
                    sequence,
                    sorted(sequence),
                    f"share-ordered picks violated at tick {tick}, priority {priority}",
                )


def audit_preemption(tc, preemptor, placement, scenario, task_by_id, active, used) -> None:
    """The victim set is the smallest lowest-priority-first prefix that fits."""
    tc.assertGreater(preemptor.priority, 0)
    node_by_id = {node.id: node for node in scenario.nodes}
    node = node_by_id[placement.node_id]
    free_here = node.capacity.minus(used[node.id])
    tc.assertFalse(preemptor.request.fits(free_here), "preemption though the task already fit")

    # No node fits the preemptor outright (else it would not preempt at all),
    # and no earlier node in id order can be rescued even by full eviction.
    for other in sorted(scenario.nodes, key=lambda item: item.id):
        free_other = other.capacity.minus(used[other.id])
        tc.assertFalse(preemptor.request.fits(free_other), f"{other.id} fit {preemptor.id} outright")
        if other.id == node.id:
            break
        rescued = free_other
        for item in active:
            candidate = task_by_id[item.task_id]
            if item.node_id == other.id and candidate.priority < preemptor.priority:
                rescued = rescued.plus(candidate.request)
        tc.assertFalse(
            preemptor.request.fits(rescued),
            f"earlier node {other.id} could have hosted {preemptor.id} by eviction",
        )

    candidates = [
        item
        for item in active
        if item.node_id == node.id and task_by_id[item.task_id].priority < preemptor.priority
    ]
    candidates.sort(
        key=lambda item: (
            task_by_id[item.task_id].priority,
            -task_by_id[item.task_id].request.cpu,
            -task_by_id[item.task_id].request.memory,
            item.task_id,
        )
    )
    count = len(placement.preempted)
    chosen = candidates[:count]
    tc.assertEqual(sorted(item.task_id for item in chosen), sorted(placement.preempted))
    for victim in chosen:
        tc.assertLess(task_by_id[victim.task_id].priority, preemptor.priority)
    # The released amount is exactly the victims' requests: the full prefix
    # frees enough, one victim fewer does not.
    freed = free_here
    for victim in chosen:
        freed = freed.plus(task_by_id[victim.task_id].request)
    tc.assertTrue(preemptor.request.fits(freed))
    if chosen:
        shortfall = free_here
        for victim in chosen[:-1]:
            shortfall = shortfall.plus(task_by_id[victim.task_id].request)
        tc.assertFalse(preemptor.request.fits(shortfall), "the victim set was not minimal")


# ---------------------------------------------------------------------------
# Oracle 3: metrics recomputed from the placement intervals alone
# ---------------------------------------------------------------------------
def check_metrics(tc: unittest.TestCase, scenario: Scenario, options: dict[str, object], result) -> None:
    task_by_id = {task.id: task for task in scenario.tasks}
    weights = options["queue_weights"]
    placements = result.placements
    metrics = result.metrics

    makespan = max((item.end for item in placements), default=0)
    tc.assertEqual(result.makespan, makespan)
    tc.assertEqual(metrics["makespan"], makespan)
    tc.assertEqual(metrics["placed"], len(placements))
    placed_ids = {item.task_id for item in placements}
    unplaced = sorted(task.id for task in scenario.tasks if task.id not in placed_ids)
    tc.assertEqual(result.unplaced, unplaced)
    tc.assertEqual(metrics["unplaced"], len(unplaced))

    waits = [item.start - task_by_id[item.task_id].arrival for item in placements]
    tc.assertEqual(metrics["averageWait"], round(sum(waits) / len(waits), 3) if waits else 0.0)
    tc.assertEqual(metrics["maxWait"], max(waits) if waits else 0)

    total_cpu = sum(node.capacity.cpu for node in scenario.nodes)
    total_memory = sum(node.capacity.memory for node in scenario.nodes)
    cpu_time = sum(task_by_id[item.task_id].request.cpu * (item.end - item.start) for item in placements)
    tc.assertEqual(
        metrics["utilization"],
        round(cpu_time / (total_cpu * makespan), 6) if total_cpu and makespan else 0.0,
    )

    # The run ends with every placement released, so fragmentation is measured
    # against full capacity for the smallest unplaced task.
    unplaced_tasks = [task for task in scenario.tasks if task.id in set(unplaced)]
    if unplaced_tasks:
        smallest = min(unplaced_tasks, key=lambda task: (task.request.cpu, task.request.memory, task.id)).request
        blocked = [
            node
            for node in scenario.nodes
            if not (smallest.cpu <= node.capacity.cpu and smallest.memory <= node.capacity.memory)
        ]
        expected = {"nodesBlocked": len(blocked), "wastedCpu": sum(node.capacity.cpu for node in blocked)}
    else:
        expected = {"nodesBlocked": 0, "wastedCpu": 0}
    tc.assertEqual(metrics["fragmentation"], expected)

    if weights is None:
        tc.assertNotIn("queues", metrics)
        return
    queues = metrics["queues"]
    tc.assertEqual(list(queues), sorted({task.queue for task in scenario.tasks}))
    for name, report in queues.items():
        own = [item for item in placements if task_by_id[item.task_id].queue == name]
        queue_cpu = sum(task_by_id[item.task_id].request.cpu * (item.end - item.start) for item in own)
        queue_memory = sum(task_by_id[item.task_id].request.memory * (item.end - item.start) for item in own)
        queue_waits = [item.start - task_by_id[item.task_id].arrival for item in own]
        weight = weights.get(name, 1)
        cpu_share = queue_cpu / (total_cpu * makespan) if total_cpu and makespan else 0.0
        memory_share = queue_memory / (total_memory * makespan) if total_memory and makespan else 0.0
        tc.assertEqual(
            report,
            {
                "weight": weight,
                "placed": len(own),
                "unplaced": sum(1 for task in scenario.tasks if task.queue == name and task.id in set(unplaced)),
                "averageWait": round(sum(queue_waits) / len(queue_waits), 3) if queue_waits else 0.0,
                "cpuTime": queue_cpu,
                "memoryTime": queue_memory,
                "dominantShare": round(max(cpu_share, memory_share) / weight, 6),
            },
        )


# ---------------------------------------------------------------------------
# Oracle 4: the public entry points agree with each other
# ---------------------------------------------------------------------------
def check_cross_entries(tc: unittest.TestCase, scenario: Scenario, options: dict[str, object], result) -> None:
    # simulate <-> trace: the successful decisions are exactly the placements.
    decided = sorted((d["task"], d["node"], d["at"]) for d in result.decisions if "node" in d)
    placed = sorted((item.task_id, item.node_id, item.start) for item in result.placements)
    tc.assertEqual(decided, placed)

    # policies: every entry equals a standalone simulation of that policy.
    rest = {key: value for key, value in options.items() if key != "policy"}
    report = compare_policies(scenario.nodes, scenario.tasks, **rest)
    tc.assertEqual([entry["policy"] for entry in report["policies"]], list(POLICIES))
    for entry in report["policies"]:
        solo = simulate(scenario.nodes, scenario.tasks, policy=entry["policy"], **rest)
        tc.assertEqual(entry["makespan"], solo.makespan)
        tc.assertEqual(entry["placed"], len(solo.placements))
        tc.assertEqual(entry["unplaced"], len(solo.unplaced))
        tc.assertEqual(entry["averageWait"], solo.metrics["averageWait"])
        tc.assertEqual(entry["utilization"], solo.metrics["utilization"])
        if rest["queue_weights"] is not None:
            tc.assertEqual(entry["queues"], solo.metrics["queues"])
        else:
            tc.assertNotIn("queues", entry)

    # replay: the same trajectory, reported as identical.
    report = replay(scenario.nodes, scenario.tasks, **options)
    tc.assertTrue(report["identical"])
    tc.assertEqual(report["differences"], [])
    tc.assertEqual(report["placements"], len(result.placements))
    tc.assertEqual(report["makespan"], result.makespan)
    tc.assertEqual(report["trace"], [list(item) for item in result.trace()][:20])


# ---------------------------------------------------------------------------
# The scenario matrix: every oracle against every option combination
# ---------------------------------------------------------------------------
class ScenarioMatrixTests(unittest.TestCase):
    def test_every_scenario_every_option_combination(self) -> None:
        for scenario in SCENARIOS:
            for options in option_grid(scenario.weights):
                with self.subTest(scenario=scenario.name, options=option_label(options)):
                    result = simulate(scenario.nodes, scenario.tasks, **options)
                    check_intervals(self, scenario, result)
                    audit_decisions(self, scenario, options, result)
                    check_metrics(self, scenario, options, result)
                    check_cross_entries(self, scenario, options, result)


# ---------------------------------------------------------------------------
# Targeted time-semantics checks
# ---------------------------------------------------------------------------
class TimeSemanticsTests(unittest.TestCase):
    def test_completion_frees_capacity_before_the_same_tick_decision(self) -> None:
        # "a" holds the whole node until t=2; "b" starts at 2, not 3: the
        # release happens before the new decision at the same tick.
        nodes = (Node("n1", Resources(2, 2)),)
        tasks = (Task("a", Resources(2, 2), duration=2), Task("b", Resources(2, 2), duration=2))
        result = simulate(nodes, tasks)
        starts = {item.task_id: item.start for item in result.placements}
        self.assertEqual(starts, {"a": 0, "b": 2})

    def test_idle_cluster_never_starts_a_task_before_arrival(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        result = simulate(nodes, (Task("late", Resources(4, 4), arrival=7, duration=1),))
        self.assertEqual(result.placements[0].start, 7)
        self.assertEqual(result.makespan, 8)


class BackfillBoundaryTests(unittest.TestCase):
    def test_boundary_equality_accepted_and_one_tick_late_rejected(self) -> None:
        # Horizon is 10 ("long" ends at 10). "b_edge" ends exactly at 10 and may
        # jump; "c_late" would end at 11 and must stay behind the head.
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            Task("long", Resources(2, 2), duration=10),
            Task("a_head", Resources(4, 4), arrival=1, duration=10),
            Task("b_edge", Resources(1, 1), arrival=1, duration=9),
            Task("c_late", Resources(1, 1), arrival=1, duration=10),
        )
        result = simulate(nodes, tasks)
        placed = {item.task_id: item for item in result.placements}
        self.assertEqual((placed["b_edge"].start, placed["b_edge"].end), (1, 10))
        edge = next(entry for entry in result.decisions if entry["task"] == "b_edge" and "node" in entry)
        self.assertEqual(edge["reason"], "backfill")
        self.assertFalse(
            any(entry["task"] == "c_late" and entry.get("reason") == "backfill" for entry in result.decisions)
        )
        self.assertEqual(placed["c_late"].start, 20)


# ---------------------------------------------------------------------------
# Refusal reasons: stable, in predicate order, tasks kept in `unplaced`
# ---------------------------------------------------------------------------
class RefusalReasonTests(unittest.TestCase):
    def test_capacity_is_reported_before_affinity_and_taints(self) -> None:
        nodes = (Node("n1", Resources(2, 2), labels=(("zone", "a"),), taints=("gpu",)),)
        tasks = (Task("x", Resources(9, 9), affinity=(("zone", "z"),)),)
        result = simulate(nodes, tasks)
        refusal = next(entry for entry in result.decisions if "no node fits" in str(entry["reason"]))
        self.assertIn("insufficient capacity", refusal["reason"])
        self.assertNotIn("affinity", refusal["reason"])
        self.assertNotIn("taint", refusal["reason"])
        self.assertEqual(result.unplaced, ["x"])

    def test_affinity_is_reported_before_taints(self) -> None:
        nodes = (Node("n1", Resources(2, 2), labels=(("zone", "a"),), taints=("gpu",)),)
        tasks = (Task("x", Resources(1, 1), affinity=(("zone", "z"),)),)
        result = simulate(nodes, tasks)
        refusal = next(entry for entry in result.decisions if "no node fits" in str(entry["reason"]))
        self.assertIn("affinity zone=z not satisfied", refusal["reason"])
        self.assertNotIn("taint", refusal["reason"])

    def test_taint_is_reported_when_capacity_and_affinity_pass(self) -> None:
        nodes = (Node("n1", Resources(2, 2), taints=("gpu",)),)
        tasks = (Task("x", Resources(1, 1)),)
        result = simulate(nodes, tasks)
        refusal = next(entry for entry in result.decisions if "no node fits" in str(entry["reason"]))
        self.assertIn("untolerated taint gpu", refusal["reason"])

    def test_unplaceable_tasks_keep_stable_reasons_across_runs(self) -> None:
        scenario = scenario_unplaceable()
        expected_unplaced = ["x-affinity", "x-cpu", "x-mixed", "x-taint"]
        for policy in POLICIES:
            first = simulate(scenario.nodes, scenario.tasks, policy=policy)
            second = simulate(scenario.nodes, scenario.tasks, policy=policy)
            self.assertEqual(first.unplaced, expected_unplaced)
            self.assertEqual(first.decisions, second.decisions)
            # "x-affinity" sorts first and stays blocked, so it is the only
            # task ever refused as head; the others are only ever probed by
            # the backfill scan, which must not leak refusals.
            reasons = [
                entry["reason"] for entry in first.decisions if "no node fits" in str(entry["reason"])
            ]
            self.assertTrue(reasons)
            self.assertTrue(all("affinity zone=c not satisfied" in reason for reason in reasons))
            probed = {"x-cpu", "x-mixed", "x-taint"}
            self.assertFalse(
                any(
                    entry["task"] in probed and "no node fits" in str(entry["reason"])
                    for entry in first.decisions
                )
            )

    def test_each_predicate_failure_is_refused_verbatim_when_head(self) -> None:
        # One unplaceable task per predicate, each alone against the cluster so
        # it becomes the head and its refusal is recorded.
        nodes = (
            Node("n1", Resources(4, 8), labels=(("zone", "a"),), taints=("gpu",)),
            Node("n2", Resources(4, 8), labels=(("zone", "b"),), taints=("gpu",)),
        )
        cases = {
            "x-cpu": (Task("x-cpu", Resources(9, 1), tolerations=("gpu",)), "insufficient capacity"),
            "x-mixed": (Task("x-mixed", Resources(9, 9), affinity=(("zone", "c"),)), "insufficient capacity"),
            "x-affinity": (
                Task("x-affinity", Resources(1, 1), affinity=(("zone", "c"),), tolerations=("gpu",)),
                "affinity zone=c not satisfied",
            ),
            "x-taint": (Task("x-taint", Resources(1, 1)), "untolerated taint gpu"),
        }
        for name, (task, fragment) in cases.items():
            with self.subTest(task=name):
                result = simulate(nodes, (task,))
                self.assertEqual(result.unplaced, [name])
                refusal = next(entry for entry in result.decisions if "no node fits" in str(entry["reason"]))
                self.assertIn(fragment, refusal["reason"])
                self.assertIn("n1:", refusal["reason"])
                self.assertIn("n2:", refusal["reason"])


# ---------------------------------------------------------------------------
# Fair-share semantics
# ---------------------------------------------------------------------------
class FairShareOrderingTests(unittest.TestCase):
    def test_equal_priority_picks_the_smaller_weighted_share_first(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            Task("running", Resources(2, 2), queue="q1", duration=10),
            Task("a", Resources(1, 1), queue="q1", arrival=1, duration=1),
            Task("b", Resources(1, 1), queue="q2", arrival=1, duration=1),
        )
        result = simulate(nodes, tasks, queue_weights={"q1": 1, "q2": 1})
        order = [entry["task"] for entry in result.decisions if "node" in entry and entry["at"] == 1]
        self.assertEqual(order, ["b", "a"])  # q2 holds nothing yet, q1 holds 0.5
        b = next(entry for entry in result.decisions if entry["task"] == "b")
        a = next(entry for entry in result.decisions if entry["task"] == "a" and "node" in entry)
        self.assertEqual(b["weightedDominantShare"], 0.0)
        self.assertEqual(a["weightedDominantShare"], 0.5)

    def test_priority_still_dominates_weighted_share(self) -> None:
        nodes = (Node("n1", Resources(2, 2)),)
        tasks = (
            Task("hi", Resources(2, 2), priority=5, queue="q1", duration=1),
            Task("lo", Resources(2, 2), priority=1, queue="q2", duration=1),
        )
        result = simulate(nodes, tasks, queue_weights={"q1": 1, "q2": 100})
        starts = {item.task_id: item.start for item in result.placements}
        self.assertEqual((starts["hi"], starts["lo"]), (0, 1))

    def test_queue_resource_time_stops_at_the_preemption_tick(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            Task("low", Resources(4, 4), priority=1, queue="q1", duration=10),
            Task("hi", Resources(4, 4), priority=9, queue="q2", arrival=3, duration=2),
        )
        result = simulate(nodes, tasks, allow_preemption=True, queue_weights={"q1": 1, "q2": 1})
        low = next(item for item in result.placements if item.task_id == "low")
        self.assertEqual(low.end, 3)
        queues = result.metrics["queues"]
        self.assertEqual(queues["q1"]["cpuTime"], 4 * 3)  # three ticks, not ten
        self.assertEqual(queues["q1"]["memoryTime"], 4 * 3)
        self.assertEqual(queues["q2"]["cpuTime"], 4 * 2)

    def test_no_weights_means_no_fair_share_fields_anywhere(self) -> None:
        scenario = scenario_mixed_bottlenecks()
        result = simulate(scenario.nodes, scenario.tasks)
        for decision in result.decisions:
            for key in ("queue", "weight", "weightedDominantShare"):
                self.assertNotIn(key, decision)
        self.assertNotIn("queues", result.metrics)


# ---------------------------------------------------------------------------
# Permutation invariance: id-broken ties make input order irrelevant
# ---------------------------------------------------------------------------
def permutations_of(items: tuple) -> list[tuple]:
    sequence = list(items)
    variants = [
        tuple(sequence),
        tuple(reversed(sequence)),
        tuple(sorted(sequence, key=lambda item: item.id, reverse=True)),
    ]
    if len(sequence) > 2:
        middle = len(sequence) // 2
        variants.append(tuple(sequence[middle:] + sequence[:middle]))
    unique: list[tuple] = []
    for variant in variants:
        if variant not in unique:
            unique.append(variant)
    return unique


class PermutationTests(unittest.TestCase):
    def test_permutations_preserve_placements_reasons_and_metrics(self) -> None:
        option_sets = (
            {"policy": "first-fit", "allow_preemption": True, "backfill": True},
            {"policy": "best-fit", "allow_preemption": False, "backfill": True},
            {"policy": "best-fit", "allow_preemption": True, "backfill": False},
        )
        for scenario in (scenario_mixed_bottlenecks(), scenario_preemption(), scenario_unplaceable()):
            for base in option_sets:
                for fair in (False, True):
                    options = dict(base)
                    options["queue_weights"] = scenario.weights if fair else None
                    reference = simulate(scenario.nodes, scenario.tasks, **options)
                    for nodes_perm in permutations_of(scenario.nodes):
                        for tasks_perm in permutations_of(scenario.tasks):
                            with self.subTest(
                                scenario=scenario.name, options=option_label(options),
                                nodes=[n.id for n in nodes_perm], tasks=[t.id for t in tasks_perm],
                            ):
                                rerun = simulate(nodes_perm, tasks_perm, **options)
                                self.assertEqual(rerun.trace(), reference.trace())
                                self.assertEqual(rerun.placements, reference.placements)
                                self.assertEqual(rerun.decisions, reference.decisions)
                                self.assertEqual(rerun.metrics, reference.metrics)
                                self.assertEqual(rerun.unplaced, reference.unplaced)


# ---------------------------------------------------------------------------
# CLI: the same cross-checks through the public command line entry points
# ---------------------------------------------------------------------------
def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def node_row(node: Node) -> dict:
    row: dict = {"id": node.id, "cpu": node.capacity.cpu, "memory": node.capacity.memory}
    if node.labels:
        row["labels"] = dict(node.labels)
    if node.taints:
        row["taints"] = list(node.taints)
    return row


def task_row(task: Task) -> dict:
    row: dict = {
        "id": task.id,
        "cpu": task.request.cpu,
        "memory": task.request.memory,
        "priority": task.priority,
        "queue": task.queue,
        "arrival": task.arrival,
        "duration": task.duration,
    }
    if task.affinity:
        row["affinity"] = dict(task.affinity)
    if task.anti_affinity:
        row["antiAffinity"] = list(task.anti_affinity)
    if task.tolerations:
        row["tolerations"] = list(task.tolerations)
    return row


class PublicEntryCrossCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        scenario = scenario_preemption()
        self.cluster = self.write("cluster.jsonl", [node_row(node) for node in scenario.nodes])
        self.tasks = self.write("tasks.jsonl", [task_row(task) for task in scenario.tasks])
        self.weights = self.write(
            "weights.jsonl", [{"queue": queue, "weight": w} for queue, w in scenario.weights.items()]
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write(self, name: str, rows: list[dict]) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def test_simulate_trace_metrics_policies_replay_agree(self) -> None:
        base = ["--cluster", self.cluster, "--tasks", self.tasks, "--preemption", "--queue-weights", self.weights]

        code, sim_out, sim_err = run_cli(["simulate", *base])
        sim = json.loads(sim_out)
        self.assertEqual(sim_err, "")
        self.assertEqual(code, EXIT_NEGATIVE if sim["unplaced"] else EXIT_OK)

        # trace: the successful decisions are exactly simulate's placements.
        code, trace_out, _ = run_cli(["trace", *base])
        trace = json.loads(trace_out)
        self.assertEqual(code, EXIT_NEGATIVE if trace["unplaced"] else EXIT_OK)
        self.assertEqual(trace["unplaced"], sim["unplaced"])
        self.assertEqual(trace["makespan"], sim["makespan"])
        decided = sorted((d["task"], d["node"], d["at"]) for d in trace["decisions"] if "node" in d)
        placed = sorted((p["task"], p["node"], p["start"]) for p in sim["placements"])
        self.assertEqual(decided, placed)

        # metrics: identical document to the one simulate reported.
        code, metrics_out, _ = run_cli(["metrics", *base])
        self.assertEqual(code, EXIT_NEGATIVE if sim["unplaced"] else EXIT_OK)
        self.assertEqual(json.loads(metrics_out)["metrics"], sim["metrics"])

        # policies: each entry equals the standalone run of that policy.
        code, policies_out, _ = run_cli(["policies", *base])
        report = json.loads(policies_out)
        self.assertEqual({entry["policy"] for entry in report["policies"]}, set(POLICIES))
        best_unplaced = min(entry["unplaced"] for entry in report["policies"])
        self.assertEqual(code, EXIT_OK if best_unplaced == 0 else EXIT_NEGATIVE)
        for entry in report["policies"]:
            solo_code, solo_out, _ = run_cli(["simulate", *base, "--policy", entry["policy"]])
            solo = json.loads(solo_out)
            self.assertEqual(solo_code, EXIT_NEGATIVE if solo["unplaced"] else EXIT_OK)
            self.assertEqual(entry["makespan"], solo["makespan"])
            self.assertEqual(entry["placed"], solo["metrics"]["placed"])
            self.assertEqual(entry["unplaced"], len(solo["unplaced"]))
            self.assertEqual(entry["averageWait"], solo["metrics"]["averageWait"])
            self.assertEqual(entry["utilization"], solo["metrics"]["utilization"])
            self.assertEqual(entry["queues"], solo["metrics"]["queues"])

        # replay: the same trajectory, reported as identical.
        code, replay_out, _ = run_cli(["replay", *base])
        report = json.loads(replay_out)
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(report["identical"])
        self.assertEqual(report["differences"], [])
        self.assertEqual(report["placements"], len(sim["placements"]))
        self.assertEqual(report["makespan"], sim["makespan"])
        expected_trace = sorted((p["task"], p["node"], p["start"], p["end"]) for p in sim["placements"])
        self.assertEqual(report["trace"], [list(item) for item in expected_trace][:20])

    def test_cli_without_weights_emits_no_fair_share_fields(self) -> None:
        base = ["--cluster", self.cluster, "--tasks", self.tasks]
        _, trace_out, _ = run_cli(["trace", *base])
        for decision in json.loads(trace_out)["decisions"]:
            for key in ("queue", "weight", "weightedDominantShare"):
                self.assertNotIn(key, decision)
        _, metrics_out, _ = run_cli(["metrics", *base])
        self.assertNotIn("queues", json.loads(metrics_out)["metrics"])
        _, policies_out, _ = run_cli(["policies", *base])
        for entry in json.loads(policies_out)["policies"]:
            self.assertNotIn("queues", entry)


if __name__ == "__main__":
    unittest.main()
