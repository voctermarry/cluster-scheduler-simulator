"""Resources, predicates, node selection, preemption and whole-simulation behaviour."""

from __future__ import annotations

import unittest

from schedsim import (
    Cluster,
    Node,
    Resources,
    Task,
    ValidationError,
    affinity_predicate,
    capacity_predicate,
    compare_policies,
    fits,
    fragmentation,
    order_candidates,
    preemption_candidates,
    replay,
    select_node,
    simulate,
    taint_predicate,
)


def nodes() -> tuple[Node, ...]:
    return (
        Node("n1", Resources(4, 8), labels=(("zone", "a"),)),
        Node("n2", Resources(4, 8), labels=(("zone", "b"),)),
        Node("n3", Resources(8, 16), labels=(("zone", "a"),), taints=("gpu",)),
    )


def task(task_id: str, cpu: int = 1, memory: int = 1, **kwargs) -> Task:
    return Task(id=task_id, request=Resources(cpu, memory), **kwargs)


class ModelTests(unittest.TestCase):
    def test_resources_reject_negatives(self) -> None:
        with self.assertRaises(ValidationError):
            Resources(-1, 0)

    def test_task_must_request_something(self) -> None:
        with self.assertRaises(ValidationError):
            Task(id="t", request=Resources(0, 0))

    def test_duplicate_node_ids_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Cluster((Node("n", Resources(1, 1)), Node("n", Resources(1, 1))))

    def test_occupy_and_release_track_free_capacity(self) -> None:
        cluster = Cluster(nodes())
        cluster.occupy("n1", Resources(3, 4))
        self.assertEqual(cluster.free("n1"), Resources(1, 4))
        cluster.release("n1", Resources(1, 4))
        self.assertEqual(cluster.free("n1"), Resources(2, 8))

    def test_release_never_goes_below_zero(self) -> None:
        cluster = Cluster(nodes())
        cluster.release("n1", Resources(1, 1))
        self.assertEqual(cluster.free("n1"), Resources(4, 8))


class PredicateTests(unittest.TestCase):
    def test_capacity_reason_names_the_numbers(self) -> None:
        ok, reason = capacity_predicate(task("t", cpu=9), Resources(4, 8))
        self.assertFalse(ok)
        self.assertIn("cpu=9", reason)

    def test_affinity_and_anti_affinity(self) -> None:
        self.assertTrue(affinity_predicate(task("t", affinity=(("zone", "a"),)), nodes()[0])[0])
        ok, reason = affinity_predicate(task("t", affinity=(("zone", "b"),)), nodes()[0])
        self.assertFalse(ok)
        self.assertIn("zone=b", reason)
        self.assertFalse(affinity_predicate(task("t", anti_affinity=("zone",)), nodes()[0])[0])

    def test_taints_need_tolerations(self) -> None:
        self.assertFalse(taint_predicate(task("t"), nodes()[2])[0])
        self.assertTrue(taint_predicate(task("t", tolerations=("gpu",)), nodes()[2])[0])

    def test_first_failure_is_stable(self) -> None:
        cluster = Cluster(nodes())
        ok, reason = fits(cluster, task("t", cpu=99), nodes()[0])
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("insufficient capacity"))

    def test_no_fit_reason_lists_every_node(self) -> None:
        cluster = Cluster(nodes())
        decision = select_node(cluster, task("t", cpu=99))
        self.assertIsNone(decision.node_id)
        self.assertIn("n1:", decision.reason)


class SelectionTests(unittest.TestCase):
    def test_first_fit_takes_the_lowest_node_id(self) -> None:
        cluster = Cluster(nodes())
        self.assertEqual(select_node(cluster, task("t"), "first-fit").node_id, "n1")

    def test_best_fit_takes_the_fullest_node_that_still_fits(self) -> None:
        cluster = Cluster(nodes())
        cluster.occupy("n2", Resources(3, 6))
        self.assertEqual(select_node(cluster, task("t"), "best-fit").node_id, "n2")

    def test_best_fit_falls_back_when_the_fullest_is_too_small(self) -> None:
        cluster = Cluster(nodes())
        cluster.occupy("n2", Resources(4, 8))
        self.assertEqual(select_node(cluster, task("t"), "best-fit").node_id, "n1")

    def test_order_candidates_is_deterministic(self) -> None:
        cluster = Cluster(nodes())
        self.assertEqual([node.id for node in order_candidates(cluster, "first-fit")], ["n1", "n2", "n3"])
        self.assertEqual([node.id for node in order_candidates(cluster, "best-fit")], ["n1", "n2", "n3"])


class PreemptionTests(unittest.TestCase):
    def setUp(self) -> None:
        # A 4x4 node with 6 units already occupied would be an impossible baseline; the node is 8x8 so
        # the scenario is one a scheduler could actually be handed.
        self.capacity = Resources(8, 8)
        self.cluster = Cluster((Node("n1", self.capacity),))
        self.low = task("low", cpu=4, memory=4, priority=1, duration=10)
        self.mid = task("mid", cpu=2, memory=2, priority=2, duration=10)
        self.high = task("high", cpu=4, memory=4, priority=5, duration=2)
        self.cluster.occupy("n1", self.low.request)
        self.cluster.occupy("n1", self.mid.request)

    def test_no_preemption_needed_when_capacity_is_free(self) -> None:
        chosen, reason = preemption_candidates(self.cluster, task("x", cpu=1, priority=9), Node("n1", self.capacity), [], {}, 0)
        self.assertEqual(chosen, [])
        self.assertIn("no preemption needed", reason)

    def test_lowest_priority_is_evicted_first(self) -> None:
        from schedsim import Placement

        placements = [Placement("low", "n1", 0, 10), Placement("mid", "n1", 0, 10)]
        tasks = {"low": self.low, "mid": self.mid}
        chosen, reason = preemption_candidates(self.cluster, self.high, Node("n1", self.capacity), placements, tasks, 1)
        self.assertEqual([item.task_id for item in chosen], ["low"])
        self.assertIn("preempting 1", reason)

    def test_equal_priority_is_never_evicted(self) -> None:
        from schedsim import Placement

        placements = [Placement("mid", "n1", 0, 10)]
        same = task("same", cpu=2, priority=2)
        chosen, _ = preemption_candidates(self.cluster, same, Node("n1", self.capacity), placements, {"mid": self.mid}, 1)
        self.assertEqual(chosen, [])

    def test_refusal_when_even_evicting_everything_would_not_fit(self) -> None:
        from schedsim import Placement

        placements = [Placement("low", "n1", 0, 10), Placement("mid", "n1", 0, 10)]
        tasks = {"low": self.low, "mid": self.mid}
        huge = task("huge", cpu=99, priority=9)
        chosen, reason = preemption_candidates(self.cluster, huge, Node("n1", self.capacity), placements, tasks, 1)
        self.assertEqual(chosen, [])
        self.assertIn("would not fit", reason)


class SimulationTests(unittest.TestCase):
    def test_simple_schedule_places_everything_and_never_overcommits(self) -> None:
        cluster_nodes = (Node("n1", Resources(2, 4)),)
        tasks = (task("a", cpu=1, memory=1, duration=3), task("b", cpu=1, memory=1, duration=2), task("c", cpu=2, memory=4, duration=1))
        result = simulate(cluster_nodes, tasks)
        self.assertEqual(len(result.placements), 3)
        self.assertEqual(result.unplaced, [])
        capacity = {}
        for item in result.placements:
            request = {t.id: t for t in tasks}[item.task_id].request
            for tick in range(item.start, item.end):
                used = capacity.get(("n1", tick), Resources(0, 0)).plus(request)
                self.assertTrue(used.fits(cluster_nodes[0].capacity), f"overcommitted at tick {tick}")
                capacity[("n1", tick)] = used

    def test_waits_are_recorded_and_arrivals_are_respected(self) -> None:
        cluster_nodes = (Node("n1", Resources(1, 1)),)
        tasks = (task("a", duration=5, arrival=0), task("b", duration=1, arrival=3))
        result = simulate(cluster_nodes, tasks)
        starts = {item.task_id: item.start for item in result.placements}
        self.assertEqual(starts["a"], 0)
        self.assertGreaterEqual(starts["b"], 5)
        self.assertEqual(result.metrics["maxWait"], starts["b"] - 3)

    def test_backfill_lets_a_short_task_pass_a_blocked_head(self) -> None:
        # Single node: "long" runs with half the node idle; the waiting head wants the whole node and
        # is blocked, while "short" fits the idle half and finishes before long completes.
        cluster_nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("long", cpu=2, memory=2, duration=10),
            task("head", cpu=4, memory=4, priority=5, duration=10, arrival=1),
            task("short", cpu=2, memory=2, duration=2, arrival=1),
        )
        with_backfill = simulate(cluster_nodes, tasks, backfill=True)
        without = simulate(cluster_nodes, tasks, backfill=False)
        starts = {item.task_id: item.start for item in with_backfill.placements}
        starts_without = {item.task_id: item.start for item in without.placements}
        self.assertEqual((starts["long"], starts["short"], starts["head"]), (0, 1, 10))
        # With backfill disabled the short task must wait behind the head it could have jumped.
        self.assertEqual(starts_without["short"], 20)
        short_decision = next(
            entry for entry in with_backfill.decisions if entry["task"] == "short" and "node" in entry
        )
        self.assertEqual(short_decision["reason"], "backfill")
        self.assertEqual(short_decision["at"], 1)

    def test_backfill_accepts_a_candidate_finishing_exactly_at_the_boundary(self) -> None:
        # long ends at 10; the candidate starts at 1 with duration 9 -> ends exactly at 10.
        cluster_nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("long", cpu=2, memory=2, duration=10),
            task("a_head", cpu=4, memory=4, duration=10, arrival=1),
            task("z_edge", cpu=1, memory=1, duration=9, arrival=1),
        )
        result = simulate(cluster_nodes, tasks)
        edge = next(item for item in result.placements if item.task_id == "z_edge")
        self.assertEqual((edge.start, edge.end), (1, 10))
        edge_decision = next(entry for entry in result.decisions if entry["task"] == "z_edge" and "node" in entry)
        self.assertEqual(edge_decision["reason"], "backfill")

    def test_backfill_rejects_a_candidate_finishing_after_the_boundary(self) -> None:
        # start 1 + duration 10 = 11 > horizon 10: the candidate may not jump.
        cluster_nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("long", cpu=2, memory=2, duration=10),
            task("a_head", cpu=4, memory=4, duration=10, arrival=1),
            task("z_late", cpu=1, memory=1, duration=10, arrival=1),
        )
        result = simulate(cluster_nodes, tasks)
        late = next(item for item in result.placements if item.task_id == "z_late")
        self.assertEqual(late.start, 20)
        self.assertFalse(any(entry.get("reason") == "backfill" for entry in result.decisions))

    def test_backfill_skips_ineligible_candidates_and_logs_no_probe_refusals(self) -> None:
        # Behind the blocked head come, in order: a task that fails the resource boundary, one that
        # fails the time boundary, and one that qualifies. Only the last may become a placement, and
        # probing the first two must not record refusals.
        cluster_nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("long", cpu=2, memory=2, duration=10),
            task("a_head", cpu=3, memory=3, duration=10, arrival=1),
            task("b_big", cpu=3, memory=3, duration=2, arrival=1),
            task("c_long", cpu=1, memory=1, duration=99, arrival=1),
            task("d_ok", cpu=1, memory=1, duration=1, arrival=1),
        )
        result = simulate(cluster_nodes, tasks)
        placed = {item.task_id: (item.start, item.end) for item in result.placements}
        self.assertEqual(placed["d_ok"], (1, 2))
        d_decision = next(entry for entry in result.decisions if entry["task"] == "d_ok")
        self.assertEqual(d_decision["reason"], "backfill")
        # No probe may leak a refusal (or anything else) at the probing tick.
        self.assertFalse(
            any(entry["task"] in {"b_big", "c_long"} and entry.get("at") == 1 for entry in result.decisions)
        )

    def test_backfill_never_starts_when_nothing_is_running(self) -> None:
        # The head is impossible right now and no task is running, so no completion can free room:
        # nothing behind it may jump, even with backfill enabled.
        cluster_nodes = (Node("n1", Resources(2, 2)),)
        tasks = (task("huge", cpu=9, memory=9, duration=1), task("short", cpu=1, memory=1, duration=1))
        result = simulate(cluster_nodes, tasks, backfill=True)
        self.assertEqual(result.unplaced, ["huge", "short"])
        self.assertFalse(any(entry.get("reason") == "backfill" for entry in result.decisions))

    def test_backfill_respects_affinity_anti_affinity_and_taints(self) -> None:
        tainted = (Node("n1", Resources(4, 4), taints=("gpu",)),)
        tasks = (
            task("long", cpu=2, memory=2, duration=10, tolerations=("gpu",)),
            task("head", cpu=4, memory=4, duration=10, arrival=1),
            task("short", cpu=2, memory=2, duration=2, arrival=1),
        )
        result = simulate(tainted, tasks)
        self.assertFalse(any(item.task_id == "short" for item in result.placements))
        self.assertFalse(any(entry.get("reason") == "backfill" for entry in result.decisions))

    def test_a_backfill_probe_never_preempts(self) -> None:
        # With preemption enabled, a candidate that could only fit by evicting a running task is
        # skipped by the probe (no eviction, no placement, no refusal); a genuinely free candidate
        # behind it is still backfilled.
        cluster_nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("long", cpu=2, memory=2, priority=1, duration=10),
            task("a_head", cpu=5, memory=5, priority=5, duration=2, arrival=1),
            task("jumper", cpu=4, memory=4, duration=2, arrival=1),
            task("z_small", cpu=1, memory=1, duration=1, arrival=1),
        )
        result = simulate(cluster_nodes, tasks, allow_preemption=True)
        self.assertEqual(result.metrics["preemptions"], 0)
        long = next(item for item in result.placements if item.task_id == "long")
        self.assertEqual((long.start, long.end, long.preempted), (0, 10, ()))
        self.assertFalse(
            any(item.task_id == "jumper" and item.start == 1 for item in result.placements)
        )
        self.assertFalse(
            any(entry["task"] == "jumper" and entry.get("at") == 1 for entry in result.decisions)
        )
        small = next(item for item in result.placements if item.task_id == "z_small")
        self.assertEqual((small.start, small.end), (1, 2))

    def test_backfill_uses_the_policy_node_order(self) -> None:
        # n1 runs half full; n2 is empty. Under best-fit the backfilled short task takes the fuller
        # node that still fits (n1); first-fit would take n1 by id anyway.
        cluster_nodes = (Node("n1", Resources(4, 4)), Node("n2", Resources(2, 2)))
        tasks = (
            task("long", cpu=2, memory=2, duration=10),
            task("head", cpu=9, memory=9, duration=10, arrival=1),
            task("short", cpu=2, memory=2, duration=2, arrival=1),
        )
        result = simulate(cluster_nodes, tasks, policy="best-fit")
        short = next(item for item in result.placements if item.task_id == "short")
        self.assertEqual(short.node_id, "n1")

    def test_several_backfills_can_land_at_the_same_tick(self) -> None:
        cluster_nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("long", cpu=2, memory=2, duration=10),
            task("head", cpu=4, memory=4, priority=5, duration=10, arrival=1),
            task("s1", cpu=1, memory=1, duration=1, arrival=1),
            task("s2", cpu=1, memory=1, duration=1, arrival=1),
        )
        result = simulate(cluster_nodes, tasks)
        shorts = {item.task_id: (item.start, item.end) for item in result.placements if item.task_id in {"s1", "s2"}}
        self.assertEqual(shorts, {"s1": (1, 2), "s2": (1, 2)})
        self.assertEqual(
            [entry["task"] for entry in result.decisions if entry.get("reason") == "backfill"], ["s1", "s2"]
        )
        # The real short intervals drive the waits, not the head's blocking.
        self.assertEqual(result.metrics["maxWait"], 9)

    def test_fair_mode_reranks_candidates_after_every_backfill(self) -> None:
        # Free room for one 2-unit backfill plus one 1-unit backfill. Queue C already holds share 0.25
        # at tick 1; queue B holds none. The first two candidates are b1, b2 (queue B, share 0), then
        # c (queue C). Frozen ordering would backfill b1 then b2; re-ranking after b1 lifts queue B to
        # 0.5 and lets c overtake b2 for the second slot.
        cluster_nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("long", cpu=1, memory=2, queue="L", duration=10),
            task("cRun", cpu=0, memory=1, queue="C", duration=10),
            task("head", cpu=4, memory=4, priority=5, duration=10, arrival=1),
            task("b1", cpu=2, memory=0, queue="B", duration=1, arrival=1),
            task("b2", cpu=1, memory=0, queue="B", duration=1, arrival=1),
            task("c", cpu=1, memory=0, queue="C", duration=1, arrival=1),
        )
        weights = {"B": 1, "C": 1}
        fair = simulate(cluster_nodes, tasks, queue_weights=weights)
        baseline = simulate(cluster_nodes, tasks)
        self.assertEqual(
            [(entry["task"], entry["at"]) for entry in fair.decisions if entry.get("reason") == "backfill"][:2],
            [("b1", 1), ("c", 1)],
        )
        self.assertEqual(
            [entry["task"] for entry in baseline.decisions if entry.get("reason") == "backfill"][:2],
            ["b1", "b2"],
        )
        # The backfilled decisions carry the same pre-decision fair-share context as normal ones.
        b1 = next(entry for entry in fair.decisions if entry["task"] == "b1")
        self.assertEqual(b1["weightedDominantShare"], 0.0)

    def test_backfilled_runs_are_deterministic_in_trace_and_decisions(self) -> None:
        cluster_nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("long", cpu=2, memory=2, queue="L", duration=10),
            task("head", cpu=4, memory=4, priority=5, duration=10, arrival=1),
            task("b_big", cpu=3, memory=3, queue="B", duration=2, arrival=1),
            task("c_long", cpu=1, memory=1, queue="C", duration=99, arrival=1),
            task("d_ok", cpu=1, memory=1, queue="C", duration=1, arrival=1),
        )
        options = {"queue_weights": {"B": 1, "C": 1}}
        first = simulate(cluster_nodes, tasks, **options)
        second = simulate(cluster_nodes, tasks, **options)
        self.assertEqual(first.trace(), second.trace())
        self.assertEqual(first.decisions, second.decisions)
        self.assertTrue(replay(cluster_nodes, tasks, **options)["identical"])

    def test_preemption_places_a_high_priority_task_that_would_otherwise_wait(self) -> None:
        cluster_nodes = (Node("n1", Resources(2, 2)),)
        tasks = (
            task("low", cpu=2, memory=2, priority=1, duration=10),
            task("high", cpu=2, memory=2, priority=9, duration=1, arrival=1),
        )
        without = simulate(cluster_nodes, tasks)
        with_preemption = simulate(cluster_nodes, tasks, allow_preemption=True)
        self.assertEqual(len(without.placements), 2)
        high_without = {item.task_id: item for item in without.placements}["high"]
        high_with = {item.task_id: item for item in with_preemption.placements}["high"]
        self.assertGreater(high_without.start, high_with.start)
        self.assertEqual(with_preemption.metrics["preemptions"], 1)

    def test_replay_is_identical(self) -> None:
        report = replay(nodes(), (task("a", cpu=1), task("b", cpu=2), task("c", cpu=8, priority=3)))
        self.assertTrue(report["identical"])
        self.assertEqual(report["differences"], [])

    def test_policy_comparison_reports_both(self) -> None:
        report = compare_policies(nodes(), (task("a", cpu=1), task("b", cpu=1)))
        policies = {entry["policy"] for entry in report["policies"]}
        self.assertEqual(policies, {"first-fit", "best-fit"})

    def test_unplaceable_task_is_reported_not_dropped(self) -> None:
        result = simulate((Node("n1", Resources(1, 1)),), (task("huge", cpu=9),))
        self.assertEqual(result.placements, [])
        self.assertEqual(result.unplaced, ["huge"])
        self.assertTrue(any("unplaced" in str(decision.get("reason", "")) for decision in result.decisions))

    def test_fragmentation_counts_blocked_nodes(self) -> None:
        cluster = Cluster((Node("n1", Resources(4, 4)), Node("n2", Resources(1, 1))))
        cluster.occupy("n1", Resources(4, 4))
        report = fragmentation(cluster, {"pending": task("pending", cpu=1)})
        self.assertEqual(report["nodesBlocked"], 1)

    def test_duplicate_task_ids_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            simulate(nodes(), (task("a"), task("a")))

    def test_unknown_policy_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            simulate(nodes(), (task("a"),), policy="round-robin")


class FairShareTests(unittest.TestCase):
    def weights(self) -> dict[str, int]:
        return {"a": 2, "b": 1}

    def test_equal_priority_prefers_the_queue_with_smaller_weighted_share(self) -> None:
        cluster_nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("a1", cpu=2, memory=2, queue="a", duration=4),
            task("a2", cpu=2, memory=2, queue="a", duration=2),
            task("b1", cpu=2, memory=2, queue="b", duration=2),
        )
        baseline = simulate(cluster_nodes, tasks)
        fair = simulate(cluster_nodes, tasks, queue_weights=self.weights())
        baseline_starts = {item.task_id: item.start for item in baseline.placements}
        fair_starts = {item.task_id: item.start for item in fair.placements}
        # Baseline serves queue "a" twice before "b"; fairness lets "b" jump the second "a" task.
        self.assertEqual([baseline_starts["a2"], baseline_starts["b1"]], [0, 2])
        self.assertEqual([fair_starts["b1"], fair_starts["a2"]], [0, 2])

    def test_priority_still_dominates_weighted_share(self) -> None:
        cluster_nodes = (Node("n1", Resources(2, 2)),)
        tasks = (
            task("a1", cpu=2, memory=2, queue="a", priority=5, duration=2),
            task("b1", cpu=2, memory=2, queue="b", priority=1, duration=2),
        )
        result = simulate(cluster_nodes, tasks, queue_weights={"a": 1, "b": 100})
        starts = {item.task_id: item.start for item in result.placements}
        self.assertLess(starts["a1"], starts["b1"])

    def test_equal_priority_and_share_break_ties_on_arrival_queue_and_id(self) -> None:
        cluster_nodes = (Node("n1", Resources(1, 1)),)
        tasks = (
            task("z", queue="z", arrival=0),
            task("a", queue="a", arrival=0),
            task("late", queue="a", arrival=1),
        )
    def test_equal_priority_and_share_break_ties_on_arrival_queue_and_id(self) -> None:
        serial = (Node("n1", Resources(1, 1)),)
        serial_tasks = (
            task("z", queue="z", arrival=0),
            task("a", queue="a", arrival=0),
            task("late", queue="a", arrival=1),
        )
        result = simulate(serial, serial_tasks, queue_weights={"a": 1, "z": 1})
        # One task per tick: at tick 0 equal shares tie-break on queue name ("a" before "z");
        # at tick 1 the earlier arrival ("z") has already run and "late" is the only waiter.
        self.assertEqual([item.task_id for item in result.placements], ["a", "z", "late"])

        # Same queue, same arrival, everything else equal: task id is the final tie-break.
        wide = (Node("n2", Resources(4, 4)),)
        wide_tasks = (
            task("y", cpu=1, memory=1, queue="q", arrival=0),
            task("x", cpu=1, memory=1, queue="q", arrival=0),
        )
        wide_result = simulate(wide, wide_tasks, queue_weights={"q": 1})
        self.assertEqual([item.task_id for item in wide_result.placements], ["x", "y"])

    def test_trace_carries_queue_weight_and_pre_decision_share(self) -> None:
        cluster_nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("a1", cpu=2, memory=2, queue="a", duration=4),
            task("b1", cpu=2, memory=2, queue="b", duration=2),
        )
        result = simulate(cluster_nodes, tasks, queue_weights=self.weights())
        first = result.decisions[0]
        self.assertEqual(first["queue"], "a")
        self.assertEqual(first["weight"], 2)
        self.assertEqual(first["weightedDominantShare"], 0.0)
        self.assertIsInstance(first["weightedDominantShare"], float)
        # The second placement records queue "a" already holding half the cluster: 0.5 / weight 2.
        second = result.decisions[1]
        self.assertEqual(second["queue"], "b")
        self.assertEqual(second["weightedDominantShare"], 0.0)

    def test_undeclared_queue_has_implicit_weight_one(self) -> None:
        cluster_nodes = (Node("n1", Resources(2, 2)),)
        tasks = (task("a1", cpu=1, memory=1, queue="a"), task("x1", cpu=1, memory=1, queue="x"))
        result = simulate(cluster_nodes, tasks, queue_weights={"a": 4})
        queues = result.metrics["queues"]
        self.assertEqual(queues["a"]["weight"], 4)
        self.assertEqual(queues["x"]["weight"], 1)

    def test_queue_metrics_are_sorted_and_complete(self) -> None:
        cluster_nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("a1", cpu=2, memory=2, queue="a", duration=4),
            task("b1", cpu=2, memory=2, queue="b", duration=2),
            task("ghost", cpu=9, queue="c"),
        )
        result = simulate(cluster_nodes, tasks, queue_weights=self.weights())
        queues = result.metrics["queues"]
        self.assertEqual(list(queues), ["a", "b", "c"])
        for report in queues.values():
            self.assertEqual(
                set(report),
                {"weight", "placed", "unplaced", "averageWait", "cpuTime", "memoryTime", "dominantShare"},
            )
        self.assertEqual(queues["a"]["placed"], 1)
        self.assertEqual(queues["a"]["cpuTime"], 8)
        self.assertEqual(queues["c"]["placed"], 0)
        self.assertEqual(queues["c"]["unplaced"], 1)
        self.assertEqual(queues["c"]["averageWait"], 0.0)
        self.assertEqual(queues["c"]["dominantShare"], 0.0)

    def test_preempted_resource_time_stops_at_the_eviction_tick(self) -> None:
        cluster_nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("low", cpu=4, memory=4, queue="a", priority=1, duration=10),
            task("hi", cpu=4, memory=4, queue="b", priority=9, duration=2, arrival=2),
        )
        result = simulate(cluster_nodes, tasks, queue_weights={"a": 1, "b": 3}, allow_preemption=True)
        queues = result.metrics["queues"]
        # "low" ran for two ticks (cpu-time 8), not the ten it asked for.
        self.assertEqual(queues["a"]["cpuTime"], 8)
        self.assertEqual(queues["a"]["memoryTime"], 8)
        self.assertEqual(queues["b"]["cpuTime"], 8)

    def test_zero_cpu_capacity_and_zero_makespan_yield_zero_share(self) -> None:
        memory_only = (Node("n1", Resources(0, 4)),)
        placed = simulate(memory_only, (task("a", cpu=0, memory=2, queue="a", duration=2),), queue_weights={"a": 2})
        self.assertEqual(placed.metrics["queues"]["a"]["dominantShare"], 0.25)
        idle = simulate(memory_only, (task("big", cpu=1, queue="a"),), queue_weights={"a": 2})
        self.assertEqual(idle.makespan, 0)
        self.assertEqual(idle.metrics["queues"]["a"]["dominantShare"], 0.0)

    def test_baseline_metrics_have_no_queues_block(self) -> None:
        result = simulate((Node("n1", Resources(2, 2)),), (task("a"),))
        self.assertNotIn("queues", result.metrics)

    def test_replay_is_identical_in_fair_mode(self) -> None:
        cluster_nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("a1", cpu=2, memory=2, queue="a", duration=4),
            task("a2", cpu=2, memory=2, queue="a", duration=2),
            task("b1", cpu=2, memory=2, queue="b", duration=2),
        )
        report = replay(cluster_nodes, tasks, queue_weights=self.weights())
        self.assertTrue(report["identical"])

    def test_mapping_rejects_bad_weights(self) -> None:
        cluster_nodes = (Node("n1", Resources(1, 1)),)
        tasks = (task("a"),)
        for bad in ({}, {"": 1}, {"a": 0}, {"a": -1}, {"a": True}):
            with self.assertRaises(ValidationError):
                simulate(cluster_nodes, tasks, queue_weights=bad)


if __name__ == "__main__":
    unittest.main()
