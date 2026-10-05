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
    static_constraints,
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


class PreemptionConstraintTests(unittest.TestCase):
    """Preemption buys resources, never affinity, anti-affinity or taint compatibility."""

    def constrained_nodes(self) -> tuple[Node, ...]:
        return (
            Node("n1", Resources(4, 4), labels=(("zone", "a"),)),
            Node("n2", Resources(4, 4), labels=(("zone", "b"),), taints=("gpu",)),
            Node("n3", Resources(4, 4), labels=(("spot", "true"), ("zone", "c"))),
        )

    def state(self, cluster: Cluster, node_id: str):
        from schedsim import Placement

        low = task(f"low-{node_id}", cpu=4, memory=4, priority=1, duration=10)
        cluster.occupy(node_id, low.request)
        placements = [Placement(low.id, node_id, 0, 10)]
        return placements, {low.id: low}, low

    def test_preemption_requires_the_same_affinity_as_ordinary_placement(self) -> None:
        cluster = Cluster(self.constrained_nodes())
        placements, tasks, _ = self.state(cluster, "n1")
        pinned = task("hi", cpu=4, memory=4, priority=9, affinity=(("zone", "b"),))
        # n1 satisfies nothing: no victims may be named there, and the reason is the predicate's.
        chosen, reason = preemption_candidates(cluster, pinned, cluster.node("n1"), placements, tasks, 1)
        self.assertEqual(chosen, [])
        self.assertIn("affinity zone=b not satisfied", reason)

    def test_preemption_requires_the_same_anti_affinity_and_taints(self) -> None:
        cluster = Cluster(self.constrained_nodes())
        placements, tasks, _ = self.state(cluster, "n3")
        avoiding = task("hi", cpu=4, memory=4, priority=9, anti_affinity=("spot",))
        chosen, reason = preemption_candidates(cluster, avoiding, cluster.node("n3"), placements, tasks, 1)
        self.assertEqual(chosen, [])
        self.assertIn("anti-affinity", reason)

        placements, tasks, _ = self.state(cluster, "n2")
        intolerant = task("hi", cpu=4, memory=4, priority=9, affinity=(("zone", "b"),))
        chosen, reason = preemption_candidates(cluster, intolerant, cluster.node("n2"), placements, tasks, 1)
        self.assertEqual(chosen, [])
        self.assertIn("untolerated taint gpu", reason)
        # the matching toleration makes the very same node eligible again
        tolerant = task("hi", cpu=4, memory=4, priority=9, affinity=(("zone", "b"),), tolerations=("gpu",))
        chosen, _ = preemption_candidates(cluster, tolerant, cluster.node("n2"), placements, tasks, 1)
        self.assertEqual([item.task_id for item in chosen], ["low-n2"])

    def test_static_constraints_match_fits_for_every_task_node_pair(self) -> None:
        cluster = Cluster(self.constrained_nodes())
        cluster.occupy("n1", Resources(4, 4))
        probes = (
            task("a", affinity=(("zone", "a"),)),
            task("b", affinity=(("zone", "b"),), tolerations=("gpu",)),
            task("c", anti_affinity=("spot",)),
            task("mismatch", affinity=(("zone", "zzz"),)),
        )
        for probe in probes:
            for node in self.constrained_nodes():
                static_ok, _ = static_constraints(probe, node)
                # capacity is held out of the static check, so feed `fits` enough free room to
                # isolate exactly the same conclusion about labels and taints
                roomy = Cluster((Node(node.id, Resources(100, 100), labels=node.labels, taints=node.taints),))
                full_ok, _ = fits(roomy, probe, roomy.node(node.id))
                self.assertIs(static_ok, full_ok, (probe.id, node.id))

    def test_smaller_id_incompatible_node_is_not_evicted(self) -> None:
        nodes = self.constrained_nodes()
        tasks = (
            task("low1", cpu=4, memory=4, priority=1, duration=10),
            task("low2", cpu=4, memory=4, priority=1, duration=10, affinity=(("zone", "b"),),
                 tolerations=("gpu",)),
            task("low3", cpu=4, memory=4, priority=1, duration=10),
            task("hi", cpu=4, memory=4, priority=9, arrival=1, duration=2,
                 affinity=(("zone", "b"),), tolerations=("gpu",)),
        )
        result = simulate(nodes, tasks, allow_preemption=True)
        placed = {p.task_id: p for p in result.placements}
        # n1 is the smallest id and its low1 is evictable, but it cannot serve a zone=b task: it
        # must be left alone and the eviction happens on n2.
        self.assertEqual(placed["hi"].node_id, "n2")
        self.assertEqual(placed["hi"].preempted, ("low2",))
        self.assertEqual((placed["low1"].start, placed["low1"].end), (0, 10))
        self.assertEqual((placed["low2"].start, placed["low2"].end), (0, 1))
        self.assertEqual(result.metrics["preemptions"], 1)
        decision = next(d for d in result.decisions if d["task"] == "hi")
        self.assertEqual(decision["node"], "n2")
        self.assertIn("on n2", decision["reason"])

    def test_search_continues_to_the_compatible_node_under_each_policy(self) -> None:
        # The preemption walk is always id-ordered, never the policy's best-fit reordering: under
        # either policy n1 (smallest id, full of an evictable victim) must be skipped on affinity
        # and the eviction must happen on n2.
        nodes = (
            Node("n1", Resources(4, 4), labels=(("zone", "a"),)),
            Node("n2", Resources(4, 4), labels=(("zone", "b"),)),
        )
        tasks = (
            task("low1", cpu=4, memory=4, priority=1, duration=10),
            task("low2", cpu=4, memory=4, priority=1, duration=10, affinity=(("zone", "b"),)),
            task("hi", cpu=4, memory=4, priority=9, arrival=1, duration=2, affinity=(("zone", "b"),)),
        )
        for policy in ("first-fit", "best-fit"):
            with self.subTest(policy=policy):
                result = simulate(nodes, tasks, policy=policy, allow_preemption=True)
                placed = {p.task_id: p for p in result.placements}
                self.assertEqual(placed["hi"].node_id, "n2")
                self.assertEqual(placed["hi"].preempted, ("low2",))
                self.assertEqual((placed["low1"].start, placed["low1"].end), (0, 10))
                self.assertEqual(result.metrics["preemptions"], 1)

    def test_task_waits_for_a_release_when_evictable_tasks_cannot_fit_it(self) -> None:
        # n1 (zone=a) is the only compatible node, but it is held by a higher-priority anchor that
        # is not an eviction victim, so no set of strictly-lower-priority tasks can free room yet.
        # The task waits on the ordinary clock and is served the tick the anchor finishes; n2 is
        # incompatible and is merely skipped, never looted.
        nodes = (
            Node("n1", Resources(4, 4), labels=(("zone", "a"),)),
            Node("n2", Resources(4, 4), labels=(("zone", "b"),)),
        )
        tasks = (
            task("anchor", cpu=4, memory=4, priority=9, duration=3, affinity=(("zone", "a"),)),
            task("hi", cpu=4, memory=4, priority=5, arrival=1, duration=2, affinity=(("zone", "a"),)),
        )
        result = simulate(nodes, tasks, allow_preemption=True)
        placed = {p.task_id: p for p in result.placements}
        self.assertEqual(result.metrics["preemptions"], 0)
        self.assertEqual(placed["hi"].node_id, "n1")
        self.assertEqual((placed["anchor"].start, placed["anchor"].end), (0, 3))
        self.assertEqual((placed["hi"].start, placed["hi"].end), (3, 5))
        # the blocked wait is recorded with the stable "no node fits" refusal and retried silently
        refusal = next(d for d in result.decisions if d["task"] == "hi" and "node" not in d)
        self.assertIn("no node fits", refusal["reason"])
        self.assertEqual(refusal["at"], 1)

    def test_no_compatible_node_even_with_full_eviction_leaves_unplaced_without_fake_records(self) -> None:
        # Every node is forbidden to the high-priority task (its affinity key exists nowhere), so
        # even with preemption enabled and evictable victims on every node, no eviction occurs and
        # the task stays waiting until the run ends.
        nodes = self.constrained_nodes()
        tasks = (
            task("low1", cpu=4, memory=4, priority=1, duration=10),
            task("low2", cpu=4, memory=4, priority=1, duration=10, affinity=(("zone", "b"),),
                 tolerations=("gpu",)),
            task("low3", cpu=4, memory=4, priority=1, duration=10),
            task("stranded", cpu=4, memory=4, priority=9, arrival=1, duration=2,
                 affinity=(("zone", "never"),)),
        )
        result = simulate(nodes, tasks, allow_preemption=True)
        self.assertEqual(result.unplaced, ["stranded"])
        self.assertEqual(result.metrics["preemptions"], 0)
        self.assertTrue(all(not p.preempted for p in result.placements))
        for victim in ("low1", "low2", "low3"):
            placement = next(p for p in result.placements if p.task_id == victim)
            self.assertEqual((placement.start, placement.end), (0, 10))
        # the trace keeps the ordinary "no node fits" refusal (capacity is the first predicate on
        # the still-full nodes, exactly the existing format) and shows no phantom preemption
        refusal = next(d for d in result.decisions if d["task"] == "stranded" and "node" not in d)
        self.assertTrue(refusal["reason"].startswith("no node fits"))
        self.assertFalse(
            any(str(d.get("reason", "")).startswith("preempting") for d in result.decisions)
        )

    def test_waiting_task_is_served_at_a_compatible_node_after_a_release(self) -> None:
        # n3 (zone=c) is the ONLY node compatible with the incoming task; it is held by a
        # higher-priority anchor that is not an eligible victim. The smaller-id n1/n2 are full of
        # evictable low-priority tasks but fail the affinity, so the scheduler may not loot them:
        # the task waits on the ordinary clock and starts exactly when the anchor finishes.
        nodes = self.constrained_nodes()
        tasks = (
            task("low1", cpu=4, memory=4, priority=1, duration=10),
            task("low2", cpu=4, memory=4, priority=1, duration=10, affinity=(("zone", "b"),),
                 tolerations=("gpu",)),
            task("anchor", cpu=4, memory=4, priority=9, duration=3, affinity=(("zone", "c"),)),
            task("hi", cpu=4, memory=4, priority=5, arrival=1, duration=2, affinity=(("zone", "c"),)),
        )
        result = simulate(nodes, tasks, allow_preemption=True)
        placed = {p.task_id: p for p in result.placements}
        self.assertEqual(result.metrics["preemptions"], 0)
        self.assertEqual(placed["hi"].node_id, "n3")
        self.assertEqual((placed["anchor"].start, placed["anchor"].end), (0, 3))
        self.assertEqual((placed["hi"].start, placed["hi"].end), (3, 5))
        self.assertEqual((placed["low1"].start, placed["low1"].end), (0, 10))


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


class ExactShareTests(unittest.TestCase):
    """Weighted shares must follow the exact rational definition, never a binary float rounding.

    The scenarios use ~1e20 integer capacities: at that magnitude two fractions differing by a
    single unit of numerator map to the SAME IEEE double, so float ordering silently falls through
    to the queue/id tie-break. Every scenario below was observed to produce the wrong order on the
    float implementation and the right order on exact arithmetic. The displayed
    ``weightedDominantShare`` / ``dominantShare`` stay rounded to six decimals -- both exact values
    can print as one number without that changing who is picked first.
    """

    CAPACITY = 10**20
    HALF = 10**20 // 2

    def node(self, cpu: int | None = None, memory: int | None = None) -> tuple[Node, ...]:
        return (Node("n1", Resources(self.CAPACITY if cpu is None else cpu, self.CAPACITY if memory is None else memory)),)

    def test_sub_ulp_difference_orders_consecutive_placements_at_one_tick(self) -> None:
        # Anchors leave queue Z one unit below queue A; each unit placed flips which queue is ahead,
        # so four same-tick placements must alternate z, a, z, a. Float shares are both 0.5 and the
        # old code served a1, a2, z1, z2 by queue name instead.
        tasks = (
            task("aRun", cpu=self.HALF - 2, memory=1, queue="A", duration=10),
            task("zRun", cpu=self.HALF - 3, memory=1, queue="Z", duration=10),
            task("z1", cpu=1, queue="Z", arrival=1),
            task("a1", cpu=1, queue="A", arrival=1),
            task("z2", cpu=1, queue="Z", arrival=1),
            task("a2", cpu=1, queue="A", arrival=1),
        )
        result = simulate(self.node(), tasks, queue_weights={"A": 1, "Z": 1})
        at_one = [entry["task"] for entry in result.decisions if "node" in entry and entry["at"] == 1]
        self.assertEqual(at_one, ["z1", "a1", "z2", "a2"])
        # All four exact shares round to the same six-decimal number; the display never drove order.
        self.assertEqual({entry["weightedDominantShare"] for entry in result.decisions if entry.get("at") == 1}, {0.5})

    def test_release_recomputes_exact_shares_before_same_tick_decisions(self) -> None:
        # At tick 2 zBlink's single unit is released: Z then holds HALF-1 versus A holding HALF, a
        # sub-ulp gap. Exact ordering serves zNext first (which refills the cluster); aNext waits one
        # more tick. Float ordering saw both queues at 0.5 and picked aNext by queue name.
        tasks = (
            task("aRun", cpu=self.HALF, memory=1, queue="A", duration=10),
            task("zRun", cpu=self.HALF - 1, memory=1, queue="Z", duration=10),
            task("zBlink", cpu=1, queue="Z", duration=2),
            task("aNext", cpu=1, queue="A", arrival=2),
            task("zNext", cpu=1, queue="Z", arrival=2),
        )
        result = simulate(self.node(), tasks, queue_weights={"A": 1, "Z": 1})
        placed = {entry["task"]: entry["at"] for entry in result.decisions if "node" in entry}
        self.assertEqual((placed["zNext"], placed["aNext"]), (2, 3))
        starts = {item.task_id: item.start for item in result.placements}
        self.assertEqual((starts["zNext"], starts["aNext"]), (2, 3))
        self.assertEqual(result.unplaced, [])

    def test_successful_preemption_recomputes_exact_shares(self) -> None:
        # hi evicts zTiny at tick 1, taking its two units from queue Z. Z then holds HALF-1 and A
        # HALF, so the equal-priority survivors pick zNext before aNext; float ties served aNext.
        tasks = (
            task("aAnchor", cpu=self.HALF, memory=1, queue="A", priority=5, duration=10),
            task("zAnchor", cpu=self.HALF - 2, memory=1, queue="Z", priority=5, duration=10),
            task("zTiny", cpu=2, queue="Z", priority=1, duration=10),
            task("hi", cpu=1, queue="H", priority=9, arrival=1),
            task("aNext", cpu=1, queue="A", arrival=1),
            task("zNext", cpu=1, queue="Z", arrival=1),
        )
        result = simulate(self.node(), tasks, queue_weights={"A": 1, "Z": 1, "H": 1}, allow_preemption=True)
        self.assertEqual(result.metrics["preemptions"], 1)
        hi = next(item for item in result.placements if item.task_id == "hi")
        self.assertEqual(hi.preempted, ("zTiny",))
        at_one = [entry["task"] for entry in result.decisions if "node" in entry and entry["at"] == 1]
        self.assertEqual(at_one, ["hi", "zNext"])
        starts = {item.task_id: item.start for item in result.placements}
        self.assertEqual(starts["aNext"], 2)

    def test_successful_backfill_recomputes_exact_shares(self) -> None:
        # Big-integer twin of the existing re-rank test: after b1 is backfilled queue B jumps from
        # zero to ~0.5; C sat at 0.5 minus a sub-ulp gap, so the second backfill is c, not b2.
        # Float shares made B and C equal and the old code took b2.
        tasks = (
            task("cRun", cpu=self.HALF - 2, memory=1, queue="C", duration=10),
            task("head", cpu=self.CAPACITY, memory=self.CAPACITY, priority=5, arrival=1, duration=10),
            task("b1", cpu=self.HALF - 1, queue="B", arrival=1, duration=1),
            task("b2", cpu=1, queue="B", arrival=1, duration=1),
            task("c", cpu=1, queue="C", arrival=1, duration=1),
        )
        result = simulate(self.node(), tasks, queue_weights={"B": 1, "C": 1})
        self.assertEqual(
            [entry["task"] for entry in result.decisions if entry.get("reason") == "backfill"],
            ["b1", "c", "b2"],
        )
        self.assertTrue(all(item.task_id != "head" or item.start >= 10 for item in result.placements))

    def test_mathematically_equal_shares_fall_back_to_queue_and_id_ties(self) -> None:
        # A holds 3 units at weight 3, Z holds 1 unit at weight 1: the weighted shares are exactly
        # 1/C both. Binary doubles disagree (3/C/3 rounds to 1.0000...01e-20), which used to promote
        # Z wrongly; exact equality must defer to the queue-name tie-break, serving A first.
        tasks = (
            task("aRun", cpu=3, memory=1, queue="A", duration=10),
            task("zRun", cpu=1, memory=1, queue="Z", duration=10),
            task("a1", cpu=1, queue="A", arrival=1),
            task("z1", cpu=1, queue="Z", arrival=1),
        )
        result = simulate(self.node(), tasks, queue_weights={"A": 3, "Z": 1})
        at_one = [entry["task"] for entry in result.decisions if "node" in entry and entry["at"] == 1]
        self.assertEqual(at_one, ["a1", "z1"])
        self.assertEqual({entry["weightedDominantShare"] for entry in result.decisions if entry.get("at") == 1}, {0.0})

    def test_large_integer_weights_are_compared_exactly(self) -> None:
        # Equal usage, weights 10**18 versus 10**18 + 1: weighted shares coincide as doubles but not
        # as rationals, so the slightly heavier queue Z must go first.
        tasks = (
            task("aRun", cpu=self.HALF - 1, memory=1, queue="A", duration=10),
            task("zRun", cpu=self.HALF - 1, memory=1, queue="Z", duration=10),
            task("a1", cpu=1, queue="A", arrival=1),
            task("z1", cpu=1, queue="Z", arrival=1),
        )
        weights = {"A": 10**18, "Z": 10**18 + 1}
        result = simulate(self.node(), tasks, queue_weights=weights)
        at_one = [entry["task"] for entry in result.decisions if "node" in entry and entry["at"] == 1]
        self.assertEqual(at_one, ["z1", "a1"])
        z_decision = next(entry for entry in result.decisions if entry["task"] == "z1")
        self.assertEqual(z_decision["weight"], 10**18 + 1)
        self.assertTrue(replay(self.node(), tasks, queue_weights=weights)["identical"])

    def test_zero_capacity_resource_contributes_zero_without_dividing_by_it(self) -> None:
        # Pure-memory cluster mirrors the CPU scenario through the memory fraction.
        memory_node = (Node("n1", Resources(0, self.CAPACITY)),)
        tasks = (
            task("qRun", cpu=0, memory=self.HALF - 2, queue="Q", duration=10),
            task("rRun", cpu=0, memory=self.HALF - 3, queue="R", duration=10),
            task("r1", cpu=0, memory=1, queue="R", arrival=1),
            task("q1", cpu=0, memory=1, queue="Q", arrival=1),
            task("r2", cpu=0, memory=1, queue="R", arrival=1),
            task("q2", cpu=0, memory=1, queue="Q", arrival=1),
        )
        result = simulate(memory_node, tasks, queue_weights={"Q": 1, "R": 1})
        at_one = [entry["task"] for entry in result.decisions if "node" in entry and entry["at"] == 1]
        self.assertEqual(at_one, ["r1", "q1", "r2", "q2"])
        self.assertTrue(replay(memory_node, tasks, queue_weights={"Q": 1, "R": 1})["identical"])
        # A cluster with zero capacity of everything raises nothing: shares stay zero and the task
        # is simply left unplaced.
        dead = (Node("n0", Resources(0, 0)),)
        stuck = simulate(dead, (task("s", cpu=1, queue="S"),), queue_weights={"S": 1})
        self.assertEqual(stuck.unplaced, ["s"])
        self.assertEqual(stuck.metrics["queues"]["S"]["dominantShare"], 0.0)

    def test_dominant_share_metric_is_a_six_decimal_float(self) -> None:
        tasks = (
            task("aRun", cpu=self.HALF - 2, memory=1, queue="A", duration=10),
            task("zRun", cpu=self.HALF - 3, memory=1, queue="Z", duration=10),
            task("z1", cpu=1, queue="Z", arrival=1),
            task("a1", cpu=1, queue="A", arrival=1),
        )
        result = simulate(self.node(), tasks, queue_weights={"A": 1, "Z": 1})
        share = result.metrics["queues"]["A"]["dominantShare"]
        self.assertIsInstance(share, float)
        self.assertEqual(share, 0.5)
        self.assertEqual(result.metrics["queues"]["A"]["placed"], 2)
        self.assertEqual(result.metrics["queues"]["A"]["unplaced"], 0)


if __name__ == "__main__":
    unittest.main()
