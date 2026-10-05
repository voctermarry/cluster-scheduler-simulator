"""Hard per-queue concurrency quotas.

The quota is checked before every placement attempt and before node selection: a task whose
running queue would exceed either its CPU or memory ceiling is refused, never probes a node, never
triggers (or bypasses) preemption, and keeps its place in the existing waiting order. Conservative
backfill may still skip it and run a later candidate whose own quota and the node/time constraints
allow. Finished and preempted tasks release the ceiling immediately. Metrics gain a sorted
``quotas`` object; the CLI accepts ``--queue-quotas`` with the same parse/validation split as queue
weights. Without the mapping every ordering, output and exit code is the baseline.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from schedsim import Node, Resources, Task, ValidationError, compare_policies, replay, simulate
from schedsim.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, load_queue_quotas, main


def node(node_id: str = "n1", cpu: int = 8, memory: int = 8) -> Node:
    return Node(node_id, Resources(cpu, memory))


def task(task_id: str, cpu: int = 1, memory: int = 1, **kwargs) -> Task:
    return Task(id=task_id, request=Resources(cpu, memory), **kwargs)


QUOTA_REASON = "queue quota exceeded"


class QuotaGatingTests(unittest.TestCase):
    def test_cpu_ceiling_blocks_the_second_task_until_the_first_finishes(self) -> None:
        tasks = (
            task("a", cpu=2, queue="q", duration=3),
            task("b", cpu=2, queue="q", duration=1),
            task("c", cpu=2, queue="q", duration=1),
        )
        result = simulate((node(),), tasks, queue_quotas={"q": Resources(2, 8)})
        starts = {p.task_id: p.start for p in result.placements}
        self.assertEqual(starts, {"a": 0, "b": 3, "c": 4})

    def test_memory_ceiling_blocks_independently_of_cpu(self) -> None:
        tasks = (
            task("a", cpu=1, memory=4, queue="q", duration=2),
            task("b", cpu=4, memory=4, queue="q", duration=1),
        )
        result = simulate((node(),), tasks, queue_quotas={"q": Resources(8, 4)})
        starts = {p.task_id: p.start for p in result.placements}
        # a holds 4 of the 4 memory; b fits on cpu but its memory would reach 8 > 4, so it waits.
        self.assertEqual(starts, {"a": 0, "b": 2})

    def test_a_request_alone_over_the_quota_stays_unplaced(self) -> None:
        tasks = (task("big", cpu=9, queue="q", duration=1),)
        result = simulate((node(),), tasks, queue_quotas={"q": Resources(2, 2)})
        self.assertEqual(result.placements, [])
        self.assertEqual(result.unplaced, ["big"])
        self.assertEqual(result.metrics["quotas"]["q"]["blocked"], 1)

    def test_unconfigured_queues_are_unlimited(self) -> None:
        tasks = (
            task("a", cpu=4, queue="limited", duration=1),
            task("b", cpu=4, queue="limited", duration=1),
            task("x", cpu=4, queue="free", duration=1),
            task("y", cpu=4, queue="free", duration=1),
        )
        result = simulate((node(cpu=16),), tasks, queue_quotas={"limited": Resources(4, 8)})
        starts = {p.task_id: p.start for p in result.placements}
        # "free" has no quota: x and y both start at 0; the second "limited" task waits a tick.
        self.assertEqual((starts["x"], starts["y"]), (0, 0))
        self.assertEqual(starts["b"], 1)

    def test_quota_runs_before_node_selection_and_never_explores_nodes(self) -> None:
        # The task fits no node anyway (cpu 9), but its queue is also over quota: the refusal must
        # be the quota sentence, not a node-probe reason, proving the gate runs first.
        tasks = (
            task("a", cpu=1, queue="q", duration=5),
            task("huge", cpu=9, queue="q", duration=1),
        )
        result = simulate((node(),), tasks, queue_quotas={"q": Resources(1, 8)})
        refusal = next(d for d in result.decisions if d["task"] == "huge" and "node" not in d)
        self.assertTrue(refusal["reason"].startswith(QUOTA_REASON))
        self.assertNotIn("no node fits", refusal["reason"])

    def test_refusal_names_queue_request_running_and_limit(self) -> None:
        tasks = (
            task("a", cpu=2, memory=3, queue="team", duration=4),
            task("b", cpu=1, memory=2, queue="team", duration=1),
        )
        result = simulate((node(),), tasks, queue_quotas={"team": Resources(2, 4)})
        refusal = next(d for d in result.decisions if d["task"] == "b" and "node" not in d)
        reason = refusal["reason"]
        self.assertTrue(reason.startswith(QUOTA_REASON))
        self.assertIn("team", reason)
        self.assertIn("request cpu=1,memory=2", reason)
        self.assertIn("running cpu=2,memory=3", reason)
        self.assertIn("limit cpu=2,memory=4", reason)

    def test_repeated_quota_refusal_is_recorded_once_per_contiguous_spell(self) -> None:
        tasks = (
            task("a", cpu=2, queue="q", duration=3),
            task("b", cpu=2, queue="q", duration=1),
        )
        result = simulate((node(),), tasks, queue_quotas={"q": Resources(2, 8)})
        refusals = [d for d in result.decisions if d["task"] == "b" and "node" not in d]
        self.assertEqual(len(refusals), 1)
        # The record is retained even after b is eventually placed.
        self.assertIn("b", {p.task_id for p in result.placements})

    def test_completion_releases_quota_for_a_decision_at_the_same_tick(self) -> None:
        tasks = (
            task("a", cpu=2, queue="q", duration=2),
            task("b", cpu=2, queue="q", arrival=1, duration=1),
        )
        result = simulate((node(),), tasks, queue_quotas={"q": Resources(2, 8)})
        # a ends at 2 and releases at 2; b starts exactly at 2, not 3.
        self.assertEqual({p.task_id: p.start for p in result.placements}, {"a": 0, "b": 2})

    def test_preemption_cannot_bypass_the_quota(self) -> None:
        tasks = (
            task("low", cpu=2, queue="q", priority=1, duration=10),
            task("high", cpu=2, queue="q", priority=9, arrival=1, duration=1),
        )
        result = simulate((node(),), tasks, allow_preemption=True, queue_quotas={"q": Resources(2, 8)})
        self.assertEqual(result.metrics["preemptions"], 0)
        self.assertEqual({p.task_id: p.start for p in result.placements}, {"low": 0, "high": 10})

    def test_preemption_of_another_queue_does_not_free_a_quota_bound_queue(self) -> None:
        # high belongs to the quota-bound queue; the only running lower-priority task is in a
        # different queue, so even evicting it cannot raise q's ceiling.
        tasks = (
            task("other", cpu=2, queue="other", priority=1, duration=10),
            task("qhold", cpu=1, queue="q", priority=1, duration=10),
            task("high", cpu=2, queue="q", priority=9, arrival=1, duration=1),
        )
        result = simulate((node(),), tasks, allow_preemption=True, queue_quotas={"q": Resources(1, 8)})
        self.assertEqual(result.metrics["preemptions"], 0)
        self.assertTrue(any(
            d["task"] == "high" and d["reason"].startswith(QUOTA_REASON) for d in result.decisions
        ))

    def test_preemption_under_the_quota_still_releases_the_victim_immediately(self) -> None:
        # The quota (4 cpu) admits the newcomer alongside the running task, but the 2-cpu node is
        # full, so the high-priority q task legitimately preempts the lower-priority q task. The
        # eviction releases quota occupancy at the eviction tick: the peak must never count both.
        tasks = (
            task("low", cpu=2, queue="q", priority=1, duration=10),
            task("high", cpu=2, queue="q", priority=9, arrival=1, duration=2),
        )
        result = simulate((node(cpu=2),), tasks, allow_preemption=True, queue_quotas={"q": Resources(4, 8)})
        self.assertEqual(result.metrics["preemptions"], 1)
        high = next(p for p in result.placements if p.task_id == "high")
        self.assertEqual((high.start, high.end), (1, 3))
        self.assertEqual(high.preempted, ("low",))
        summary = result.metrics["quotas"]["q"]
        self.assertEqual(summary["peakCpu"], 2)
        self.assertEqual(summary["blocked"], 0)

    def test_backfill_runs_a_later_candidate_whose_own_quota_allows_it(self) -> None:
        # The head (b, queue q) is quota-blocked by a long-running q task; behind it sits a short
        # task in an unlimited queue that fits and finishes before the running tasks end, so it
        # must backfill past the blocked head.
        tasks = (
            task("qhold", cpu=2, queue="q", duration=10),
            task("long", cpu=1, queue="runner", duration=10),
            task("b", cpu=2, queue="q", arrival=1, duration=4),
            task("jump", cpu=1, queue="free", arrival=1, duration=2),
        )
        result = simulate((node(cpu=4),), tasks, queue_quotas={"q": Resources(2, 8)})
        jump = next(p for p in result.placements if p.task_id == "jump")
        self.assertEqual((jump.start, jump.end), (1, 3))
        jump_decision = next(d for d in result.decisions if d["task"] == "jump")
        self.assertEqual(jump_decision["reason"], "backfill")

    def test_backfill_skips_a_candidate_blocked_by_its_own_quota_without_a_trace_entry(self) -> None:
        # Head b is cluster-blocked; the first candidate behind it (bq) is over its own quota, so it
        # must be silently skipped (no refusal) and the next candidate (ok) backfills instead.
        tasks = (
            task("qhold", cpu=2, queue="q", duration=10),
            task("long", cpu=1, queue="runner", duration=10),
            task("b", cpu=9, queue="free", arrival=1, duration=4),
            task("bq", cpu=1, queue="q", arrival=1, duration=2),
            task("ok", cpu=1, queue="free", arrival=1, duration=2),
        )
        result = simulate((node(cpu=4),), tasks, queue_quotas={"q": Resources(2, 8)})
        backfills = [d["task"] for d in result.decisions if d.get("reason") == "backfill"]
        self.assertEqual(backfills, ["ok"])
        # The quota probe of bq is invisible: no refusal for it at the probing tick.
        self.assertFalse(
            any(d["task"] == "bq" and d.get("at") == 1 and "node" not in d for d in result.decisions)
        )
        # But bq was still blocked by quota, so it is counted once.
        self.assertEqual(result.metrics["quotas"]["q"]["blocked"], 1)

    def test_no_backfill_forbids_jumping_the_quota_blocked_head(self) -> None:
        tasks = (
            task("qhold", cpu=2, queue="q", duration=10),
            task("b", cpu=2, queue="q", arrival=1, duration=4),
            task("jump", cpu=1, queue="free", arrival=1, duration=2),
        )
        result = simulate((node(cpu=4),), tasks, backfill=False, queue_quotas={"q": Resources(2, 8)})
        self.assertFalse(any(d.get("reason") == "backfill" for d in result.decisions))
        self.assertGreaterEqual(next(p for p in result.placements if p.task_id == "jump").start, 10)

    def test_peak_concurrency_never_exceeds_the_limit(self) -> None:
        tasks = (
            task("a", cpu=2, memory=2, queue="q", duration=2),
            task("b", cpu=1, memory=3, queue="q", duration=1),
            task("c", cpu=1, memory=1, queue="q", duration=1),
        )
        result = simulate((node(),), tasks, queue_quotas={"q": Resources(3, 5)})
        summary = result.metrics["quotas"]["q"]
        self.assertEqual(summary["cpu"], 3)
        self.assertEqual(summary["memory"], 5)
        self.assertLessEqual(summary["peakCpu"], 3)
        self.assertLessEqual(summary["peakMemory"], 5)
        # a runs at ticks 0,1 (cpu 2, mem 2); b joins at tick 2 (mem +3 => 5); c joins at tick 3.
        self.assertEqual(summary["peakCpu"], 3)
        self.assertEqual(summary["peakMemory"], 5)

    def test_blocked_counts_distinct_tasks_not_ticks(self) -> None:
        tasks = (
            task("a", cpu=2, queue="q", duration=5),
            task("b", cpu=2, queue="q", duration=1),
            task("c", cpu=2, queue="q", duration=1),
        )
        result = simulate((node(),), tasks, queue_quotas={"q": Resources(2, 8)})
        self.assertEqual(result.metrics["quotas"]["q"]["blocked"], 2)


class QuotaMetricsTests(unittest.TestCase):
    def test_quotas_object_is_sorted_and_complete(self) -> None:
        tasks = (
            task("a", cpu=1, queue="z"),
            task("b", cpu=1, queue="a"),
            task("ghost", cpu=1, queue="unmentioned"),
        )
        result = simulate((node(),), tasks, queue_quotas={"z": Resources(2, 2), "a": Resources(2, 2)})
        quotas = result.metrics["quotas"]
        # Only configured queues appear, sorted by name.
        self.assertEqual(list(quotas), ["a", "z"])
        for summary in quotas.values():
            self.assertEqual(set(summary), {"cpu", "memory", "peakCpu", "peakMemory", "blocked"})
        self.assertEqual(quotas["a"], {"cpu": 2, "memory": 2, "peakCpu": 1, "peakMemory": 1, "blocked": 0})

    def test_without_quotas_metrics_have_no_quotas_block(self) -> None:
        result = simulate((node(),), (task("a"),))
        self.assertNotIn("quotas", result.metrics)

    def test_fair_share_fields_coexist_with_quotas(self) -> None:
        tasks = (
            task("a", cpu=2, queue="q", duration=2),
            task("b", cpu=2, queue="q", duration=1),
        )
        result = simulate(
            (node(),), tasks, queue_weights={"q": 1}, queue_quotas={"q": Resources(2, 8)}
        )
        self.assertIn("queues", result.metrics)
        self.assertIn("quotas", result.metrics)
        self.assertTrue(all("weightedDominantShare" in d for d in result.decisions if "at" in d))

    def test_policies_carry_the_quota_summary(self) -> None:
        tasks = (task("a", cpu=2, queue="q", duration=1), task("b", cpu=2, queue="q", duration=1))
        report = compare_policies((node(),), tasks, queue_quotas={"q": Resources(2, 8)})
        for entry in report["policies"]:
            self.assertEqual(entry["quotas"]["q"]["cpu"], 2)
            self.assertEqual(entry["quotas"]["q"]["peakCpu"], 2)

    def test_replay_compares_the_full_trajectory_with_quotas(self) -> None:
        tasks = (
            task("a", cpu=2, queue="q", duration=2),
            task("b", cpu=2, queue="q", duration=1),
            task("c", cpu=1, queue="free", duration=1),
        )
        report = replay((node(cpu=4),), tasks, queue_quotas={"q": Resources(2, 8)})
        self.assertTrue(report["identical"])
        self.assertEqual(report["differences"], [])


class QuotaValidationTests(unittest.TestCase):
    def test_mapping_rejects_bad_quotas(self) -> None:
        good_nodes = (node(),)
        good_tasks = (task("a"),)
        for bad in ({}, {"": Resources(1, 1)}, {"q": Resources(0, 1)}, {"q": Resources(1, 0)}):
            with self.assertRaises(ValidationError):
                simulate(good_nodes, good_tasks, queue_quotas=bad)

    def test_non_mapping_values_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            simulate((node(),), (task("a"),), queue_quotas={"q": (1, 1)})  # type: ignore[dict-item]


class QuotaCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.cluster = self.write("cluster.jsonl", [{"id": "n1", "cpu": 8, "memory": 8}])
        self.tasks = self.write(
            "tasks.jsonl",
            [
                {"id": "a", "cpu": 2, "queue": "q", "duration": 3},
                {"id": "b", "cpu": 2, "queue": "q", "duration": 1},
                {"id": "c", "cpu": 2, "queue": "free", "duration": 1},
            ],
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write(self, name: str, rows: list[dict]) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_loader_returns_resources_keyed_by_queue(self) -> None:
        path = self.write("quotas.jsonl", [{"queue": "q", "cpu": 2, "memory": 4}])
        self.assertEqual(load_queue_quotas(path), {"q": Resources(2, 4)})

    def test_simulate_metrics_and_trace_carry_quota_outputs(self) -> None:
        quotas = self.write("quotas.jsonl", [{"queue": "q", "cpu": 2, "memory": 8}])
        code, out, err = self.run_cli(
            ["metrics", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", quotas]
        )
        self.assertEqual((code, err), (EXIT_OK, ""))
        self.assertEqual(json.loads(out)["metrics"]["quotas"]["q"]["cpu"], 2)

        code, out, _ = self.run_cli(
            ["trace", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", quotas]
        )
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(
            any(d["reason"].startswith(QUOTA_REASON) for d in json.loads(out)["decisions"])
        )

    def test_quota_leaving_tasks_unplaced_is_exit_3_with_unplaced_and_refusal(self) -> None:
        # A single task whose request alone exceeds the quota is never placed.
        stuck = self.write("stuck.jsonl", [{"id": "big", "cpu": 9, "queue": "q"}])
        quotas = self.write("q2.jsonl", [{"queue": "q", "cpu": 2, "memory": 2}])
        code, out, _ = self.run_cli(
            ["simulate", "--cluster", self.cluster, "--tasks", stuck, "--queue-quotas", quotas]
        )
        self.assertEqual(code, EXIT_NEGATIVE)
        document = json.loads(out)
        self.assertEqual(document["unplaced"], ["big"])
        code, out, _ = self.run_cli(
            ["trace", "--cluster", self.cluster, "--tasks", stuck, "--queue-quotas", quotas]
        )
        self.assertEqual(code, EXIT_NEGATIVE)
        self.assertTrue(
            any(d["reason"].startswith(QUOTA_REASON) for d in json.loads(out)["decisions"])
        )

    def test_policies_and_replay_accept_the_flag(self) -> None:
        quotas = self.write("quotas.jsonl", [{"queue": "q", "cpu": 2, "memory": 8}])
        code, out, _ = self.run_cli(
            ["policies", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", quotas]
        )
        self.assertEqual(code, EXIT_OK)
        for entry in json.loads(out)["policies"]:
            self.assertIn("quotas", entry)
        code, out, _ = self.run_cli(
            ["replay", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", quotas]
        )
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(json.loads(out)["identical"])

    def test_without_the_flag_outputs_no_quota_fields(self) -> None:
        _, out, _ = self.run_cli(["metrics", "--cluster", self.cluster, "--tasks", self.tasks])
        self.assertNotIn("quotas", json.loads(out)["metrics"])

    def test_unreadable_file_is_a_validation_error_with_empty_stdout(self) -> None:
        code, out, err = self.run_cli(
            ["metrics", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", "/no/such/q.jsonl"]
        )
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_empty_file_is_a_validation_error(self) -> None:
        quotas = self.write("empty.jsonl", [])
        code, out, err = self.run_cli(
            ["metrics", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", quotas]
        )
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_duplicate_queue_is_a_validation_error_with_line(self) -> None:
        path = os.path.join(self.directory.name, "dup.jsonl")
        with open(path, "w") as handle:
            handle.write('{"queue": "q", "cpu": 1, "memory": 1}\n{"queue": "q", "cpu": 2, "memory": 2}\n')
        code, out, err = self.run_cli(
            ["metrics", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", path]
        )
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")
        self.assertEqual(document["line"], 2)

    def test_non_positive_quota_is_a_validation_error(self) -> None:
        quotas = self.write("zero.jsonl", [{"queue": "q", "cpu": 0, "memory": 2}])
        code, out, err = self.run_cli(
            ["metrics", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", quotas]
        )
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")
        self.assertEqual(document["line"], 1)

    def test_parse_errors_name_the_raw_line(self) -> None:
        cases = {
            "badjson": ['{"queue": "q", "cpu": 1, "memory": 1}', "{broken"],
            "nonobject": ['{"queue": "q", "cpu": 1, "memory": 1}', "[1]"],
            "missingcpu": ['{"queue": "q", "memory": 1}'],
            "missingmemory": ['{"queue": "q", "cpu": 1}'],
            "missingqueue": ['{"cpu": 1, "memory": 1}'],
            "strtype": ['{"queue": "q", "cpu": "1", "memory": 1}'],
            "booltype": ['{"queue": "q", "cpu": true, "memory": 1}'],
            "unknown": ['{"queue": "q", "cpu": 1, "memory": 1, "extra": 2}'],
            "emptystring": ['{"queue": "", "cpu": 1, "memory": 1}'],
        }
        for name, lines in cases.items():
            path = os.path.join(self.directory.name, f"q-{name}.jsonl")
            with open(path, "w") as handle:
                handle.write("\n".join(lines) + "\n")
            code, out, err = self.run_cli(
                ["metrics", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", path]
            )
            self.assertEqual((code, out), (EXIT_ERROR, ""), name)
            document = json.loads(err)
            self.assertEqual(document["error"], "parse_error", name)
            self.assertEqual(document["line"], 2 if name in ("badjson", "nonobject") else 1, name)


if __name__ == "__main__":
    unittest.main()
