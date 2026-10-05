"""Hard per-queue concurrency quotas.

A quota caps the CPU and memory a queue's *running* tasks may hold concurrently. The gate runs
before node selection and preemption, so a blocked task probes no node, triggers no eviction, and
cannot bypass the ceiling by emptying another queue. These tests cover the gate, the immediate
release on completion and preemption, the (silent) interaction with conservative backfill, the
trace de-duplication, the sorted quota metrics, the CLI error taxonomy and the unchanged baseline
when no file is supplied.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from schedsim import Node, Resources, Task, compare_policies, replay, simulate
from schedsim.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, main
from schedsim.errors import ValidationError


def task(task_id: str, cpu: int = 1, memory: int = 1, **kwargs) -> Task:
    return Task(id=task_id, request=Resources(cpu, memory), **kwargs)


class QuotaGateTests(unittest.TestCase):
    def test_quota_blocks_before_node_selection_and_reports_a_summary(self) -> None:
        # Two idle 4x4 nodes exist, so node capacity never binds: a2 is refused purely by queue a's
        # 4-cpu ceiling. It runs after a1 completes, never alongside it.
        nodes = (Node("n1", Resources(4, 4)), Node("n2", Resources(4, 4)),)
        tasks = (
            task("a1", cpu=3, memory=1, queue="a", duration=2),
            task("a2", cpu=3, memory=1, queue="a", duration=2),
        )
        result = simulate(nodes, tasks, queue_quotas={"a": Resources(4, 4)})
        starts = {p.task_id: p.start for p in result.placements}
        self.assertEqual(starts, {"a1": 0, "a2": 2})
        quotas = result.metrics["quotas"]
        self.assertEqual(list(quotas), ["a"])
        self.assertEqual(
            quotas["a"],
            {"cpu": 4, "memory": 4, "peakCpu": 3, "peakMemory": 1, "blocked": 1},
        )
        refusal = next(d for d in result.decisions if d["task"] == "a2" and "node" not in d)
        self.assertTrue(refusal["reason"].startswith("queue quota exceeded"))
        self.assertIn("queue a", refusal["reason"])

    def test_either_resource_over_the_limit_blocks(self) -> None:
        nodes = (Node("n1", Resources(8, 8)),)
        tasks = (
            task("a1", cpu=1, memory=4, queue="a", duration=2),
            task("a2", cpu=1, memory=1, queue="a", duration=2),
        )
        result = simulate(nodes, tasks, queue_quotas={"a": Resources(8, 4)})
        # cpu is free; memory (4 running + 1 requested > 4) is the binding resource.
        self.assertEqual({p.task_id: p.start for p in result.placements}, {"a1": 0, "a2": 2})
        self.assertEqual(result.metrics["quotas"]["a"]["blocked"], 1)
        self.assertEqual(result.metrics["quotas"]["a"]["peakMemory"], 4)

    def test_a_request_larger_than_its_quota_is_never_placed(self) -> None:
        nodes = (Node("n1", Resources(8, 8)),)
        tasks = (task("big", cpu=5, queue="a"),)
        result = simulate(nodes, tasks, queue_quotas={"a": Resources(4, 4)})
        self.assertEqual(result.placements, [])
        self.assertEqual(result.unplaced, ["big"])
        self.assertEqual(result.metrics["quotas"]["a"]["blocked"], 1)

    def test_unconfigured_queue_is_unrestricted(self) -> None:
        # Only queue a is limited; queue b fills the whole cluster past a's ceiling.
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("a1", cpu=1, queue="a", duration=3),
            task("b1", cpu=4, queue="b", duration=1),
        )
        result = simulate(nodes, tasks, queue_quotas={"a": Resources(1, 1)})
        self.assertEqual(result.unplaced, [])
        self.assertEqual(set(result.metrics["quotas"]), {"a"})

    def test_peak_tracks_half_open_intervals_and_never_exceeds_the_limit(self) -> None:
        nodes = (Node("n1", Resources(10, 10)),)
        tasks = (
            task("a1", cpu=2, queue="a", duration=2),
            task("a2", cpu=2, queue="a", duration=2),
            task("a3", cpu=1, queue="a", arrival=1, duration=2),
        )
        result = simulate(nodes, tasks, queue_quotas={"a": Resources(5, 10)})
        # at tick 1 a1+a2+a3 run concurrently: peak cpu 5; at tick 2 a1/a2 end and release first.
        self.assertEqual(result.metrics["quotas"]["a"]["peakCpu"], 5)
        self.assertLessEqual(result.metrics["quotas"]["a"]["peakCpu"], 5)


class QuotaReleaseTests(unittest.TestCase):
    def test_completion_releases_at_the_same_tick(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("a1", cpu=4, queue="a", duration=3),
            task("a2", cpu=1, queue="a", arrival=1, duration=1),
        )
        result = simulate(nodes, tasks, queue_quotas={"a": Resources(4, 4)})
        # a2 is blocked at tick 1 and 2 (one de-duplicated refusal) and admitted exactly at tick 3,
        # the tick a1's half-open interval ends -- the release is visible to the same-tick decision.
        self.assertEqual({p.task_id: p.start for p in result.placements}, {"a1": 0, "a2": 3})
        refusals = [d for d in result.decisions if d["task"] == "a2" and "node" not in d]
        self.assertEqual(len(refusals), 1)
        self.assertEqual(refusals[0]["at"], 1)

    def test_preemption_releases_the_victim_quota_immediately(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("l1", cpu=2, queue="a", priority=1, duration=10),
            task("l2", cpu=2, queue="a", priority=1, duration=10),
            task("hA", cpu=2, queue="a", priority=9, arrival=1, duration=1),
        )
        result = simulate(
            nodes, tasks, allow_preemption=True, queue_quotas={"a": Resources(6, 6)}
        )
        self.assertEqual(result.metrics["preemptions"], 1)
        # the evicted task's usage is released before hA charges, so the post-eviction peak (4) is
        # what is recorded, and it never crosses the 6-unit ceiling.
        self.assertEqual(result.metrics["quotas"]["a"]["peakCpu"], 4)


class QuotaPreemptionBypassTests(unittest.TestCase):
    def test_a_blocked_task_cannot_evict_another_queue(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("lowA", cpu=4, queue="a", priority=1, duration=10),
            task("highB", cpu=4, queue="b", priority=9, arrival=1, duration=1),
        )
        blocked = simulate(
            nodes, tasks, allow_preemption=True, queue_quotas={"b": Resources(1, 1)}
        )
        # highB's own request (4) exceeds queue b's ceiling: no eviction of lowA may manufacture it.
        self.assertEqual(blocked.metrics["preemptions"], 0)
        self.assertEqual(blocked.unplaced, ["highB"])
        self.assertTrue(
            all(p.task_id != "lowA" or (p.start, p.end) == (0, 10) for p in blocked.placements)
        )
        # the same scenario without a binding quota preempts normally.
        control = simulate(nodes, tasks, allow_preemption=True, queue_quotas={"a": Resources(8, 8)})
        self.assertEqual(control.metrics["preemptions"], 1)
        self.assertEqual({p.task_id: p.start for p in control.placements}, {"lowA": 0, "highB": 1})

    def test_a_blocked_task_cannot_evict_its_own_queue_either(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("l1", cpu=2, queue="a", priority=1, duration=10),
            task("l2", cpu=2, queue="a", priority=1, duration=10),
            task("hA", cpu=2, queue="a", priority=9, arrival=1, duration=1),
        )
        # l1+l2 (2+2) run legally under the 5-unit ceiling. At tick 1 hA sees current use 4 and a
        # 6-unit post-placement total; evicting one same-queue victim would make it fit, but the gate
        # runs on the *current* running use before any preemption search, so no eviction is attempted.
        result = simulate(nodes, tasks, allow_preemption=True, queue_quotas={"a": Resources(5, 5)})
        self.assertEqual(result.metrics["preemptions"], 0)
        self.assertEqual({p.task_id: p.start for p in result.placements}.get("hA"), 10)
        self.assertEqual(result.metrics["quotas"]["a"]["blocked"], 1)


class QuotaBackfillTests(unittest.TestCase):
    def test_backfill_jumps_a_quota_blocked_head_for_an_admitted_candidate(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("long", cpu=2, queue="a", duration=10),
            task("head", cpu=2, queue="a", arrival=1, duration=2),
            task("jumper", cpu=1, queue="b", arrival=1, duration=2),
        )
        result = simulate(nodes, tasks, queue_quotas={"a": Resources(2, 2)})
        # head is quota-blocked (a already holds 2); queue b has no quota, fits and ends by the
        # horizon (long's completion at 10), so it takes the jump.
        jumper = next(p for p in result.placements if p.task_id == "jumper")
        self.assertEqual((jumper.start, jumper.end), (1, 3))
        decision = next(d for d in result.decisions if d["task"] == "jumper")
        self.assertEqual(decision["reason"], "backfill")
        self.assertEqual(result.metrics["quotas"]["a"]["blocked"], 1)

    def test_backfill_skips_a_candidate_blocked_by_its_own_quota_silently(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        anchors = (
            task("aAnchor1", cpu=1, queue="a", duration=10),
            task("aAnchor2", cpu=1, queue="a", duration=10),
        )
        tasks = (
            task("long", cpu=1, queue="L", duration=10),
            task("aHead", cpu=4, queue="H", arrival=1, duration=10),
            task("blockedJumper", cpu=1, queue="a", arrival=1, duration=1),
            task("freeJumper", cpu=1, queue="b", arrival=1, duration=1),
        )
        # At tick 0 the two anchors pin queue a at its 2-unit ceiling and long leaves exactly one
        # unit free. At tick 1 aHead (id-sorted first) is the waiting head and is capacity-blocked;
        # behind it blockedJumper fails the *quota* probe (a still holds 2) invisibly -- exactly like
        # a node-capacity probe failure, with no trace refusal and no blocked count -- and the next
        # candidate freeJumper (unlimited queue b, one free unit) takes the backfill.
        result = simulate(nodes, anchors + tasks, queue_quotas={"a": Resources(2, 2)})
        self.assertFalse(
            any(p.task_id == "blockedJumper" and p.start == 1 for p in result.placements)
        )
        self.assertFalse(
            any(d["task"] == "blockedJumper" and d.get("at") == 1 for d in result.decisions)
        )
        free = next(p for p in result.placements if p.task_id == "freeJumper")
        self.assertEqual((free.start, free.end), (1, 2))
        self.assertEqual(
            next(d["reason"] for d in result.decisions if d["task"] == "freeJumper"), "backfill"
        )
        # A purely-probed quota failure is invisible to the blocked counter as well; queue a never
        # had a task *served* and refused at its gate.
        self.assertEqual(result.metrics["quotas"]["a"]["blocked"], 0)
        self.assertEqual(result.metrics["quotas"]["a"]["peakCpu"], 2)


class QuotaTraceTests(unittest.TestCase):
    def test_one_refusal_per_unchanged_segment_retained_after_placement(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("a1", cpu=3, queue="a", duration=2),
            task("a2", cpu=3, queue="a", duration=2),
        )
        result = simulate(nodes, tasks, queue_quotas={"a": Resources(4, 4)})
        quota_entries = [
            d for d in result.decisions if str(d["reason"]).startswith("queue quota exceeded")
        ]
        self.assertEqual(len(quota_entries), 1)  # blocked across tick 0 only, recorded once
        refusal = quota_entries[0]
        self.assertEqual(refusal["at"], 0)
        # the record is still present even though a2 is subsequently placed
        placed = next(d for d in result.decisions if d["task"] == "a2" and "node" in d)
        self.assertEqual(placed["at"], 2)
        self.assertEqual(len([d for d in result.decisions if d["task"] == "a2"]), 2)

    def test_refusal_names_queue_request_current_use_and_limit(self) -> None:
        nodes = (Node("n1", Resources(8, 8)),)
        tasks = (
            task("a1", cpu=3, memory=2, queue="team-a", duration=2),
            task("a2", cpu=2, memory=3, queue="team-a", duration=2),
        )
        result = simulate(nodes, tasks, queue_quotas={"team-a": Resources(4, 4)})
        reason = next(
            d["reason"] for d in result.decisions if d["task"] == "a2" and "node" not in d
        )
        self.assertIn("queue team-a", reason)
        self.assertIn("cpu=2,memory=3", reason)  # the task's request
        self.assertIn("cpu=3,memory=2", reason)  # current running use
        self.assertIn("cpu=4,memory=4", reason)  # the ceiling


class QuotaMetricsAndEntriesTests(unittest.TestCase):
    def test_blocked_counts_distinct_tasks(self) -> None:
        nodes = (Node("n1", Resources(8, 8)),)
        tasks = (
            task("a1", cpu=3, queue="a", duration=1),
            task("a2", cpu=3, queue="a", duration=3),
            task("a3", cpu=3, queue="a", arrival=1, duration=1),
        )
        result = simulate(nodes, tasks, queue_quotas={"a": Resources(4, 4)})
        # a2 is refused repeatedly (ticks 0,1) as one distinct task; a3 once; blocked is 2 not N.
        self.assertEqual(result.metrics["quotas"]["a"]["blocked"], 2)

    def test_quotas_are_sorted_and_every_policy_carries_the_summary(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (task("z1", cpu=3, queue="z"), task("a1", cpu=3, queue="a"))
        report = compare_policies(
            nodes, tasks, queue_quotas={"z": Resources(4, 4), "a": Resources(4, 4)}
        )
        for entry in report["policies"]:
            self.assertEqual(list(entry["quotas"]), ["a", "z"])
            self.assertEqual(
                set(entry["quotas"]["a"]),
                {"cpu", "memory", "peakCpu", "peakMemory", "blocked"},
            )

    def test_replay_compares_the_full_trace_under_quotas(self) -> None:
        nodes = (Node("n1", Resources(4, 4)), Node("n2", Resources(2, 2)))
        tasks = (
            task("a1", cpu=3, queue="a", duration=2),
            task("a2", cpu=3, queue="a", arrival=1, duration=1),
            task("b1", cpu=1, queue="b", duration=2),
        )
        report = replay(nodes, tasks, policy="best-fit", queue_quotas={"a": Resources(4, 4)})
        self.assertTrue(report["identical"])
        self.assertEqual(report["differences"], [])

    def test_quotas_compose_with_fair_weights(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (
            task("a1", cpu=3, queue="a", duration=2),
            task("a2", cpu=3, queue="a", duration=2),
            task("b1", cpu=1, queue="b", duration=2),
        )
        result = simulate(
            nodes,
            tasks,
            queue_weights={"a": 1, "b": 1},
            queue_quotas={"a": Resources(4, 4)},
        )
        self.assertIn("queues", result.metrics)
        self.assertIn("quotas", result.metrics)
        # b still jumps the quota-blocked a2 through backfill regardless of fair mode.
        self.assertEqual({p.task_id: p.start for p in result.placements},
                         {"a1": 0, "b1": 0, "a2": 2})

    def test_baseline_without_quotas_is_unchanged(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (task("a1", cpu=3, queue="a", duration=2), task("a2", cpu=3, queue="a", duration=2))
        baseline = simulate(nodes, tasks)
        self.assertNotIn("quotas", baseline.metrics)
        # without a quota a2 starts immediately on the same node's free... it fits? 3+3>4, so it
        # waits only on node capacity; the trace carries no quota reason.
        self.assertFalse(
            any("queue quota exceeded" in str(d.get("reason", "")) for d in baseline.decisions)
        )


class QuotaMappingValidationTests(unittest.TestCase):
    def test_bad_mappings_are_rejected(self) -> None:
        nodes = (Node("n1", Resources(4, 4)),)
        tasks = (task("a"),)
        bad_quotas = (
            {},
            {"": Resources(1, 1)},
            {"a": Resources(0, 1)},
            {"a": Resources(1, 0)},
            {"a": Resources(0, 0)},
        )
        for quotas in bad_quotas:
            with self.subTest(quotas=quotas):
                with self.assertRaises(ValidationError):
                    simulate(nodes, tasks, queue_quotas=quotas)


# -- CLI ------------------------------------------------------------------------------------------
def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class QuotaCLITests(unittest.TestCase):
    CLUSTER = [{"id": "n1", "cpu": 4, "memory": 4}, {"id": "n2", "cpu": 4, "memory": 4}]
    TASKS = [
        {"id": "a1", "cpu": 3, "queue": "a", "duration": 2},
        {"id": "a2", "cpu": 3, "queue": "a", "duration": 2},
    ]

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.cluster = self.write("cluster.jsonl", self.CLUSTER)
        self.tasks = self.write("tasks.jsonl", self.TASKS)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write(self, name: str, rows: list) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def write_raw(self, name: str, text: str) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_every_scheduling_command_accepts_quotas(self) -> None:
        quotas = self.write("quotas.jsonl", [{"queue": "a", "cpu": 4, "memory": 4}])
        for command in ("simulate", "trace", "metrics", "policies", "replay"):
            code, out, err = run_cli(
                [command, "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", quotas]
            )
            self.assertEqual((code, err), (EXIT_OK, ""), command)
        # trace and replay carry decisions/identity, not metrics; the metrics-bearing commands add
        # the sorted quotas summary.
        for command in ("simulate", "metrics"):
            code, out, err = run_cli(
                [command, "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", quotas]
            )
            self.assertEqual((code, err), (EXIT_OK, ""), command)
            self.assertIn("quotas", json.loads(out)["metrics"])
        code, out, _ = run_cli(
            ["policies", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", quotas]
        )
        for entry in json.loads(out)["policies"]:
            self.assertIn("quotas", entry)

    def test_quota_driven_unplacement_is_a_negative_verdict(self) -> None:
        quotas = self.write("quotas.jsonl", [{"queue": "a", "cpu": 2, "memory": 2}])
        code, out, err = run_cli(
            ["simulate", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", quotas]
        )
        self.assertEqual((code, err), (EXIT_NEGATIVE, ""))
        self.assertEqual(json.loads(out)["unplaced"], ["a1", "a2"])
        # trace still carries the quota refusal and exits 3
        code, out, _ = run_cli(
            ["trace", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", quotas]
        )
        self.assertEqual(code, EXIT_NEGATIVE)
        reasons = [d["reason"] for d in json.loads(out)["decisions"]]
        self.assertTrue(any(r.startswith("queue quota exceeded") for r in reasons))

    def test_parse_errors_name_the_raw_line(self) -> None:
        cases = {
            "badjson": ('{"queue": "a", "cpu": 4, "memory": 4}\n{broken\n', 2),
            "nonobject": ('{"queue": "a", "cpu": 4, "memory": 4}\n[1]\n', 2),
            "missingcpu": ('{"queue": "a", "memory": 4}\n', 1),
            "missingmemory": ('{"queue": "a", "cpu": 4}\n', 1),
            "missingqueue": ('{"cpu": 4, "memory": 4}\n', 1),
            "typestr": ('{"queue": "a", "cpu": "4", "memory": 4}\n', 1),
            "typebool": ('{"queue": "a", "cpu": true, "memory": 4}\n', 1),
            "unknown": ('{"queue": "a", "cpu": 4, "memory": 4, "gpu": 1}\n', 1),
            "emptyname": ('{"queue": "", "cpu": 4, "memory": 4}\n', 1),
            "nameint": ('{"queue": 3, "cpu": 4, "memory": 4}\n', 1),
        }
        for name, (text, line) in cases.items():
            path = self.write_raw(f"q-{name}.jsonl", text)
            code, out, err = run_cli(
                ["metrics", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", path]
            )
            self.assertEqual((code, out), (EXIT_ERROR, ""), name)
            document = json.loads(err)
            self.assertEqual(document["error"], "parse_error", name)
            self.assertEqual(document["line"], line, name)

    def test_validation_errors(self) -> None:
        # non-positive cpu / memory
        for name, row in (
            ("zero", {"queue": "a", "cpu": 0, "memory": 4}),
            ("neg", {"queue": "a", "cpu": 4, "memory": -1}),
        ):
            path = self.write(f"q-{name}.jsonl", [row])
            code, out, err = run_cli(
                ["metrics", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", path]
            )
            self.assertEqual((code, out), (EXIT_ERROR, ""), name)
            self.assertEqual(json.loads(err)["error"], "validation_error", name)
        # duplicate queue carries the second raw line
        dup = self.write_raw(
            "dup.jsonl", '{"queue": "a", "cpu": 4, "memory": 4}\n{"queue": "a", "cpu": 1, "memory": 1}\n'
        )
        code, out, err = run_cli(
            ["metrics", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", dup]
        )
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")
        self.assertEqual(document["line"], 2)
        # empty file
        empty = self.write("empty.jsonl", [])
        code, out, err = run_cli(
            ["metrics", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", empty]
        )
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        self.assertEqual(json.loads(err)["error"], "validation_error")
        # unreadable file
        code, out, err = run_cli(
            ["metrics", "--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", "/no/such/q.jsonl"]
        )
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_without_the_flag_outputs_no_quota_fields(self) -> None:
        code, out, err = run_cli(["metrics", "--cluster", self.cluster, "--tasks", self.tasks])
        self.assertEqual((code, err), (EXIT_OK, ""))
        self.assertNotIn("quotas", json.loads(out)["metrics"])

    def test_cli_documents_agree_with_the_api_and_each_other(self) -> None:
        from schedsim.cli import load_cluster, load_tasks

        # a2 is transiently quota-blocked at tick 0 and placed at tick 2: everything still places,
        # so every entry exits OK and tells one story.
        quotas = self.write("quotas.jsonl", [{"queue": "a", "cpu": 4, "memory": 4}])
        argv = ["--cluster", self.cluster, "--tasks", self.tasks, "--queue-quotas", quotas]
        code, sim_out, err = run_cli(["simulate", *argv])
        self.assertEqual((code, err), (EXIT_OK, ""))
        code, met_out, err = run_cli(["metrics", *argv])
        self.assertEqual((code, err), (EXIT_OK, ""))
        sim_doc = json.loads(sim_out)
        self.assertEqual(json.loads(met_out)["metrics"], sim_doc["metrics"])
        # byte-for-byte the API document
        api = simulate(
            load_cluster(self.cluster),
            load_tasks(self.tasks),
            queue_quotas={"a": Resources(4, 4)},
        )
        self.assertEqual(sim_doc, api.to_document())
        # policies entries carry the same quotas summary, replay keeps an identical trace
        code, pol_out, err = run_cli(["policies", *argv])
        self.assertEqual((code, err), (EXIT_OK, ""))
        for entry in json.loads(pol_out)["policies"]:
            self.assertEqual(entry["quotas"], sim_doc["metrics"]["quotas"])
        code, rep_out, err = run_cli(["replay", *argv])
        self.assertEqual((code, err), (EXIT_OK, ""))
        self.assertTrue(json.loads(rep_out)["identical"])


if __name__ == "__main__":
    unittest.main()
