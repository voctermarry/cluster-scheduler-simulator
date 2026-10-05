"""CLI contract: JSON on stdout, documented exit codes, and the determinism report."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from schedsim.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, main

CLUSTER = [
    {"id": "n1", "cpu": 4, "memory": 8, "labels": {"zone": "a"}},
    {"id": "n2", "cpu": 4, "memory": 8, "labels": {"zone": "b"}},
]

TASKS = [
    {"id": "t1", "cpu": 2, "memory": 2, "priority": 1, "duration": 4},
    {"id": "t2", "cpu": 2, "memory": 2, "priority": 1, "duration": 2},
    {"id": "t3", "cpu": 9, "memory": 9, "priority": 1, "duration": 1},
]


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class CLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.cluster = self.write("cluster.jsonl", CLUSTER)
        self.tasks = self.write("tasks.jsonl", TASKS)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write(self, name: str, rows: list[dict]) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def test_describe_publishes_the_contract(self) -> None:
        code, out, err = run_cli(["describe"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertEqual(document["exitCodes"], {"ok": 0, "error": 2, "negativeVerdict": 3})
        self.assertEqual(document["policies"], ["first-fit", "best-fit"])

    def test_validate_totals_capacity_and_request(self) -> None:
        code, out, _ = run_cli(["validate", "--cluster", self.cluster, "--tasks", self.tasks])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertEqual(document["totalCapacity"], {"cpu": 8, "memory": 16})
        self.assertEqual(document["nodes"], 2)

    def test_simulate_reports_unplaced_as_a_negative_verdict(self) -> None:
        code, out, _ = run_cli(["simulate", "--cluster", self.cluster, "--tasks", self.tasks])
        self.assertEqual(code, EXIT_NEGATIVE)
        document = json.loads(out)
        self.assertEqual(document["unplaced"], ["t3"])

    def test_simulate_full_placement_is_ok(self) -> None:
        small = self.write("small.jsonl", [{"id": "a", "cpu": 1, "memory": 1, "duration": 1}])
        code, out, _ = run_cli(["simulate", "--cluster", self.cluster, "--tasks", small])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out)["metrics"]["placed"], 1)

    def test_trace_explains_each_decision(self) -> None:
        small = self.write("trace.jsonl", [{"id": "a", "cpu": 1, "memory": 1}, {"id": "b", "cpu": 9}])
        code, out, _ = run_cli(["trace", "--cluster", self.cluster, "--tasks", small])
        self.assertEqual(code, EXIT_NEGATIVE)
        document = json.loads(out)
        reasons = [entry["reason"] for entry in document["decisions"]]
        self.assertTrue(any("placed by first-fit" in reason for reason in reasons))
        self.assertTrue(any("insufficient capacity" in reason for reason in reasons))

    def test_metrics_only(self) -> None:
        small = self.write("m.jsonl", [{"id": "a", "cpu": 4, "memory": 8, "duration": 2}])
        code, out, _ = run_cli(["metrics", "--cluster", self.cluster, "--tasks", small])
        self.assertEqual(code, EXIT_OK)
        metrics = json.loads(out)["metrics"]
        self.assertEqual(metrics["makespan"], 2)
        self.assertGreater(metrics["utilization"], 0.0)

    def test_replay_reports_identical_traces(self) -> None:
        small = self.write("r.jsonl", [{"id": "a", "cpu": 1, "memory": 1}, {"id": "b", "cpu": 2, "memory": 2}])
        code, out, _ = run_cli(["replay", "--cluster", self.cluster, "--tasks", small, "--policy", "best-fit"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertTrue(document["identical"])
        self.assertEqual(document["differences"], [])

    def test_policies_compares_both(self) -> None:
        small = self.write("p.jsonl", [{"id": "a", "cpu": 3, "memory": 3, "duration": 2}])
        code, out, _ = run_cli(["policies", "--cluster", self.cluster, "--tasks", small])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual({entry["policy"] for entry in json.loads(out)["policies"]}, {"first-fit", "best-fit"})

    def test_affinity_restricts_placement(self) -> None:
        pinned = self.write("pinned.jsonl", [{"id": "a", "cpu": 1, "memory": 1, "affinity": {"zone": "b"}}])
        code, out, _ = run_cli(["simulate", "--cluster", self.cluster, "--tasks", pinned])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out)["placements"][0]["node"], "n2")

    def test_unknown_field_is_reported_with_a_line_number(self) -> None:
        broken = self.write("broken.jsonl", [{"id": "a", "cpu": 1, "memory": 1, "gpu": 1}])
        code, _, err = run_cli(["validate", "--cluster", self.cluster, "--tasks", broken])
        self.assertEqual(code, EXIT_ERROR)
        document = json.loads(err)
        self.assertEqual(document["error"], "parse_error")
        self.assertEqual(document["line"], 1)

    def test_missing_id_is_rejected(self) -> None:
        broken = self.write("noid.jsonl", [{"cpu": 1, "memory": 1}])
        code, _, err = run_cli(["validate", "--cluster", broken, "--tasks", self.tasks])
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("missing id", json.loads(err)["message"])

    def test_empty_cluster_is_rejected(self) -> None:
        empty = self.write("empty.jsonl", [])
        code, _, err = run_cli(["validate", "--cluster", empty, "--tasks", self.tasks])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_preemption_flag_changes_the_schedule(self) -> None:
        # One node only: with two nodes the high-priority task simply lands on the other one and no
        # preemption is needed (the first version of this test used the two-node cluster and failed
        # for that reason -- the product was right).
        rows = [
            {"id": "low", "cpu": 4, "memory": 8, "priority": 1, "duration": 6},
            {"id": "high", "cpu": 4, "memory": 8, "priority": 9, "duration": 1, "arrival": 1},
        ]
        tasks = self.write("preempt.jsonl", rows)
        one_node = self.write("one.jsonl", [{"id": "n1", "cpu": 4, "memory": 8}])
        _, without, _ = run_cli(["simulate", "--cluster", one_node, "--tasks", tasks])
        _, with_preemption, _ = run_cli(["simulate", "--cluster", one_node, "--tasks", tasks, "--preemption"])
        start_without = {item["task"]: item["start"] for item in json.loads(without)["placements"]}
        start_with = {item["task"]: item["start"] for item in json.loads(with_preemption)["placements"]}
        self.assertLess(start_with["high"], start_without["high"])
        self.assertEqual(json.loads(with_preemption)["metrics"]["preemptions"], 1)

    # -- preemption may not bypass affinity, anti-affinity or taints --------------------------------
    CONSTRAINT_CLUSTER = [
        {"id": "n1", "cpu": 4, "memory": 4, "labels": {"zone": "a"}},
        {"id": "n2", "cpu": 4, "memory": 4, "labels": {"zone": "b"}},
        {"id": "n3", "cpu": 4, "memory": 4, "labels": {"zone": "b", "spot": "x"}},
        {"id": "n4", "cpu": 4, "memory": 4, "labels": {"zone": "b"}, "taints": ["gpu"]},
    ]

    def constraint_tasks(self) -> str:
        rows = [
            {"id": "low1", "cpu": 4, "memory": 4, "priority": 1, "duration": 10},
            {"id": "low2", "cpu": 4, "memory": 4, "priority": 1, "duration": 10},
            {"id": "low3", "cpu": 4, "memory": 4, "priority": 1, "duration": 10},
            {"id": "low4", "cpu": 4, "memory": 4, "priority": 1, "duration": 10, "tolerations": ["gpu"]},
            {"id": "hi", "cpu": 4, "memory": 4, "priority": 9, "arrival": 1, "duration": 1,
             "affinity": {"zone": "b"}, "antiAffinity": ["spot"]},
        ]
        return self.write("constraint_tasks.jsonl", rows)

    def test_preemption_skips_smaller_id_incompatible_nodes_end_to_end(self) -> None:
        cluster = self.write("constraint_cluster.jsonl", self.CONSTRAINT_CLUSTER)
        tasks = self.constraint_tasks()
        # n1 is the smallest id and resource-wise preemptable, but its zone=a label fails affinity;
        # n3 carries spot and n4 an untolerated taint, so the only valid victim is low2 on n2.
        code, out, err = run_cli(
            ["simulate", "--cluster", cluster, "--tasks", tasks, "--preemption"]
        )
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        placements = {p["task"]: p for p in document["placements"]}
        self.assertEqual(placements["hi"]["node"], "n2")
        self.assertEqual(placements["hi"]["preempted"], ["low2"])
        self.assertEqual(document["metrics"]["preemptions"], 1)
        low1 = placements["low1"]
        self.assertEqual((low1["start"], low1["end"]), (0, 10))
        self.assertNotIn("preempted", low1)
        # trace agrees: node, preempted list, reason and at all match the simulate document
        code, out, _ = run_cli(["trace", "--cluster", cluster, "--tasks", tasks, "--preemption"])
        self.assertEqual(code, EXIT_OK)
        decision = next(d for d in json.loads(out)["decisions"] if d["task"] == "hi")
        self.assertEqual(
            decision,
            {"task": "hi", "node": "n2",
             "reason": "preempting 1 lower-priority task(s) on n2", "at": 1},
        )
        # metrics and replay carry the same single, real eviction
        code, out, _ = run_cli(["metrics", "--cluster", cluster, "--tasks", tasks, "--preemption"])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out)["metrics"]["preemptions"], 1)
        code, out, _ = run_cli(["replay", "--cluster", cluster, "--tasks", tasks, "--preemption"])
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(json.loads(out)["identical"])

    def test_constraint_impossible_task_is_unplaced_exit_3_without_preemption_records(self) -> None:
        # Every node fails the incoming task's zone=c affinity. The task must wait, then end up
        # unplaced: exit code 3, the existing "no node fits" refusal, and zero preemptions.
        cluster_rows = [
            {"id": "n1", "cpu": 4, "memory": 4, "labels": {"zone": "a"}},
            {"id": "n2", "cpu": 4, "memory": 4, "labels": {"zone": "b"}},
        ]
        tasks_rows = [
            {"id": "low", "cpu": 4, "memory": 4, "priority": 1, "duration": 2},
            {"id": "hi", "cpu": 4, "memory": 4, "priority": 9, "arrival": 1, "duration": 1,
             "affinity": {"zone": "c"}},
        ]
        cluster = self.write("impossible_cluster.jsonl", cluster_rows)
        tasks = self.write("impossible_tasks.jsonl", tasks_rows)
        code, out, err = run_cli(
            ["simulate", "--cluster", cluster, "--tasks", tasks, "--preemption"]
        )
        self.assertEqual((code, err), (EXIT_NEGATIVE, ""))
        document = json.loads(out)
        self.assertEqual(document["unplaced"], ["hi"])
        self.assertEqual(document["metrics"]["preemptions"], 0)
        self.assertFalse(
            any(p["task"] == "low" and "preempted" in p for p in document["placements"])
        )
        code, out, _ = run_cli(["trace", "--cluster", cluster, "--tasks", tasks, "--preemption"])
        self.assertEqual(code, EXIT_NEGATIVE)
        decisions = json.loads(out)["decisions"]
        refusals = [d for d in decisions if d["task"] == "hi" and "at" in d]
        self.assertTrue(all("no node fits" in d["reason"] for d in refusals))
        self.assertTrue(any("affinity zone=c" in d["reason"] for d in refusals))
        self.assertFalse(any("preempt" in str(d["reason"]) for d in decisions))
        # preemption off behaves identically for this constraint-impossible task
        code, out, _ = run_cli(["simulate", "--cluster", cluster, "--tasks", tasks])
        self.assertEqual(code, EXIT_NEGATIVE)
        self.assertEqual(json.loads(out)["unplaced"], ["hi"])


    def test_validate_rejects_duplicate_node_ids(self) -> None:
        # The end-to-end run found `validate` exiting 0 for a cluster with a repeated node id, because
        # it only built Node objects and never a Cluster (whose constructor owns that check).
        duplicate = self.write("dup.jsonl", [{"id": "n1", "cpu": 4, "memory": 8}, {"id": "n1", "cpu": 4, "memory": 8}])
        code, _, err = run_cli(["validate", "--cluster", duplicate, "--tasks", self.tasks])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_validate_rejects_duplicate_task_ids(self) -> None:
        duplicate = self.write("duptask.jsonl", [{"id": "a", "cpu": 1, "memory": 1}, {"id": "a", "cpu": 1, "memory": 1}])
        code, _, err = run_cli(["validate", "--cluster", self.cluster, "--tasks", duplicate])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_repeated_refusals_are_reported_once_per_reason(self) -> None:
        # The scheduler retries a blocked task every tick; the trace used to repeat one sentence seven
        # times and bury the decisions that mattered.
        rows = [{"id": "big", "cpu": 9, "memory": 9}, {"id": "small", "cpu": 1, "memory": 1, "duration": 3}]
        tasks = self.write("blocked.jsonl", rows)
        _, out, _ = run_cli(["trace", "--cluster", self.cluster, "--tasks", tasks])
        refusals = [entry for entry in json.loads(out)["decisions"] if entry["task"] == "big" and "no node fits" in entry["reason"]]
        self.assertEqual(len(refusals), 1)

    # -- conservative backfill -------------------------------------------------------------------
    BACKFILL_CLUSTER = [{"id": "n1", "cpu": 4, "memory": 4}]
    BACKFILL_TASKS = [
        {"id": "long", "cpu": 2, "memory": 2, "duration": 10},
        {"id": "head", "cpu": 4, "memory": 4, "duration": 10, "arrival": 1},
        {"id": "short", "cpu": 2, "memory": 2, "duration": 2, "arrival": 1},
    ]

    def test_backfill_starts_the_short_task_and_marks_the_reason(self) -> None:
        cluster = self.write("bf_cluster.jsonl", self.BACKFILL_CLUSTER)
        tasks = self.write("bf_tasks.jsonl", self.BACKFILL_TASKS)
        code, out, err = run_cli(["trace", "--cluster", cluster, "--tasks", tasks])
        self.assertEqual((code, err), (EXIT_OK, ""))
        short = next(d for d in json.loads(out)["decisions"] if d["task"] == "short")
        self.assertEqual(short["reason"], "backfill")
        self.assertEqual(short["at"], 1)

    def test_no_backfill_keeps_the_short_task_behind_the_head(self) -> None:
        cluster = self.write("nbf_cluster.jsonl", self.BACKFILL_CLUSTER)
        tasks = self.write("nbf_tasks.jsonl", self.BACKFILL_TASKS)
        _, with_flag, _ = run_cli(["simulate", "--cluster", cluster, "--tasks", tasks, "--no-backfill"])
        starts = {p["task"]: p["start"] for p in json.loads(with_flag)["placements"]}
        self.assertEqual(starts, {"long": 0, "head": 10, "short": 20})

    # -- weighted fair share ---------------------------------------------------------------------
    def write_weights(self, rows: list, name: str = "weights.jsonl") -> str:
        return self.write(name, rows)

    def FAIR_TASKS(self) -> list:
        return [
            {"id": "a1", "cpu": 2, "memory": 2, "queue": "a", "duration": 4},
            {"id": "a2", "cpu": 2, "memory": 2, "queue": "a", "duration": 2},
            {"id": "b1", "cpu": 2, "memory": 2, "queue": "b", "duration": 2},
        ]

    def test_queue_weights_change_the_schedule_and_annotate_the_trace(self) -> None:
        tasks = self.write("fair.jsonl", self.FAIR_TASKS())
        weights = self.write_weights([{"queue": "a", "weight": 2}, {"queue": "b", "weight": 1}])
        code, out, err = run_cli(["trace", "--cluster", self.cluster, "--tasks", tasks, "--queue-weights", weights])
        self.assertEqual((code, err), (EXIT_OK, ""))
        decisions = json.loads(out)["decisions"]
        self.assertEqual([d["task"] for d in decisions if "node" in d], ["a1", "b1", "a2"])
        for decision in decisions:
            self.assertIn("queue", decision)
            self.assertIn("weight", decision)
            self.assertIn("weightedDominantShare", decision)
            self.assertIsInstance(decision["weightedDominantShare"], float)
        first_b = next(d for d in decisions if d["task"] == "b1" and "node" in d)
        self.assertEqual(first_b["weightedDominantShare"], 0.0)

    def test_metrics_report_sorted_queues(self) -> None:
        tasks = self.write("fairm.jsonl", self.FAIR_TASKS())
        weights = self.write_weights([{"queue": "b", "weight": 1}, {"queue": "a", "weight": 2}])
        code, out, _ = run_cli(["metrics", "--cluster", self.cluster, "--tasks", tasks, "--queue-weights", weights])
        self.assertEqual(code, EXIT_OK)
        queues = json.loads(out)["metrics"]["queues"]
        self.assertEqual(list(queues), ["a", "b"])
        self.assertEqual(set(queues["a"]), {"weight", "placed", "unplaced", "averageWait", "cpuTime", "memoryTime", "dominantShare"})

    def test_policies_include_queue_metrics_in_fair_mode(self) -> None:
        tasks = self.write("fairp.jsonl", self.FAIR_TASKS())
        weights = self.write_weights([{"queue": "a", "weight": 2}])
        code, out, _ = run_cli(["policies", "--cluster", self.cluster, "--tasks", tasks, "--queue-weights", weights])
        self.assertEqual(code, EXIT_OK)
        for entry in json.loads(out)["policies"]:
            self.assertIn("queues", entry)
            self.assertEqual(entry["queues"]["b"]["weight"], 1)

    def test_replay_is_identical_in_fair_mode(self) -> None:
        tasks = self.write("fairr.jsonl", self.FAIR_TASKS())
        weights = self.write_weights([{"queue": "a", "weight": 2}, {"queue": "b", "weight": 1}])
        code, out, _ = run_cli(["replay", "--cluster", self.cluster, "--tasks", tasks, "--queue-weights", weights])
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(json.loads(out)["identical"])

    def test_without_weights_outputs_no_fair_fields(self) -> None:
        tasks = self.write("plain.jsonl", self.FAIR_TASKS())
        _, out, _ = run_cli(["trace", "--cluster", self.cluster, "--tasks", tasks])
        for decision in json.loads(out)["decisions"]:
            self.assertNotIn("weightedDominantShare", decision)
        _, out, _ = run_cli(["metrics", "--cluster", self.cluster, "--tasks", tasks])
        self.assertNotIn("queues", json.loads(out)["metrics"])

    def test_unreadable_weights_file_is_a_validation_error(self) -> None:
        tasks = self.write("fairx.jsonl", self.FAIR_TASKS())
        code, out, err = run_cli(["metrics", "--cluster", self.cluster, "--tasks", tasks, "--queue-weights", "/no/such/file.jsonl"])
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_empty_weights_file_is_a_validation_error(self) -> None:
        tasks = self.write("faire.jsonl", self.FAIR_TASKS())
        weights = self.write_weights([])
        code, out, err = run_cli(["metrics", "--cluster", self.cluster, "--tasks", tasks, "--queue-weights", weights])
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_duplicate_queue_is_a_validation_error_with_line(self) -> None:
        tasks = self.write("faird.jsonl", self.FAIR_TASKS())
        with open(weights_path := os.path.join(self.directory.name, "dupw.jsonl"), "w") as handle:
            handle.write('{"queue": "a", "weight": 1}\n{"queue": "a", "weight": 3}\n')
        code, out, err = run_cli(["metrics", "--cluster", self.cluster, "--tasks", tasks, "--queue-weights", weights_path])
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")
        self.assertEqual(document["line"], 2)

    def test_non_positive_weight_is_a_validation_error(self) -> None:
        tasks = self.write("fairz.jsonl", self.FAIR_TASKS())
        weights = self.write_weights([{"queue": "a", "weight": 0}])
        code, _, err = run_cli(["metrics", "--cluster", self.cluster, "--tasks", tasks, "--queue-weights", weights])
        self.assertEqual(code, EXIT_ERROR)
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")
        self.assertEqual(document["line"], 1)

    def test_big_integer_inputs_order_by_exact_share_everywhere(self) -> None:
        # 1e20 capacity: queue Z's anchor holds one unit fewer than A's, a difference no binary
        # double can represent at 0.5. The exact ordering must serve z1 before a1 through every
        # public entry point, while the displayed share stays the rounded float 0.5.
        big = 10**20
        half = big // 2
        cluster = self.write(
            "big_cluster.jsonl", [{"id": "n1", "cpu": big, "memory": big}]
        )
        rows = [
            {"id": "aRun", "cpu": half - 2, "memory": 1, "queue": "A", "duration": 10},
            {"id": "zRun", "cpu": half - 3, "memory": 1, "queue": "Z", "duration": 10},
            {"id": "z1", "cpu": 1, "queue": "Z", "arrival": 1},
            {"id": "a1", "cpu": 1, "queue": "A", "arrival": 1},
            {"id": "z2", "cpu": 1, "queue": "Z", "arrival": 1},
            {"id": "a2", "cpu": 1, "queue": "A", "arrival": 1},
        ]
        tasks = self.write("big_tasks.jsonl", rows)
        weights = self.write_weights(
            [{"queue": "A", "weight": 1}, {"queue": "Z", "weight": 1}], "big_weights.jsonl"
        )
        # simulate: every task is placed; placement documents are sorted by (start, task id), so
        # the scheduling order itself is asserted through `trace` below.
        code, out, err = run_cli(["simulate", "--cluster", cluster, "--tasks", tasks, "--queue-weights", weights])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertEqual(document["unplaced"], [])
        self.assertEqual([p["task"] for p in document["placements"] if p["start"] == 1], ["a1", "a2", "z1", "z2"])

        # trace: decisions carry a six-decimal float that displays identically for both queues
        code, out, _ = run_cli(["trace", "--cluster", cluster, "--tasks", tasks, "--queue-weights", weights])
        self.assertEqual(code, EXIT_OK)
        decisions = json.loads(out)["decisions"]
        self.assertEqual([d["task"] for d in decisions if d.get("at") == 1 and "node" in d], ["z1", "a1", "z2", "a2"])
        for decision in decisions:
            if decision.get("at") == 1:
                self.assertIsInstance(decision["weightedDominantShare"], float)
        self.assertEqual({d["weightedDominantShare"] for d in decisions if d.get("at") == 1}, {0.5})

        # metrics: exact resource accounting survives, dominantShare remains a rounded JSON number
        code, out, _ = run_cli(["metrics", "--cluster", cluster, "--tasks", tasks, "--queue-weights", weights])
        self.assertEqual(code, EXIT_OK)
        queues = json.loads(out)["metrics"]["queues"]
        self.assertEqual(list(queues), ["A", "Z"])
        self.assertEqual(queues["A"]["cpuTime"], (half - 2) * 10 + 2)
        self.assertEqual(queues["Z"]["cpuTime"], (half - 3) * 10 + 2)
        self.assertEqual(queues["A"]["dominantShare"], 0.5)

        # policies: both node selectors keep the same queue ordering (node choice is unchanged)
        code, out, _ = run_cli(["policies", "--cluster", cluster, "--tasks", tasks, "--queue-weights", weights])
        self.assertEqual(code, EXIT_OK)
        for entry in json.loads(out)["policies"]:
            self.assertEqual(entry["unplaced"], 0)

        # replay: the exact ordering is deterministic run to run
        code, out, _ = run_cli(["replay", "--cluster", cluster, "--tasks", tasks, "--queue-weights", weights])
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(json.loads(out)["identical"])

        # best-fit changes only node selection, never the queue order
        code, out, _ = run_cli(
            ["trace", "--cluster", cluster, "--tasks", tasks, "--queue-weights", weights, "--policy", "best-fit"]
        )
        self.assertEqual(code, EXIT_OK)
        self.assertEqual([d["task"] for d in json.loads(out)["decisions"] if d.get("at") == 1 and "node" in d],
                         ["z1", "a1", "z2", "a2"])

    def test_weights_parse_errors_name_the_raw_line(self) -> None:
        tasks = self.write("fairb.jsonl", self.FAIR_TASKS())
        cases = {
            "badjson": ['{"queue": "a", "weight": 1}', "{broken"],
            "nonobject": ['{"queue": "a", "weight": 1}', "[1]"],
            "missing": ['{"queue": "a"}'],
            "type": ['{"queue": "a", "weight": "2"}'],
            "unknown": ['{"queue": "a", "weight": 1, "extra": 2}'],
            "emptystring": ['{"queue": "", "weight": 1}'],
            "boolweight": ['{"queue": "a", "weight": true}'],
        }
        for name, lines in cases.items():
            path = os.path.join(self.directory.name, f"w-{name}.jsonl")
            with open(path, "w") as handle:
                handle.write("\n".join(lines) + "\n")
            code, out, err = run_cli(["metrics", "--cluster", self.cluster, "--tasks", tasks, "--queue-weights", path])
            self.assertEqual((code, out), (EXIT_ERROR, ""), name)
            document = json.loads(err)
            self.assertEqual(document["error"], "parse_error", name)
            self.assertEqual(document["line"], 2 if name in ("badjson", "nonobject") else 1, name)


if __name__ == "__main__":
    unittest.main()
