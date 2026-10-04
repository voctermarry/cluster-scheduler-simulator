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


if __name__ == "__main__":
    unittest.main()
