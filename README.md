# cluster-scheduler-simulator

Deterministic cluster scheduling simulation: resource model, placement predicates, quotas and
priorities, preemption, conservative backfill, and reproducibility (Python standard library only).

## Install and entry point

```
python3 -m pip install -e .
cluster-scheduler-simulator --help
cluster-scheduler-simulator describe
```

Package `schedsim`; console script `cluster-scheduler-simulator`. Inputs are a **JSONL node list** and a
**JSONL task list**; every subcommand writes one JSON document to stdout, and a failure writes one JSON
document to stderr with the offending **line number**.

## Input

```json
// cluster.jsonl
{"id": "n1", "cpu": 4, "memory": 8, "labels": {"zone": "a"}, "taints": ["gpu"]}

// tasks.jsonl
{"id": "t1", "cpu": 2, "memory": 2, "priority": 5, "queue": "team-a",
 "arrival": 0, "duration": 4, "affinity": {"zone": "a"},
 "antiAffinity": ["spot"], "tolerations": ["gpu"]}
```

`priority`, `queue`, `arrival`, `duration`, `affinity`, `antiAffinity` and `tolerations` are optional.
Unknown fields are rejected.

## Commands

| Command | Purpose | Exit codes |
|---|---|---|
| `describe` | capabilities, policies, input fields, exit codes | 0 |
| `validate --cluster … --tasks …` | parse and validate; report capacity and total request | 0 / 2 |
| `simulate …` | the schedule: placements and metrics | 0 / **3** (tasks unplaced) / 2 |
| `trace …` | every scheduling decision with its reason | 0 / **3** / 2 |
| `metrics …` | metrics only | 0 / **3** / 2 |
| `policies …` | first-fit and best-fit side by side | 0 / **3** / 2 |
| `replay …` | run twice and compare the traces | 0 / **3** (traces differ) / 2 |

Common options: `--policy first-fit|best-fit`, `--preemption`, `--no-backfill`, `--cluster -` for stdin.

## What the scheduler promises

* **Every rejection says why.** Predicates are checked in a fixed order (capacity, affinity,
  anti-affinity, taints) and the failure names the constraint and its numbers — e.g.
  `insufficient capacity: needs cpu=9,memory=9 has cpu=4,memory=8`, or `affinity zone=b not satisfied`.
  A refusal that cannot explain itself is useless when a schedule looks wrong.
* **Placement is deterministic.** No wall clock, no iteration over unordered containers, and every tie
  breaks on node id (best-fit: smallest remaining CPU, then memory, then id).
* **Preemption is conservative and minimal.** Only strictly lower priorities are eligible, they are
  ordered lowest-priority-first and then largest-request-first, and the search stops as soon as the
  incoming task fits. When even evicting everything would not free enough, the request is refused with
  that sentence instead of evicting and still failing.
* **Capacity is never over-committed.** The test suite re-walks a schedule tick by tick and asserts the
  sum of concurrent requests fits each node.
* **Backfill is bounded.** A task may pass a blocked head only when it finishes before the earliest
  running completion (`now + duration <= horizon`), which is the conservative rule: it cannot delay the
  head it passed.
* **Reproducibility is checked, not asserted.** `replay` runs the same input through the simulator twice
  and compares the full `(task, node, start, end)` trace, reporting `identical` and any difference, and
  exiting **3** when the two runs disagree.
* **Nothing is dropped silently.** Tasks that cannot be placed appear in `unplaced` and in the decision
  trace, and the process exits 3.

## Metrics

`makespan`, `placed`, `unplaced`, `preemptions`, `averageWait`, `maxWait`, CPU `utilization` (used
CPU-time over capacity × makespan) and `fragmentation` (`nodesBlocked` and `wastedCpu` for the smallest
pending request).

## Layout

```
schedsim/model.py       resources, tasks, nodes, cluster bookkeeping, placements
schedsim/policies.py    predicates with reasons, node ordering, preemption selection, fragmentation
schedsim/simulator.py   event clock, conservative backfill, metrics, replay, policy comparison
schedsim/cli.py         eight subcommands and the exit-code contract
tests/                  model, predicates, selection, preemption, simulation, CLI
```
