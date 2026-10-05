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

Common options: `--policy first-fit|best-fit`, `--preemption`, `--no-backfill`,
`--queue-weights PATH` (fair share), `--queue-quotas PATH` (hard per-queue limits),
`--cluster -` for stdin.

## Weighted fair share

`simulate`, `trace`, `metrics`, `policies` and `replay` accept `--queue-weights PATH`, a UTF-8
JSONL file with one object per line:

```json
{"queue": "team-a", "weight": 2}
{"queue": "team-b", "weight": 1}
```

Each valid row contains exactly the non-empty string `queue` and the positive integer `weight`;
queues must not repeat. A queue used by a task but absent from the file has an implicit weight of
1. Supplying the file enables **fair mode**; without it every ordering, output field and exit code
is unchanged.

In fair mode each simulation tick first releases finished tasks, then waiting tasks are ordered by
higher `priority` first; at equal priority the task comes from the queue with the smallest
**weighted dominant share** — `max(used CPU / total CPU, used memory / total memory) / weight` over
the running tasks — and remaining ties break on `arrival`, queue name and task id. Shares are
recomputed after every release, placement and preemption. Node selection (first-fit / best-fit),
affinity, anti-affinity, taints, preemption candidates and the conservative backfill boundary keep
their existing semantics. The Python entry points (`simulate`, `replay`, `compare_policies`,
`Simulation`) take the same configuration as a mapping, `queue_weights={"team-a": 2}`.

The `trace` decisions gain `queue`, `weight` and `weightedDominantShare` (the pre-decision share,
six decimals); `replay` produces an identical placement trace. Metrics gain a `queues` object keyed
by queue name (sorted); each queue reports `weight`, `placed`, `unplaced`, `averageWait`,
`cpuTime`, `memoryTime` and `dominantShare`. Resource time accumulates over the actual running
interval (a preempted task counts only up to its eviction tick), and `dominantShare` is
`max(cpu-time / cpu-capacity, memory-time / memory-capacity) / makespan / weight`, or `0` when
capacity or makespan is zero. `policies` attaches the same per-queue metrics to every policy.

An unreadable, empty, or duplicate-queue file, or a non-positive weight, is a `validation_error`;
broken JSON, a non-object row, a missing field, a wrong type or an unknown field is a
`parse_error` carrying the raw line number. Both exit **2** with nothing written to stdout.

## Queue concurrency quotas

`simulate`, `trace`, `metrics`, `policies` and `replay` also accept `--queue-quotas PATH`, a
UTF-8 JSONL file with one object per line:

```json
{"queue": "team-a", "cpu": 16, "memory": 32}
{"queue": "team-b", "cpu": 8, "memory": 16}
```

Each valid row contains exactly the non-empty string `queue` and the positive integers `cpu` and
`memory`; queues must not repeat. The values are a **hard ceiling** on the CPU and memory held at
once by that queue's running tasks. A queue not named in the file is unlimited, and supplying no
file at all leaves every ordering, output field and exit code unchanged. The Python entry points
(`simulate`, `replay`, `compare_policies`, `Simulation`) take the same configuration as a mapping
of queue name to `Resources`, `queue_quotas={"team-a": Resources(16, 32)}`.

Before **every** placement the scheduler sums the CPU and memory of the queue's currently running
tasks and refuses the task when adding its request would exceed either limit. The check runs
**before node selection**: a blocked task probes no node, triggers no preemption, and cannot evade
the ceiling by evicting tasks of other queues (the limit is on its own queue). It keeps its place
in the existing waiting order — priority, fair share and arrival ordering are unchanged. With
backfill enabled, the scheduler may still start the first later candidate that satisfies the node
predicates, the conservative time boundary **and its own quota**; a candidate over its own quota
is probed without side effects, exactly like one that fails a node predicate. A finished or
preempted task releases its quota occupancy immediately, so a decision at the same tick already
sees the release.

A quota refusal appears in the `trace` once per contiguous unchanged spell (the same deduping as
every refusal), even if the task is placed later. Its reason starts with
`queue quota exceeded` and names the queue, the task request, the current running occupancy and
the limit, e.g.
`queue quota exceeded for queue team-a: request cpu=2,memory=4 running cpu=14,memory=20 limit cpu=16,memory=32`.

When the file is supplied, `simulate` and `metrics` gain a `quotas` object keyed by queue name
(sorted). Each queue reports:

| Field | Meaning |
|---|---|
| `cpu`, `memory` | the configured limit |
| `peakCpu`, `peakMemory` | most CPU / memory held concurrently over the half-open run intervals; never above the limit |
| `blocked` | number of distinct tasks ever refused by the queue's quota gate |

`policies` attaches the same `quotas` summary to every policy result; `replay` keeps comparing the
full `(task, node, start, end)` placement trajectory. If quotas leave tasks unplaced, they appear
in `unplaced` with their refusal in the trace and the command exits **3** as usual.

An unreadable, empty, or duplicate-queue file, or a non-positive `cpu`/`memory`, is a
`validation_error`; broken JSON, a non-object row, a missing field, a wrong type or an unknown
field is a `parse_error` carrying the raw line number. Both exit **2** with nothing written to
stdout.

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
* **Backfill is bounded.** When the waiting head cannot be placed, and only then, the scheduler
  looks at the tasks behind it in the same waiting order and starts the **first** one that already
  fits a node and finishes by the earliest running completion (`now + duration <= horizon`) -- the
  conservative rule, so a backfilled task can never delay the head it passed. The boundary is
  inclusive: ending exactly at the horizon is allowed, one tick later is not. Backfill never runs
  with no task active (there is no completion to bound against), and `--no-backfill` forbids
  jumping altogether. A candidate that fails the resource check or the time boundary is probed
  without side effects: no placement, no refusal in the trace, no preemption. A successful
  backfill is recorded with the fixed reason `backfill`; afterwards the scheduler re-ranks the
  waiters on the fresh occupancy and weighted shares and may place or backfill again at the same
  tick. Affinity, anti-affinity, taints, first-fit / best-fit node choice and the preemption rules
  are unchanged inside a backfill.
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
