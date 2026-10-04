"""Deterministic scheduler simulation: an event clock, conservative backfill, and metrics.

The simulator is a pure function of its input. It never consults the wall clock, never iterates a
`set` in a way that affects a decision, and never breaks a tie by insertion order -- so the same
cluster and task list always produce the same trace, and `replay` proves it instead of asserting it.

One list holds every placement ever made; "running" is derived from the clock (`start <= t < end`),
which removes any chance of the bookkeeping disagreeing with the schedule. Capacity is released
exactly once per placement, tracked by index.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .errors import ValidationError
from .model import Cluster, Node, Placement, Task
from .policies import POLICIES, fragmentation, preemption_candidates, select_node


def validate_queue_weights(weights: Mapping[str, int]) -> dict[str, int]:
    """The one place queue-weight rules live, so the CLI file format and the Python mapping agree."""
    if not isinstance(weights, Mapping) or not weights:
        raise ValidationError("queue weights must be a non-empty mapping of queue name to positive integer")
    clean: dict[str, int] = {}
    for queue, weight in weights.items():
        if not isinstance(queue, str) or not queue:
            raise ValidationError("queue names must be non-empty strings", value=queue)
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            raise ValidationError("queue weights must be positive integers", queue=queue, value=weight)
        clean[queue] = int(weight)
    return clean


@dataclass(slots=True)
class SimulationResult:
    policy: str
    placements: list[Placement]
    decisions: list[dict[str, object]]
    unplaced: list[str]
    makespan: int
    metrics: dict[str, object]

    def trace(self) -> list[tuple[str, str, int, int]]:
        """The identity of a run: (task, node, start, end) sorted -- what determinism is checked on."""
        return sorted((item.task_id, item.node_id, item.start, item.end) for item in self.placements)

    def to_document(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "makespan": self.makespan,
            "placed": len(self.placements),
            "unplaced": self.unplaced,
            "metrics": self.metrics,
            "placements": [item.to_document() for item in self.placements],
        }


@dataclass(slots=True)
class Simulation:
    nodes: tuple[Node, ...]
    tasks: tuple[Task, ...]
    policy: str = "first-fit"
    allow_preemption: bool = False
    backfill: bool = True
    queue_weights: Mapping[str, int] | None = None
    _cluster: Cluster = field(init=False)
    _placements: list[Placement] = field(default_factory=list, init=False)
    _released: set[int] = field(default_factory=set, init=False)
    _decisions: list[dict[str, object]] = field(default_factory=list, init=False)
    _preemptions: int = field(default=0, init=False)
    _waits: list[int] = field(default_factory=list, init=False)
    _wait_by_task: dict[str, int] = field(default_factory=dict, init=False)
    _last_refusal: dict[str, str] = field(default_factory=dict, init=False)
    _fair: bool = field(init=False)
    _weights: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValidationError(f"policy must be one of {', '.join(POLICIES)}", value=self.policy)
        ids = [task.id for task in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValidationError("task ids must be unique")
        self._fair = self.queue_weights is not None
        self._weights = validate_queue_weights(self.queue_weights) if self._fair else {}
        self._cluster = Cluster(self.nodes)

    # -- bookkeeping -----------------------------------------------------------------------------
    @property
    def task_index(self) -> dict[str, Task]:
        return {task.id: task for task in self.tasks}

    def _pending_order(self, pending: list[Task]) -> list[Task]:
        return sorted(pending, key=lambda task: (-task.priority, task.arrival, task.id))

    # -- fair share ------------------------------------------------------------------------------
    def _weight(self, queue: str) -> int:
        """Queues the tasks use but the config never declares weigh 1."""
        return self._weights.get(queue, 1)

    def _queue_shares(self, clock: int) -> dict[str, float]:
        """Weighted dominant share per queue, from what is running right now.

        The dominant share is the larger of the queue's CPU fraction and memory fraction of the
        whole cluster; dividing by the queue's weight is what makes the schedule *fair* rather
        than merely proportional. Recomputed on every call, so a placement, preemption or release
        is reflected in the very next ordering.
        """
        total = self._cluster.total_capacity()
        cpu_used: dict[str, int] = {}
        memory_used: dict[str, int] = {}
        for item in self._active(clock):
            task = self.task_index[item.task_id]
            cpu_used[task.queue] = cpu_used.get(task.queue, 0) + task.request.cpu
            memory_used[task.queue] = memory_used.get(task.queue, 0) + task.request.memory
        shares: dict[str, float] = {}
        for queue in set(cpu_used) | set(memory_used):
            cpu_share = cpu_used.get(queue, 0) / total.cpu if total.cpu else 0.0
            memory_share = memory_used.get(queue, 0) / total.memory if total.memory else 0.0
            shares[queue] = max(cpu_share, memory_share) / self._weight(queue)
        return shares

    def _fair_order(self, pending: list[Task], clock: int) -> list[Task]:
        shares = self._queue_shares(clock)
        return sorted(
            pending,
            key=lambda task: (-task.priority, shares.get(task.queue, 0.0), task.arrival, task.queue, task.id),
        )

    def _fair_fields(self, task: Task, clock: int) -> dict[str, object]:
        if not self._fair:
            return {}
        return {
            "queue": task.queue,
            "weight": self._weight(task.queue),
            "weightedDominantShare": round(self._queue_shares(clock).get(task.queue, 0.0), 6),
        }

    def _active(self, clock: int) -> list[Placement]:
        return [item for item in self._placements if item.start <= clock < item.end]

    def _running_end(self, clock: int) -> int | None:
        ends = [item.end for item in self._active(clock)]
        return min(ends) if ends else None

    def _release_finished(self, clock: int) -> None:
        """Free capacity for everything that has ended by `clock`, once per placement.

        This runs *before* placement at the same tick: capacity freed at t must be available to the
        task considered at t. The first version placed first and released afterwards, which held
        resources one tick too long.
        """
        for index, item in enumerate(self._placements):
            if index in self._released or item.end > clock:
                continue
            self._cluster.release(item.node_id, self.task_index[item.task_id].request)
            self._released.add(index)

    def _place(self, task: Task, node_id: str, clock: int, preempted: tuple[str, ...]) -> None:
        self._cluster.occupy(node_id, task.request)
        self._placements.append(Placement(task.id, node_id, clock, clock + task.duration, preempted))
        self._waits.append(clock - task.arrival)
        self._wait_by_task[task.id] = clock - task.arrival

    # -- run -------------------------------------------------------------------------------------
    def run(self) -> SimulationResult:
        clock = 0
        waiting: list[Task] = [task for task in self.tasks if task.arrival == 0]
        not_arrived: list[Task] = [task for task in self.tasks if task.arrival > 0]
        guard = 0
        limit = max(1, len(self.tasks)) * 1000 + 1000

        while (waiting or not_arrived or self._active(clock)) and guard < limit:
            guard += 1
            self._release_finished(clock)
            waiting.extend(task for task in list(not_arrived) if task.arrival <= clock)
            not_arrived = [task for task in not_arrived if task.arrival > clock]

            if not waiting and not self._active(clock) and not_arrived:
                clock = min(task.arrival for task in not_arrived)
                continue
            if not waiting and not self._active(clock):
                break

            progressed = False
            if self._fair:
                progressed = self._fair_pass(waiting, clock)
            else:
                for task in self._pending_order(waiting):
                    if self._try_place(task, clock):
                        waiting.remove(task)
                        progressed = True
                        continue
                    if self.backfill:
                        horizon = self._running_end(clock)
                        if horizon is not None and clock + task.duration <= horizon and self._try_place(task, clock, backfill=True):
                            waiting.remove(task)
                            progressed = True
                            continue
                    break  # the head of the queue is blocked; nothing behind it may jump the line

            if not self._active(clock) and not not_arrived:
                break
            candidates = [item.end for item in self._active(clock) if item.end > clock]
            candidates += [task.arrival for task in not_arrived if task.arrival > clock]
            if not candidates:
                break
            if progressed and waiting and self._running_end(clock) is not None and not not_arrived:
                # capacity is full but something is still waiting: jump straight to the next completion
                clock = min(item.end for item in self._active(clock))
            else:
                clock = min(candidates)

        placed_ids = {item.task_id for item in self._placements}
        unplaced = sorted(task.id for task in self.tasks if task.id not in placed_ids)
        for task in self._pending_order(waiting):
            self._decisions.append({"task": task.id, "reason": "left unplaced when the simulation ended"})
        makespan = max((item.end for item in self._placements), default=0)
        return SimulationResult(
            policy=self.policy,
            placements=sorted(self._placements, key=lambda item: (item.start, item.task_id)),
            decisions=self._decisions,
            unplaced=unplaced,
            makespan=makespan,
            metrics=self._metrics(makespan, unplaced),
        )

    def _fair_pass(self, waiting: list[Task], clock: int) -> bool:
        """One fair-mode scheduling round: always reconsider the head, because shares move.

        The queue order is recomputed before every attempt -- a placement, preemption or release
        changes the weighted dominant shares, and the next task must be chosen against the new
        shares. Head-of-line blocking and the conservative backfill boundary are unchanged.
        """
        progressed = False
        while waiting:
            task = self._fair_order(waiting, clock)[0]
            if self._try_place(task, clock):
                waiting.remove(task)
                progressed = True
                continue
            if self.backfill:
                horizon = self._running_end(clock)
                if horizon is not None and clock + task.duration <= horizon and self._try_place(task, clock, backfill=True):
                    waiting.remove(task)
                    progressed = True
                    continue
            break  # the head of the queue is blocked; nothing behind it may jump the line
        return progressed

    def _record_refusal(self, task: Task, reason: str, clock: int, extra: dict[str, object] | None = None) -> None:
        """Append a refusal only when it differs from the previous one for this task.

        The scheduler retries the same blocked task at every tick, so logging each attempt verbatim
        repeated one sentence seven times in the trace and buried the decisions that mattered.
        """
        previous = self._last_refusal.get(task.id)
        if previous == reason:
            return
        self._last_refusal[task.id] = reason
        decision: dict[str, object] = {"task": task.id, "reason": reason, "at": clock}
        if extra:
            decision.update(extra)
        self._decisions.append(decision)

    def _try_place(self, task: Task, clock: int, backfill: bool = False) -> bool:
        fair = self._fair_fields(task, clock)
        decision = select_node(self._cluster, task, self.policy)
        if decision.node_id is not None:
            self._place(task, decision.node_id, clock, ())
            self._decisions.append({"task": task.id, "node": decision.node_id, "reason": "backfill" if backfill else decision.reason, "at": clock, **fair})
            self._last_refusal.pop(task.id, None)
            return True
        if not self.allow_preemption:
            self._record_refusal(task, decision.reason, clock, fair)
            return False
        for node in sorted(self._cluster.nodes, key=lambda item: item.id):
            chosen, reason = preemption_candidates(self._cluster, task, node, self._placements, self.task_index, clock)
            if not chosen:
                continue
            for item in chosen:
                index = self._placements.index(item)
                self._cluster.release(item.node_id, self.task_index[item.task_id].request)
                self._placements[index] = Placement(item.task_id, item.node_id, item.start, clock, item.preempted)
                self._released.add(index)
                self._preemptions += 1
            self._place(task, node.id, clock, tuple(sorted(item.task_id for item in chosen)))
            self._decisions.append({"task": task.id, "node": node.id, "reason": reason, "at": clock, **fair})
            self._last_refusal.pop(task.id, None)
            return True
        self._record_refusal(task, decision.reason, clock, fair)
        return False

    def _metrics(self, makespan: int, unplaced: list[str]) -> dict[str, object]:
        total_cpu = self._cluster.total_capacity().cpu
        used_cpu_time = sum(self.task_index[item.task_id].request.cpu * (item.end - item.start) for item in self._placements)
        metrics: dict[str, object] = {
            "makespan": makespan,
            "placed": len(self._placements),
            "unplaced": len(unplaced),
            "preemptions": self._preemptions,
            "averageWait": round(sum(self._waits) / len(self._waits), 3) if self._waits else 0.0,
            "maxWait": max(self._waits) if self._waits else 0,
            "utilization": round(used_cpu_time / (total_cpu * makespan), 6) if total_cpu and makespan else 0.0,
            "fragmentation": fragmentation(self._cluster, {task.id: task for task in self.tasks if task.id in set(unplaced)}),
        }
        if self._fair:
            metrics["queues"] = self._queue_metrics(makespan, set(unplaced))
        return metrics

    def _queue_metrics(self, makespan: int, unplaced: set[str]) -> dict[str, object]:
        """Per-queue rollup, reported for every declared queue and every queue the tasks use.

        Resource time accumulates over actual running intervals, so a preempted task contributes
        only up to its termination tick. `dominantShare` is the same weighted-dominant measure the
        scheduler ordered by, taken over the whole makespan; a zero capacity or makespan yields 0.
        """
        total = self._cluster.total_capacity()
        report: dict[str, object] = {}
        queues = sorted(set(self._weights) | {task.queue for task in self.tasks})
        for queue in queues:
            weight = self._weight(queue)
            task_ids = {task.id for task in self.tasks if task.queue == queue}
            placements = [item for item in self._placements if item.task_id in task_ids]
            waits = [self._wait_by_task[item.task_id] for item in placements]
            cpu_time = sum(self.task_index[item.task_id].request.cpu * (item.end - item.start) for item in placements)
            memory_time = sum(self.task_index[item.task_id].request.memory * (item.end - item.start) for item in placements)
            if makespan and total.cpu and total.memory:
                dominant = max(cpu_time / (total.cpu * makespan), memory_time / (total.memory * makespan)) / weight
            else:
                dominant = 0.0
            report[queue] = {
                "weight": weight,
                "placed": len(placements),
                "unplaced": len(task_ids & unplaced),
                "averageWait": round(sum(waits) / len(waits), 3) if waits else 0.0,
                "cpuTime": cpu_time,
                "memoryTime": memory_time,
                "dominantShare": round(dominant, 6),
            }
        return report


def simulate(nodes: tuple[Node, ...], tasks: tuple[Task, ...], **options: object) -> SimulationResult:
    return Simulation(nodes=nodes, tasks=tasks, **options).run()  # type: ignore[arg-type]


def replay(nodes: tuple[Node, ...], tasks: tuple[Task, ...], **options: object) -> dict[str, object]:
    """Run twice and compare the traces: the determinism guarantee, checked rather than asserted."""
    first = simulate(nodes, tasks, **options)
    second = simulate(nodes, tasks, **options)
    identical = first.trace() == second.trace()
    return {
        "identical": identical,
        "policy": first.policy,
        "placements": len(first.placements),
        "makespan": first.makespan,
        "trace": [list(item) for item in first.trace()][:20],
        "differences": [] if identical else [list(item) for item in first.trace() if item not in second.trace()][:20],
    }


def compare_policies(
    nodes: tuple[Node, ...],
    tasks: tuple[Task, ...],
    policies: tuple[str, ...] = POLICIES,
    **options: object,
) -> dict[str, object]:
    """Same cluster, same tasks, different placement policies: metrics side by side."""
    results: list[dict[str, object]] = []
    for policy in policies:
        result = simulate(nodes, tasks, policy=policy, **options)
        entry: dict[str, object] = {
            "policy": policy,
            "makespan": result.makespan,
            "placed": len(result.placements),
            "unplaced": len(result.unplaced),
            "averageWait": result.metrics["averageWait"],
            "utilization": result.metrics["utilization"],
        }
        if "queues" in result.metrics:
            entry["queues"] = result.metrics["queues"]
        results.append(entry)
    return {"policies": results}
