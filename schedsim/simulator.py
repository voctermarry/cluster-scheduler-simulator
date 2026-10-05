"""Deterministic scheduler simulation: an event clock, conservative backfill, and metrics.

The simulator is a pure function of its input. It never consults the wall clock, never iterates a
`set` in a way that affects a decision, and never breaks a tie by insertion order -- so the same
cluster and task list always produce the same trace, and `replay` proves it instead of asserting it.

One list holds every placement ever made; "running" is derived from the clock (`start <= t < end`),
which removes any chance of the bookkeeping disagreeing with the schedule. Capacity is released
exactly once per placement, tracked by index.

With ``queue_weights`` supplied the run is in fair-share mode: priorities still order the waiting
list, but tasks of equal priority are picked from the queue with the smallest weighted dominant
share -- max(used CPU / total CPU, used memory / total memory) / weight -- and shares are
recomputed after every release, placement and preemption. Without that mapping, nothing changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .errors import ValidationError
from .model import Cluster, Node, Placement, Resources, Task
from .policies import POLICIES, fragmentation, preemption_candidates, select_node


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
    queue_weights: dict[str, int] | None = None
    _cluster: Cluster = field(init=False)
    _placements: list[Placement] = field(default_factory=list, init=False)
    _released: set[int] = field(default_factory=set, init=False)
    _decisions: list[dict[str, object]] = field(default_factory=list, init=False)
    _preemptions: int = field(default=0, init=False)
    _waits: list[int] = field(default_factory=list, init=False)
    _last_refusal: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValidationError(f"policy must be one of {', '.join(POLICIES)}", value=self.policy)
        ids = [task.id for task in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValidationError("task ids must be unique")
        if self.queue_weights is not None:
            if not self.queue_weights:
                raise ValidationError("queue weights must declare at least one queue")
            for name, weight in self.queue_weights.items():
                if not isinstance(name, str) or not name:
                    raise ValidationError("queue weights must use non-empty queue names", value=name)
                if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
                    raise ValidationError("queue weights must be positive integers", queue=name, value=weight)
        self._cluster = Cluster(self.nodes)

    # -- bookkeeping -----------------------------------------------------------------------------
    @property
    def task_index(self) -> dict[str, Task]:
        return {task.id: task for task in self.tasks}

    @property
    def fair(self) -> bool:
        return self.queue_weights is not None

    def _weight(self, queue: str) -> int:
        # A queue used by a task but absent from the config carries an implicit weight of one.
        assert self.queue_weights is not None
        return self.queue_weights.get(queue, 1)

    def _shares(self, clock: int) -> dict[str, Fraction]:
        """Weighted dominant share of every queue from the tasks running at ``clock``.

        The dominant resource is the larger of the queue's CPU and memory fractions of total cluster
        capacity; dividing by the queue weight yields its weighted dominant share. Shares are exact
        ratios for ordering -- two queues whose shares differ by less than binary floating point can
        resolve still sort by the mathematical value -- and are rounded only when they enter a trace
        document.
        """
        assert self.queue_weights is not None
        running = self._active(clock)
        used: dict[str, Resources] = {}
        for item in running:
            queue = self.task_index[item.task_id].queue
            used[queue] = used.get(queue, Resources(0, 0)).plus(self.task_index[item.task_id].request)
        capacity = self._cluster.total_capacity()
        shares: dict[str, Fraction] = {}
        for queue, request in used.items():
            cpu_fraction = Fraction(request.cpu, capacity.cpu) if capacity.cpu else Fraction(0)
            memory_fraction = Fraction(request.memory, capacity.memory) if capacity.memory else Fraction(0)
            shares[queue] = max(cpu_fraction, memory_fraction) / self._weight(queue)
        return shares

    def _pending_order(self, pending: list[Task], clock: int) -> list[Task]:
        if not self.fair:
            return sorted(pending, key=lambda task: (-task.priority, task.arrival, task.id))
        shares = self._shares(clock)
        return sorted(
            pending,
            key=lambda task: (
                -task.priority,
                shares.get(task.queue, Fraction(0)),
                task.arrival,
                task.queue,
                task.id,
            ),
        )

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
            # The head is re-picked after every attempt: a placement (or a preemption inside
            # `_try_place`) changes queue shares, so fair mode re-ranks the waiting list each time.
            # Without weights the ordering keys never change, which leaves the baseline walk identical.
            # A task already refused as head at THIS clock is never refused a second time after one of
            # its own backfills (or a fair-share re-rank) tightened the free numbers: the re-check the
            # backfill loop performs must not write a duplicate refusal.
            refused_at_clock: set[str] = set()
            while waiting:
                ordered = self._pending_order(waiting, clock)
                head = ordered[0]
                if head.id not in refused_at_clock:
                    if self._try_place(head, clock):
                        waiting.remove(head)
                        progressed = True
                        continue
                    refused_at_clock.add(head.id)
                # The head cannot be served right now. Conservative backfill looks PAST it -- at the
                # tasks behind it in this same deterministic order -- and starts the first one that
                # already fits a node and finishes by the earliest running completion, so it cannot
                # delay the head it jumped. Nothing may jump when nothing is running (the capacity
                # the head waits for cannot reappear before a completion), and --no-backfill forbids
                # jumping altogether.
                if not self.backfill:
                    break
                horizon = self._running_end(clock)
                if horizon is None:
                    break
                jumped = False
                # The order and horizon are rebuilt after every jump: shares and occupancy changed.
                for candidate in self._pending_order(waiting, clock)[1:]:
                    if clock + candidate.duration > horizon:
                        continue  # past the time boundary: not a placement and not a refusal
                    if self._try_backfill(candidate, clock):
                        waiting.remove(candidate)
                        progressed = True
                        jumped = True
                        break
                    # A failed probe changes nothing: no pseudo-placement, no logged refusal, no
                    # preemption. The scan continues with the next candidate in the same order.
                if not jumped:
                    break  # nobody behind the head qualified this tick

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
        for task in self._pending_order(waiting, clock):
            self._decisions.append(
                self._annotate(
                    {"task": task.id, "reason": "left unplaced when the simulation ended"},
                    self._fair_context(task, clock),
                )
            )
        makespan = max((item.end for item in self._placements), default=0)
        return SimulationResult(
            policy=self.policy,
            placements=sorted(self._placements, key=lambda item: (item.start, item.task_id)),
            decisions=self._decisions,
            unplaced=unplaced,
            makespan=makespan,
            metrics=self._metrics(makespan, unplaced),
        )

    def _record_refusal(self, task: Task, reason: str, clock: int, context: dict[str, object] | None = None) -> None:
        """Append a refusal only when it differs from the previous one for this task.

        The scheduler retries the same blocked task at every tick, so logging each attempt verbatim
        repeated one sentence seven times in the trace and buried the decisions that mattered.
        """
        previous = self._last_refusal.get(task.id)
        if previous == reason:
            return
        self._last_refusal[task.id] = reason
        self._decisions.append(
            self._annotate({"task": task.id, "reason": reason, "at": clock}, context or self._fair_context(task, clock))
        )

    def _fair_context(self, task: Task, clock: int) -> dict[str, object]:
        """The fair-share fields recorded *before* the decision mutates anything.

        Captured at the top of a placement attempt -- after finished tasks were released this tick
        but ahead of the occupy or preemption the decision performs -- so a placement records the
        share it was actually chosen against.
        """
        if not self.fair:
            return {}
        shares = self._shares(clock)
        return {
            "queue": task.queue,
            "weight": self._weight(task.queue),
            "weightedDominantShare": round(float(shares.get(task.queue, Fraction(0))), 6),
        }

    def _annotate(self, document: dict[str, object], context: dict[str, object]) -> dict[str, object]:
        document.update(context)
        return document

    def _try_place(self, task: Task, clock: int) -> bool:
        context = self._fair_context(task, clock)
        decision = select_node(self._cluster, task, self.policy)
        if decision.node_id is not None:
            self._place(task, decision.node_id, clock, ())
            self._decisions.append(
                self._annotate(
                    {
                        "task": task.id,
                        "node": decision.node_id,
                        "reason": decision.reason,
                        "at": clock,
                    },
                    context,
                )
            )
            self._last_refusal.pop(task.id, None)
            return True
        if not self.allow_preemption:
            self._record_refusal(task, decision.reason, clock, context)
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
            self._decisions.append(
                self._annotate({"task": task.id, "node": node.id, "reason": reason, "at": clock}, context)
            )
            self._last_refusal.pop(task.id, None)
            return True
        self._record_refusal(task, decision.reason, clock, context)
        return False

    def _try_backfill(self, task: Task, clock: int) -> bool:
        """Place a task jumping a blocked head, or fail with NO side effects.

        The caller has already checked the time boundary (``clock + duration`` must not pass the
        earliest running completion). The probe applies exactly the same node selection as a normal
        attempt -- capacity, affinity, anti-affinity, taints and the first-fit / best-fit node order
        are unchanged. A candidate that does not fit right now produces neither a placement nor a
        refusal and never triggers preemption: probing is free and invisible, so a failed candidate
        can still be placed later as an ordinary head.
        """
        decision = select_node(self._cluster, task, self.policy)
        if decision.node_id is None:
            return False
        context = self._fair_context(task, clock)
        self._place(task, decision.node_id, clock, ())
        self._decisions.append(
            self._annotate(
                {
                    "task": task.id,
                    "node": decision.node_id,
                    "reason": "backfill",
                    "at": clock,
                },
                context,
            )
        )
        self._last_refusal.pop(task.id, None)
        return True

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
        if self.fair:
            metrics["queues"] = self._queue_metrics(makespan, set(unplaced))
        return metrics

    def _queue_metrics(self, makespan: int, unplaced: set[str]) -> dict[str, object]:
        """Per-queue accounting over the actual run intervals.

        Resource time is summed over each placement's real ``[start, end)`` interval, so a preempted
        task counts only up to the tick it was evicted. The dominant share integrates the same
        intervals over the whole makespan: max(cpu-time/capacity, memory-time/memory) / makespan /
        weight. Zero capacity or zero makespan yields zero.
        """
        assert self.queue_weights is not None
        names = {task.queue for task in self.tasks}
        cpu_capacity = self._cluster.total_capacity().cpu
        memory_capacity = self._cluster.total_capacity().memory
        cpu_time: dict[str, int] = {}
        memory_time: dict[str, int] = {}
        waits: dict[str, list[int]] = {}
        for item in self._placements:
            task = self.task_index[item.task_id]
            duration = item.end - item.start
            cpu_time[task.queue] = cpu_time.get(task.queue, 0) + task.request.cpu * duration
            memory_time[task.queue] = memory_time.get(task.queue, 0) + task.request.memory * duration
            waits.setdefault(task.queue, []).append(item.start - task.arrival)
        report: dict[str, object] = {}
        for name in sorted(names):
            queue_waits = waits.get(name, [])
            cpu_share = Fraction(cpu_time.get(name, 0), cpu_capacity * makespan) if cpu_capacity and makespan else Fraction(0)
            memory_share = (
                Fraction(memory_time.get(name, 0), memory_capacity * makespan) if memory_capacity and makespan else Fraction(0)
            )
            report[name] = {
                "weight": self._weight(name),
                "placed": len(queue_waits),
                "unplaced": sum(1 for task in self.tasks if task.queue == name and task.id in unplaced),
                "averageWait": round(sum(queue_waits) / len(queue_waits), 3) if queue_waits else 0.0,
                "cpuTime": cpu_time.get(name, 0),
                "memoryTime": memory_time.get(name, 0),
                "dominantShare": round(float(max(cpu_share, memory_share) / self._weight(name)), 6),
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
