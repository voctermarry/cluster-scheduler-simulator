"""Placement predicates, node selection and preemption.

Predicates are pure functions returning `(ok, reason)`: a rejection that cannot say why is useless
when a schedule looks wrong, so every branch names the constraint it hit. Node selection is
deterministic -- ties always break on node id -- and preemption picks the *smallest* set of the
lowest-priority tasks that makes the incoming task fit.

Preemption buys room, never legitimacy: it searches only nodes that already satisfy the static
placement constraints (affinity, anti-affinity, taints), the same predicates ordinary placement
applies. A temporary resource shortage can be solved by an eviction; a missing or mismatched
affinity label, a present anti-affinity label or an untolerated taint cannot, so such a node is
skipped without evicting anything that runs on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ValidationError
from .model import Cluster, Node, Placement, Resources, Task

POLICIES = ("first-fit", "best-fit")


@dataclass(frozen=True, slots=True)
class Decision:
    task_id: str
    node_id: str | None
    reason: str
    preempted: tuple[str, ...] = ()

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {"task": self.task_id, "reason": self.reason}
        if self.node_id is not None:
            document["node"] = self.node_id
        if self.preempted:
            document["preempted"] = list(self.preempted)
        return document


def capacity_predicate(task: Task, free: Resources) -> tuple[bool, str]:
    if not task.request.fits(free):
        return False, f"insufficient capacity: needs cpu={task.request.cpu},memory={task.request.memory} has cpu={free.cpu},memory={free.memory}"
    return True, "capacity ok"


def affinity_predicate(task: Task, node: Node) -> tuple[bool, str]:
    labels = node.label_map()
    for key, value in task.affinity:
        if labels.get(key) != value:
            return False, f"affinity {key}={value} not satisfied (node has {key}={labels.get(key)!r})"
    for key in task.anti_affinity:
        if key in labels:
            return False, f"anti-affinity: node carries label {key}"
    return True, "affinity ok"


def taint_predicate(task: Task, node: Node) -> tuple[bool, str]:
    for taint in node.taints:
        if taint not in task.tolerations:
            return False, f"untolerated taint {taint}"
    return True, "taints ok"


def static_constraints(task: Task, node: Node) -> tuple[bool, str]:
    """The placement constraints an eviction can never change, in the documented predicate order.

    Capacity is deliberately absent: a resource shortage is exactly what preemption exists to
    resolve. Affinity, anti-affinity and taints are properties of the task and the node itself, so
    ordinary placement and a preempting placement must reach the same conclusion about them for the
    same task/node pair. The failure reason is the one ``fits`` would report on that node.
    """
    ok, reason = affinity_predicate(task, node)
    if not ok:
        return False, reason
    return taint_predicate(task, node)


PREDICATES = (capacity_predicate, affinity_predicate, taint_predicate)


def fits(cluster: Cluster, task: Task, node: Node) -> tuple[bool, str]:
    """All predicates, reported in a stable order so the first failure is always the same one."""
    free = cluster.free(node.id)
    ok, reason = capacity_predicate(task, free)
    if not ok:
        return False, reason
    return static_constraints(task, node)


def order_candidates(cluster: Cluster, policy: str) -> list[Node]:
    if policy not in POLICIES:
        raise ValidationError(f"policy must be one of {', '.join(POLICIES)}", value=policy)
    nodes = sorted(cluster.nodes, key=lambda node: node.id)
    if policy == "first-fit":
        return nodes
    # best-fit: smallest remaining CPU, then smallest remaining memory, then node id for determinism
    return sorted(nodes, key=lambda node: (cluster.free(node.id).cpu, cluster.free(node.id).memory, node.id))


def select_node(cluster: Cluster, task: Task, policy: str = "first-fit") -> Decision:
    """The first candidate that passes every predicate, in the order the policy defines."""
    reasons: list[str] = []
    for node in order_candidates(cluster, policy):
        ok, reason = fits(cluster, task, node)
        if ok:
            return Decision(task.id, node.id, f"placed by {policy}")
        reasons.append(f"{node.id}: {reason}")
    detail = "; ".join(reasons) if reasons else "no nodes in the cluster"
    return Decision(task.id, None, f"no node fits ({detail})")


def running_on(placements: list[Placement], tasks: dict[str, Task], node_id: str, now: int) -> list[Placement]:
    return [item for item in placements if item.node_id == node_id and item.start <= now < item.end and item.task_id in tasks]


def preemption_candidates(
    cluster: Cluster,
    task: Task,
    node: Node,
    placements: list[Placement],
    tasks: dict[str, Task],
    now: int,
) -> tuple[list[Placement], str]:
    """The smallest set of lower-priority running tasks whose eviction frees room for `task`.

    The static placement constraints are checked first and shared verbatim with ordinary
    placement: a node that misses an affinity label, carries an anti-affinity label or holds a
    taint the task does not tolerate returns no candidates, so the caller never evicts anything on
    it and moves on to the next compatible node -- preemption cannot buy compatibility.

    Candidates are ordered lowest priority first, then largest request, then task id, and the search
    stops as soon as the request fits -- so the answer is deterministic and is never larger than needed.
    A task is never a candidate for eviction of something at its own priority or higher.
    """
    if task.priority <= 0:
        return [], "preemption only applies to positive priorities"
    ok, reason = static_constraints(task, node)
    if not ok:
        return [], reason
    clock_free = cluster.free(node.id)
    if task.request.fits(clock_free):
        return [], "no preemption needed"
    candidates = [item for item in running_on(placements, tasks, node.id, now) if tasks[item.task_id].priority < task.priority]
    candidates.sort(key=lambda item: (tasks[item.task_id].priority, -tasks[item.task_id].request.cpu, -tasks[item.task_id].request.memory, item.task_id))
    freed = clock_free
    chosen: list[Placement] = []
    for candidate in candidates:
        chosen.append(candidate)
        freed = freed.plus(tasks[candidate.task_id].request)
        if task.request.fits(freed):
            return chosen, f"preempting {len(chosen)} lower-priority task(s) on {node.id}"
    return [], f"even evicting every lower-priority task on {node.id} would not fit cpu={task.request.cpu},memory={task.request.memory}"


def fragmentation(cluster: Cluster, tasks: dict[str, Task]) -> dict[str, int]:
    """How many nodes could not accept the smallest pending task, and their wasted CPU."""
    if not tasks:
        return {"nodesBlocked": 0, "wastedCpu": 0}
    smallest = min(tasks.values(), key=lambda item: (item.request.cpu, item.request.memory, item.id)).request
    blocked = 0
    wasted = 0
    for node in cluster.nodes:
        free = cluster.free(node.id)
        if not smallest.fits(free):
            blocked += 1
            wasted += free.cpu
    return {"nodesBlocked": blocked, "wastedCpu": wasted}
