"""Placement predicates, node selection and preemption.

Predicates are pure functions returning `(ok, reason)`: a rejection that cannot say why is useless
when a schedule looks wrong, so every branch names the constraint it hit. Node selection is
deterministic -- ties always break on node id -- and preemption picks the *smallest* set of the
lowest-priority tasks that makes the incoming task fit.
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


PREDICATES = (capacity_predicate, affinity_predicate, taint_predicate)


def fits(cluster: Cluster, task: Task, node: Node) -> tuple[bool, str]:
    """All predicates, reported in a stable order so the first failure is always the same one."""
    free = cluster.free(node.id)
    ok, reason = capacity_predicate(task, free)
    if not ok:
        return False, reason
    ok, reason = affinity_predicate(task, node)
    if not ok:
        return False, reason
    return taint_predicate(task, node)


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

    Candidates are ordered lowest priority first, then largest request, then task id, and the search
    stops as soon as the request fits -- so the answer is deterministic and is never larger than needed.
    A task is never a candidate for eviction of something at its own priority or higher.
    """
    if task.priority <= 0:
        return [], "preemption only applies to positive priorities"
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
