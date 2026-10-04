"""Resource model: capacities, requests, nodes, tasks and placements.

Everything is integer arithmetic on CPU units and memory units. Tasks and nodes are validated at
construction, so the simulator never has to defend against negative requests halfway through a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ValidationError


@dataclass(frozen=True, slots=True)
class Resources:
    cpu: int
    memory: int

    def __post_init__(self) -> None:
        if self.cpu < 0 or self.memory < 0:
            raise ValidationError("resources must be non-negative", cpu=self.cpu, memory=self.memory)

    def fits(self, available: "Resources") -> bool:
        return self.cpu <= available.cpu and self.memory <= available.memory

    def plus(self, other: "Resources") -> "Resources":
        return Resources(self.cpu + other.cpu, self.memory + other.memory)

    def minus(self, other: "Resources") -> "Resources":
        return Resources(max(0, self.cpu - other.cpu), max(0, self.memory - other.memory))

    def to_document(self) -> dict[str, int]:
        return {"cpu": self.cpu, "memory": self.memory}


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    request: Resources
    priority: int = 0
    queue: str = "default"
    arrival: int = 0
    duration: int = 1
    affinity: tuple[tuple[str, str], ...] = ()
    anti_affinity: tuple[str, ...] = ()
    tolerations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValidationError("task id must be non-empty")
        if self.duration <= 0:
            raise ValidationError("duration must be > 0", task=self.id, value=self.duration)
        if self.arrival < 0:
            raise ValidationError("arrival must be >= 0", task=self.id, value=self.arrival)
        if self.request.cpu == 0 and self.request.memory == 0:
            raise ValidationError("a task must request something", task=self.id)

    def affinity_map(self) -> dict[str, str]:
        return dict(self.affinity)

    def to_document(self) -> dict[str, object]:
        return {
            "id": self.id,
            "queue": self.queue,
            "priority": self.priority,
            "arrival": self.arrival,
            "duration": self.duration,
            "request": self.request.to_document(),
            "affinity": dict(self.affinity),
            "antiAffinity": list(self.anti_affinity),
            "tolerations": list(self.tolerations),
        }


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    capacity: Resources
    labels: tuple[tuple[str, str], ...] = ()
    taints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValidationError("node id must be non-empty")

    def label_map(self) -> dict[str, str]:
        return dict(self.labels)

    def to_document(self) -> dict[str, object]:
        return {
            "id": self.id,
            "capacity": self.capacity.to_document(),
            "labels": dict(self.labels),
            "taints": list(self.taints),
        }


@dataclass(frozen=True, slots=True)
class Placement:
    task_id: str
    node_id: str
    start: int
    end: int
    preempted: tuple[str, ...] = ()

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {"task": self.task_id, "node": self.node_id, "start": self.start, "end": self.end}
        if self.preempted:
            document["preempted"] = list(self.preempted)
        return document


@dataclass(slots=True)
class Cluster:
    nodes: tuple[Node, ...]
    _used: dict[str, Resources] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        ids = [node.id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValidationError("node ids must be unique", ids=ids)
        for node in self.nodes:
            self._used[node.id] = Resources(0, 0)

    def node(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise ValidationError(f"unknown node: {node_id}", known=[node.id for node in self.nodes])

    def used(self, node_id: str) -> Resources:
        self.node(node_id)
        return self._used[node_id]

    def free(self, node_id: str) -> Resources:
        return self.node(node_id).capacity.minus(self.used(node_id))

    def occupy(self, node_id: str, resources: Resources) -> None:
        self.node(node_id)
        self._used[node_id] = self._used[node_id].plus(resources)

    def release(self, node_id: str, resources: Resources) -> None:
        self.node(node_id)
        self._used[node_id] = self._used[node_id].minus(resources)

    def total_capacity(self) -> Resources:
        total = Resources(0, 0)
        for node in self.nodes:
            total = total.plus(node.capacity)
        return total

    def total_used(self) -> Resources:
        total = Resources(0, 0)
        for node in self.nodes:
            total = total.plus(self._used[node.id])
        return total
