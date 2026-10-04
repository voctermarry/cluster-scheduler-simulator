"""Deterministic cluster scheduling simulation: resources, predicates, preemption, metrics."""

from .errors import OutputError, ParseError, SchedulerError, SchedulingError, ValidationError
from .model import Cluster, Node, Placement, Resources, Task
from .policies import (
    POLICIES,
    Decision,
    affinity_predicate,
    capacity_predicate,
    fits,
    fragmentation,
    order_candidates,
    preemption_candidates,
    select_node,
    taint_predicate,
)
from .simulator import Simulation, SimulationResult, compare_policies, replay, simulate

__all__ = [
    "Cluster",
    "Decision",
    "Node",
    "OutputError",
    "POLICIES",
    "ParseError",
    "Placement",
    "Resources",
    "SchedulerError",
    "SchedulingError",
    "Simulation",
    "SimulationResult",
    "Task",
    "ValidationError",
    "affinity_predicate",
    "capacity_predicate",
    "compare_policies",
    "fits",
    "fragmentation",
    "order_candidates",
    "preemption_candidates",
    "replay",
    "select_node",
    "simulate",
    "taint_predicate",
]

__version__ = "0.1.0"
