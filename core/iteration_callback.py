"""Typed iteration telemetry shared by the solver and run writer."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class IterationSnapshot:
    """Serializable scalar metrics captured after one solver step."""

    algo_step: int
    best_fitness: float
    best_makespan: float
    iteration_fitness: float
    iteration_makespan: float
    best_solution_snapshot: Optional[dict] = None


@runtime_checkable
class IterationCallback(Protocol):
    """Callback contract for solver iteration telemetry."""

    def __call__(
        self,
        algo_step: int,
        best_fitness: float,
        best_makespan: float,
        iteration_fitness: float,
        iteration_makespan: float,
        best_solution_snapshot: Optional[dict] = None,
    ) -> None:
        ...


def serialize_solution(solution: Any) -> Optional[dict]:
    """Return a native, detached representation of a candidate solution."""
    if solution is None or not isinstance(solution, dict):
        return None
    raw = copy.deepcopy(solution)
    if "job_sequence_base" in raw:
        raw["job_sequence_base"] = [int(job_id) for job_id in raw["job_sequence_base"]]
    if "room_assignment" in raw:
        normalized: dict[int, dict[int, str]] = {}
        for job_id, operations in raw["room_assignment"].items():
            normalized[int(job_id)] = {
                int(operation_id): str(room)
                for operation_id, room in operations.items()
            }
        raw["room_assignment"] = normalized
    return raw


__all__ = ["IterationCallback", "IterationSnapshot", "serialize_solution"]
