from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping
from core.iteration_callback import IterationSnapshot


def _deep_freeze_mapping(data: Mapping[Any, Any] | None) -> MappingProxyType | None:
    """Recursively freeze mappings into read-only MappingProxyType structures."""
    if data is None:
        return None
    frozen: dict[Any, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            frozen[key] = _deep_freeze_mapping(value)
        else:
            frozen[key] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class ScheduleEntry:
    job_id: int
    operation: int
    room: str
    personnel: str
    start: float
    processing_end: float
    finish: float
    setup: float
    transition: float
    cleanup: float

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, int):
            raise TypeError(f"job_id must be int, got {type(self.job_id)}")
        if not isinstance(self.operation, int):
            raise TypeError(f"operation must be int, got {type(self.operation)}")
        if not isinstance(self.room, str) or not self.room:
            raise ValueError(f"room must be non-empty str, got {self.room!r}")
        if not isinstance(self.personnel, str) or not self.personnel:
            raise ValueError(f"personnel must be non-empty str, got {self.personnel!r}")
        for field_name in ("start", "processing_end", "finish", "setup", "transition", "cleanup"):
            val = getattr(self, field_name)
            if not isinstance(val, (int, float)):
                raise TypeError(f"{field_name} must be numeric, got {type(val)}")
            if val < 0.0:
                raise ValueError(f"{field_name} must be non-negative, got {val}")


@dataclass(frozen=True, slots=True)
class ScheduleSolution:
    """Deeply immutable candidate schedule solution replacing generic dicts."""
    job_sequence_base: tuple[int, ...]
    room_assignment: Mapping[int, Mapping[int, str]]
    personnel_assignment: Mapping[int, Mapping[int, str]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_sequence_base, tuple):
            object.__setattr__(self, "job_sequence_base", tuple(self.job_sequence_base))
        object.__setattr__(self, "room_assignment", _deep_freeze_mapping(self.room_assignment))
        if self.personnel_assignment is not None:
            object.__setattr__(self, "personnel_assignment", _deep_freeze_mapping(self.personnel_assignment))

    def __reduce__(self) -> tuple[Any, ...]:
        personnel = (
            {k: dict(v) for k, v in self.personnel_assignment.items()}
            if self.personnel_assignment is not None
            else None
        )
        return (
            ScheduleSolution,
            (
                self.job_sequence_base,
                {k: dict(v) for k, v in self.room_assignment.items()},
                personnel,
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduleSolution":
        return cls(
            job_sequence_base=tuple(data.get("job_sequence_base", ())),
            room_assignment=dict(data.get("room_assignment", {})),
            personnel_assignment=dict(data["personnel_assignment"]) if data.get("personnel_assignment") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "job_sequence_base": list(self.job_sequence_base),
            "room_assignment": {k: dict(v) for k, v in self.room_assignment.items()},
        }
        if self.personnel_assignment is not None:
            res["personnel_assignment"] = {k: dict(v) for k, v in self.personnel_assignment.items()}
        return res


@dataclass(frozen=True, slots=True)
class SolverOutput:
    """Typed output from an instance solver execution with auto-converted solution."""
    combined_objective: float
    solution: ScheduleSolution
    best_fitness_history: tuple[float, ...] = ()
    average_fitness_history: tuple[float, ...] = ()
    iterations: tuple[IterationSnapshot, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.solution, Mapping):
            object.__setattr__(self, "solution", ScheduleSolution.from_dict(self.solution))
        elif not isinstance(self.solution, ScheduleSolution):
            raise TypeError(f"solution must be ScheduleSolution or Mapping, got {type(self.solution)}")
        if not isinstance(self.best_fitness_history, tuple):
            object.__setattr__(self, "best_fitness_history", tuple(float(x) for x in self.best_fitness_history))
        if not isinstance(self.average_fitness_history, tuple):
            object.__setattr__(self, "average_fitness_history", tuple(float(x) for x in self.average_fitness_history))
        if not isinstance(self.iterations, tuple):
            object.__setattr__(self, "iterations", tuple(self.iterations))
        for item in self.iterations:
            if not isinstance(item, IterationSnapshot):
                raise TypeError(f"All iteration items must be IterationSnapshot instances, got {type(item)}")


@dataclass(frozen=True, slots=True)
class SimulationResult:
    simulation_index: int
    solver_seed: int
    instance_id: str
    instance_digest: str
    status: Literal["completed", "failed"]
    error: str | None
    combined_objective: float
    makespan: float
    schedule: tuple[ScheduleEntry, ...]
    algorithm_seconds: float
    wall_clock_seconds: float
    best_fitness_history: tuple[float, ...]
    average_fitness_history: tuple[float, ...]
    iterations: tuple[IterationSnapshot, ...]

    def __post_init__(self) -> None:
        if self.status not in ("completed", "failed"):
            raise ValueError(f"status must be 'completed' or 'failed', got {self.status!r}")
        if not isinstance(self.instance_id, str) or not self.instance_id:
            raise ValueError("instance_id must be a non-empty string")
        if not isinstance(self.instance_digest, str) or len(self.instance_digest) != 64:
            raise ValueError("instance_digest must be a 64-character SHA-256 hex string")
        if not isinstance(self.schedule, tuple):
            object.__setattr__(self, "schedule", tuple(self.schedule))
        for schedule_entry in self.schedule:
            if not isinstance(schedule_entry, ScheduleEntry):
                raise TypeError(
                    "All schedule items must be ScheduleEntry instances, "
                    f"got {type(schedule_entry)}"
                )
        if not isinstance(self.best_fitness_history, tuple):
            object.__setattr__(self, "best_fitness_history", tuple(float(x) for x in self.best_fitness_history))
        if not isinstance(self.average_fitness_history, tuple):
            object.__setattr__(self, "average_fitness_history", tuple(float(x) for x in self.average_fitness_history))
        if not isinstance(self.iterations, tuple):
            object.__setattr__(self, "iterations", tuple(self.iterations))
        for snapshot in self.iterations:
            if not isinstance(snapshot, IterationSnapshot):
                raise TypeError(
                    "All iteration items must be IterationSnapshot instances, "
                    f"got {type(snapshot)}"
                )
        if self.status == "completed" and self.error is not None:
            raise ValueError("Completed simulation result must have error=None")
        if self.status == "failed" and self.error is None:
            raise ValueError("Failed simulation result must have an error message")

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @classmethod
    def from_context(
        cls,
        context: Any,
        simulation_index: int,
        solver_seed: int,
        status: Literal["completed", "failed"],
        combined_objective: float = float("inf"),
        makespan: float = float("inf"),
        schedule: tuple[ScheduleEntry, ...] = (),
        algorithm_seconds: float = 0.0,
        wall_clock_seconds: float = 0.0,
        best_fitness_history: tuple[float, ...] = (),
        average_fitness_history: tuple[float, ...] = (),
        iterations: tuple[IterationSnapshot, ...] = (),
        error: str | None = None,
    ) -> "SimulationResult":
        """Create a SimulationResult propagating instance identity and digest from an InstanceContext."""
        return cls(
            simulation_index=simulation_index,
            solver_seed=solver_seed,
            instance_id=context.instance_id,
            instance_digest=context.digest,
            status=status,
            error=error,
            combined_objective=combined_objective,
            makespan=makespan,
            schedule=schedule,
            algorithm_seconds=algorithm_seconds,
            wall_clock_seconds=wall_clock_seconds,
            best_fitness_history=best_fitness_history,
            average_fitness_history=average_fitness_history,
            iterations=iterations,
        )
