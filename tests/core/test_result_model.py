import pickle
from dataclasses import FrozenInstanceError, fields

import pytest
from simulation.result_model import (
    IterationSnapshot,
    ScheduleEntry,
    ScheduleSolution,
    SimulationResult,
    SolverOutput,
)


def test_simulation_result_has_frozen_typed_contract_and_round_trips():
    entry = ScheduleEntry(1, 1, "R1", "A1", 0.0, 10.0, 10.0, 0.0, 0.0, 0.0)
    iteration = IterationSnapshot(1, 30.0, 25.0, 31.0, 26.0)
    result = SimulationResult(
        0, 101, "HOSP-DIDACT-03-01", "a" * 64, "completed", None,
        30.0, 25.0, (entry,), 0.5, 0.7, (30.0,), (32.0,), (iteration,),
    )
    assert tuple(field.name for field in fields(result)) == (
        "simulation_index", "solver_seed", "instance_id", "instance_digest",
        "status", "error", "combined_objective", "makespan", "schedule",
        "algorithm_seconds", "wall_clock_seconds", "best_fitness_history",
        "average_fitness_history", "iterations",
    )
    assert pickle.loads(pickle.dumps(result)) == result
    assert result.schedule[0].personnel == "A1"
    with pytest.raises(FrozenInstanceError):
        result.status = "failed"


def test_simulation_result_failed_status_contract():
    result = SimulationResult(
        0, 101, "HOSP-DIDACT-03-01", "b" * 64, "failed", "Infeasible assignment",
        float("inf"), float("inf"), (), 0.01, 0.01, (), (), (),
    )
    assert result.status == "failed"
    assert result.error == "Infeasible assignment"
    assert result.is_failed is True
    assert result.is_completed is False
    assert result.schedule == ()
    assert pickle.loads(pickle.dumps(result)) == result
    with pytest.raises(FrozenInstanceError):
        result.error = "other"


def test_simulation_result_rejects_generic_dicts_and_untyped_schedule():
    # schedule must not contain raw dicts
    with pytest.raises(TypeError, match="ScheduleEntry"):
        SimulationResult(
            0, 101, "HOSP-DIDACT-03-01", "c" * 64, "completed", None,
            10.0, 10.0, ({"job_id": 1},), 0.1, 0.1, (), (), (),
        )

    # status must be 'completed' or 'failed'
    with pytest.raises(ValueError, match="status"):
        SimulationResult(
            0, 101, "HOSP-DIDACT-03-01", "c" * 64, "unknown", None,
            10.0, 10.0, (), 0.1, 0.1, (), (), (),
        )

    # completed result must have error=None
    with pytest.raises(ValueError, match="error=None"):
        SimulationResult(
            0, 101, "HOSP-DIDACT-03-01", "c" * 64, "completed", "error occurred",
            10.0, 10.0, (), 0.1, 0.1, (), (), (),
        )

    # failed result must have non-None error
    with pytest.raises(ValueError, match="error message"):
        SimulationResult(
            0, 101, "HOSP-DIDACT-03-01", "c" * 64, "failed", None,
            float("inf"), float("inf"), (), 0.1, 0.1, (), (), (),
        )


def test_simulation_result_from_context_metadata_propagation():
    class DummyContext:
        instance_id = "HOSP-DIDACT-03-01"
        digest = "d" * 64

    ctx = DummyContext()
    entry = ScheduleEntry(1, 1, "OR-1", "AN-1", 0.0, 10.0, 10.0, 0.0, 0.0, 0.0)
    success = SimulationResult.from_context(
        context=ctx,
        simulation_index=1,
        solver_seed=202,
        status="completed",
        combined_objective=42.0,
        makespan=40.0,
        schedule=(entry,),
    )
    assert success.instance_id == "HOSP-DIDACT-03-01"
    assert success.instance_digest == "d" * 64
    assert success.simulation_index == 1
    assert success.solver_seed == 202
    assert success.is_completed is True

    failure = SimulationResult.from_context(
        context=ctx,
        simulation_index=2,
        solver_seed=303,
        status="failed",
        error="Resource exhausted",
    )
    assert failure.instance_id == "HOSP-DIDACT-03-01"
    assert failure.instance_digest == "d" * 64
    assert failure.error == "Resource exhausted"
    assert failure.is_failed is True


def test_schedule_entry_invariants_and_immutability():
    entry = ScheduleEntry(1, 2, "OR-2", "SU-1", 10.0, 20.0, 22.0, 1.0, 0.5, 2.0)
    assert entry.room == "OR-2"
    assert entry.personnel == "SU-1"
    with pytest.raises(FrozenInstanceError):
        entry.room = "OR-3"

    with pytest.raises(TypeError):
        ScheduleEntry("1", 2, "OR-2", "SU-1", 10.0, 20.0, 22.0, 1.0, 0.5, 2.0)

    with pytest.raises(ValueError, match="non-negative"):
        ScheduleEntry(1, 2, "OR-2", "SU-1", -1.0, 20.0, 22.0, 1.0, 0.5, 2.0)


def test_schedule_solution_typed_boundary():
    solution = ScheduleSolution(
        job_sequence_base=(1, 2, 3),
        room_assignment={1: {1: "OR-1", 2: "OR-1"}},
        personnel_assignment={1: {1: "AN-1", 2: "SU-1"}},
    )
    assert solution.job_sequence_base == (1, 2, 3)
    dict_repr = solution.to_dict()
    assert dict_repr["job_sequence_base"] == [1, 2, 3]
    restored = ScheduleSolution.from_dict(dict_repr)
    assert restored.job_sequence_base == (1, 2, 3)
    assert restored.room_assignment == {1: {1: "OR-1", 2: "OR-1"}}


def test_schedule_solution_deep_immutability_and_pickle():
    from types import MappingProxyType

    solution = ScheduleSolution(
        job_sequence_base=[1, 2],
        room_assignment={1: {1: "OR-1"}},
        personnel_assignment={1: {1: "AN-1"}},
    )
    assert isinstance(solution.room_assignment, MappingProxyType)
    assert isinstance(solution.room_assignment[1], MappingProxyType)
    assert isinstance(solution.personnel_assignment, MappingProxyType)
    assert isinstance(solution.personnel_assignment[1], MappingProxyType)

    with pytest.raises(TypeError):
        solution.room_assignment[1][1] = "OR-2"

    with pytest.raises(TypeError):
        solution.room_assignment[1] = {}

    if solution.personnel_assignment is not None:
        with pytest.raises(TypeError):
            solution.personnel_assignment[1][1] = "SU-1"

    unpickled = pickle.loads(pickle.dumps(solution))
    assert unpickled == solution
    assert isinstance(unpickled.room_assignment, MappingProxyType)
    assert isinstance(unpickled.room_assignment[1], MappingProxyType)


def test_solver_output_narrow_type_and_auto_conversion():
    from types import MappingProxyType

    # Auto-conversion from dict
    raw_dict_solution = {
        "job_sequence_base": (1, 2),
        "room_assignment": {1: {1: "OR-1"}},
    }
    output_from_dict = SolverOutput(
        combined_objective=35.0,
        solution=raw_dict_solution,
        best_fitness_history=[35.0],
    )
    assert isinstance(output_from_dict.solution, ScheduleSolution)
    assert isinstance(output_from_dict.solution.room_assignment, MappingProxyType)
    assert output_from_dict.solution.room_assignment[1][1] == "OR-1"
    assert output_from_dict.best_fitness_history == (35.0,)

    # Already ScheduleSolution
    solution = ScheduleSolution(
        job_sequence_base=(1,),
        room_assignment={1: {1: "OR-1"}},
    )
    output_typed = SolverOutput(
        combined_objective=42.0,
        solution=solution,
    )
    assert output_typed.solution is solution
    assert pickle.loads(pickle.dumps(output_typed)) == output_typed

    # Invalid solution type rejected
    with pytest.raises(TypeError, match="ScheduleSolution"):
        SolverOutput(combined_objective=1.0, solution="invalid_solution")


