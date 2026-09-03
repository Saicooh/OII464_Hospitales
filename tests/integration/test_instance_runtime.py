import inspect
from pathlib import Path

import pytest

from data.instance_loader import InstanceValidationError
from simulation.result_model import SimulationResult


INSTANCE = Path(__file__).parents[2] / "instances/didactic/HOSP-DIDACT-03-01.yaml"


def first_eligible_solver(context, solver_seed):
    solution = {
        "job_sequence_base": [job.job_id for job in context.jobs],
        "room_assignment": {
            job.job_id: {
                operation.operation_id: operation.eligible_rooms[solver_seed % len(operation.eligible_rooms)]
                for operation in job.operations
            }
            for job in context.jobs
        },
    }
    score = float(solver_seed)
    return score, solution, [score], [score]


def test_two_process_simulations_share_digest_and_vary_only_solver_seed(monkeypatch):
    import core.simulation_runner as runtime

    loads = 0
    original_load = runtime.load_instance

    def counted_load(path):
        nonlocal loads
        loads += 1
        return original_load(path)

    monkeypatch.setattr(runtime, "load_instance", counted_load)
    runner = runtime.SimulationRunner.from_instance(
        INSTANCE,
        first_eligible_solver,
        num_simulations=2,
        n_jobs=2,
        solver_seeds=(101, 202),
    )

    first_results = runner.run_instance_mode()
    second_results = runner.run_instance_mode()

    def stable(result):
        return (
            result.simulation_index, result.solver_seed, result.instance_digest,
            result.status, result.error, result.combined_objective,
            result.makespan, result.schedule, result.best_fitness_history,
            result.average_fitness_history, result.iterations,
        )

    assert loads == 1
    assert tuple(map(stable, first_results)) == tuple(map(stable, second_results))
    assert all(isinstance(result, SimulationResult) for result in first_results)
    assert all(result.status == "completed" for result in first_results)
    assert {result.instance_digest for result in first_results} == {runner.context.digest}
    assert [result.solver_seed for result in first_results] == [101, 202]
    assert all(len(result.schedule) == 6 for result in first_results)


@pytest.mark.parametrize("contents", ("schema_version: 999\n", None))
def test_invalid_selection_fails_before_parallel_pool(monkeypatch, tmp_path, contents):
    import core.simulation_runner as runtime

    invalid = tmp_path / "selected.yaml"
    if contents is not None:
        invalid.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(runtime, "Parallel", lambda *args, **kwargs: pytest.fail("pool created"))

    with pytest.raises(InstanceValidationError):
        runtime.SimulationRunner.from_instance(invalid, first_eligible_solver)


def test_supported_worker_has_no_runtime_data_selection_path():
    from simulation.workers.elective_worker import InstanceWorker

    source = inspect.getsource(InstanceWorker).lower()
    assert "load_instance" not in source
    assert "instancecontext" in source


def test_instance_runtime_produces_no_generic_dicts_and_preserves_metadata():
    import core.simulation_runner as runtime
    from simulation.result_model import ScheduleEntry

    runner = runtime.SimulationRunner.from_instance(
        INSTANCE,
        first_eligible_solver,
        num_simulations=2,
        n_jobs=1,
        solver_seeds=(11, 22),
    )
    results = runner.run_instance_mode()
    assert len(results) == 2
    for res in results:
        assert isinstance(res, SimulationResult)
        assert res.instance_id == runner.context.instance_id
        assert res.instance_digest == runner.context.digest
        assert res.status == "completed"
        assert res.error is None
        assert isinstance(res.schedule, tuple)
        assert len(res.schedule) > 0
        for entry in res.schedule:
            assert isinstance(entry, ScheduleEntry)
            assert not isinstance(entry, dict)
        assert isinstance(res.best_fitness_history, tuple)
        assert isinstance(res.average_fitness_history, tuple)
        assert isinstance(res.iterations, tuple)


def test_instance_runtime_accepts_typed_schedule_solution_and_solver_output():
    import core.simulation_runner as runtime
    from simulation.result_model import ScheduleSolution, SolverOutput

    def typed_solver(context, solver_seed):
        solution = ScheduleSolution(
            job_sequence_base=tuple(job.job_id for job in context.jobs),
            room_assignment={
                job.job_id: {
                    operation.operation_id: operation.eligible_rooms[0]
                    for operation in job.operations
                }
                for job in context.jobs
            },
            personnel_assignment={
                job.job_id: {
                    operation.operation_id: operation.eligible_personnel[0]
                    for operation in job.operations
                }
                for job in context.jobs
            },
        )
        return SolverOutput(
            combined_objective=10.0 + solver_seed,
            solution=solution,
            best_fitness_history=(10.0 + solver_seed,),
            average_fitness_history=(12.0 + solver_seed,),
        )

    runner = runtime.SimulationRunner.from_instance(
        INSTANCE,
        typed_solver,
        num_simulations=2,
        n_jobs=1,
        solver_seeds=(42, 43),
    )
    results = runner.run_instance_mode()
    assert len(results) == 2
    assert all(res.status == "completed" for res in results)
    assert results[0].solver_seed == 42
    assert results[1].solver_seed == 43
    assert results[0].combined_objective == 52.0
    assert results[1].combined_objective == 53.0
    assert results[0].instance_digest == runner.context.digest
