from pathlib import Path

from data.instance_loader import load_instance
from simulation.result_model import SolverOutput


INSTANCE = Path(__file__).parents[2] / "instances/didactic/HOSP-DIDACT-03-01.yaml"


def test_loader_exposes_one_supported_dmshoa_runner():
    from algorithms.dmshoa import run

    assert run.__module__ == "algorithms.dmshoa"


def test_context_solver_is_seeded_and_returns_typed_output(monkeypatch):
    import config.config as config
    from algorithms import dmshoa

    monkeypatch.setattr(config, "MSHOA_POP_SIZE", 4)
    monkeypatch.setattr(config, "MAX_ITERATIONS_MSHOA", 4)
    context = load_instance(INSTANCE)

    first = dmshoa.run(context, seed=7)
    second = dmshoa.run(context, seed=7)

    assert isinstance(first, SolverOutput)
    assert first == second
    assert sorted(first.solution.job_sequence_base) == [job.job_id for job in context.jobs]
    assert first.combined_objective >= 0.0
    assert len(first.best_fitness_history) == 4
