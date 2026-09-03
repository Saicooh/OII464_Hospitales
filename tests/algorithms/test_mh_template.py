from pathlib import Path

from data.instance_loader import load_instance
from simulation.result_model import SolverOutput


INSTANCE = Path(__file__).parents[2] / "instances/didactic/HOSP-DIDACT-03-01.yaml"


def test_mh_template_returns_reproducible_typed_output(monkeypatch):
    import config.config as config
    from algorithms import mh

    monkeypatch.setattr(config, "MH_POP_SIZE", 3)
    monkeypatch.setattr(config, "MH_MAX_ITERATIONS", 2)
    context = load_instance(INSTANCE)

    first = mh.run(context, seed=17)
    second = mh.run(context, seed=17)

    assert isinstance(first, SolverOutput)
    assert first == second
    assert sorted(first.solution.job_sequence_base) == [
        job.job_id for job in context.jobs
    ]
    assert first.combined_objective >= 0.0
    assert len(first.best_fitness_history) == 2


def test_loader_exposes_mh_as_an_optional_context_solver():
    from config.algorithms_loader import load_algorithms

    specs = load_algorithms(
        ga_enabled=False,
        dpso_enabled=False,
        sboa_enabled=False,
        mshoa_enabled=False,
        max_generations=1,
        max_iterations_dpso=1,
        sboa_max_iter=1,
        max_iterations_mshoa=1,
        all_rooms=["OR-1"],
        mh_enabled=True,
        mh_max_iterations=2,
    )

    assert [spec["name"] for spec in specs] == ["MH"]
    assert specs[0]["runner"].__module__ == "algorithms.mh"
    assert specs[0]["interface"] == "context"
