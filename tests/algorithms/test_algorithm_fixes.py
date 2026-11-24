"""Regression tests for implementation errors found in the historical solvers."""

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import algorithms.ga as ga
import algorithms.sboa as sboa
from algorithms._eligibility import eligible_room_table
from core.legacy_runner import LegacyInstanceData
from data.instance_loader import load_instance
from utils import statistics

ROOT = Path(__file__).resolve().parents[2]
INSTANCE = ROOT / "instances/standard/HOSP-STD-15-01.yaml"


def _restricted_context():
    """Instance where every operation may use only the first two rooms."""
    context = load_instance(INSTANCE)
    allowed = context.rooms[:2]
    jobs = tuple(
        replace(job, operations=tuple(replace(op, eligible_rooms=allowed) for op in job.operations))
        for job in context.jobs
    )
    return replace(context, jobs=jobs), set(allowed)


@pytest.mark.parametrize("seed", range(20))
def test_order_crossover_returns_a_permutation_with_the_kept_segment_in_place(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 30)
    donor = rng.sample(range(n), n)
    other = rng.sample(range(n), n)
    p1, p2 = sorted(rng.sample(range(n), 2))
    child = ga.order_crossover(donor, other, p1, p2)
    assert sorted(child) == list(range(n))
    assert child[p1 : p2 + 1] == donor[p1 : p2 + 1]


def test_ga_individuals_and_offspring_keep_a_single_permutation(monkeypatch):
    context = load_instance(INSTANCE)
    jobs = [job.job_id for job in context.jobs]
    monkeypatch.setattr(ga, "PABELLONES", list(context.rooms))
    random.seed(0)
    population = [ga.create_individual(jobs) for _ in range(10)]
    for _ in range(200):
        a, b = random.sample(population, 2)
        for child in ga.crossover(a, b):
            child = ga.mutate(child)
            assert sorted(child["job_sequence_base"]) == sorted(jobs)


def test_ga_respects_room_eligibility(monkeypatch):
    context, allowed = _restricted_context()
    data = LegacyInstanceData(context)
    jobs = [job.job_id for job in context.jobs]
    monkeypatch.setattr(ga, "PABELLONES", list(context.rooms))
    eligible = eligible_room_table(data, list(context.rooms))
    random.seed(1)
    for _ in range(20):
        individual = ga.mutate(ga.create_individual(jobs, eligible), eligible)
        rooms = {room for ops in individual["room_assignment"].values() for room in ops.values()}
        assert rooms <= allowed


def test_sboa_decoding_is_deterministic_and_respects_eligibility(monkeypatch):
    context, allowed = _restricted_context()
    data = LegacyInstanceData(context)
    jobs = [job.job_id for job in context.jobs]
    monkeypatch.setattr(sboa, "PABELLONES", list(context.rooms))
    eligible = eligible_room_table(data, list(context.rooms))
    rng = np.random.default_rng(0)
    for _ in range(50):
        position = rng.uniform(-5, 5, 3 * len(jobs))
        position[len(jobs):] = rng.uniform(-5, -4.9, 2 * len(jobs))  # concentrated rooms
        first = sboa.discretize_bird(position, jobs, eligible=eligible)
        second = sboa.discretize_bird(position, jobs, eligible=eligible)
        assert first == second
        rooms = {room for ops in first["room_assignment"].values() for room in ops.values()}
        assert rooms <= allowed


def test_independent_test_matches_mann_whitney_and_holm():
    rng = np.random.default_rng(3)
    results = {
        "A": {"makespan": list(rng.normal(100, 5, 20))},
        "B": {"makespan": list(rng.normal(105, 5, 20))},
        "C": {"makespan": list(rng.normal(100, 5, 20)) + [float("inf")]},
    }
    out = statistics.perform_independent_statistical_test(results, 0.05, verbose=False)
    assert out["n_excluded_by_algorithm"]["C"] == 1
    assert out["kruskal"]["p_value"] == pytest.approx(
        stats.kruskal(*[np.array(r["makespan"])[np.isfinite(r["makespan"])] for r in results.values()]).pvalue
    )
    pair = next(p for p in out["pairwise"] if (p["algo_a"], p["algo_b"]) == ("A", "B"))
    assert pair["p_value"] == pytest.approx(
        stats.mannwhitneyu(results["A"]["makespan"], results["B"]["makespan"]).pvalue
    )
    raw = sorted(p["p_value"] for p in out["pairwise"])
    assert min(p["p_adjusted"] for p in out["pairwise"]) == pytest.approx(min(1.0, 3 * raw[0]))


def test_runner_marks_infeasible_solver_output_as_failed(monkeypatch):
    from core import legacy_runner
    from simulation.result_model import ScheduleSolution, SolverOutput

    context = load_instance(INSTANCE)
    solution = {
        "job_sequence_base": [job.job_id for job in context.jobs],
        "room_assignment": {
            job.job_id: {op.operation_id: op.eligible_rooms[0] for op in job.operations}
            for job in context.jobs
        },
    }

    def infeasible_solver(ctx, seed):
        return SolverOutput(
            combined_objective=float("inf"),
            solution=ScheduleSolution.from_dict(solution),
            best_fitness_history=(),
            average_fitness_history=(),
        )

    spec = {"name": "X", "runner": infeasible_solver, "interface": "context"}
    _, results = legacy_runner._run_one_simulation(context, (spec,), 0)
    assert results["X"]["makespan"] == float("inf")
    assert "error" in results["X"]


@pytest.mark.parametrize("module_name", ["ga", "dpso", "sboa", "dmshoa", "mh"])
def test_every_algorithm_stops_at_the_same_evaluation_budget(monkeypatch, module_name):
    """Count only the charged evaluations (bookkeeping re-decodes of the best
    solution are excluded) and check that each algorithm spends exactly the
    configured budget."""
    import importlib

    import config.config as config

    context = load_instance(INSTANCE)
    module = importlib.import_module(f"algorithms.{module_name}")
    monkeypatch.setattr(config, "EVALUATION_BUDGET", 240)
    calls = {"n": 0}

    if module_name in ("dmshoa", "mh"):
        import simulation.scheduler as scheduler

        original = scheduler.objective_from

        def counting(decoded):
            calls["n"] += 1
            return original(decoded)

        monkeypatch.setattr(scheduler, "objective_from", counting)
        if module_name == "mh":
            monkeypatch.setattr(module, "objective_from", counting)
        output = module.run(context, 1)
        objective = output.combined_objective
    else:
        original = module.calculate_schedule_fitness

        def counting(solution, data, return_details=False):
            if not return_details:
                calls["n"] += 1
            return original(solution, data, return_details=return_details)

        monkeypatch.setattr(module, "calculate_schedule_fitness", counting)
        objective = module.run(context, 1)[0]

    assert objective < float("inf")
    assert calls["n"] == 240
