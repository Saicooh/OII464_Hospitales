"""
Tests for on_iteration callback instrumentation in all algorithms.

Covers tasks 3.1 and 3.2:
- GA, DPSO, DMSHOA, SBOA must accept optional `on_iteration` parameter in `run()`.
- Callback is invoked once per generation/iteration of the main loop.
- Callback receives (algo_step, best_fitness, combined_obj=None).
- Normal mode (no callback) behavior is completely unchanged.
"""

import os
import yaml
import sys
import pytest


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _make_minimal_config(num_procedures=5):
    return {
        "experiment": {
            "num_simulations": 1,
            "std_factor_times": 0.0,
            "alpha_test": 0.05,
            "num_procedures": num_procedures,
            "output_dirs": {"plots": "results/plots", "csv": "results/csv"},
        },
        "real_data": {"enabled": False},
        "logging": {"verbose_mode": False},
        "times": {
            "setup": {"1": 10, "2": 10, "3": 10},
            "cleanup": {"1": 5, "2": 5, "3": 5},
            "max_wait": {"1": 1000, "2": 1000},
        },
        "jobs": {"types": {"1": 1, "2": 2, "3": 3, "4": 1, "5": 2}},
        "resources": {"num_pabellones": 2},
        "personnel": {
            "num_anesthesiologists": 1,
            "num_surgeons": 1,
        },
        "algorithms": {
            "alpha": 1e-6,
            "beta": 0.7,
            "gamma": 1.4,
            "delta": 100.0,
            "ga": {
                "enabled": True,
                "population_size": 2,
                "max_generations": 3,  # 3 generations for counting
                "crossover_probability": 0.8,
                "mutation_probability": 0.3,
                "elitism_count": 1,
            },
            "dpso": {
                "enabled": True,
                "swarm_size": 2,
                "max_iterations": 3,
                "w": 0.7,
                "c1": 1.5,
                "c2": 1.5,
                "vel_high": 4.0,
                "vel_low": -4.0,
            },
            "sboa": {
                "enabled": True,
                "population_size": 3,  # Must be >= 3 for SBOA's choose-2-excluding-self
                "max_iterations": 3,
                "lower_bound": -5.0,
                "upper_bound": 5.0,
            },
            "dmshoa": {
                "enabled": True,
                "population_size": 2,
                "max_iterations": 3,
                "k": 0.3,
                "lower_bound": -5.0,
                "upper_bound": 5.0,
            },
        },
    }


def _setup_env(tmp_path):
    cfg = _make_minimal_config()
    cfg_file = tmp_path / "config_test.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    _PREFIXES = ("config", "algorithms", "simulation")
    for mod_name in list(sys.modules.keys()):
        if any(mod_name == p or mod_name.startswith(p + ".") for p in _PREFIXES):
            del sys.modules[mod_name]


def _get_surgeries_data_and_job_ids(tmp_path):
    """Returns minimal surgeries_data and job_ids for algorithm tests."""
    _setup_env(tmp_path)
    from data.data_generator import generate_day_surgeries_data

    job_ids = list(range(1, 6))
    surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)
    return surgeries_data, job_ids


# ---------------------------------------------------------------------------
# GA on_iteration tests (Task 3.1)
# ---------------------------------------------------------------------------


class TestGAOnIteration:
    """Verify GA.run() accepts and invokes on_iteration callback."""

    def test_ga_run_accepts_on_iteration_parameter(self, tmp_path):
        """GA.run() must accept an optional on_iteration parameter without error."""
        surgeries_data, job_ids = _get_surgeries_data_and_job_ids(tmp_path)
        from algorithms import ga

        calls = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            calls.append(algo_step)

        # Should not raise TypeError
        ga.run(surgeries_data, job_ids, seed=42, on_iteration=cb)
        assert len(calls) > 0, "on_iteration callback was never called"

    def test_ga_callback_called_once_per_generation(self, tmp_path):
        """on_iteration is called exactly MAX_GENERATIONS times."""
        _setup_env(tmp_path)
        from algorithms import ga
        from config.config import MAX_GENERATIONS
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        calls = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            calls.append(algo_step)

        ga.run(surgeries_data, job_ids, seed=0, on_iteration=cb)
        assert len(calls) == MAX_GENERATIONS, (
            f"Expected {MAX_GENERATIONS} calls, got {len(calls)}"
        )

    def test_ga_callback_receives_correct_generation_numbers(self, tmp_path):
        """Callback generation numbers start at 1 and are sequential."""
        _setup_env(tmp_path)
        from algorithms import ga
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        generations = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            generations.append(algo_step)

        ga.run(surgeries_data, job_ids, seed=1, on_iteration=cb)
        assert generations[0] == 1, (
            f"First generation should be 1, got {generations[0]}"
        )
        assert generations[-1] == len(generations), (
            f"Generations should be sequential 1..N, got {generations}"
        )

    def test_ga_callback_receives_finite_best_fitness(self, tmp_path):
        """Callback best_fitness should be a finite float (never inf on valid data)."""
        _setup_env(tmp_path)
        from algorithms import ga
        from data.data_generator import generate_day_surgeries_data
        import math

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        fitnesses = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            fitnesses.append(best_fitness)

        ga.run(surgeries_data, job_ids, seed=2, on_iteration=cb)
        assert all(math.isfinite(f) for f in fitnesses), (
            f"All best_fitness values should be finite, got: {fitnesses}"
        )

    def test_ga_without_callback_returns_same_signature(self, tmp_path):
        """Without on_iteration, GA.run() returns same 4-tuple as before."""
        _setup_env(tmp_path)
        from algorithms import ga
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        result = ga.run(surgeries_data, job_ids, seed=42)
        assert isinstance(result, tuple), "run() should return a tuple"
        assert len(result) == 4, f"run() should return 4 elements, got {len(result)}"


# ---------------------------------------------------------------------------
# DPSO on_iteration tests (Task 3.2)
# ---------------------------------------------------------------------------


class TestDPSOOnIteration:
    """Verify DPSO.run() accepts and invokes on_iteration callback."""

    def test_dpso_run_accepts_on_iteration_parameter(self, tmp_path):
        """DPSO.run() must accept optional on_iteration without error."""
        surgeries_data, job_ids = _get_surgeries_data_and_job_ids(tmp_path)
        from algorithms import dpso

        calls = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            calls.append(algo_step)

        dpso.run(surgeries_data, job_ids, seed=42, on_iteration=cb)
        assert len(calls) > 0, "on_iteration callback was never called"

    def test_dpso_callback_called_once_per_iteration(self, tmp_path):
        """on_iteration is called exactly MAX_ITERATIONS_DPSO times."""
        _setup_env(tmp_path)
        from algorithms import dpso
        from config.config import MAX_ITERATIONS_DPSO
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        calls = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            calls.append(algo_step)

        dpso.run(surgeries_data, job_ids, seed=0, on_iteration=cb)
        assert len(calls) == MAX_ITERATIONS_DPSO, (
            f"Expected {MAX_ITERATIONS_DPSO} calls, got {len(calls)}"
        )

    def test_dpso_without_callback_returns_same_signature(self, tmp_path):
        """Without on_iteration, DPSO.run() returns same 4-tuple as before."""
        _setup_env(tmp_path)
        from algorithms import dpso
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        result = dpso.run(surgeries_data, job_ids, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# DMSHOA on_iteration tests (Task 3.2)
# ---------------------------------------------------------------------------


class TestDMSHOAOnIteration:
    """Verify DMSHOA.run() accepts and invokes on_iteration callback."""

    def test_dmshoa_run_accepts_on_iteration_parameter(self, tmp_path):
        """DMSHOA.run() must accept optional on_iteration without error."""
        surgeries_data, job_ids = _get_surgeries_data_and_job_ids(tmp_path)
        from algorithms import dmshoa

        calls = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            calls.append(algo_step)

        dmshoa.run(surgeries_data, job_ids, seed=42, on_iteration=cb)
        assert len(calls) > 0, "on_iteration callback was never called"

    def test_dmshoa_callback_called_once_per_iteration(self, tmp_path):
        """on_iteration is called exactly MAX_ITERATIONS_MSHOA times."""
        _setup_env(tmp_path)
        from algorithms import dmshoa
        from config.config import MAX_ITERATIONS_MSHOA
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        calls = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            calls.append(algo_step)

        dmshoa.run(surgeries_data, job_ids, seed=0, on_iteration=cb)
        assert len(calls) == MAX_ITERATIONS_MSHOA, (
            f"Expected {MAX_ITERATIONS_MSHOA} calls, got {len(calls)}"
        )

    def test_dmshoa_without_callback_returns_same_signature(self, tmp_path):
        """Without on_iteration, DMSHOA.run() returns same 4-tuple."""
        _setup_env(tmp_path)
        from algorithms import dmshoa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        result = dmshoa.run(surgeries_data, job_ids, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# SBOA on_iteration tests (Task 3.2)
# ---------------------------------------------------------------------------


class TestSBOAOnIteration:
    """Verify SBOA.run() accepts and invokes on_iteration callback."""

    def test_sboa_run_accepts_on_iteration_parameter(self, tmp_path):
        """SBOA.run() must accept optional on_iteration without error."""
        surgeries_data, job_ids = _get_surgeries_data_and_job_ids(tmp_path)
        from algorithms import sboa

        calls = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            calls.append(algo_step)

        sboa.run(surgeries_data, job_ids, seed=42, on_iteration=cb)
        assert len(calls) > 0, "on_iteration callback was never called"

    def test_sboa_callback_called_once_per_iteration(self, tmp_path):
        """on_iteration is called exactly SBOA_MAX_ITER times."""
        _setup_env(tmp_path)
        from algorithms import sboa
        from config.config import SBOA_MAX_ITER
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        calls = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            calls.append(algo_step)

        sboa.run(surgeries_data, job_ids, seed=0, on_iteration=cb)
        assert len(calls) == SBOA_MAX_ITER, (
            f"Expected {SBOA_MAX_ITER} calls, got {len(calls)}"
        )

    def test_sboa_without_callback_returns_same_signature(self, tmp_path):
        """Without on_iteration, SBOA.run() returns same 4-tuple."""
        _setup_env(tmp_path)
        from algorithms import sboa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        result = sboa.run(surgeries_data, job_ids, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Task 2.1 & 2.2 — makespan kwarg propagated by all algorithms
# ---------------------------------------------------------------------------


class TestAlgorithmsMakespanKwarg:
    """All algorithms must pass makespan= kwarg to the on_iteration callback."""

    def test_ga_callback_receives_makespan_kwarg(self, tmp_path):
        """GA must pass makespan= kwarg; callback receives it as keyword arg."""
        surgeries_data, job_ids = _get_surgeries_data_and_job_ids(tmp_path)
        from algorithms import ga

        received = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            received.append({"algo_step": algo_step, "makespan": best_makespan})

        ga.run(surgeries_data, job_ids, seed=42, on_iteration=cb)
        assert len(received) > 0
        # All snapshots must have makespan != None (GA tracks best_makespan_overall)
        for item in received:
            assert item["makespan"] is not None, (
                f"GA callback at gen {item['generation']} received makespan=None"
            )

    def test_sboa_callback_receives_makespan_kwarg(self, tmp_path):
        """SBOA must pass makespan= kwarg to callback."""
        surgeries_data, job_ids = _get_surgeries_data_and_job_ids(tmp_path)
        from algorithms import sboa

        received = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            received.append(best_makespan)

        sboa.run(surgeries_data, job_ids, seed=42, on_iteration=cb)
        assert len(received) > 0
        assert all(m is not None for m in received), (
            f"SBOA: some makespan values were None: {received}"
        )

    def test_dmshoa_callback_receives_makespan_kwarg(self, tmp_path):
        """DMSHOA must pass makespan= kwarg to callback."""
        surgeries_data, job_ids = _get_surgeries_data_and_job_ids(tmp_path)
        from algorithms import dmshoa

        received = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            received.append(best_makespan)

        dmshoa.run(surgeries_data, job_ids, seed=42, on_iteration=cb)
        assert len(received) > 0
        assert all(m is not None for m in received), (
            f"DMSHOA: some makespan values were None: {received}"
        )

    def test_dpso_callback_receives_makespan_kwarg(self, tmp_path):
        """DPSO must pass makespan= kwarg to callback."""
        surgeries_data, job_ids = _get_surgeries_data_and_job_ids(tmp_path)
        from algorithms import dpso

        received = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            received.append(best_makespan)

        dpso.run(surgeries_data, job_ids, seed=42, on_iteration=cb)
        assert len(received) > 0
        assert all(m is not None for m in received), (
            f"DPSO: some makespan values were None: {received}"
        )


# ---------------------------------------------------------------------------
# iteration_makespan semantic correctness — must be real makespan, not fitness
# ---------------------------------------------------------------------------


def _make_penalized_config(num_procedures=5):
    """Config with non-zero penalty weights so fitness > makespan."""
    cfg = _make_minimal_config(num_procedures)
    cfg["algorithms"]["alpha"] = 1.0  # penalty weights > 0
    cfg["algorithms"]["beta"] = 1.0
    cfg["algorithms"]["gamma"] = 1.0
    cfg["algorithms"]["delta"] = 100.0
    return cfg


def _setup_penalized_env(tmp_path):
    import yaml, os, sys

    cfg = _make_penalized_config()
    cfg_file = tmp_path / "config_penalized.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    _PREFIXES = ("config", "algorithms", "simulation")
    for mod_name in list(sys.modules.keys()):
        if any(mod_name == p or mod_name.startswith(p + ".") for p in _PREFIXES):
            del sys.modules[mod_name]


class TestIterationMakespanIsRealMakespan:
    """
    iteration_makespan in the callback must be the real scheduling makespan
    (the max completion time of the best individual/particle of the iteration),
    NOT the combined objective fitness value.

    Proof method: with alpha/beta/gamma/delta > 0, the combined objective includes
    penalties and is strictly greater than the raw makespan for any non-trivial
    schedule. If iteration_makespan == iteration_fitness, the algorithm is passing
    the fitness instead of the real makespan — this test catches that bug.
    """

    def test_ga_iteration_makespan_differs_from_iteration_fitness(self, tmp_path):
        """GA: iteration_makespan must be real makespan (not equal to iteration_fitness)."""
        _setup_penalized_env(tmp_path)
        from algorithms import ga
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        snapshots = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            if iteration_fitness is not None and iteration_makespan is not None:
                if iteration_fitness != float("inf") and iteration_makespan != float(
                    "inf"
                ):
                    snapshots.append(
                        {
                            "iteration_fitness": iteration_fitness,
                            "iteration_makespan": iteration_makespan,
                        }
                    )

        ga.run(surgeries_data, job_ids, seed=42, on_iteration=cb)

        assert len(snapshots) > 0, "No finite snapshots received from GA"
        # With penalties, fitness > makespan for any non-trivial schedule
        at_least_one_differs = any(
            s["iteration_makespan"] != s["iteration_fitness"] for s in snapshots
        )
        assert at_least_one_differs, (
            f"GA: iteration_makespan always equals iteration_fitness — "
            f"likely passing fitness instead of real makespan. "
            f"Snapshots: {snapshots[:3]}"
        )

    def test_dpso_iteration_makespan_differs_from_iteration_fitness(self, tmp_path):
        """DPSO: iteration_makespan must be real makespan (not equal to iteration_fitness)."""
        _setup_penalized_env(tmp_path)
        from algorithms import dpso
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        snapshots = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            if iteration_fitness is not None and iteration_makespan is not None:
                if iteration_fitness != float("inf") and iteration_makespan != float(
                    "inf"
                ):
                    snapshots.append(
                        {
                            "iteration_fitness": iteration_fitness,
                            "iteration_makespan": iteration_makespan,
                        }
                    )

        dpso.run(surgeries_data, job_ids, seed=42, on_iteration=cb)

        assert len(snapshots) > 0, "No finite snapshots received from DPSO"
        at_least_one_differs = any(
            s["iteration_makespan"] != s["iteration_fitness"] for s in snapshots
        )
        assert at_least_one_differs, (
            f"DPSO: iteration_makespan always equals iteration_fitness — "
            f"likely passing fitness instead of real makespan. "
            f"Snapshots: {snapshots[:3]}"
        )

    def test_sboa_iteration_makespan_differs_from_iteration_fitness(self, tmp_path):
        """SBOA: iteration_makespan must be real makespan (not equal to iteration_fitness)."""
        _setup_penalized_env(tmp_path)
        from algorithms import sboa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        snapshots = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            if iteration_fitness is not None and iteration_makespan is not None:
                if iteration_fitness != float("inf") and iteration_makespan != float(
                    "inf"
                ):
                    snapshots.append(
                        {
                            "iteration_fitness": iteration_fitness,
                            "iteration_makespan": iteration_makespan,
                        }
                    )

        sboa.run(surgeries_data, job_ids, seed=42, on_iteration=cb)

        assert len(snapshots) > 0, "No finite snapshots received from SBOA"
        at_least_one_differs = any(
            s["iteration_makespan"] != s["iteration_fitness"] for s in snapshots
        )
        assert at_least_one_differs, (
            f"SBOA: iteration_makespan always equals iteration_fitness — "
            f"likely passing fitness instead of real makespan. "
            f"Snapshots: {snapshots[:3]}"
        )

    def test_dmshoa_iteration_makespan_differs_from_iteration_fitness(self, tmp_path):
        """DMSHOA: iteration_makespan must be real makespan (not equal to iteration_fitness)."""
        _setup_penalized_env(tmp_path)
        from algorithms import dmshoa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        snapshots = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            if iteration_fitness is not None and iteration_makespan is not None:
                if iteration_fitness != float("inf") and iteration_makespan != float(
                    "inf"
                ):
                    snapshots.append(
                        {
                            "iteration_fitness": iteration_fitness,
                            "iteration_makespan": iteration_makespan,
                        }
                    )

        dmshoa.run(surgeries_data, job_ids, seed=42, on_iteration=cb)

        assert len(snapshots) > 0, "No finite snapshots received from DMSHOA"
        at_least_one_differs = any(
            s["iteration_makespan"] != s["iteration_fitness"] for s in snapshots
        )
        assert at_least_one_differs, (
            f"DMSHOA: iteration_makespan always equals iteration_fitness — "
            f"likely passing fitness instead of real makespan. "
            f"Snapshots: {snapshots[:3]}"
        )

    def test_iteration_makespan_is_less_than_or_equal_to_iteration_fitness(
        self, tmp_path
    ):
        """
        Triangulation: makespan <= fitness because fitness includes penalties.
        This is the mathematical invariant that proves makespan is real.
        """
        _setup_penalized_env(tmp_path)
        from algorithms import ga
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        violations = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            if (
                iteration_fitness is not None
                and iteration_makespan is not None
                and iteration_fitness != float("inf")
                and iteration_makespan != float("inf")
            ):
                if iteration_makespan > iteration_fitness:
                    violations.append(
                        {
                            "gen": algo_step,
                            "iteration_makespan": iteration_makespan,
                            "iteration_fitness": iteration_fitness,
                        }
                    )

        ga.run(surgeries_data, job_ids, seed=42, on_iteration=cb)

        assert len(violations) == 0, (
            f"GA: iteration_makespan > iteration_fitness in {len(violations)} snapshots "
            f"(makespan must be <= fitness since fitness includes penalties). "
            f"First violation: {violations[0]}"
        )


# ---------------------------------------------------------------------------
# Task 2.1 — GA: iteration_fitness must isolate offspring only
# ---------------------------------------------------------------------------


class TestGAIterationIsolation:
    """
    GA: iteration_fitness/iteration_makespan must reflect ONLY the offspring
    generated in this algo_step, NOT the elite individuals carried from
    the previous generation.

    Key invariant: when the global best is strictly better than any offspring
    in at least one algo_step, best_fitness < iteration_fitness must hold
    in that generation.
    """

    def test_ga_iteration_fitness_can_be_worse_than_best_fitness(self, tmp_path):
        """
        GA: iteration_fitness CAN be > best_fitness when offspring are worse
        than the accumulated best (elite survivor).

        This test verifies that iteration_fitness is NOT always identical to
        best_fitness — proving the offspring isolation is real. Specifically,
        we look for at least one generation where iteration_fitness > best_fitness,
        which is the natural consequence of tracking offspring-only quality.
        """
        _setup_penalized_env(tmp_path)
        from algorithms import ga
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        snapshots = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            if (
                iteration_fitness is not None
                and best_fitness is not None
                and iteration_fitness != float("inf")
                and best_fitness != float("inf")
            ):
                snapshots.append(
                    {
                        "algo_step": algo_step,
                        "best_fitness": best_fitness,
                        "iteration_fitness": iteration_fitness,
                    }
                )

        ga.run(surgeries_data, job_ids, seed=42, on_iteration=cb)

        assert len(snapshots) > 0, "No finite snapshots received from GA"
        # iteration_fitness should NOT always equal best_fitness —
        # it reflects offspring only, which may be worse in some generations.
        at_least_one_worse = any(
            s["iteration_fitness"] > s["best_fitness"] for s in snapshots
        )
        assert at_least_one_worse, (
            f"GA: iteration_fitness never exceeds best_fitness — "
            f"offspring isolation may be broken (elite contaminating iteration metrics). "
            f"Snapshots: {snapshots[:5]}"
        )

    def test_ga_iteration_fitness_differs_from_best_fitness_in_some_generation(
        self, tmp_path
    ):
        """
        Triangulation: with elitism=1 and multiple generations, the elite individual
        (historical best) is expected to be better than offspring in at least some
        generation. So iteration_fitness > best_fitness must occur at least once.

        If iteration_fitness always equals best_fitness, the code is passing the
        accumulated population best instead of the offspring-only best.
        """
        _setup_penalized_env(tmp_path)
        from algorithms import ga
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        snapshots = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            if (
                iteration_fitness is not None
                and best_fitness is not None
                and iteration_fitness != float("inf")
                and best_fitness != float("inf")
            ):
                snapshots.append(
                    {
                        "algo_step": algo_step,
                        "best_fitness": best_fitness,
                        "iteration_fitness": iteration_fitness,
                    }
                )

        ga.run(surgeries_data, job_ids, seed=42, on_iteration=cb)

        assert len(snapshots) > 0, "No finite snapshots received from GA"
        at_least_one_differs = any(
            s["iteration_fitness"] > s["best_fitness"] for s in snapshots
        )
        assert at_least_one_differs, (
            f"GA: iteration_fitness always equals best_fitness — "
            f"offspring isolation is broken (elite is contaminating iteration metrics). "
            f"Snapshots: {snapshots[:5]}"
        )


# ---------------------------------------------------------------------------
# Task 2.2 — SBOA: iteration_fitness must reflect new evaluations only
# ---------------------------------------------------------------------------


class TestSBOAIterationIsolation:
    """
    SBOA: iteration_fitness must reflect ONLY the fitness of new candidate
    positions evaluated in this iteration (new_fit_p1, new_fit_p2), NOT
    the fitness of survivors/winners after greedy replacement.
    """

    def _setup_sboa_isolation_env(self, tmp_path):
        """Setup with more iterations to ensure diversity in observations."""
        import yaml, os, sys

        cfg = _make_penalized_config()
        cfg["algorithms"]["sboa"]["max_iterations"] = 10
        cfg["algorithms"]["sboa"]["population_size"] = 4
        cfg_file = tmp_path / "config_sboa_isolation.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        _PREFIXES = ("config", "algorithms", "simulation")
        for mod_name in list(sys.modules.keys()):
            if any(mod_name == p or mod_name.startswith(p + ".") for p in _PREFIXES):
                del sys.modules[mod_name]

    def test_sboa_iteration_fitness_is_not_always_equal_to_best_fitness(self, tmp_path):
        """
        SBOA: when new positions are worse than gbest, iteration_fitness > best_fitness.
        If iteration_fitness always equals best_fitness, isolation is broken.
        """
        self._setup_sboa_isolation_env(tmp_path)
        from algorithms import sboa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        snapshots = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            if (
                iteration_fitness is not None
                and best_fitness is not None
                and iteration_fitness != float("inf")
                and best_fitness != float("inf")
            ):
                snapshots.append(
                    {
                        "iteration": algo_step,
                        "best_fitness": float(best_fitness),
                        "iteration_fitness": float(iteration_fitness),
                    }
                )

        sboa.run(surgeries_data, job_ids, seed=7, on_iteration=cb)

        assert len(snapshots) > 0, "No finite snapshots received from SBOA"
        at_least_one_differs = any(
            s["iteration_fitness"] > s["best_fitness"] for s in snapshots
        )
        assert at_least_one_differs, (
            f"SBOA: iteration_fitness always equals best_fitness — "
            f"new-eval isolation is broken. Snapshots: {snapshots[:5]}"
        )

    def test_sboa_iteration_fitness_never_below_best_fitness(self, tmp_path):
        """
        Triangulation: iteration_fitness >= best_fitness always, because best_fitness
        is the accumulated global minimum (gbest), while iteration_fitness reflects
        only the candidates newly evaluated in this iteration.
        """
        self._setup_sboa_isolation_env(tmp_path)
        from algorithms import sboa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        violations = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            if (
                iteration_fitness is not None
                and best_fitness is not None
                and iteration_fitness != float("inf")
                and best_fitness != float("inf")
            ):
                if iteration_fitness < best_fitness - 1e-9:
                    violations.append(
                        {
                            "iteration": algo_step,
                            "best_fitness": best_fitness,
                            "iteration_fitness": iteration_fitness,
                        }
                    )

        sboa.run(surgeries_data, job_ids, seed=42, on_iteration=cb)

        assert len(violations) == 0, (
            f"SBOA: iteration_fitness < best_fitness in {len(violations)} iterations — "
            f"iteration_fitness must be >= best_fitness (it reflects new evals only). "
            f"First violation: {violations[0]}"
        )


# ---------------------------------------------------------------------------
# Task 2.3 — dMShOA: iteration_fitness must reflect new evaluations only
# ---------------------------------------------------------------------------


class TestDMShOAIterationIsolation:
    """
    dMShOA: iteration_fitness must reflect ONLY the fitness of new_sol
    evaluated in this iteration, NOT the fitness of survivors after greedy update.
    """

    def _setup_dmshoa_isolation_env(self, tmp_path):
        """Setup with more iterations to ensure diversity in observations."""
        import yaml, os, sys

        cfg = _make_penalized_config()
        cfg["algorithms"]["dmshoa"]["max_iterations"] = 10
        cfg["algorithms"]["dmshoa"]["population_size"] = 3
        cfg_file = tmp_path / "config_dmshoa_isolation.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        _PREFIXES = ("config", "algorithms", "simulation")
        for mod_name in list(sys.modules.keys()):
            if any(mod_name == p or mod_name.startswith(p + ".") for p in _PREFIXES):
                del sys.modules[mod_name]

    def test_dmshoa_iteration_fitness_is_not_always_equal_to_best_fitness(
        self, tmp_path
    ):
        """
        dMShOA: when new solutions are worse than gbest, iteration_fitness > best_fitness.
        If iteration_fitness always equals best_fitness, isolation is broken.
        """
        self._setup_dmshoa_isolation_env(tmp_path)
        from algorithms import dmshoa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        snapshots = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            if (
                iteration_fitness is not None
                and best_fitness is not None
                and iteration_fitness != float("inf")
                and best_fitness != float("inf")
            ):
                snapshots.append(
                    {
                        "iteration": algo_step,
                        "best_fitness": float(best_fitness),
                        "iteration_fitness": float(iteration_fitness),
                    }
                )

        dmshoa.run(surgeries_data, job_ids, seed=7, on_iteration=cb)

        assert len(snapshots) > 0, "No finite snapshots received from dMShOA"
        at_least_one_differs = any(
            s["iteration_fitness"] > s["best_fitness"] for s in snapshots
        )
        assert at_least_one_differs, (
            f"dMShOA: iteration_fitness always equals best_fitness — "
            f"new-eval isolation is broken. Snapshots: {snapshots[:5]}"
        )

    def test_dmshoa_iteration_fitness_never_below_best_fitness(self, tmp_path):
        """
        Triangulation: iteration_fitness >= best_fitness always (gbest is historical minimum).
        """
        self._setup_dmshoa_isolation_env(tmp_path)
        from algorithms import dmshoa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        violations = []

        def cb(
            algo_step,
            best_fitness,
            best_makespan=None,
            iteration_fitness=None,
            iteration_makespan=None,
            **kwargs,
        ):
            if (
                iteration_fitness is not None
                and best_fitness is not None
                and iteration_fitness != float("inf")
                and best_fitness != float("inf")
            ):
                if iteration_fitness < best_fitness - 1e-9:
                    violations.append(
                        {
                            "iteration": algo_step,
                            "best_fitness": best_fitness,
                            "iteration_fitness": iteration_fitness,
                        }
                    )

        dmshoa.run(surgeries_data, job_ids, seed=42, on_iteration=cb)

        assert len(violations) == 0, (
            f"dMShOA: iteration_fitness < best_fitness in {len(violations)} iterations — "
            f"iteration_fitness must be >= best_fitness (it reflects new evals only). "
            f"First violation: {violations[0]}"
        )
