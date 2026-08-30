"""
Tests for worker snapshot accumulation (Task 3.3).

Covers:
- In analysis mode (wall_clock_start set), sim_results[algo]['iteration_snapshots']
  contains a list of IterationSnapshot objects with generation > 0.
- In normal mode, 'iteration_snapshots' is absent or empty.
- Worker does not break normal mode structure.
"""

import os
import yaml
import sys
import time
import pytest


def _make_minimal_config(num_procedures=5):
    return {
        "experiment": {
            "num_simulations": 1,
            "std_factor_times": 0.0,
            "alpha_test": 0.05,
            "num_procedures": num_procedures,
            "output_dirs": {"plots": "results/plots", "csv": "results/csv"},
            "n_jobs": 1,
        },
        "real_data": {"enabled": False},
        "logging": {"verbose_mode": False},
        "times": {
            "setup": {"1": 10, "2": 10, "3": 10},
            "cleanup": {"1": 5, "2": 5, "3": 5},
            "max_wait": {"1": 500, "2": 500},
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
                "max_generations": 2,  # small for speed
                "crossover_probability": 0.8,
                "mutation_probability": 0.3,
                "elitism_count": 1,
            },
            "dpso": {
                "enabled": False,
                "swarm_size": 2,
                "max_iterations": 2,
                "w": 0.7,
                "c1": 1.5,
                "c2": 1.5,
                "vel_high": 4.0,
                "vel_low": -4.0,
            },
            "sboa": {
                "enabled": False,
                "population_size": 2,
                "max_iterations": 2,
                "lower_bound": -5.0,
                "upper_bound": 5.0,
            },
            "dmshoa": {
                "enabled": False,
                "population_size": 2,
                "max_iterations": 2,
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
    for mod_name in list(sys.modules.keys()):
        if (
            mod_name.startswith("config.")
            or mod_name.startswith("simulation.workers.")
            or mod_name.startswith("algorithms.")
            or mod_name.startswith("simulation.scheduler")
            or mod_name.startswith("core.iteration_callback")
        ):
            del sys.modules[mod_name]


class TestWorkerSnapshotNormalMode:
    """Normal mode: no iteration_snapshots in sim_results."""

    def test_normal_mode_result_has_no_iteration_snapshots(self, tmp_path):
        """In normal mode, iteration_snapshots key is absent from sim_results per algo."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        algorithms = get_algorithms()
        job_ids = list(range(1, 6))
        worker = ElectiveWorker(job_ids, algorithms, std_factor=0.0)
        _, sim_results = worker.run(0)

        for spec in algorithms:
            result = sim_results[spec["name"]]
            # iteration_snapshots must NOT be present in normal mode
            assert "iteration_snapshots" not in result, (
                f"Normal mode must not include 'iteration_snapshots' in sim_results[{spec['name']}]. "
                "This key breaks the backward-compatible tuple interface."
            )


class TestWorkerSnapshotAnalysisMode:
    """Analysis mode: sim_results[algo]['iteration_snapshots'] populated."""

    def test_analysis_mode_has_iteration_snapshots_key(self, tmp_path):
        """In analysis mode, each algo result contains 'iteration_snapshots'."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        algorithms = get_algorithms()
        job_ids = list(range(1, 6))
        t0 = time.time()
        worker = ElectiveWorker(
            job_ids, algorithms, std_factor=0.0, wall_clock_start=t0
        )
        _, sim_results, *rest = worker.run(0)

        for spec in algorithms:
            result = sim_results[spec["name"]]
            assert "iteration_snapshots" in result, (
                f"Analysis mode must include 'iteration_snapshots' in sim_results[{spec['name']}]"
            )

    def test_analysis_mode_iteration_snapshots_is_list(self, tmp_path):
        """iteration_snapshots is a list."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        algorithms = get_algorithms()
        job_ids = list(range(1, 6))
        t0 = time.time()
        worker = ElectiveWorker(
            job_ids, algorithms, std_factor=0.0, wall_clock_start=t0
        )
        _, sim_results, *rest = worker.run(0)

        for spec in algorithms:
            snapshots = sim_results[spec["name"]]["iteration_snapshots"]
            assert isinstance(snapshots, list), (
                f"iteration_snapshots for {spec['name']} must be a list, got {type(snapshots)}"
            )

    def test_analysis_mode_iteration_snapshots_not_empty(self, tmp_path):
        """iteration_snapshots contains at least one snapshot (best_only policy still saves first)."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        algorithms = get_algorithms()
        job_ids = list(range(1, 6))
        t0 = time.time()
        worker = ElectiveWorker(
            job_ids, algorithms, std_factor=0.0, wall_clock_start=t0
        )
        _, sim_results, *rest = worker.run(0)

        for spec in algorithms:
            snapshots = sim_results[spec["name"]]["iteration_snapshots"]
            assert len(snapshots) >= 1, (
                f"iteration_snapshots for {spec['name']} must have >= 1 snapshot, got {len(snapshots)}"
            )

    def test_analysis_mode_snapshots_have_valid_generation(self, tmp_path):
        """Each snapshot has generation >= 1."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        algorithms = get_algorithms()
        job_ids = list(range(1, 6))
        t0 = time.time()
        worker = ElectiveWorker(
            job_ids, algorithms, std_factor=0.0, wall_clock_start=t0
        )
        _, sim_results, *rest = worker.run(0)

        for spec in algorithms:
            snapshots = sim_results[spec["name"]]["iteration_snapshots"]
            for snap in snapshots:
                assert snap.algo_step >= 1, (
                    f"Snapshot algo_step must be >= 1, got {snap.algo_step}"
                )

    def test_analysis_mode_snapshots_have_finite_fitness(self, tmp_path):
        """Each snapshot has finite best_fitness."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms
        import math

        algorithms = get_algorithms()
        job_ids = list(range(1, 6))
        t0 = time.time()
        worker = ElectiveWorker(
            job_ids, algorithms, std_factor=0.0, wall_clock_start=t0
        )
        _, sim_results, *rest = worker.run(0)

        for spec in algorithms:
            snapshots = sim_results[spec["name"]]["iteration_snapshots"]
            for snap in snapshots:
                assert math.isfinite(snap.best_fitness), (
                    f"Snapshot best_fitness must be finite, got {snap.best_fitness}"
                )
