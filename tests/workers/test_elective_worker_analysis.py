"""
Tests para workers/elective_worker.py — modo análisis.

Cubre:
- Contrato dual de retorno: (sim_i, sim_results) modo normal, (sim_i, sim_results, wall_clock_elapsed_s) modo análisis.
- wall_clock_start opcional en constructor.
- wall_clock_elapsed_s > 0 y reflejando tiempo transcurrido desde wall_clock_start.
"""

import os
import yaml
import sys
import time
import tempfile
import pytest


# ---------------------------------------------------------------------------
# Helpers de config mínima (sin real_data, 1 algoritmo GA fast)
# ---------------------------------------------------------------------------


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
                "max_generations": 1,
                "crossover_probability": 0.8,
                "mutation_probability": 0.3,
                "elitism_count": 1,
            },
            "dpso": {
                "enabled": False,
                "swarm_size": 2,
                "max_iterations": 1,
                "w": 0.7,
                "c1": 1.5,
                "c2": 1.5,
                "vel_high": 4.0,
                "vel_low": -4.0,
            },
            "sboa": {
                "enabled": False,
                "population_size": 2,
                "max_iterations": 1,
                "lower_bound": -5.0,
                "upper_bound": 5.0,
            },
            "dmshoa": {
                "enabled": False,
                "population_size": 2,
                "max_iterations": 1,
                "k": 0.3,
                "lower_bound": -5.0,
                "upper_bound": 5.0,
            },
        },
    }


def _setup_env(tmp_path, num_procedures=5):
    cfg = _make_minimal_config(num_procedures)
    cfg_file = tmp_path / "config_test.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    # Invalida todos los módulos cacheados para limpiar imports
    for mod_name in list(sys.modules.keys()):
        if (
            mod_name.startswith("config.")
            or mod_name.startswith("simulation.workers.")
            or mod_name.startswith("simulation.scheduler")
            or mod_name.startswith("algorithms.")
        ):
            del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestElectiveWorkerNormalMode:
    """Modo normal: sin wall_clock_start → retorna tupla de 2 elementos."""

    def test_normal_mode_returns_two_tuple(self, tmp_path):
        """Sin wall_clock_start, run() retorna (sim_i, sim_results) — exactamente 2 elementos."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        job_ids = list(range(1, 6))
        worker = ElectiveWorker(job_ids, get_algorithms(), std_factor=0.0)
        result = worker.run(0)

        assert len(result) == 2, (
            f"Modo normal debe retornar (sim_i, sim_results), pero retornó tupla de {len(result)} elementos"
        )

    def test_normal_mode_first_element_is_sim_i(self, tmp_path):
        """El primer elemento de la tupla es el mismo sim_i pasado."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        job_ids = list(range(1, 6))
        worker = ElectiveWorker(job_ids, get_algorithms(), std_factor=0.0)
        result = worker.run(3)

        assert result[0] == 3

    def test_normal_mode_second_element_is_dict_with_algo_keys(self, tmp_path):
        """sim_results es un dict con claves = nombres de algoritmos habilitados."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        algorithms = get_algorithms()
        job_ids = list(range(1, 6))
        worker = ElectiveWorker(job_ids, algorithms, std_factor=0.0)
        _, sim_results = worker.run(0)

        for spec in algorithms:
            assert spec["name"] in sim_results, (
                f"Algoritmo '{spec['name']}' ausente en sim_results"
            )


class TestElectiveWorkerAnalysisMode:
    """Modo análisis: con wall_clock_start → retorna tupla de 4 elementos con elapsed_s."""

    def test_analysis_mode_returns_four_tuple(self, tmp_path):
        """Con wall_clock_start, run() retorna (sim_i, sim_results, batch_trace, wall_clock_elapsed_s)."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        job_ids = list(range(1, 6))
        t0 = time.time()
        worker = ElectiveWorker(
            job_ids, get_algorithms(), std_factor=0.0, wall_clock_start=t0
        )
        result = worker.run(0)

        assert len(result) == 4, (
            f"Modo análisis debe retornar (sim_i, sim_results, batch_trace, elapsed_s), "
            f"pero retornó tupla de {len(result)} elementos"
        )

    def test_analysis_mode_elapsed_is_positive(self, tmp_path):
        """wall_clock_elapsed_s debe ser positivo (tiempo transcurrido > 0)."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        job_ids = list(range(1, 6))
        t0 = time.time() - 1.0  # Simulamos que la corrida empezó 1 s antes
        worker = ElectiveWorker(
            job_ids, get_algorithms(), std_factor=0.0, wall_clock_start=t0
        )
        _, _, _, elapsed = worker.run(0)

        assert elapsed >= 1.0, (
            f"elapsed_s debería ser >= 1.0 s (wall_clock_start 1 s atrás), got {elapsed}"
        )

    def test_analysis_mode_elapsed_reflects_wall_clock_start(self, tmp_path):
        """elapsed_s ≈ time.time() - wall_clock_start al momento de llamar run()."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        job_ids = list(range(1, 6))
        offset = 5.0
        t0 = time.time() - offset
        worker = ElectiveWorker(
            job_ids, get_algorithms(), std_factor=0.0, wall_clock_start=t0
        )
        t_before = time.time()
        _, _, _, elapsed = worker.run(0)
        t_after = time.time()

        expected_min = t_before - t0
        expected_max = t_after - t0
        assert expected_min <= elapsed <= expected_max, (
            f"elapsed_s={elapsed:.4f} fuera del rango [{expected_min:.4f}, {expected_max:.4f}]"
        )

    def test_analysis_mode_sim_i_preserved(self, tmp_path):
        """El primer elemento sigue siendo sim_i en modo análisis."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        job_ids = list(range(1, 6))
        t0 = time.time()
        worker = ElectiveWorker(
            job_ids, get_algorithms(), std_factor=0.0, wall_clock_start=t0
        )
        result = worker.run(7)

        assert result[0] == 7

    def test_analysis_mode_sim_results_still_valid(self, tmp_path):
        """sim_results en modo análisis contiene los mismos datos que en modo normal."""
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
            algo_name = spec["name"]
            assert algo_name in sim_results
            assert "makespan" in sim_results[algo_name]
            assert "solution" in sim_results[algo_name]
            assert "time" in sim_results[algo_name]


class TestElectiveWorkerSnapshotContract:
    """Task 5.3 — Analysis mode deve retornar iteration_snapshots acumulados."""

    def test_analysis_mode_returns_iteration_snapshots_key(self, tmp_path):
        """En modo análisis, sim_results[algo]['iteration_snapshots'] debe existir."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        algorithms = get_algorithms()
        job_ids = list(range(1, 6))
        t0 = __import__("time").time()
        worker = ElectiveWorker(
            job_ids, algorithms, std_factor=0.0, wall_clock_start=t0
        )
        _, sim_results, *rest = worker.run(0)

        for spec in algorithms:
            algo_name = spec["name"]
            assert "iteration_snapshots" in sim_results[algo_name], (
                f"'{algo_name}' no tiene 'iteration_snapshots' en modo análisis. "
                "ElectiveWorker.run() debe adjuntar handler.snapshots al resultado."
            )

    def test_analysis_mode_iteration_snapshots_is_list(self, tmp_path):
        """iteration_snapshots debe ser una lista (vacía o con elementos)."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        algorithms = get_algorithms()
        job_ids = list(range(1, 6))
        t0 = __import__("time").time()
        worker = ElectiveWorker(
            job_ids, algorithms, std_factor=0.0, wall_clock_start=t0
        )
        _, sim_results, *rest = worker.run(0)

        for spec in algorithms:
            algo_name = spec["name"]
            snapshots = sim_results[algo_name]["iteration_snapshots"]
            assert isinstance(snapshots, list), (
                f"'{algo_name}': iteration_snapshots debe ser una lista, got {type(snapshots)}"
            )

    def test_analysis_mode_snapshots_have_generation_and_fitness(self, tmp_path):
        """Cada snapshot debe tener 'generation' y 'best_fitness' como atributos."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms
        from core.iteration_callback import IterationSnapshot

        algorithms = get_algorithms()
        job_ids = list(range(1, 6))
        t0 = __import__("time").time()
        worker = ElectiveWorker(
            job_ids, algorithms, std_factor=0.0, wall_clock_start=t0
        )
        _, sim_results, *rest = worker.run(0)

        for spec in algorithms:
            algo_name = spec["name"]
            snapshots = sim_results[algo_name]["iteration_snapshots"]
            # GA con max_generations=1 debe producir al menos 1 snapshot
            if snapshots:
                first = snapshots[0]
                assert hasattr(first, "algo_step"), (
                    f"Snapshot de '{algo_name}' no tiene atributo 'algo_step'"
                )
                assert hasattr(first, "best_fitness"), (
                    f"Snapshot de '{algo_name}' no tiene atributo 'best_fitness'"
                )
                assert first.algo_step >= 1, (
                    f"algo_step debe ser >= 1, got {first.algo_step}"
                )
                assert isinstance(first.best_fitness, float), (
                    f"best_fitness debe ser float, got {type(first.best_fitness)}"
                )

    def test_normal_mode_has_no_iteration_snapshots(self, tmp_path):
        """En modo normal (sin wall_clock_start), iteration_snapshots NO debe estar en sim_results."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        algorithms = get_algorithms()
        job_ids = list(range(1, 6))
        worker = ElectiveWorker(job_ids, algorithms, std_factor=0.0)
        _, sim_results = worker.run(0)

        for spec in algorithms:
            algo_name = spec["name"]
            assert "iteration_snapshots" not in sim_results[algo_name], (
                f"'{algo_name}': 'iteration_snapshots' no debe estar en modo normal."
            )

    def test_analysis_mode_job_grupo_map_none_with_synthetic_data(self, tmp_path):
        """Con datos sintéticos (USE_REAL_DATA=False), job_grupo_map debe ser None."""
        _setup_env(tmp_path)
        from simulation.workers.elective_worker import ElectiveWorker
        from config.config import get_algorithms

        algorithms = get_algorithms()
        job_ids = list(range(1, 6))
        t0 = __import__("time").time()
        worker = ElectiveWorker(
            job_ids, algorithms, std_factor=0.0, wall_clock_start=t0
        )
        _, sim_results, *rest = worker.run(0)

        for spec in algorithms:
            algo_name = spec["name"]
            assert "job_grupo_map" in sim_results[algo_name], (
                f"'{algo_name}' no tiene 'job_grupo_map' en modo análisis."
            )
            assert sim_results[algo_name]["job_grupo_map"] is None, (
                f"Con datos sintéticos, 'job_grupo_map' debe ser None, "
                f"got {sim_results[algo_name]['job_grupo_map']}"
            )
