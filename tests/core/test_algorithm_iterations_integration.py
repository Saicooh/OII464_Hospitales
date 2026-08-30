"""
Tests for algorithm_iterations population in analysis mode (Task 4.1 partial).

Covers:
- run_elective_analysis_mode() must populate algorithm_iterations table
  with snapshot rows after algo instrumentation is wired.
- algorithm_iterations rows are linked to correct sim_id.
- Each row has valid generation, best_fitness fields.
"""

import os
import yaml
import sys
import sqlite3
from pathlib import Path
import pytest


def _find_analysis_db(base_dir: str) -> str:
    """Find the analysis.db created by run_elective_analysis_mode under base_dir."""
    dbs = list(Path(base_dir).rglob("analysis.db"))
    if not dbs:
        raise FileNotFoundError(
            f"No analysis.db found under {base_dir}. "
            f"Contents: {list(Path(base_dir).rglob('*'))}"
        )
    return str(dbs[0])


def _make_analysis_config(tmp_path, db_path, num_procedures=5, sims_per_run=2):
    cfg = {
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
        "personnel": {"num_anesthesiologists": 1, "num_surgeons": 1},
        "algorithms": {
            "alpha": 1e-6,
            "beta": 0.7,
            "gamma": 1.4,
            "delta": 100.0,
            "ga": {
                "enabled": True,
                "population_size": 2,
                "max_generations": 2,
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
        "analysis_mode": {
            "enabled": True,
            "num_runs": 1,
            "sims_per_run": sims_per_run,
            "checkpoint_interval_seconds": 300,
            "sqlite_path": db_path,
            "sweep_enabled": False,
            "sweep_num_procedures": [],
            "sweep_sims_per_x": 2,
            "export_csv_after_run": False,
            "artifact_save_mode": "all",
        },
    }
    cfg_file = tmp_path / "config_analysis.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if (
            mod_name.startswith("config.")
            or mod_name.startswith("core.simulation_runner")
            or mod_name.startswith("core.analysis_persistence")
            or mod_name.startswith("simulation.workers.")
            or mod_name.startswith("simulation.scheduler")
            or mod_name.startswith("algorithms.")
        ):
            del sys.modules[mod_name]


class TestAlgorithmIterationsPopulation:
    """Tests that algorithm_iterations is populated after analysis run."""

    def test_algorithm_iterations_has_rows_after_analysis_run(self, tmp_path):
        """After run_elective_analysis_mode(), algorithm_iterations has >= 1 row."""
        db_path = str(tmp_path / "algo_iter_test.db")
        _make_analysis_config(tmp_path, db_path, sims_per_run=2)

        from core.simulation_runner import SimulationRunner
        from core.file_manager import FileManager

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        actual_db = _find_analysis_db(results_dir)
        conn = sqlite3.connect(actual_db)
        algo_iter_rows = conn.execute(
            "SELECT COUNT(*) FROM algorithm_iterations"
        ).fetchone()[0]
        conn.close()

        assert algo_iter_rows >= 1, (
            f"algorithm_iterations must have >= 1 row after instrumented run, got {algo_iter_rows}"
        )

    def test_algorithm_iterations_linked_to_valid_sim_ids(self, tmp_path):
        """All algorithm_iterations rows reference valid sim_ids in simulations table."""
        db_path = str(tmp_path / "algo_iter_fk.db")
        _make_analysis_config(tmp_path, db_path, sims_per_run=2)

        from core.simulation_runner import SimulationRunner
        from core.file_manager import FileManager

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        actual_db = _find_analysis_db(results_dir)
        conn = sqlite3.connect(actual_db)
        orphan_rows = conn.execute("""
            SELECT COUNT(*) FROM algorithm_iterations ai
            WHERE NOT EXISTS (
                SELECT 1 FROM simulations s WHERE s.sim_id = ai.sim_id
            )
        """).fetchone()[0]
        conn.close()

        assert orphan_rows == 0, (
            f"Found {orphan_rows} algorithm_iterations rows with invalid sim_id FK"
        )

    def test_algorithm_iterations_generation_starts_at_one(self, tmp_path):
        """All algorithm_iterations rows have generation >= 1."""
        db_path = str(tmp_path / "algo_iter_gen.db")
        _make_analysis_config(tmp_path, db_path, sims_per_run=1)

        from core.simulation_runner import SimulationRunner
        from core.file_manager import FileManager

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        actual_db = _find_analysis_db(results_dir)
        conn = sqlite3.connect(actual_db)
        bad_gen_rows = conn.execute(
            "SELECT COUNT(*) FROM algorithm_iterations WHERE algo_step < 1"
        ).fetchone()[0]
        conn.close()

        assert bad_gen_rows == 0, (
            f"Found {bad_gen_rows} rows with algo_step < 1 in algorithm_iterations"
        )

    def test_algorithm_iterations_best_fitness_is_valid(self, tmp_path):
        """All algorithm_iterations rows have finite best_fitness (not NULL or inf)."""
        db_path = str(tmp_path / "algo_iter_fit.db")
        _make_analysis_config(tmp_path, db_path, sims_per_run=1)

        from core.simulation_runner import SimulationRunner
        from core.file_manager import FileManager

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        actual_db = _find_analysis_db(results_dir)
        conn = sqlite3.connect(actual_db)
        null_fit_rows = conn.execute(
            "SELECT COUNT(*) FROM algorithm_iterations WHERE best_fitness IS NULL"
        ).fetchone()[0]
        conn.close()

        assert null_fit_rows == 0, (
            f"Found {null_fit_rows} rows with NULL best_fitness in algorithm_iterations"
        )

    def test_algorithm_iterations_best_makespan_greater_than_zero(self, tmp_path):
        """All algorithm_iterations rows must have best_makespan > 0 after wiring."""
        db_path = str(tmp_path / "algo_iter_wall.db")
        _make_analysis_config(tmp_path, db_path, sims_per_run=2)

        from core.simulation_runner import SimulationRunner
        from core.file_manager import FileManager

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        actual_db = _find_analysis_db(results_dir)
        conn = sqlite3.connect(actual_db)
        total_rows = conn.execute(
            "SELECT COUNT(*) FROM algorithm_iterations"
        ).fetchone()[0]
        zero_makespan_rows = conn.execute(
            "SELECT COUNT(*) FROM algorithm_iterations WHERE best_makespan IS NULL OR best_makespan <= 0.0"
        ).fetchone()[0]
        conn.close()

        assert total_rows >= 1, "No rows found in algorithm_iterations"
        assert zero_makespan_rows == 0, (
            f"Found {zero_makespan_rows}/{total_rows} rows with NULL or zero best_makespan"
        )

    def test_algorithm_iterations_best_fitness_monotonic_within_simulation(
        self, tmp_path
    ):
        """Within a single sim_id, best_fitness must be non-increasing across generations."""
        db_path = str(tmp_path / "algo_iter_mono.db")
        # Use ALL policy and multiple generations to get multiple rows per sim
        cfg_file_path = str(tmp_path / "config_analysis.yaml")
        import yaml as _yaml

        cfg = {
            "experiment": {
                "num_simulations": 1,
                "std_factor_times": 0.0,
                "alpha_test": 0.05,
                "num_procedures": 5,
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
            "personnel": {"num_anesthesiologists": 1, "num_surgeons": 1},
            "algorithms": {
                "alpha": 1e-6,
                "beta": 0.7,
                "gamma": 1.4,
                "delta": 100.0,
                "ga": {
                    "enabled": True,
                    "population_size": 3,
                    "max_generations": 5,  # more generations → more rows per sim
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
            "analysis_mode": {
                "enabled": True,
                "num_runs": 1,
                "sims_per_run": 1,
                "checkpoint_interval_seconds": 300,
                "sqlite_path": db_path,
                "sweep_enabled": False,
                "sweep_num_procedures": [],
                "sweep_sims_per_x": 2,
                "export_csv_after_run": False,
                "artifact_save_mode": "all",  # ALL so every generation is captured
            },
        }
        import os, sys

        with open(cfg_file_path, "w", encoding="utf-8") as f:
            _yaml.safe_dump(cfg, f)
        os.environ["HOSPITAL_CONFIG_PATH"] = cfg_file_path
        for mod_name in list(sys.modules.keys()):
            if (
                mod_name.startswith("config.")
                or mod_name.startswith("core.simulation_runner")
                or mod_name.startswith("core.analysis_persistence")
                or mod_name.startswith("simulation.workers.")
                or mod_name.startswith("simulation.scheduler")
                or mod_name.startswith("algorithms.")
            ):
                del sys.modules[mod_name]

        from core.simulation_runner import SimulationRunner
        from core.file_manager import FileManager

        results_dir = str(tmp_path / "results_mono")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        actual_db = _find_analysis_db(results_dir)
        conn = sqlite3.connect(actual_db)
        # Find a sim_id with more than 1 row
        rows = conn.execute(
            "SELECT sim_id, algo_step, best_fitness FROM algorithm_iterations ORDER BY sim_id, algo_step"
        ).fetchall()
        conn.close()

        # Group by sim_id and check monotonicity
        from collections import defaultdict

        by_sim: dict = defaultdict(list)
        for sim_id, step, bf in rows:
            by_sim[sim_id].append((step, bf))

        # At least one sim must have rows to validate monotonicity
        assert len(rows) >= 1, "No rows found — cannot check monotonicity"

        for sim_id, entries in by_sim.items():
            if len(entries) <= 1:
                continue  # skip single-row sims
            fitnesses = [e[1] for e in entries]
            for i in range(1, len(fitnesses)):
                assert fitnesses[i] <= fitnesses[i - 1] + 1e-9, (
                    f"sim_id={sim_id}: best_fitness not non-increasing at index {i}: "
                    f"{fitnesses}"
                )


# ---------------------------------------------------------------------------
# Task 3.1 — runner populates makespan from snap.makespan (not snap.best_fitness)
# ---------------------------------------------------------------------------


class TestRunnerMakespanColumn:
    """Runner must populate best_makespan and iteration_makespan columns."""

    def test_algorithm_iterations_makespan_column_populated(self, tmp_path):
        """algorithm_iterations.best_makespan must not be NULL after analysis run."""
        db_path = str(tmp_path / "algo_makespan_test.db")
        _make_analysis_config(tmp_path, db_path, sims_per_run=1)

        from core.simulation_runner import SimulationRunner
        from core.file_manager import FileManager

        results_dir = str(tmp_path / "results_mk")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        actual_db = _find_analysis_db(results_dir)
        conn = sqlite3.connect(actual_db)
        null_makespan = conn.execute(
            "SELECT COUNT(*) FROM algorithm_iterations WHERE best_makespan IS NULL"
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM algorithm_iterations").fetchone()[0]
        conn.close()

        assert total >= 1, "No rows in algorithm_iterations"
        assert null_makespan == 0, (
            f"Found {null_makespan}/{total} rows with NULL best_makespan — "
            "runner must populate best_makespan from snapshots"
        )
