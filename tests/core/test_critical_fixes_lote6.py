"""
Tests for Lote 6 critical fixes:
1. combined_obj propagation end-to-end (algorithms → callback → persistence)
2. Checkpoint reports produce real artifacts (schedule populated, files created)
3. results/<timestamp>/... directory structure in analysis mode

Strict TDD: tests written BEFORE implementation.
"""

import math
import os
import yaml
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_config(num_procedures=5, max_generations=3, population_size=2):
    return {
        "experiment": {
            "num_simulations": 2,
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
            "max_wait": {"1": 100, "2": 100},
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
                "population_size": population_size,
                "max_generations": max_generations,
                "crossover_probability": 0.8,
                "mutation_probability": 0.3,
                "elitism_count": 1,
            },
            "dpso": {
                "enabled": False,
                "swarm_size": 2,
                "max_iterations": 3,
                "w": 0.7,
                "c1": 1.5,
                "c2": 1.5,
                "vel_high": 4.0,
                "vel_low": -4.0,
            },
            "sboa": {
                "enabled": False,
                "population_size": 3,
                "max_iterations": 3,
                "lower_bound": -5.0,
                "upper_bound": 5.0,
            },
            "dmshoa": {
                "enabled": False,
                "population_size": 2,
                "max_iterations": 3,
                "k": 0.3,
                "lower_bound": -5.0,
                "upper_bound": 5.0,
            },
        },
    }


def _setup_env(tmp_path, **kwargs):
    cfg = _make_minimal_config(**kwargs)
    cfg_file = tmp_path / "config_test.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if any(
            mod_name.startswith(p)
            for p in ["config.", "algorithms.", "core.", "simulation.workers.", "simulation."]
        ):
            del sys.modules[mod_name]


def _get_surgeries_data_and_job_ids(tmp_path, **kwargs):
    _setup_env(tmp_path, **kwargs)
    from data.data_generator import generate_day_surgeries_data

    job_ids = list(range(1, 6))
    surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)
    return surgeries_data, job_ids


# ===========================================================================
# Fix 1: combined_obj propagation end-to-end
# ===========================================================================


class TestCombinedObjPropagation:
    """best_fitness and best_makespan must flow from algorithm callback → snapshot → persistence."""

    def test_ga_callback_receives_non_none_best_fitness(self, tmp_path):
        """GA on_iteration callback must pass best_fitness (not None)."""
        surgeries_data, job_ids = _get_surgeries_data_and_job_ids(tmp_path)
        from algorithms import ga

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

        ga.run(surgeries_data, job_ids, seed=42, on_iteration=cb)

        assert len(fitnesses) > 0, "No callbacks received"
        non_none = [v for v in fitnesses if v is not None]
        assert len(non_none) > 0, (
            f"GA never passed best_fitness; all values are None: {fitnesses}"
        )

    def test_ga_best_fitness_monotonically_non_increasing(self, tmp_path):
        """In GA, best_fitness passed to callback is monotonically non-increasing."""
        surgeries_data, job_ids = _get_surgeries_data_and_job_ids(tmp_path)
        from algorithms import ga

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

        ga.run(surgeries_data, job_ids, seed=7, on_iteration=cb)
        assert len(fitnesses) > 0

        for i in range(1, len(fitnesses)):
            assert fitnesses[i] <= fitnesses[i - 1] + 1e-9, (
                f"best_fitness is not monotonically non-increasing at index {i}: "
                f"{fitnesses[i - 1]} -> {fitnesses[i]}"
            )

    def test_snapshot_best_fitness_non_none_after_ga_run(self, tmp_path):
        """AnalysisIterationHandler accumulates snapshots with non-None best_fitness."""
        surgeries_data, job_ids = _get_surgeries_data_and_job_ids(tmp_path)
        from algorithms import ga
        from core.iteration_callback import AnalysisIterationHandler

        handler = AnalysisIterationHandler(policy="best_only")
        ga.run(surgeries_data, job_ids, seed=0, on_iteration=handler)

        assert len(handler.snapshots) > 0, "No snapshots captured"
        non_none = [s for s in handler.snapshots if s.best_fitness is not None]
        assert len(non_none) > 0, (
            f"All snapshots have best_fitness=None; snapshots: {handler.snapshots}"
        )

    def test_best_fitness_persisted_non_null_in_db(self, tmp_path):
        """After analysis mode run, algorithm_iterations.best_fitness is not NULL."""
        _setup_env(tmp_path)

        db_path = str(tmp_path / "test_analysis.db")
        os.environ["ANALYSIS_SQLITE_PATH"] = db_path
        os.environ["ANALYSIS_NUM_RUNS"] = "1"
        os.environ["ANALYSIS_SIMS_PER_RUN"] = "1"
        os.environ["ANALYSIS_EXPORT_CSV"] = "false"
        os.environ["ANALYSIS_FULL_REPORTS_ENABLED"] = "false"

        # Reload modules with new env
        for mod_name in list(sys.modules.keys()):
            if any(mod_name.startswith(p) for p in ["config.", "core.", "simulation.workers."]):
                del sys.modules[mod_name]

        from core.analysis_persistence import AnalysisPersistence
        from data.data_generator import generate_day_surgeries_data

        # Run minimal analysis scenario manually
        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)

        from core.iteration_callback import AnalysisIterationHandler
        from algorithms import ga

        handler = AnalysisIterationHandler(policy="best_only")
        ga.run(surgeries_data, job_ids, seed=1, on_iteration=handler)

        persistence = AnalysisPersistence(db_path)
        persistence.init_db()
        run_id = persistence.insert_run(num_sims=1, num_procs=5, config="{}")
        last_snap = handler.snapshots[-1] if handler.snapshots else None
        sim_id = persistence.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=1.0,
            final_makespan=last_snap.best_makespan if last_snap else None,
            combined_obj=None,
            algo_time_s=0.5,
        )

        algo_iter_rows = [
            {
                "algo_step": s.algo_step,
                "best_fitness": s.best_fitness,
                "best_makespan": s.best_makespan,
                "iteration_fitness": s.iteration_fitness,
                "iteration_makespan": s.iteration_makespan,
            }
            for s in handler.snapshots
        ]
        persistence.save_algorithm_iterations_batch(sim_id, algo_iter_rows)

        # Verify best_fitness is not NULL in DB
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT best_fitness FROM algorithm_iterations WHERE sim_id = ?", (sim_id,)
        ).fetchall()
        conn.close()

        assert len(rows) > 0, "No algorithm_iterations rows inserted"
        null_count = sum(1 for r in rows if r[0] is None)
        assert null_count == 0, f"{null_count}/{len(rows)} rows have NULL best_fitness"

    def test_simulations_final_makespan_non_null_in_db(self, tmp_path):
        """After analysis mode run, simulations.final_makespan is not NULL."""
        _setup_env(tmp_path)
        db_path = str(tmp_path / "test_sims_combined.db")

        from core.analysis_persistence import AnalysisPersistence
        from core.iteration_callback import AnalysisIterationHandler
        from data.data_generator import generate_day_surgeries_data
        from algorithms import ga

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)
        handler = AnalysisIterationHandler(policy="best_only")
        ga.run(surgeries_data, job_ids, seed=3, on_iteration=handler)

        best_makespan_val = (
            handler.snapshots[-1].best_makespan if handler.snapshots else None
        )
        assert best_makespan_val is not None, (
            "Handler produced no snapshots with best_makespan"
        )

        persistence = AnalysisPersistence(db_path)
        persistence.init_db()
        run_id = persistence.insert_run(num_sims=1, num_procs=5, config="{}")
        persistence.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=1.0,
            final_makespan=best_makespan_val,
            combined_obj=None,
            algo_time_s=0.5,
        )

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT final_makespan FROM simulations WHERE run_id = ?", (run_id,)
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] is not None, "simulations.final_makespan is NULL"


# ===========================================================================
# Fix 2: Checkpoint reports with real schedule artifacts
# ===========================================================================


class TestCheckpointReportsRealArtifacts:
    """generate_checkpoint_report must produce real files when schedule is present."""

    def test_generate_checkpoint_report_creates_csv_when_schedule_nonempty(
        self, tmp_path
    ):
        """generate_checkpoint_report creates schedule CSV when best_run has schedule."""
        _setup_env(tmp_path)
        from core.report_generator import ReportGenerator
        from config.config import ALL_ROOMS

        # Build a minimal schedule (one entry per operation)
        schedule = [
            {
                "Job": 1,
                "Operation": 1,
                "Room": "P1",
                "Start": 0.0,
                "Finish": 10.0,
                "ProcessingEnd": 10.0,
                "SetupUsed": 5.0,
                "TransitionUsed": 0.0,
                "CleanupUsed": 0.0,
            },
            {
                "Job": 1,
                "Operation": 2,
                "Room": "P1",
                "Start": 10.0,
                "Finish": 50.0,
                "ProcessingEnd": 45.0,
                "SetupUsed": 0.0,
                "TransitionUsed": 5.0,
                "CleanupUsed": 5.0,
            },
        ]
        best_run = {
            "algo_name": "GA",
            "makespan": 50.0,
            "schedule": schedule,
            "sim_num": 0,
            "job_label_map": None,
        }
        output_dirs = {
            "csv": str(tmp_path / "csv"),
            "plots": str(tmp_path / "plots"),
        }
        os.makedirs(output_dirs["csv"], exist_ok=True)
        os.makedirs(os.path.join(output_dirs["plots"], "gantt"), exist_ok=True)

        rg = ReportGenerator()
        rg.generate_checkpoint_report(best_run, output_dirs, ALL_ROOMS)

        # CSV must be created
        csv_files = list(Path(output_dirs["csv"]).rglob("*.csv"))
        assert len(csv_files) > 0, (
            f"No CSV files created in {output_dirs['csv']} even though schedule was provided"
        )

    def test_generate_checkpoint_report_skips_when_schedule_empty(self, tmp_path):
        """generate_checkpoint_report gracefully handles empty schedule (no crash)."""
        _setup_env(tmp_path)
        from core.report_generator import ReportGenerator
        from config.config import ALL_ROOMS

        best_run = {
            "algo_name": "GA",
            "makespan": float("inf"),
            "schedule": [],
            "sim_num": 0,
            "job_label_map": None,
        }
        output_dirs = {
            "csv": str(tmp_path / "csv"),
            "plots": str(tmp_path / "plots"),
        }
        os.makedirs(output_dirs["csv"], exist_ok=True)
        os.makedirs(os.path.join(output_dirs["plots"], "gantt"), exist_ok=True)

        rg = ReportGenerator()
        # Must NOT raise
        rg.generate_checkpoint_report(best_run, output_dirs, ALL_ROOMS)

    def test_analysis_runner_stores_schedule_for_checkpoint(self, tmp_path):
        """SimulationRunner._generate_checkpoint_reports uses real schedule from in-memory store."""
        _setup_env(tmp_path)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()

        # Inject a fake schedule store as if populated during analysis run
        schedule_store = {
            ("GA", 0): [  # (algo_name, sim_i): schedule
                {
                    "Job": 1,
                    "Operation": 1,
                    "Room": "P1",
                    "Start": 0.0,
                    "Finish": 10.0,
                    "ProcessingEnd": 10.0,
                    "SetupUsed": 5.0,
                    "TransitionUsed": 0.0,
                    "CleanupUsed": 0.0,
                }
            ]
        }

        # Build minimal DB with checkpoint data
        db_path = str(tmp_path / "cp_test.db")
        from core.analysis_persistence import AnalysisPersistence

        persistence = AnalysisPersistence(db_path)
        persistence.init_db()
        run_id = persistence.insert_run(num_sims=1, num_procs=5, config="{}")
        sim_id = persistence.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=1.0,
            final_makespan=50.0,
            combined_obj=50.0,
            algo_time_s=0.5,
        )
        persistence.reconstruct_checkpoints(run_id, 1)  # 1s interval

        # generate_checkpoint_reports must accept schedule_store kwarg
        runner._generate_checkpoint_reports(
            persistence,
            run_id,
            run_idx=0,
            schedule_store=schedule_store,
        )
        # We verify no exception raised and the method accepts schedule_store


# ===========================================================================
# Fix 3: results/<timestamp>/... directory structure
# ===========================================================================


class TestTimestampedOutputDirs:
    """Analysis mode must use results/<timestamp>/... directory structure."""

    def test_setup_analysis_directories_uses_timestamp(self, tmp_path):
        """FileManager.setup_analysis_directories creates timestamped path."""
        from core.file_manager import FileManager

        fm = FileManager(base_dir=str(tmp_path))
        timestamp = "20260416_120000"
        output_dirs = fm.setup_analysis_directories(timestamp, "elective")

        assert timestamp in output_dirs["csv"], (
            f"Timestamp not in csv path: {output_dirs['csv']}"
        )
        assert timestamp in output_dirs["plots"], (
            f"Timestamp not in plots path: {output_dirs['plots']}"
        )
        assert os.path.isdir(output_dirs["csv"])
        assert os.path.isdir(output_dirs["plots"])

    def test_analysis_mode_creates_timestamped_results_dir(self, tmp_path):
        """run_elective_analysis_mode generates a timestamp and passes it to _generate_checkpoint_reports."""
        _setup_env(tmp_path)
        import re
        from core.simulation_runner import SimulationRunner
        import core.simulation_runner as sr_module

        runner = SimulationRunner()

        # Mock _generate_checkpoint_reports to capture timestamp argument
        captured_timestamps = []

        def mock_generate_cp(
            persistence, run_id, run_idx, schedule_store=None, timestamp=None
        ):
            captured_timestamps.append(timestamp)

        runner._generate_checkpoint_reports = mock_generate_cp

        # Patch module-level constants so only 1 run of 1 sim executes and
        # ANALYSIS_FULL_REPORTS_ENABLED=True triggers the timestamp path
        db_path = str(tmp_path / "test_ts.db")
        old_vals = {
            "ANALYSIS_SQLITE_PATH": sr_module.ANALYSIS_SQLITE_PATH,
            "ANALYSIS_NUM_RUNS": sr_module.ANALYSIS_NUM_RUNS,
            "ANALYSIS_SIMS_PER_RUN": sr_module.ANALYSIS_SIMS_PER_RUN,
            "ANALYSIS_EXPORT_CSV": sr_module.ANALYSIS_EXPORT_CSV,
            "ANALYSIS_FULL_REPORTS_ENABLED": sr_module.ANALYSIS_FULL_REPORTS_ENABLED,
            "ANALYSIS_CHECKPOINT_INTERVAL": sr_module.ANALYSIS_CHECKPOINT_INTERVAL,
            "ANALYSIS_ARTIFACT_SAVE_MODE": sr_module.ANALYSIS_ARTIFACT_SAVE_MODE,
        }

        sr_module.ANALYSIS_SQLITE_PATH = db_path
        sr_module.ANALYSIS_NUM_RUNS = 1
        sr_module.ANALYSIS_SIMS_PER_RUN = 1
        sr_module.ANALYSIS_EXPORT_CSV = False
        sr_module.ANALYSIS_FULL_REPORTS_ENABLED = True
        sr_module.ANALYSIS_CHECKPOINT_INTERVAL = 9999  # large interval → 0 checkpoints
        sr_module.ANALYSIS_ARTIFACT_SAVE_MODE = "all"

        try:
            runner.run_elective_analysis_mode()
        finally:
            for k, v in old_vals.items():
                setattr(sr_module, k, v)

        # _generate_checkpoint_reports was called (even if 0 checkpoints)
        assert len(captured_timestamps) == 1, (
            f"Expected 1 call to _generate_checkpoint_reports, got {len(captured_timestamps)}"
        )
        ts = captured_timestamps[0]
        assert ts is not None, "_generate_checkpoint_reports called without timestamp"
        assert re.match(r"\d{8}_\d{6}", ts), (
            f"Timestamp does not match YYYYMMDD_HHMMSS: {ts}"
        )

    def test_analysis_mode_checkpoint_reports_use_timestamped_dirs(self, tmp_path):
        """_generate_checkpoint_reports creates output in results/<timestamp>/checkpoints/."""
        _setup_env(tmp_path)
        from core.simulation_runner import SimulationRunner
        from core.analysis_persistence import AnalysisPersistence
        from core.file_manager import FileManager

        results_dir = str(tmp_path / "results_ts")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)

        db_path = str(tmp_path / "ts_test.db")
        persistence = AnalysisPersistence(db_path)
        persistence.init_db()
        run_id = persistence.insert_run(num_sims=1, num_procs=5, config="{}")
        sim_id = persistence.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=1.0,
            final_makespan=50.0,
            combined_obj=50.0,
            algo_time_s=0.5,
        )
        persistence.reconstruct_checkpoints(run_id, 1)  # 1s interval

        # Build schedule_store
        schedule_store = {("GA", 0): []}

        # _generate_checkpoint_reports must accept timestamp kwarg
        runner._generate_checkpoint_reports(
            persistence,
            run_id,
            run_idx=0,
            schedule_store=schedule_store,
            timestamp="20260416_120000",
        )

        # Verify timestamped directory was used (directory exists)
        ts_dir = os.path.join(results_dir, "20260416_120000")
        assert os.path.isdir(ts_dir), f"Timestamped directory not created: {ts_dir}"
