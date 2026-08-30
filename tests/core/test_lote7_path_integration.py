"""
Tests for Lote 7 corrective fixes:
1. SQLite and CSV exports land under results/<timestamp>/...
2. Checkpoint path is results/<timestamp>/checkpoints/<id>/ (no run<N> prefix)
3. generate_analysis_reports() uses v2 schema (simulations, sim_id)
4. Integration tests for final runtime paths

Schema v6 additions (TDD RED — tasks 1.1–1.2):
5. generate_analysis_reports() exports patient_wait_metrics.csv,
   schedule_quality_metrics.csv and simulation_summary.csv from v6 DB.

Strict TDD: tests written BEFORE implementation.
"""

import os
import yaml
import sqlite3
import sys
import tempfile
from pathlib import Path

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


def _build_minimal_db(db_path: str, num_algo_iters: int = 2) -> tuple[int, int]:
    """Creates a minimal v2 DB with one run, one simulation, and breakdowns.

    Returns (run_id, sim_id).
    """
    from core.analysis_persistence import AnalysisPersistence

    p = AnalysisPersistence(db_path)
    p.init_db()
    run_id = p.insert_run(num_sims=1, num_procs=5, config="{}")
    sim_id = p.insert_simulation(
        run_id=run_id,
        sim_index=0,
        algo_name="GA",
        wall_clock_s=1.0,
        final_makespan=100.0,
        combined_obj=100.0,
        algo_time_s=0.5,
    )
    algo_rows = [
        {
            "algo_step": i,
            "best_fitness": 100.0 - i,
            "best_makespan": 100.0 - i,
            "iteration_fitness": 100.0 - i,
            "iteration_makespan": 100.0 - i,
        }
        for i in range(num_algo_iters)
    ]
    p.save_algorithm_iterations_batch(sim_id, algo_rows)
    p.save_breakdowns_batch(
        sim_id,
        [
            {
                "job_id": 1,
                "codigo_cie10": "J12.0",
                "grupo": "top20",
                "setup_min": 10.0,
                "proc_time_min": 30.0,
                "transition_min": 5.0,
                "cleanup_min": 5.0,
            },
        ],
    )
    p.close()
    return run_id, sim_id


# ===========================================================================
# Issue 1: SQLite DB must be placed inside results/<timestamp>/
# ===========================================================================


class TestSQLiteTimestampedPath:
    """The SQLite file created by run_elective_analysis_mode must be inside
    results/<timestamp>/ not at a global path like results/analysis.db."""

    def test_analysis_mode_db_path_contains_timestamp(self, tmp_path):
        """run_elective_analysis_mode must create SQLite inside results/<timestamp>/."""
        _setup_env(tmp_path)
        import re
        from core.simulation_runner import SimulationRunner
        import core.simulation_runner as sr_module
        from core.file_manager import FileManager

        results_dir = str(tmp_path / "results")
        os.makedirs(results_dir, exist_ok=True)

        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)

        old_vals = {
            "ANALYSIS_NUM_RUNS": sr_module.ANALYSIS_NUM_RUNS,
            "ANALYSIS_SIMS_PER_RUN": sr_module.ANALYSIS_SIMS_PER_RUN,
            "ANALYSIS_EXPORT_CSV": sr_module.ANALYSIS_EXPORT_CSV,
            "ANALYSIS_FULL_REPORTS_ENABLED": sr_module.ANALYSIS_FULL_REPORTS_ENABLED,
            "ANALYSIS_CHECKPOINT_INTERVAL": sr_module.ANALYSIS_CHECKPOINT_INTERVAL,
            "ANALYSIS_ARTIFACT_SAVE_MODE": sr_module.ANALYSIS_ARTIFACT_SAVE_MODE,
        }
        sr_module.ANALYSIS_NUM_RUNS = 1
        sr_module.ANALYSIS_SIMS_PER_RUN = 1
        sr_module.ANALYSIS_EXPORT_CSV = False
        sr_module.ANALYSIS_FULL_REPORTS_ENABLED = False
        sr_module.ANALYSIS_CHECKPOINT_INTERVAL = 9999
        sr_module.ANALYSIS_ARTIFACT_SAVE_MODE = "all"

        try:
            runner.run_elective_analysis_mode()
        finally:
            for k, v in old_vals.items():
                setattr(sr_module, k, v)

        # After the run, a timestamped subdir with analysis.db must exist
        db_files = list(Path(results_dir).rglob("analysis.db"))
        assert len(db_files) >= 1, (
            f"No analysis.db found under {results_dir}. "
            f"Contents: {list(Path(results_dir).rglob('*'))}"
        )
        db_path_used = str(db_files[0])
        assert re.search(r"\d{8}_\d{6}", db_path_used), (
            f"DB path does not contain a timestamp: {db_path_used}"
        )

    def test_csv_exports_use_timestamped_paths(self, tmp_path):
        """When ANALYSIS_EXPORT_CSV=True, CSV files must be inside results/<timestamp>/."""
        _setup_env(tmp_path)
        import re
        from core.simulation_runner import SimulationRunner
        import core.simulation_runner as sr_module
        from core.file_manager import FileManager

        results_dir = str(tmp_path / "results")
        os.makedirs(results_dir, exist_ok=True)

        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)

        old_vals = {
            "ANALYSIS_NUM_RUNS": sr_module.ANALYSIS_NUM_RUNS,
            "ANALYSIS_SIMS_PER_RUN": sr_module.ANALYSIS_SIMS_PER_RUN,
            "ANALYSIS_EXPORT_CSV": sr_module.ANALYSIS_EXPORT_CSV,
            "ANALYSIS_FULL_REPORTS_ENABLED": sr_module.ANALYSIS_FULL_REPORTS_ENABLED,
            "ANALYSIS_CHECKPOINT_INTERVAL": sr_module.ANALYSIS_CHECKPOINT_INTERVAL,
            "ANALYSIS_ARTIFACT_SAVE_MODE": sr_module.ANALYSIS_ARTIFACT_SAVE_MODE,
        }
        sr_module.ANALYSIS_NUM_RUNS = 1
        sr_module.ANALYSIS_SIMS_PER_RUN = 1
        sr_module.ANALYSIS_EXPORT_CSV = True
        sr_module.ANALYSIS_FULL_REPORTS_ENABLED = False
        sr_module.ANALYSIS_CHECKPOINT_INTERVAL = 9999
        sr_module.ANALYSIS_ARTIFACT_SAVE_MODE = "all"

        try:
            runner.run_elective_analysis_mode()
        finally:
            for k, v in old_vals.items():
                setattr(sr_module, k, v)

        csv_files = list(Path(results_dir).rglob("*.csv"))
        assert len(csv_files) >= 3, f"Expected at least 3 CSV files, found: {csv_files}"
        for csv_path in csv_files:
            path_str = str(csv_path)
            assert re.search(r"\d{8}_\d{6}", path_str), (
                f"CSV file path does not contain timestamp: {path_str}"
            )


# ===========================================================================
# Issue 2: Checkpoint path must be results/<timestamp>/checkpoints/<id>/
#          (NOT results/<timestamp>/run<N>/checkpoints/<id>/)
# ===========================================================================


class TestCheckpointPathWithRunPrefix:
    """Checkpoint output dirs must include run<N> in the path to isolate runs."""

    def test_checkpoint_path_has_run_prefix(self, tmp_path):
        """_generate_checkpoint_reports must call setup_analysis_directories
        with a scenario that includes 'run<N>' prefix."""
        _setup_env(tmp_path)
        from core.simulation_runner import SimulationRunner
        from core.analysis_persistence import AnalysisPersistence
        from core.file_manager import FileManager

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)

        captured_scenarios = []
        original_setup = runner.file_manager.setup_analysis_directories

        def _capturing_setup(timestamp, scenario, **kwargs):
            captured_scenarios.append(scenario)
            return original_setup(timestamp, scenario, **kwargs)

        runner.file_manager.setup_analysis_directories = _capturing_setup

        db_path = str(tmp_path / "chk_test.db")
        persistence = AnalysisPersistence(db_path)
        persistence.init_db()
        run_id = persistence.insert_run(num_sims=1, num_procs=5, config="{}")
        persistence.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=1.0,
            final_makespan=50.0,
            combined_obj=50.0,
            algo_time_s=0.5,
        )
        persistence.reconstruct_checkpoints(run_id, 1)

        schedule_store = {("GA", 0): []}
        runner._generate_checkpoint_reports(
            persistence,
            run_id,
            run_idx=0,
            schedule_store=schedule_store,
            timestamp="20260417_090000",
        )

        # All scenarios must include 'run<N>' in the path
        import re

        for scenario in captured_scenarios:
            assert re.search(r"run\d+", scenario), (
                f"Checkpoint scenario missing run<N> prefix: '{scenario}'"
            )

    def test_checkpoint_dir_structure_matches_spec(self, tmp_path):
        """Checkpoint directory must follow results/<timestamp>/checkpoints/<id>/ spec."""
        _setup_env(tmp_path)
        from core.simulation_runner import SimulationRunner
        from core.analysis_persistence import AnalysisPersistence
        from core.file_manager import FileManager

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)

        db_path = str(tmp_path / "spec_test.db")
        persistence = AnalysisPersistence(db_path)
        persistence.init_db()
        run_id = persistence.insert_run(num_sims=1, num_procs=5, config="{}")
        persistence.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=1.0,
            final_makespan=50.0,
            combined_obj=50.0,
            algo_time_s=0.5,
        )
        persistence.reconstruct_checkpoints(run_id, 1)

        schedule_store = {("GA", 0): []}
        runner._generate_checkpoint_reports(
            persistence,
            run_id,
            run_idx=0,
            schedule_store=schedule_store,
            timestamp="20260417_090000",
        )

        ts_dir = os.path.join(results_dir, "20260417_090000")
        assert os.path.isdir(ts_dir), f"Timestamped dir not created: {ts_dir}"

        # Must have checkpoints/run1 subdirectory under <timestamp>/
        chk_parent = os.path.join(ts_dir, "checkpoints", "run1")
        assert os.path.isdir(chk_parent), (
            f"Expected {chk_parent} to exist (with run<N> prefix). "
            f"Dir contents: {list(Path(ts_dir).iterdir())}"
        )


# ===========================================================================
# Issue 3: generate_analysis_reports() must use v2 schema
# ===========================================================================


class TestGenerateAnalysisReportsV2Schema:
    """generate_analysis_reports() must query v2 tables (simulations, sim_id)
    instead of v1 tables (iterations, iter_id)."""

    def test_generate_analysis_reports_does_not_produce_breakdown_csv(self, tmp_path):
        """generate_analysis_reports must NOT produce breakdown CSV as it was removed."""
        _setup_env(tmp_path)

        db_path = str(tmp_path / "v2_test.db")
        _build_minimal_db(db_path, num_algo_iters=2)

        output_dir = str(tmp_path / "analysis_out")
        from core.report_generator import ReportGenerator

        rg = ReportGenerator()
        rg.generate_analysis_reports(db_path, output_dir)

        bd_csv = os.path.join(output_dir, "analysis_breakdown.csv")
        assert not os.path.exists(bd_csv), f"Breakdown CSV was created but it should be removed: {bd_csv}"

    def test_generate_analysis_reports_exports_checkpoints_v2(self, tmp_path):
        """generate_analysis_reports checkpoints CSV uses v2 schema."""
        _setup_env(tmp_path)

        db_path = str(tmp_path / "v2_chk.db")
        from core.analysis_persistence import AnalysisPersistence

        p = AnalysisPersistence(db_path)
        p.init_db()
        run_id = p.insert_run(num_sims=1, num_procs=5, config="{}")
        p.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=1.0,
            final_makespan=50.0,
            combined_obj=50.0,
            algo_time_s=0.5,
        )
        p.reconstruct_checkpoints(run_id, 1)
        p.close()

        output_dir = str(tmp_path / "analysis_out2")
        from core.report_generator import ReportGenerator

        rg = ReportGenerator()
        rg.generate_analysis_reports(db_path, output_dir)

        chk_csv = os.path.join(output_dir, "analysis_checkpoints.csv")
        assert os.path.exists(chk_csv), f"Checkpoints CSV not created: {chk_csv}"

        import csv

        with open(chk_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0, "Checkpoints CSV is empty"
        assert "algo_name" in reader.fieldnames

    def test_generate_analysis_reports_with_no_v1_tables_works(self, tmp_path):
        """generate_analysis_reports must not crash when v1 tables (iterations) absent."""
        _setup_env(tmp_path)

        db_path = str(tmp_path / "fresh_v2.db")
        _build_minimal_db(db_path)

        # Verify there's no 'iterations' table (pure v2)
        conn = sqlite3.connect(db_path)
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "iterations" not in tables, (
            "DB should be pure v2 (no 'iterations' table)"
        )

        output_dir = str(tmp_path / "no_v1_out")
        from core.report_generator import ReportGenerator

        rg = ReportGenerator()
        # Must not raise OperationalError about missing 'iterations' table
        rg.generate_analysis_reports(db_path, output_dir)


# ===========================================================================
# Issue 4: Integration test — full analysis run produces timestamped artifacts
# ===========================================================================


class TestFullAnalysisRunTimestampedArtifacts:
    """End-to-end integration: run_elective_analysis_mode must produce all
    artifacts (DB, CSVs) inside results/<timestamp>/, not global paths."""

    def test_full_analysis_run_artifacts_under_timestamp_dir(self, tmp_path):
        """After a minimal analysis run, at least the DB file must be
        inside a timestamped subdirectory of results/."""
        _setup_env(tmp_path)
        import re
        from core.simulation_runner import SimulationRunner
        import core.simulation_runner as sr_module
        from core.file_manager import FileManager

        results_dir = str(tmp_path / "results")
        os.makedirs(results_dir, exist_ok=True)

        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)

        old_vals = {
            "ANALYSIS_NUM_RUNS": sr_module.ANALYSIS_NUM_RUNS,
            "ANALYSIS_SIMS_PER_RUN": sr_module.ANALYSIS_SIMS_PER_RUN,
            "ANALYSIS_EXPORT_CSV": sr_module.ANALYSIS_EXPORT_CSV,
            "ANALYSIS_FULL_REPORTS_ENABLED": sr_module.ANALYSIS_FULL_REPORTS_ENABLED,
            "ANALYSIS_CHECKPOINT_INTERVAL": sr_module.ANALYSIS_CHECKPOINT_INTERVAL,
            "ANALYSIS_ARTIFACT_SAVE_MODE": sr_module.ANALYSIS_ARTIFACT_SAVE_MODE,
        }

        sr_module.ANALYSIS_NUM_RUNS = 1
        sr_module.ANALYSIS_SIMS_PER_RUN = 1
        sr_module.ANALYSIS_EXPORT_CSV = True
        sr_module.ANALYSIS_FULL_REPORTS_ENABLED = False
        sr_module.ANALYSIS_CHECKPOINT_INTERVAL = 9999
        sr_module.ANALYSIS_ARTIFACT_SAVE_MODE = "all"

        try:
            runner.run_elective_analysis_mode()
        finally:
            for k, v in old_vals.items():
                setattr(sr_module, k, v)

        # At least one timestamped directory must exist under results_dir
        subdirs = [
            d
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
            and re.match(r"\d{8}_\d{6}", d)
        ]
        assert len(subdirs) >= 1, (
            f"No timestamped subdirectory found under {results_dir}. "
            f"Contents: {os.listdir(results_dir)}"
        )

        # Inside the timestamped dir, there must be a .db file or CSV
        ts_dir = os.path.join(results_dir, subdirs[0])
        all_files = list(Path(ts_dir).rglob("*"))
        data_files = [f for f in all_files if f.suffix in (".db", ".csv")]
        assert len(data_files) >= 1, (
            f"No .db or .csv files found under timestamped dir {ts_dir}. "
            f"Files: {all_files}"
        )


# ===========================================================================
# Issue 5 (RED — task 1.1/1.2): generate_analysis_reports() exports v6 artefacts
# ===========================================================================


def _build_v6_db(db_path: str) -> tuple[int, int]:
    """Creates a v6 DB with patient_wait_metrics and schedule_quality_metrics populated.

    Returns (run_id, sim_id).
    """
    from core.analysis_persistence import AnalysisPersistence

    p = AnalysisPersistence(db_path)
    p.init_db()
    run_id = p.insert_run(num_sims=1, num_procs=3, config={})
    sim_id = p.insert_simulation(
        run_id=run_id,
        sim_index=0,
        algo_name="GA",
        wall_clock_s=2.0,
        final_makespan=200.0,
        combined_obj=200.0,
        algo_time_s=1.0,
    )
    # Populate patient_wait_metrics
    p.save_patient_wait_batch(
        sim_id,
        [
            {
                "job_id": 1,
                "op1_room": "Q1",
                "op2_room": "Q2",
                "op1_finish": 60.0,
                "op2_start": 65.0,
                "transition_used": 5.0,
                "extra_wait_min": 0.0,
            },
            {
                "job_id": 2,
                "op1_room": "Q1",
                "op2_room": "Q2",
                "op1_finish": 100.0,
                "op2_start": 115.0,
                "transition_used": 5.0,
                "extra_wait_min": 10.0,
            },
        ],
    )
    # Populate schedule_quality_metrics
    p.save_quality_metrics(
        sim_id,
        {
            "rooms_used": 2,
            "total_overtime_min": 15.0,
            "max_room_overtime_min": 15.0,
            "personnel_count": 4,
            "workload_std_min": 3.0,
            "workload_max_min": 120.0,
            "workload_min_min": 90.0,
            "idle_gap_count": 2,
            "idle_gap_total_min": 20.0,
            "avg_idle_gap_min": 10.0,
            "value_added_ratio": 0.75,
        },
    )
    p.close()
    return run_id, sim_id


class TestGenerateAnalysisReportsV6:
    """generate_analysis_reports() must export v6 artefacts:
    patient_wait_metrics.csv, schedule_quality_metrics.csv, simulation_summary.csv."""

    def test_generate_analysis_reports_exports_patient_wait_metrics_csv(self, tmp_path):
        """generate_analysis_reports must create patient_wait_metrics.csv from v6 DB."""
        _setup_env(tmp_path)
        db_path = str(tmp_path / "v6_test.db")
        _build_v6_db(db_path)

        output_dir = str(tmp_path / "v6_out")
        from core.report_generator import ReportGenerator

        rg = ReportGenerator()
        rg.generate_analysis_reports(db_path, output_dir)

        pw_csv = os.path.join(output_dir, "patient_wait_metrics.csv")
        assert os.path.exists(pw_csv), (
            f"patient_wait_metrics.csv not created at {pw_csv}. "
            f"Dir contents: {os.listdir(output_dir)}"
        )
        import csv
        with open(pw_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1, "patient_wait_metrics.csv must have at least one data row"
        assert "job_id" in reader.fieldnames, (
            f"Expected 'job_id' column, got: {reader.fieldnames}"
        )
        assert "extra_wait_min" in reader.fieldnames, (
            f"Expected 'extra_wait_min' column, got: {reader.fieldnames}"
        )

    def test_generate_analysis_reports_exports_schedule_quality_metrics_csv(self, tmp_path):
        """generate_analysis_reports must create schedule_quality_metrics.csv from v6 DB."""
        _setup_env(tmp_path)
        db_path = str(tmp_path / "v6_test2.db")
        _build_v6_db(db_path)

        output_dir = str(tmp_path / "v6_out2")
        from core.report_generator import ReportGenerator

        rg = ReportGenerator()
        rg.generate_analysis_reports(db_path, output_dir)

        sq_csv = os.path.join(output_dir, "schedule_quality_metrics.csv")
        assert os.path.exists(sq_csv), (
            f"schedule_quality_metrics.csv not created at {sq_csv}. "
            f"Dir contents: {os.listdir(output_dir)}"
        )
        import csv
        with open(sq_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1, "schedule_quality_metrics.csv must have at least one data row"
        assert "total_overtime_min" in reader.fieldnames, (
            f"Expected 'total_overtime_min' column, got: {reader.fieldnames}"
        )
        assert "value_added_ratio" in reader.fieldnames, (
            f"Expected 'value_added_ratio' column, got: {reader.fieldnames}"
        )

    def test_generate_analysis_reports_exports_simulation_summary_csv(self, tmp_path):
        """generate_analysis_reports must create simulation_summary.csv from v6 DB view."""
        _setup_env(tmp_path)
        db_path = str(tmp_path / "v6_test3.db")
        _build_v6_db(db_path)

        output_dir = str(tmp_path / "v6_out3")
        from core.report_generator import ReportGenerator

        rg = ReportGenerator()
        rg.generate_analysis_reports(db_path, output_dir)

        ss_csv = os.path.join(output_dir, "simulation_summary.csv")
        assert os.path.exists(ss_csv), (
            f"simulation_summary.csv not created at {ss_csv}. "
            f"Dir contents: {os.listdir(output_dir)}"
        )
        import csv
        with open(ss_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1, "simulation_summary.csv must have at least one data row"
        assert "final_makespan" in reader.fieldnames, (
            f"Expected 'final_makespan' column, got: {reader.fieldnames}"
        )

    def test_simulation_summary_csv_includes_nullable_wait_columns(self, tmp_path):
        """simulation_summary.csv must include LEFT JOIN columns even when null."""
        _setup_env(tmp_path)
        # DB with quality metrics but WITHOUT patient_wait_metrics
        from core.analysis_persistence import AnalysisPersistence
        db_path = str(tmp_path / "v6_nullwait.db")
        p = AnalysisPersistence(db_path)
        p.init_db()
        run_id = p.insert_run(num_sims=1, num_procs=2, config={})
        sim_id = p.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=1.0,
            final_makespan=100.0,
            combined_obj=100.0,
            algo_time_s=0.5,
        )
        # quality metrics present, patient_wait intentionally absent
        p.save_quality_metrics(
            sim_id,
            {
                "rooms_used": 1,
                "total_overtime_min": 0.0,
                "max_room_overtime_min": 0.0,
                "personnel_count": 2,
                "workload_std_min": 0.0,
                "workload_max_min": 50.0,
                "workload_min_min": 50.0,
                "idle_gap_count": 0,
                "idle_gap_total_min": 0.0,
                "avg_idle_gap_min": 0.0,
                "value_added_ratio": 1.0,
            },
        )
        p.close()

        output_dir = str(tmp_path / "v6_nullwait_out")
        from core.report_generator import ReportGenerator
        rg = ReportGenerator()
        rg.generate_analysis_reports(db_path, output_dir)

        ss_csv = os.path.join(output_dir, "simulation_summary.csv")
        assert os.path.exists(ss_csv), f"simulation_summary.csv missing: {os.listdir(output_dir)}"
        import csv
        with open(ss_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        # The LEFT JOIN columns for patient wait may be empty/None — that's fine
        assert "final_makespan" in reader.fieldnames
        assert "rooms_used" in reader.fieldnames

    def test_generate_analysis_reports_exports_best_runs_by_mh_csv(self, tmp_path):
        """generate_analysis_reports must create best_runs_by_mh.csv from v6 DB view."""
        _setup_env(tmp_path)
        db_path = str(tmp_path / "v6_test_best_runs.db")
        _build_v6_db(db_path)

        output_dir = str(tmp_path / "v6_out_best_runs")
        from core.report_generator import ReportGenerator

        rg = ReportGenerator()
        rg.generate_analysis_reports(db_path, output_dir)

        best_csv = os.path.join(output_dir, "best_runs_by_mh.csv")
        assert os.path.exists(best_csv), (
            f"best_runs_by_mh.csv not created at {best_csv}. "
            f"Dir contents: {os.listdir(output_dir)}"
        )
        import csv
        with open(best_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1, "best_runs_by_mh.csv must have at least one data row"
        assert "combined_obj" in reader.fieldnames, (
            f"Expected 'combined_obj' column, got: {reader.fieldnames}"
        )
        assert "algo_name" in reader.fieldnames
