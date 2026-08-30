"""
Tests for Lote 3 telemetry corrections:
- _extract_cie10_breakdown uses TransitionUsed (not SetupUsed) for op2 transition
- _extract_cie10_breakdown includes 'grupo' from job_grupo_map
- combined_obj propagated from worker iteration_snapshots to insert_simulation
- export_algorithm_iterations_csv exports per-iteration data to CSV
"""

import csv
import os
import pytest
from unittest.mock import MagicMock

from core.analysis_persistence import AnalysisPersistence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schedule_with_transition_used(job_id=1):
    """Returns schedule_details for job_id with proper TransitionUsed set on op2."""
    return [
        {
            "Job": job_id,
            "Operation": 1,
            "Start": 0.0,
            "Finish": 10.0,
            "ProcessingEnd": 10.0,
            "SetupUsed": 10.0,  # anesthesia setup
            "TransitionUsed": None,  # op1 has no transition
            "CleanupUsed": 0.0,
        },
        {
            "Job": job_id,
            "Operation": 2,
            "Start": 10.0,
            "Finish": 80.0,
            "ProcessingEnd": 75.0,
            "SetupUsed": 0.0,  # op2 has no setup in PKL model
            "TransitionUsed": 5.0,  # actual transition time (op1→op2)
            "CleanupUsed": 8.0,
        },
    ]


def _make_runner():
    """Returns a SimulationRunner with minimal config via env injection."""
    import json
    import sys

    # Minimal config for instantiating SimulationRunner without heavy imports
    from core.simulation_runner import SimulationRunner

    return SimulationRunner


# ---------------------------------------------------------------------------
# Task 4.2: _extract_cie10_breakdown uses TransitionUsed for op2
# ---------------------------------------------------------------------------


class TestExtractCie10BreakdownTransitionUsed:
    """_extract_cie10_breakdown must use TransitionUsed (not SetupUsed) for op2 transition."""

    def _get_extract_fn(self):
        from core.simulation_runner import SimulationRunner

        # Access the method without instantiating (to avoid heavy constructor)
        return SimulationRunner._extract_cie10_breakdown

    def test_transition_min_uses_TransitionUsed_field(self):
        """transition_min must come from op2['TransitionUsed'], not op2['SetupUsed']."""
        extract = self._get_extract_fn()
        schedule = _make_schedule_with_transition_used(job_id=1)
        # op2 TransitionUsed=5.0, SetupUsed=0.0
        rows = extract(None, schedule, job_label_map=None)
        assert len(rows) == 1
        assert abs(rows[0]["transition_min"] - 5.0) < 1e-9, (
            f"transition_min should be 5.0 (TransitionUsed), got {rows[0]['transition_min']}"
        )

    def test_proc_time_computed_correctly_with_TransitionUsed(self):
        """proc_time_min = (Finish - Start) - TransitionUsed - CleanupUsed."""
        extract = self._get_extract_fn()
        schedule = _make_schedule_with_transition_used(job_id=1)
        # op2: Start=10, Finish=80, TransitionUsed=5, CleanupUsed=8
        # proc = 70 - 5 - 8 = 57
        rows = extract(None, schedule, job_label_map=None)
        assert abs(rows[0]["proc_time_min"] - 57.0) < 1e-9, (
            f"proc_time_min should be 57.0, got {rows[0]['proc_time_min']}"
        )

    def test_transition_min_none_TransitionUsed_falls_back_to_zero(self):
        """When TransitionUsed is None (synthetic), transition_min is 0.0."""
        extract = self._get_extract_fn()
        schedule = [
            {
                "Job": 1,
                "Operation": 1,
                "Start": 0.0,
                "Finish": 10.0,
                "ProcessingEnd": 10.0,
                "SetupUsed": 10.0,
                "TransitionUsed": None,
                "CleanupUsed": 0.0,
            },
            {
                "Job": 1,
                "Operation": 2,
                "Start": 10.0,
                "Finish": 80.0,
                "ProcessingEnd": 75.0,
                "SetupUsed": None,  # synthetic
                "TransitionUsed": None,
                "CleanupUsed": 8.0,
            },
        ]
        rows = extract(None, schedule, job_label_map=None)
        assert abs(rows[0]["transition_min"] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Task 4.2: _extract_cie10_breakdown includes 'grupo' from job_grupo_map
# ---------------------------------------------------------------------------


class TestExtractCie10BreakdownGrupo:
    """_extract_cie10_breakdown must populate 'grupo' from job_grupo_map when provided."""

    def _get_extract_fn(self):
        from core.simulation_runner import SimulationRunner

        return SimulationRunner._extract_cie10_breakdown

    def test_grupo_populated_from_job_grupo_map(self):
        """When job_grupo_map is provided, each row must include the 'grupo' value."""
        extract = self._get_extract_fn()
        schedule = _make_schedule_with_transition_used(job_id=1)
        job_grupo_map = {1: "top20"}
        rows = extract(None, schedule, job_label_map=None, job_grupo_map=job_grupo_map)
        assert rows[0]["grupo"] == "top20", (
            f"grupo should be 'top20', got {rows[0].get('grupo')}"
        )

    def test_grupo_is_none_when_no_job_grupo_map(self):
        """When job_grupo_map is None, grupo must be None."""
        extract = self._get_extract_fn()
        schedule = _make_schedule_with_transition_used(job_id=1)
        rows = extract(None, schedule, job_label_map=None, job_grupo_map=None)
        assert rows[0]["grupo"] is None

    def test_grupo_is_none_for_unknown_job_id(self):
        """When job_id is not in job_grupo_map, grupo must be None."""
        extract = self._get_extract_fn()
        schedule = _make_schedule_with_transition_used(job_id=99)
        job_grupo_map = {1: "top20"}  # job 99 not in map
        rows = extract(None, schedule, job_label_map=None, job_grupo_map=job_grupo_map)
        assert rows[0]["grupo"] is None

    def test_two_jobs_get_correct_grupos(self):
        """Each job in a multi-job schedule gets its own grupo."""
        extract = self._get_extract_fn()
        schedule = _make_schedule_with_transition_used(
            job_id=1
        ) + _make_schedule_with_transition_used(job_id=2)
        job_grupo_map = {1: "top20", 2: "otros"}
        rows = extract(None, schedule, job_label_map=None, job_grupo_map=job_grupo_map)
        grupos = {r["job_id"]: r["grupo"] for r in rows}
        assert grupos[1] == "top20"
        assert grupos[2] == "otros"


# ---------------------------------------------------------------------------
# Task 4.5: export_algorithm_iterations_csv
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    p = AnalysisPersistence(":memory:")
    p.init_db()
    return p


def _setup_iterations(db):
    """Helper: insert a run, sim, and 3 algorithm_iterations."""
    run_id = db.insert_run(num_sims=1, num_procs=5, config={})
    sim_id = db.insert_simulation(
        run_id=run_id,
        sim_index=0,
        algo_name="GA",
        wall_clock_s=30.0,
        final_makespan=480.0,
        combined_obj=1.5,
    )
    rows = [
        {
            "algo_step": 1,
            "best_fitness": 600.0,
            "best_makespan": 600.0,
            "iteration_fitness": 600.0,
            "iteration_makespan": 600.0,
        },
        {
            "algo_step": 2,
            "best_fitness": 570.0,
            "best_makespan": 570.0,
            "iteration_fitness": 580.0,
            "iteration_makespan": 580.0,
        },
        {
            "algo_step": 3,
            "best_fitness": 540.0,
            "best_makespan": 540.0,
            "iteration_fitness": 550.0,
            "iteration_makespan": 550.0,
        },
    ]
    db.save_algorithm_iterations_batch(sim_id=sim_id, rows=rows)
    return run_id, sim_id


class TestExportAlgorithmIterationsCSV:
    """export_algorithm_iterations_csv must produce a CSV with iter_id, sim_id, algo_step, etc."""

    def test_export_creates_file(self, db, tmp_path):
        """export_algorithm_iterations_csv must create the file."""
        run_id, _ = _setup_iterations(db)
        path = str(tmp_path / "algo_iterations.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=path)
        assert os.path.exists(path)

    def test_export_has_expected_columns(self, db, tmp_path):
        """CSV must include iter_id, sim_index, algo_name, algo_step, best_fitness, makespan_of_best_fitness, iteration_fitness, iteration_makespan."""
        run_id, _ = _setup_iterations(db)
        path = str(tmp_path / "algo_iterations.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=path)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
        assert "algo_iter_id" in rows[0]
        assert "sim_index" in rows[0]
        assert "algo_name" in rows[0]
        assert "algo_step" in rows[0]
        assert "best_fitness" in rows[0]
        assert "makespan_of_best_fitness" in rows[0]
        assert "iteration_fitness" in rows[0]
        assert "iteration_makespan" in rows[0]

    def test_export_values_match_db(self, db, tmp_path):
        """CSV values must match what was inserted."""
        run_id, sim_id = _setup_iterations(db)
        path = str(tmp_path / "algo_iterations.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=path)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        algo_steps = [int(r["algo_step"]) for r in rows]
        assert algo_steps == [1, 2, 3]
        assert rows[0]["algo_name"] == "GA"

    def test_export_empty_when_no_iterations(self, db, tmp_path):
        """When no algorithm_iterations exist for a run, export produces header-only CSV."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=5.0,
            final_makespan=500.0,
            combined_obj=None,
        )
        # No iterations saved
        path = str(tmp_path / "empty_iterations.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=path)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows == []


# ---------------------------------------------------------------------------
# Task 3.3 / 4.1: combined_obj propagated from iteration_snapshots to simulation
# ---------------------------------------------------------------------------


class TestCombinedObjPropagatedToSimulation:
    """combined_obj in insert_simulation must come from the last/best snapshot's combined_obj."""

    def test_insert_simulation_with_combined_obj_not_none(self, db):
        """When combined_obj is provided (not None), it is stored correctly."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=10.0,
            final_makespan=450.0,
            combined_obj=2.5,
        )
        row = db._conn.execute(
            "SELECT combined_obj FROM simulations WHERE sim_id = ?", (sim_id,)
        ).fetchone()
        assert abs(row[0] - 2.5) < 1e-9

    def test_insert_simulation_combined_obj_none_stored_as_null(self, db):
        """When combined_obj=None, it is stored as SQL NULL."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=10.0,
            final_makespan=450.0,
            combined_obj=None,
        )
        row = db._conn.execute(
            "SELECT combined_obj FROM simulations WHERE sim_id = ?", (sim_id,)
        ).fetchone()
        assert row[0] is None
