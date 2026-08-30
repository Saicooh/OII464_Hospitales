"""
Tests para core/analysis_persistence.py — schema v2 (actualizado desde v1).

Migrado para usar la API v2:
- 'iterations' → 'simulations'
- save_iterations_batch() → insert_simulation() (singular)
- save_breakdowns_batch() usa sim_id en vez de iter_id_map
- _make_run_with_iterations → _make_run_with_simulations

Estrategia: SQLite en memoria (:memory:) para aislamiento total.
"""

import sqlite3
import pytest

from core.analysis_persistence import AnalysisPersistence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """Instancia con base de datos en memoria; inicializada."""
    persistence = AnalysisPersistence(":memory:")
    persistence.init_db()
    return persistence


# ---------------------------------------------------------------------------
# Schema: init_db crea las tablas esperadas
# ---------------------------------------------------------------------------


class TestSchema:
    def test_runs_table_exists(self, db):
        """init_db debe crear la tabla 'runs'."""
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "runs" in tables

    def test_simulations_table_exists(self, db):
        """init_db debe crear la tabla 'simulations' (schema v2)."""
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "simulations" in tables

    def test_cie10_breakdown_table_exists(self, db):
        """init_db debe crear la tabla 'cie10_breakdown'."""
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "cie10_breakdown" in tables

    def test_checkpoints_table_exists(self, db):
        """init_db debe crear la tabla 'checkpoints'."""
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "checkpoints" in tables

    def test_init_db_is_idempotent(self, db):
        """Llamar init_db dos veces no debe lanzar error ni duplicar tablas."""
        db.init_db()
        table_count = db._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN "
            "('runs','simulations','algorithm_iterations','cie10_breakdown','checkpoints','schema_version')"
        ).fetchone()[0]
        assert table_count == 6


# ---------------------------------------------------------------------------
# insert_run — crea un registro en 'runs' y devuelve run_id
# ---------------------------------------------------------------------------


class TestInsertRun:
    def test_insert_run_returns_integer_id(self, db):
        """insert_run debe devolver un entero > 0."""
        run_id = db.insert_run(num_sims=10, num_procs=5, config={})
        assert isinstance(run_id, int)
        assert run_id > 0

    def test_two_runs_have_different_ids(self, db):
        """Dos llamadas a insert_run deben devolver IDs distintos."""
        id1 = db.insert_run(num_sims=10, num_procs=5, config={})
        id2 = db.insert_run(num_sims=20, num_procs=10, config={})
        assert id1 != id2

    def test_insert_run_persists_num_sims(self, db):
        """num_sims debe almacenarse en la fila de 'runs'."""
        run_id = db.insert_run(num_sims=42, num_procs=5, config={})
        row = db._conn.execute(
            "SELECT num_simulations FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row is not None
        assert row[0] == 42

    def test_insert_run_persists_num_procs(self, db):
        """num_procs debe almacenarse en la fila de 'runs'."""
        run_id = db.insert_run(num_sims=10, num_procs=99, config={})
        row = db._conn.execute(
            "SELECT num_procedures FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row is not None
        assert row[0] == 99

    def test_insert_run_serializes_config_as_json(self, db):
        """config dict debe almacenarse como JSON en la columna config_snapshot."""
        import json

        cfg = {"alpha": 0.001, "num_runs": 4}
        run_id = db.insert_run(num_sims=5, num_procs=3, config=cfg)
        row = db._conn.execute(
            "SELECT config_snapshot FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        stored = json.loads(row[0])
        assert stored["alpha"] == 0.001
        assert stored["num_runs"] == 4


# ---------------------------------------------------------------------------
# insert_simulation — inserta simulaciones en tabla 'simulations' y devuelve sim_id
# ---------------------------------------------------------------------------


class TestInsertSimulation:
    def test_returns_integer_sim_id(self, db):
        """insert_simulation debe devolver un entero > 0."""
        run_id = db.insert_run(num_sims=3, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=10.0,
            final_makespan=500.0,
            combined_obj=None,
        )
        assert isinstance(sim_id, int)
        assert sim_id > 0

    def test_two_simulations_different_ids(self, db):
        """Dos llamadas a insert_simulation deben devolver IDs distintos."""
        run_id = db.insert_run(num_sims=2, num_procs=5, config={})
        id1 = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=10.0,
            final_makespan=500.0,
            combined_obj=None,
        )
        id2 = db.insert_simulation(
            run_id=run_id,
            sim_index=1,
            algo_name="GA",
            wall_clock_s=20.0,
            final_makespan=490.0,
            combined_obj=None,
        )
        assert id1 != id2

    def test_simulations_persisted_to_db(self, db):
        """Los registros deben existir en la tabla 'simulations' tras insert."""
        run_id = db.insert_run(num_sims=3, num_procs=5, config={})
        for i in range(3):
            db.insert_simulation(
                run_id=run_id,
                sim_index=i,
                algo_name="GA",
                wall_clock_s=float(i * 10),
                final_makespan=500.0 + i,
                combined_obj=None,
            )
        count = db._conn.execute(
            "SELECT COUNT(*) FROM simulations WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert count == 3

    def test_makespan_stored_correctly(self, db):
        """El makespan de cada simulación debe almacenarse exactamente."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="DPSO",
            wall_clock_s=5.0,
            final_makespan=777.5,
            combined_obj=1.2,
        )
        row = db._conn.execute(
            "SELECT final_makespan FROM simulations WHERE sim_id = ?", (sim_id,)
        ).fetchone()
        assert abs(row[0] - 777.5) < 1e-9


# ---------------------------------------------------------------------------
# save_breakdowns_batch — inserta desglose CIE-10 por simulación
# ---------------------------------------------------------------------------


class TestSaveBreakdownsBatch:
    def test_breakdown_rows_persisted(self, db):
        """Tras save_breakdowns_batch, las filas deben existir en cie10_breakdown."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=1.0,
            final_makespan=400.0,
            combined_obj=0.5,
        )
        breakdowns = [
            {
                "job_id": 1,
                "codigo_cie10": "C50",
                "grupo": "Oncología",
                "setup_min": 10.0,
                "proc_time_min": 60.0,
                "transition_min": 5.0,
                "cleanup_min": 8.0,
            },
            {
                "job_id": 2,
                "codigo_cie10": "K40",
                "grupo": "Cirugía General",
                "setup_min": 12.0,
                "proc_time_min": 45.0,
                "transition_min": 6.0,
                "cleanup_min": 7.0,
            },
        ]
        db.save_breakdowns_batch(sim_id=sim_id, breakdowns=breakdowns)
        count = db._conn.execute(
            "SELECT COUNT(*) FROM cie10_breakdown WHERE sim_id = ?", (sim_id,)
        ).fetchone()[0]
        assert count == 2

    def test_breakdown_fields_stored_correctly(self, db):
        """Los campos de cada fila de breakdown deben almacenarse con precisión."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=2.0,
            final_makespan=300.0,
            combined_obj=0.3,
        )
        breakdowns = [
            {
                "job_id": 5,
                "codigo_cie10": "Z99",
                "grupo": "Misc",
                "setup_min": 3.0,
                "proc_time_min": 90.0,
                "transition_min": 2.0,
                "cleanup_min": 4.0,
            }
        ]
        db.save_breakdowns_batch(sim_id=sim_id, breakdowns=breakdowns)
        row = db._conn.execute(
            "SELECT job_id, codigo_cie10, proc_time_min FROM cie10_breakdown"
        ).fetchone()
        assert row[0] == 5
        assert row[1] == "Z99"
        assert abs(row[2] - 90.0) < 1e-9


# ---------------------------------------------------------------------------
# reconstruct_checkpoints — reconstrucción post-hoc de checkpoints cada N s
# ---------------------------------------------------------------------------


def _make_run_with_simulations(db, interval_s=300.0):
    """Helper: crea una run con simulaciones de GA con timestamps diversos."""
    run_id = db.insert_run(num_sims=6, num_procs=5, config={})
    sim_data = [
        (0, "GA", 10.0, 600.0),
        (1, "GA", 120.0, 550.0),
        (2, "GA", 250.0, 510.0),
        (3, "GA", 310.0, 490.0),
        (4, "GA", 450.0, 480.0),
        (5, "GA", 620.0, 470.0),
    ]
    for sim_i, algo, wall, makespan in sim_data:
        db.insert_simulation(
            run_id=run_id,
            sim_index=sim_i,
            algo_name=algo,
            wall_clock_s=wall,
            final_makespan=makespan,
            combined_obj=None,
        )
    return run_id


class TestReconstructCheckpoints:
    def test_checkpoints_table_populated_after_reconstruct(self, db):
        """reconstruct_checkpoints debe insertar filas en la tabla 'checkpoints'."""
        run_id = _make_run_with_simulations(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        count = db._conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert count > 0

    def test_checkpoint_at_300s_has_best_makespan_up_to_that_point(self, db):
        """
        Con intervalo 300s, el checkpoint en t≤300 debe registrar el mejor
        makespan de las simulaciones con wall_clock_elapsed_s ≤ 300.
        El best-so-far en [0..300] para GA es min(600, 550, 510) = 510.
        """
        run_id = _make_run_with_simulations(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        row = db._conn.execute(
            "SELECT best_makespan FROM checkpoints "
            "WHERE run_id=? AND algo_name='GA' AND checkpoint_wall_s <= 301 "
            "ORDER BY checkpoint_wall_s ASC LIMIT 1",
            (run_id,),
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 510.0) < 1e-9

    def test_checkpoint_at_600s_reflects_cumulative_best(self, db):
        """El checkpoint en t≤600 debe incluir simulaciones hasta ese momento."""
        run_id = _make_run_with_simulations(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        row = db._conn.execute(
            "SELECT best_makespan FROM checkpoints "
            "WHERE run_id=? AND algo_name='GA' AND checkpoint_wall_s <= 601 "
            "ORDER BY checkpoint_wall_s DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 480.0) < 1e-9

    def test_checkpoints_only_for_algo_in_run(self, db):
        """Sólo deben crearse checkpoints para el/los algoritmos que existen en la run."""
        run_id = _make_run_with_simulations(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        algos = {
            row[0]
            for row in db._conn.execute(
                "SELECT DISTINCT algo_name FROM checkpoints WHERE run_id=?", (run_id,)
            ).fetchall()
        }
        assert algos == {"GA"}

    def test_reconstruct_twice_does_not_duplicate_checkpoints(self, db):
        """Llamar reconstruct_checkpoints dos veces no debe duplicar registros."""
        run_id = _make_run_with_simulations(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        count_first = db._conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        count_second = db._conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        assert count_second == count_first


# ---------------------------------------------------------------------------
# export_checkpoints_csv — exporta checkpoints como CSV
# ---------------------------------------------------------------------------


class TestExportCheckpointsCsv:
    def test_export_creates_csv_file(self, db, tmp_path):
        """export_checkpoints_csv debe crear el archivo en la ruta indicada."""
        run_id = _make_run_with_simulations(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        csv_path = str(tmp_path / "checkpoints.csv")
        db.export_checkpoints_csv(run_id=run_id, path=csv_path)
        import os

        assert os.path.exists(csv_path)

    def test_export_csv_has_header_and_rows(self, db, tmp_path):
        """El CSV debe tener cabecera y al menos una fila de datos."""
        run_id = _make_run_with_simulations(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        csv_path = str(tmp_path / "checkpoints.csv")
        db.export_checkpoints_csv(run_id=run_id, path=csv_path)
        import csv

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0
        assert "algo_name" in rows[0]
        assert "makespan_of_best_fitness" in rows[0]

    def test_export_csv_values_match_db(self, db, tmp_path):
        """Los valores del CSV deben coincidir con los de la tabla 'checkpoints'."""
        run_id = _make_run_with_simulations(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        csv_path = str(tmp_path / "checkpoints.csv")
        db.export_checkpoints_csv(run_id=run_id, path=csv_path)
        import csv

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        makespans_csv = {float(r["makespan_of_best_fitness"]) for r in rows}
        db_rows = db._conn.execute(
            "SELECT best_makespan FROM checkpoints WHERE run_id=?", (run_id,)
        ).fetchall()
        makespans_db = {r[0] for r in db_rows}
        assert makespans_csv == makespans_db


# ---------------------------------------------------------------------------
# export_breakdown_csv — exporta cie10_breakdown como CSV
# ---------------------------------------------------------------------------


class TestExportBreakdownCsv:
    def _setup_breakdown(self, db):
        run_id = db.insert_run(num_sims=1, num_procs=3, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=5.0,
            final_makespan=400.0,
            combined_obj=None,
        )
        breakdowns = [
            {
                "job_id": 1,
                "codigo_cie10": "C50",
                "grupo": "Oncología",
                "setup_min": 10.0,
                "proc_time_min": 60.0,
                "transition_min": 5.0,
                "cleanup_min": 8.0,
            },
            {
                "job_id": 2,
                "codigo_cie10": "K40",
                "grupo": "Cirugía",
                "setup_min": 12.0,
                "proc_time_min": 45.0,
                "transition_min": 6.0,
                "cleanup_min": 7.0,
            },
        ]
        db.save_breakdowns_batch(sim_id=sim_id, breakdowns=breakdowns)
        return run_id

    def test_export_breakdown_creates_file(self, db, tmp_path):
        """export_breakdown_csv debe crear el archivo CSV indicado."""
        run_id = self._setup_breakdown(db)
        csv_path = str(tmp_path / "breakdown.csv")
        db.export_breakdown_csv(run_id=run_id, path=csv_path)
        import os

        assert os.path.exists(csv_path)

    def test_export_breakdown_has_expected_columns(self, db, tmp_path):
        """El CSV de breakdown debe incluir columnas: job_id, codigo_cie10, proc_time_min."""
        run_id = self._setup_breakdown(db)
        csv_path = str(tmp_path / "breakdown.csv")
        db.export_breakdown_csv(run_id=run_id, path=csv_path)
        import csv

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert "job_id" in rows[0]
        assert "codigo_cie10" in rows[0]
        assert "proc_time_min" in rows[0]


# ---------------------------------------------------------------------------
# Phase 4 (RED): schema v4 — tabla iteration_schedules
# ---------------------------------------------------------------------------


class TestSchemaV4:
    """El schema debe estar en versión >= 4 y tener tabla iteration_schedules."""

    def test_schema_version_is_4(self, db):
        """La DB inicializada debe reportar schema_version >= 4 (mínimo requerido para iteration_schedules)."""
        version = db._conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()[0]
        assert version >= 4

    def test_iteration_schedules_table_exists(self, db):
        """Debe existir la tabla 'iteration_schedules' en el schema v4."""
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "iteration_schedules" in tables

    def test_algorithm_iterations_unique_constraint(self, db):
        """algorithm_iterations debe tener UNIQUE(sim_id, algo_step)."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=1.0,
            final_makespan=500.0,
            combined_obj=None,
        )
        db.save_algorithm_iterations_batch(
            sim_id,
            [{"algo_step": 1, "best_fitness": 100.0, "best_makespan": 90.0,
              "iteration_fitness": 110.0, "iteration_makespan": 105.0}],
        )
        import pytest as _pytest
        import sqlite3 as _sqlite3
        with _pytest.raises(_sqlite3.IntegrityError):
            # Same sim_id + algo_step → should violate UNIQUE constraint
            db._conn.execute(
                "INSERT INTO algorithm_iterations"
                " (sim_id, algo_step, best_fitness, best_makespan, iteration_fitness, iteration_makespan)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (sim_id, 1, 200.0, 190.0, 210.0, 205.0),
            )
            db._conn.commit()


class TestSaveIterationSchedulesBatch:
    """save_iteration_schedules_batch debe insertar JSON + SHA-256 por algo_iter_id."""

    def _setup_iterations(self, db):
        """Helper: crea run, sim e iteraciones base."""
        run_id = db.insert_run(num_sims=1, num_procs=3, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=1.0,
            final_makespan=500.0,
            combined_obj=None,
        )
        algo_iter_ids = db.save_algorithm_iterations_batch(
            sim_id,
            [
                {"algo_step": 1, "best_fitness": 100.0, "best_makespan": 90.0,
                 "iteration_fitness": 110.0, "iteration_makespan": 105.0},
                {"algo_step": 2, "best_fitness": 90.0, "best_makespan": 80.0,
                 "iteration_fitness": 95.0, "iteration_makespan": 88.0},
            ],
        )
        return sim_id, algo_iter_ids

    def test_save_iteration_schedules_inserts_rows(self, db):
        """save_iteration_schedules_batch debe insertar una fila por algo_iter_id."""
        _, algo_iter_ids = self._setup_iterations(db)
        schedules = [
            {"algo_iter_id": algo_iter_ids[0],
             "solution_json": '{"job_sequence_base": [1, 2, 3]}'},
            {"algo_iter_id": algo_iter_ids[1],
             "solution_json": '{"job_sequence_base": [2, 1, 3]}'},
        ]
        db.save_iteration_schedules_batch(schedules)
        count = db._conn.execute(
            "SELECT COUNT(*) FROM iteration_schedules WHERE algo_iter_id IN (?, ?)",
            (algo_iter_ids[0], algo_iter_ids[1]),
        ).fetchone()[0]
        assert count == 2

    def test_save_iteration_schedules_stores_sha256(self, db):
        """Cada fila debe tener un solution_sha256 no vacío."""
        _, algo_iter_ids = self._setup_iterations(db)
        schedules = [
            {"algo_iter_id": algo_iter_ids[0],
             "solution_json": '{"job_sequence_base": [1, 2]}'},
        ]
        db.save_iteration_schedules_batch(schedules)
        row = db._conn.execute(
            "SELECT solution_sha256 FROM iteration_schedules WHERE algo_iter_id = ?",
            (algo_iter_ids[0],),
        ).fetchone()
        assert row is not None
        assert len(row[0]) == 64  # SHA-256 hex = 64 chars

    def test_iteration_schedules_unique_per_algo_iter(self, db):
        """iteration_schedules.algo_iter_id debe ser UNIQUE."""
        _, algo_iter_ids = self._setup_iterations(db)
        schedules = [
            {"algo_iter_id": algo_iter_ids[0],
             "solution_json": '{"job_sequence_base": [1, 2]}'},
        ]
        db.save_iteration_schedules_batch(schedules)
        import pytest as _pytest
        import sqlite3 as _sqlite3
        with _pytest.raises(_sqlite3.IntegrityError):
            db._conn.execute(
                "INSERT INTO iteration_schedules (algo_iter_id, solution_json, solution_format, solution_sha256)"
                " VALUES (?, ?, ?, ?)",
                (algo_iter_ids[0], '{"x": 1}', "scheduler_solution_v1", "a" * 64),
            )
            db._conn.commit()

    def test_get_iteration_schedules_for_sim(self, db):
        """get_iteration_schedules_for_sim debe retornar filas con algo_step y solution_json."""
        sim_id, algo_iter_ids = self._setup_iterations(db)
        schedules = [
            {"algo_iter_id": algo_iter_ids[0],
             "solution_json": '{"job_sequence_base": [1, 2, 3]}'},
        ]
        db.save_iteration_schedules_batch(schedules)
        rows = db.get_iteration_schedules_for_sim(sim_id)
        assert len(rows) == 1
        assert rows[0]["algo_step"] == 1
        assert "job_sequence_base" in rows[0]["solution_json"]


# ---------------------------------------------------------------------------
# Phase 1 RED (task 1.3): v6 export methods
# ---------------------------------------------------------------------------


def _make_run_with_v6_data(db):
    """Helper: crea run, sim, patient_wait_metrics y schedule_quality_metrics."""
    run_id = db.insert_run(num_sims=1, num_procs=2, config={})
    sim_id = db.insert_simulation(
        run_id=run_id,
        sim_index=0,
        algo_name="GA",
        wall_clock_s=2.0,
        final_makespan=150.0,
        combined_obj=150.0,
        algo_time_s=1.0,
    )
    db.save_patient_wait_batch(
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
    db.save_quality_metrics(
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
    return run_id, sim_id


class TestExportPatientWaitCsv:
    """export_patient_wait_csv debe exportar patient_wait_metrics de un run a CSV."""

    def test_export_creates_csv_file(self, db, tmp_path):
        """export_patient_wait_csv debe crear el archivo CSV."""
        import os
        run_id, _ = _make_run_with_v6_data(db)
        csv_path = str(tmp_path / "patient_wait.csv")
        db.export_patient_wait_csv(run_id=run_id, path=csv_path)
        assert os.path.exists(csv_path)

    def test_export_has_expected_columns(self, db, tmp_path):
        """El CSV debe tener columnas: job_id, op1_room, extra_wait_min."""
        import csv as csv_mod
        run_id, _ = _make_run_with_v6_data(db)
        csv_path = str(tmp_path / "patient_wait_cols.csv")
        db.export_patient_wait_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert "job_id" in reader.fieldnames
        assert "extra_wait_min" in reader.fieldnames
        assert "op1_room" in reader.fieldnames

    def test_export_filtered_by_run_id(self, db, tmp_path):
        """export_patient_wait_csv debe filtrar por run_id: no mezcla datos de otro run."""
        import csv as csv_mod
        # run 1
        run_id1, _ = _make_run_with_v6_data(db)
        # run 2 with different data
        run_id2 = db.insert_run(num_sims=1, num_procs=1, config={})
        sim_id2 = db.insert_simulation(
            run_id=run_id2,
            sim_index=0,
            algo_name="DPSO",
            wall_clock_s=1.0,
            final_makespan=99.0,
            combined_obj=99.0,
            algo_time_s=0.5,
        )
        db.save_patient_wait_batch(
            sim_id2,
            [
                {
                    "job_id": 99,
                    "op1_room": "Q9",
                    "op2_room": "Q9",
                    "op1_finish": 10.0,
                    "op2_start": 20.0,
                    "transition_used": 5.0,
                    "extra_wait_min": 5.0,
                }
            ],
        )
        csv_path = str(tmp_path / "patient_wait_filtered.csv")
        db.export_patient_wait_csv(run_id=run_id1, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv_mod.DictReader(f))
        # Only run 1 data (2 rows), not run 2 (job_id=99)
        assert len(rows) == 2
        job_ids = {int(r["job_id"]) for r in rows}
        assert 99 not in job_ids


class TestExportQualityMetricsCsv:
    """export_quality_metrics_csv debe exportar schedule_quality_metrics de un run a CSV."""

    def test_export_creates_csv_file(self, db, tmp_path):
        """export_quality_metrics_csv debe crear el archivo CSV."""
        import os
        run_id, _ = _make_run_with_v6_data(db)
        csv_path = str(tmp_path / "quality_metrics.csv")
        db.export_quality_metrics_csv(run_id=run_id, path=csv_path)
        assert os.path.exists(csv_path)

    def test_export_has_expected_columns(self, db, tmp_path):
        """El CSV debe tener columnas clave de schedule_quality_metrics."""
        import csv as csv_mod
        run_id, _ = _make_run_with_v6_data(db)
        csv_path = str(tmp_path / "quality_metrics_cols.csv")
        db.export_quality_metrics_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert "total_overtime_min" in reader.fieldnames
        assert "value_added_ratio" in reader.fieldnames
        assert "rooms_used" in reader.fieldnames

    def test_export_values_match_inserted(self, db, tmp_path):
        """Los valores exportados deben coincidir con los insertados."""
        import csv as csv_mod
        run_id, _ = _make_run_with_v6_data(db)
        csv_path = str(tmp_path / "quality_metrics_vals.csv")
        db.export_quality_metrics_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv_mod.DictReader(f))
        row = rows[0]
        assert abs(float(row["total_overtime_min"]) - 15.0) < 1e-6
        assert abs(float(row["value_added_ratio"]) - 0.75) < 1e-6


class TestExportSimulationSummaryCsv:
    """export_simulation_summary_csv debe exportar v_simulation_summary a CSV."""

    def test_export_creates_csv_file(self, db, tmp_path):
        """export_simulation_summary_csv debe crear el archivo CSV."""
        import os
        run_id, _ = _make_run_with_v6_data(db)
        csv_path = str(tmp_path / "sim_summary.csv")
        db.export_simulation_summary_csv(run_id=run_id, path=csv_path)
        assert os.path.exists(csv_path)

    def test_export_has_final_makespan_and_rooms_used(self, db, tmp_path):
        """El CSV debe incluir final_makespan y rooms_used (del LEFT JOIN)."""
        import csv as csv_mod
        run_id, _ = _make_run_with_v6_data(db)
        csv_path = str(tmp_path / "sim_summary_cols.csv")
        db.export_simulation_summary_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert "final_makespan" in reader.fieldnames
        assert "rooms_used" in reader.fieldnames

    def test_export_no_rows_when_no_simulations(self, db, tmp_path):
        """Si el run no tiene simulaciones, el CSV no se crea (método retorna sin escribir)."""
        import os
        run_id = db.insert_run(num_sims=0, num_procs=0, config={})
        csv_path = str(tmp_path / "sim_summary_empty.csv")
        db.export_simulation_summary_csv(run_id=run_id, path=csv_path)
        # Method returns early when rows is empty — file should NOT exist
        assert not os.path.exists(csv_path)
