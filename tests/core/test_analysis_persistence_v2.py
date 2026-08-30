"""
Tests para core/analysis_persistence.py — Schema v6 (actualizado desde v2/v3).

Cubre:
- schema_version table (v6)
- Tabla 'simulations' (renombrada desde 'iterations')
- Tabla 'algorithm_iterations' (NUEVA en v2, estable en v6)
- DBs legacy (v1, v2) rechazadas con RuntimeError
- Nuevos métodos: insert_simulation(), save_algorithm_iterations_batch()
- Backward compat: insert_run(), save_breakdowns_batch(), reconstruct_checkpoints(), exports

Estrategia: SQLite en memoria (:memory:) para aislamiento total.
"""

import csv
import json
import os
import sqlite3
import pytest

from core.analysis_persistence import AnalysisPersistence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """Instancia con base de datos en memoria; inicializada con schema v2."""
    persistence = AnalysisPersistence(":memory:")
    persistence.init_db()
    return persistence


# ---------------------------------------------------------------------------
# Schema v2: tablas esperadas
# ---------------------------------------------------------------------------


class TestSchemaV2:
    def _get_tables(self, db):
        return {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    def test_runs_table_exists(self, db):
        """init_db debe crear la tabla 'runs'."""
        assert "runs" in self._get_tables(db)

    def test_simulations_table_exists(self, db):
        """init_db debe crear la tabla 'simulations' (antes 'iterations')."""
        assert "simulations" in self._get_tables(db)

    def test_algorithm_iterations_table_exists(self, db):
        """init_db debe crear la tabla NUEVA 'algorithm_iterations'."""
        assert "algorithm_iterations" in self._get_tables(db)

    def test_cie10_breakdown_table_exists(self, db):
        """init_db debe crear la tabla 'cie10_breakdown'."""
        assert "cie10_breakdown" in self._get_tables(db)

    def test_checkpoints_table_exists(self, db):
        """init_db debe crear la tabla 'checkpoints'."""
        assert "checkpoints" in self._get_tables(db)

    def test_schema_version_table_exists(self, db):
        """init_db debe crear la tabla 'schema_version'."""
        assert "schema_version" in self._get_tables(db)

    def test_schema_version_is_2(self, db):
        """La versión del schema debe ser 6 (schema v6 actual)."""
        version = db._conn.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == 6

    def test_init_db_is_idempotent(self, db):
        """Llamar init_db dos veces no debe lanzar error ni duplicar tablas."""
        db.init_db()
        count = db._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN "
            "('runs','simulations','algorithm_iterations','cie10_breakdown','checkpoints','schema_version')"
        ).fetchone()[0]
        assert count == 6

    def test_old_iterations_table_not_present(self, db):
        """La tabla 'iterations' del schema v1 NO debe existir en v2."""
        tables = self._get_tables(db)
        assert "iterations" not in tables


# ---------------------------------------------------------------------------
# insert_run — backward compatible
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
        run_id = db.insert_run(num_sims=10, num_procs=99, config={})
        row = db._conn.execute(
            "SELECT num_procedures FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row[0] == 99

    def test_insert_run_serializes_config_as_json(self, db):
        cfg = {"alpha": 0.001, "num_runs": 4}
        run_id = db.insert_run(num_sims=5, num_procs=3, config=cfg)
        row = db._conn.execute(
            "SELECT config_snapshot FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        stored = json.loads(row[0])
        assert stored["alpha"] == 0.001
        assert stored["num_runs"] == 4


# ---------------------------------------------------------------------------
# insert_simulation — NEW in v2
# ---------------------------------------------------------------------------


class TestInsertSimulation:
    def test_returns_integer_sim_id(self, db):
        """insert_simulation debe devolver un sim_id > 0."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
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

    def test_two_simulations_have_different_ids(self, db):
        """Dos simulaciones deben tener sim_id distintos."""
        run_id = db.insert_run(num_sims=2, num_procs=5, config={})
        sim_id_1 = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=10.0,
            final_makespan=500.0,
            combined_obj=None,
        )
        sim_id_2 = db.insert_simulation(
            run_id=run_id,
            sim_index=1,
            algo_name="GA",
            wall_clock_s=20.0,
            final_makespan=490.0,
            combined_obj=None,
        )
        assert sim_id_1 != sim_id_2

    def test_simulation_persisted_to_db(self, db):
        """La simulación debe existir en la tabla 'simulations' tras insert."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="DPSO",
            wall_clock_s=15.5,
            final_makespan=455.0,
            combined_obj=1.2,
        )
        row = db._conn.execute(
            "SELECT algo_name, final_makespan, combined_obj "
            "FROM simulations WHERE sim_id = ?",
            (sim_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == "DPSO"
        assert abs(row[1] - 455.0) < 1e-9
        assert abs(row[2] - 1.2) < 1e-9

    def test_simulation_fk_to_run(self, db):
        """sim_id debe tener FK válida hacia runs."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=5.0,
            final_makespan=500.0,
            combined_obj=None,
        )
        stored_run_id = db._conn.execute(
            "SELECT run_id FROM simulations WHERE sim_id = ?", (sim_id,)
        ).fetchone()[0]
        assert stored_run_id == run_id

    def test_simulation_combined_obj_can_be_none(self, db):
        """combined_obj=None debe almacenarse sin error (columna nullable)."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=5.0,
            final_makespan=500.0,
            combined_obj=None,
        )
        row = db._conn.execute(
            "SELECT combined_obj FROM simulations WHERE sim_id = ?", (sim_id,)
        ).fetchone()
        assert row[0] is None


# ---------------------------------------------------------------------------
# save_algorithm_iterations_batch — NEW in v2
# ---------------------------------------------------------------------------


class TestSaveAlgorithmIterationsBatch:
    def _setup_sim(self, db):
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=30.0,
            final_makespan=480.0,
            combined_obj=None,
        )
        return sim_id

    def test_returns_list_of_iter_ids(self, db):
        """save_algorithm_iterations_batch debe devolver lista de iter_ids."""
        sim_id = self._setup_sim(db)
        rows = [
            {
                "algo_step": 1,
                "best_fitness": 500.0,
                "best_makespan": 480.0,
                "iteration_fitness": 510.0,
                "iteration_makespan": 495.0,
            },
            {
                "algo_step": 2,
                "best_fitness": 490.0,
                "best_makespan": 470.0,
                "iteration_fitness": 495.0,
                "iteration_makespan": 475.0,
            },
        ]
        result = db.save_algorithm_iterations_batch(sim_id=sim_id, rows=rows)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(i, int) and i > 0 for i in result)

    def test_iterations_persisted_to_db(self, db):
        """Los registros deben existir en 'algorithm_iterations' tras el batch."""
        sim_id = self._setup_sim(db)
        rows = [
            {
                "algo_step": 1,
                "best_fitness": 600.0,
                "best_makespan": 580.0,
                "iteration_fitness": 610.0,
                "iteration_makespan": 590.0,
            },
            {
                "algo_step": 2,
                "best_fitness": 580.0,
                "best_makespan": 560.0,
                "iteration_fitness": 590.0,
                "iteration_makespan": 570.0,
            },
            {
                "algo_step": 3,
                "best_fitness": 570.0,
                "best_makespan": 550.0,
                "iteration_fitness": 575.0,
                "iteration_makespan": 555.0,
            },
        ]
        db.save_algorithm_iterations_batch(sim_id=sim_id, rows=rows)
        count = db._conn.execute(
            "SELECT COUNT(*) FROM algorithm_iterations WHERE sim_id = ?", (sim_id,)
        ).fetchone()[0]
        assert count == 3

    def test_iteration_values_stored_correctly(self, db):
        """Los valores de cada iteración deben almacenarse con precisión."""
        sim_id = self._setup_sim(db)
        rows = [
            {
                "algo_step": 5,
                "best_fitness": 333.3,
                "best_makespan": 320.0,
                "iteration_fitness": 340.0,
                "iteration_makespan": 325.0,
            },
        ]
        iter_ids = db.save_algorithm_iterations_batch(sim_id=sim_id, rows=rows)
        row = db._conn.execute(
            "SELECT algo_step, best_fitness, best_makespan, iteration_fitness, iteration_makespan "
            "FROM algorithm_iterations WHERE algo_iter_id = ?",
            (iter_ids[0],),
        ).fetchone()
        assert row[0] == 5
        assert abs(row[1] - 333.3) < 1e-9
        assert abs(row[2] - 320.0) < 1e-9
        assert abs(row[3] - 340.0) < 1e-9
        assert abs(row[4] - 325.0) < 1e-9

    def test_fk_to_simulations(self, db):
        """algo_iter_id debe tener FK válida hacia simulations."""
        sim_id = self._setup_sim(db)
        rows = [
            {
                "algo_step": 1,
                "best_fitness": 500.0,
                "best_makespan": 480.0,
                "iteration_fitness": 510.0,
                "iteration_makespan": 495.0,
            }
        ]
        iter_ids = db.save_algorithm_iterations_batch(sim_id=sim_id, rows=rows)
        stored_sim_id = db._conn.execute(
            "SELECT sim_id FROM algorithm_iterations WHERE algo_iter_id = ?",
            (iter_ids[0],),
        ).fetchone()[0]
        assert stored_sim_id == sim_id

    def test_empty_batch_returns_empty_list(self, db):
        """Un batch vacío debe devolver una lista vacía sin error."""
        sim_id = self._setup_sim(db)
        result = db.save_algorithm_iterations_batch(sim_id=sim_id, rows=[])
        assert result == []


# ---------------------------------------------------------------------------
# save_breakdowns_batch — updated to use sim_id
# ---------------------------------------------------------------------------


class TestSaveBreakdownsBatchV2:
    def test_breakdown_rows_persisted_via_sim_id(self, db):
        """save_breakdowns_batch debe persistir filas usando sim_id como FK."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=5.0,
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
        """Los campos de cada fila de breakdown deben almacenarse correctamente."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=2.0,
            final_makespan=300.0,
            combined_obj=None,
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
# Migration v1 → v2
# ---------------------------------------------------------------------------


def _create_v1_db_in_memory():
    """Helper: crea una conexión con schema v1 (iterations, sin schema_version)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            num_simulations INTEGER NOT NULL,
            num_procedures INTEGER NOT NULL,
            config_snapshot TEXT
        );
        CREATE TABLE iterations (
            iter_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES runs(run_id),
            sim_i INTEGER NOT NULL,
            algo_name TEXT NOT NULL,
            wall_clock_elapsed_s REAL NOT NULL,
            makespan REAL NOT NULL,
            combined_obj REAL,
            algo_time_s REAL NOT NULL
        );
        CREATE TABLE cie10_breakdown (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iter_id INTEGER NOT NULL REFERENCES iterations(iter_id),
            job_id INTEGER NOT NULL,
            codigo_cie10 TEXT,
            grupo TEXT,
            setup_min REAL NOT NULL,
            proc_time_min REAL NOT NULL,
            transition_min REAL NOT NULL,
            cleanup_min REAL NOT NULL
        );
        CREATE TABLE checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES runs(run_id),
            algo_name TEXT NOT NULL,
            checkpoint_wall_s REAL NOT NULL,
            best_makespan REAL NOT NULL,
            best_sim_i INTEGER NOT NULL
        );
        INSERT INTO runs (started_at, num_simulations, num_procedures, config_snapshot)
        VALUES ('2026-01-01T00:00:00+00:00', 5, 15, '{}');
        INSERT INTO iterations (run_id, sim_i, algo_name, wall_clock_elapsed_s, makespan, combined_obj, algo_time_s)
        VALUES (1, 0, 'GA', 10.0, 500.0, NULL, 1.0),
               (1, 1, 'GA', 20.0, 490.0, NULL, 1.0);
    """)
    conn.commit()
    return conn


class TestLegacyDBRejected:
    """Verifica que init_db rechaza DBs legacy (v1 o v2) con RuntimeError — sin migración automática."""

    def test_v1_db_raises_runtime_error(self):
        """Una DB v1 (sin schema_version) debe lanzar RuntimeError al llamar init_db."""
        v1_conn = _create_v1_db_in_memory()
        persistence = AnalysisPersistence.__new__(AnalysisPersistence)
        persistence._db_path = ":memory:"
        persistence._conn = v1_conn
        with pytest.raises(RuntimeError, match="legacy"):
            persistence.init_db()

    def test_v1_db_error_message_mentions_analysis_db(self):
        """El mensaje de error debe mencionar 'analysis.db' para orientar al usuario."""
        v1_conn = _create_v1_db_in_memory()
        persistence = AnalysisPersistence.__new__(AnalysisPersistence)
        persistence._db_path = ":memory:"
        persistence._conn = v1_conn
        with pytest.raises(RuntimeError, match="analysis.db"):
            persistence.init_db()

    def test_v1_db_does_not_corrupt_tables(self):
        """Tras RuntimeError, las tablas v1 originales deben seguir intactas."""
        v1_conn = _create_v1_db_in_memory()
        persistence = AnalysisPersistence.__new__(AnalysisPersistence)
        persistence._db_path = ":memory:"
        persistence._conn = v1_conn
        try:
            persistence.init_db()
        except RuntimeError:
            pass
        # La tabla 'iterations' (v1) debe seguir ahí — no fue mutada
        count = v1_conn.execute("SELECT COUNT(*) FROM iterations").fetchone()[0]
        assert count == 2

    def test_v3_db_init_is_idempotent_no_migration(self):
        """Una DB ya en v6 no debe re-migrarse ni duplicar datos."""
        db = AnalysisPersistence(":memory:")
        db.init_db()
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=5.0,
            final_makespan=400.0,
            combined_obj=None,
        )
        # Llamar init_db de nuevo
        db.init_db()
        count = db._conn.execute("SELECT COUNT(*) FROM simulations").fetchone()[0]
        assert count == 1  # no duplicado


# ---------------------------------------------------------------------------
# reconstruct_checkpoints — updated to work with simulations table
# ---------------------------------------------------------------------------


def _make_run_with_simulations(db, interval_s=300.0):
    """Helper: crea una run con simulaciones (tabla simulations) para GA."""
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


class TestReconstructCheckpointsV2:
    def test_checkpoints_populated_after_reconstruct(self, db):
        """reconstruct_checkpoints debe insertar filas en 'checkpoints'."""
        run_id = _make_run_with_simulations(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        count = db._conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert count > 0

    def test_checkpoint_at_300s_has_best_makespan(self, db):
        """El checkpoint en t≤300 debe reflejar el mejor makespan hasta ese punto (510.0)."""
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

    def test_reconstruct_twice_does_not_duplicate(self, db):
        """Llamar reconstruct_checkpoints dos veces no debe duplicar registros."""
        run_id = _make_run_with_simulations(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        count1 = db._conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        count2 = db._conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        assert count2 == count1


# ---------------------------------------------------------------------------
# export_checkpoints_csv — backward compatible
# ---------------------------------------------------------------------------


class TestExportCheckpointsCsvV2:
    def test_export_creates_csv_file(self, db, tmp_path):
        run_id = _make_run_with_simulations(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        csv_path = str(tmp_path / "checkpoints.csv")
        db.export_checkpoints_csv(run_id=run_id, path=csv_path)
        assert os.path.exists(csv_path)

    def test_export_csv_has_header_and_rows(self, db, tmp_path):
        run_id = _make_run_with_simulations(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        csv_path = str(tmp_path / "checkpoints.csv")
        db.export_checkpoints_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        assert "algo_name" in rows[0]
        assert "makespan_of_best_fitness" in rows[0]


# ---------------------------------------------------------------------------
# export_breakdown_csv — updated to join via simulations
# ---------------------------------------------------------------------------


class TestExportBreakdownCsvV2:
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
        run_id = self._setup_breakdown(db)
        csv_path = str(tmp_path / "breakdown.csv")
        db.export_breakdown_csv(run_id=run_id, path=csv_path)
        assert os.path.exists(csv_path)

    def test_export_breakdown_has_expected_columns(self, db, tmp_path):
        run_id = self._setup_breakdown(db)
        csv_path = str(tmp_path / "breakdown.csv")
        db.export_breakdown_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert "job_id" in rows[0]
        assert "codigo_cie10" in rows[0]
        assert "proc_time_min" in rows[0]


# ---------------------------------------------------------------------------
# Task 4.1 — CSV export excludes combined_obj; DB still has it
# ---------------------------------------------------------------------------


class TestAlgorithmIterationsCSVExport:
    """export_algorithm_iterations_csv must reflect new 4-column schema."""

    def _setup_algo_iter(self, db):
        """Insert run, simulation and algo_iterations rows with new schema."""
        run_id = db.insert_run(
            num_sims=1,
            num_procs=5,
            config={},
        )
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=1,
            algo_name="GA",
            wall_clock_s=1.0,
            final_makespan=200.0,
            combined_obj=None,
            algo_time_s=0.5,
        )
        rows = [
            {
                "algo_step": 1,
                "best_fitness": 250.0,
                "best_makespan": 200.0,
                "iteration_fitness": 260.0,
                "iteration_makespan": 210.0,
            },
            {
                "algo_step": 2,
                "best_fitness": 240.0,
                "best_makespan": 195.0,
                "iteration_fitness": 245.0,
                "iteration_makespan": 198.0,
            },
        ]
        db.save_algorithm_iterations_batch(sim_id, rows)
        return run_id, sim_id

    def test_csv_export_excludes_combined_obj_column(self, db, tmp_path):
        """exported CSV must NOT contain combined_obj column."""
        run_id, _ = self._setup_algo_iter(db)
        csv_path = str(tmp_path / "algo_iters.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
        assert "combined_obj" not in header

    def test_csv_export_excludes_wall_clock_s_column(self, db, tmp_path):
        """exported CSV must NOT contain wall_clock_s column."""
        run_id, _ = self._setup_algo_iter(db)
        csv_path = str(tmp_path / "algo_iters_wc.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = csv.DictReader(f).fieldnames
        assert "wall_clock_s" not in header

    def test_csv_export_contains_makespan_and_best_fitness_columns(self, db, tmp_path):
        """exported CSV must contain best_makespan, iteration_fitness, iteration_makespan."""
        run_id, _ = self._setup_algo_iter(db)
        csv_path = str(tmp_path / "algo_iters2.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = list(reader)
        assert "best_fitness" in header
        assert "makespan_of_best_fitness" in header
        assert "iteration_fitness" in header
        assert "iteration_makespan" in header
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Phase 1 (RED → GREEN): Schema v3 — naming limpio sin legacy
# ---------------------------------------------------------------------------


class TestSchemaV3Clean:
    """
    Schema v3: renombre integral de identificadores ambiguos.

    Columnas esperadas en v3:
    - simulations: sim_index (en lugar de sim_i)
    - algorithm_iterations: algo_iter_id (PK), algo_step (en lugar de generation)
    - checkpoints: best_sim_index (en lugar de best_sim_i)
    - schema_version: version = 3
    """

    def _get_columns(self, db, table_name):
        rows = db._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {row[1] for row in rows}  # row[1] = column name

    def test_schema_version_is_3(self, db):
        """La versión del schema debe ser 6 (schema v6 actual)."""
        version = db._conn.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == 6

    def test_simulations_has_sim_index_not_sim_i(self, db):
        """La tabla 'simulations' debe tener columna 'sim_index' (no 'sim_i')."""
        cols = self._get_columns(db, "simulations")
        assert "sim_index" in cols, f"sim_index not in {cols}"
        assert "sim_i" not in cols, f"Legacy 'sim_i' must NOT exist in {cols}"

    def test_algorithm_iterations_has_algo_step_not_generation(self, db):
        """La tabla 'algorithm_iterations' debe tener 'algo_step' (no 'generation')."""
        cols = self._get_columns(db, "algorithm_iterations")
        assert "algo_step" in cols, f"algo_step not in {cols}"
        assert "generation" not in cols, f"Legacy 'generation' must NOT exist in {cols}"

    def test_algorithm_iterations_pk_is_algo_iter_id_not_iter_id(self, db):
        """La PK de 'algorithm_iterations' debe llamarse 'algo_iter_id' (no 'iter_id')."""
        cols = self._get_columns(db, "algorithm_iterations")
        assert "algo_iter_id" in cols, f"algo_iter_id not in {cols}"
        assert "iter_id" not in cols, f"Legacy 'iter_id' must NOT exist in {cols}"

    def test_checkpoints_has_best_sim_index_not_best_sim_i(self, db):
        """La tabla 'checkpoints' debe tener 'best_sim_index' (no 'best_sim_i')."""
        cols = self._get_columns(db, "checkpoints")
        assert "best_sim_index" in cols, f"best_sim_index not in {cols}"
        assert "best_sim_i" not in cols, f"Legacy 'best_sim_i' must NOT exist in {cols}"

    def test_fresh_db_no_legacy_columns(self, db):
        """Una DB nueva no debe contener NINGÚN identificador legacy."""
        all_legacy = ["sim_i", "generation", "iter_id", "best_sim_i"]
        for table in ["simulations", "algorithm_iterations", "checkpoints"]:
            cols = self._get_columns(db, table)
            for legacy in all_legacy:
                assert legacy not in cols, (
                    f"Legacy '{legacy}' found in table '{table}': {cols}"
                )


class TestV3InsertSimulationWithSimIndex:
    """insert_simulation en v3 debe usar el parámetro sim_index (no sim_i)."""

    def test_insert_simulation_accepts_sim_index_param(self, db):
        """insert_simulation debe aceptar 'sim_index' como parámetro nombrado."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
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

    def test_sim_index_stored_in_db(self, db):
        """El valor de sim_index debe almacenarse en la columna sim_index."""
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=7,
            algo_name="GA",
            wall_clock_s=10.0,
            final_makespan=500.0,
            combined_obj=None,
        )
        row = db._conn.execute(
            "SELECT sim_index FROM simulations WHERE sim_id = ?", (sim_id,)
        ).fetchone()
        assert row is not None
        assert row[0] == 7


class TestV3AlgorithmIterationsBatchWithAlgoStep:
    """save_algorithm_iterations_batch en v3 debe usar 'algo_step' (no 'generation')."""

    def _setup_sim(self, db):
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=30.0,
            final_makespan=480.0,
            combined_obj=None,
        )
        return sim_id

    def test_batch_accepts_algo_step_key(self, db):
        """save_algorithm_iterations_batch debe aceptar dicts con clave 'algo_step'."""
        sim_id = self._setup_sim(db)
        rows = [
            {
                "algo_step": 1,
                "best_fitness": 500.0,
                "best_makespan": 480.0,
                "iteration_fitness": 510.0,
                "iteration_makespan": 495.0,
            }
        ]
        result = db.save_algorithm_iterations_batch(sim_id=sim_id, rows=rows)
        assert len(result) == 1
        assert result[0] > 0

    def test_algo_step_stored_in_db(self, db):
        """El valor de algo_step debe almacenarse en la columna algo_step."""
        sim_id = self._setup_sim(db)
        rows = [
            {
                "algo_step": 42,
                "best_fitness": 300.0,
                "best_makespan": 280.0,
                "iteration_fitness": 310.0,
                "iteration_makespan": 290.0,
            }
        ]
        result = db.save_algorithm_iterations_batch(sim_id=sim_id, rows=rows)
        row = db._conn.execute(
            "SELECT algo_step FROM algorithm_iterations WHERE algo_iter_id = ?",
            (result[0],),
        ).fetchone()
        assert row is not None
        assert row[0] == 42


class TestV3CheckpointsWithBestSimIndex:
    """reconstruct_checkpoints y export_checkpoints_csv deben usar best_sim_index."""

    def _setup_run_v3(self, db):
        run_id = db.insert_run(num_sims=4, num_procs=5, config={})
        sim_data = [
            (0, "GA", 10.0, 600.0),
            (1, "GA", 120.0, 550.0),
            (2, "GA", 250.0, 510.0),
            (3, "GA", 310.0, 490.0),
        ]
        for sim_index, algo, wall, makespan in sim_data:
            db.insert_simulation(
                run_id=run_id,
                sim_index=sim_index,
                algo_name=algo,
                wall_clock_s=wall,
                final_makespan=makespan,
                combined_obj=None,
            )
        return run_id

    def test_checkpoints_has_best_sim_index_column(self, db):
        """La tabla checkpoints debe tener columna best_sim_index (no best_sim_i)."""
        run_id = self._setup_run_v3(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        row = db._conn.execute(
            "SELECT best_sim_index FROM checkpoints WHERE run_id=?", (run_id,)
        ).fetchone()
        assert row is not None

    def test_checkpoints_csv_header_has_best_sim_index(self, db, tmp_path):
        """El CSV de checkpoints debe tener columna 'best_sim_index' (no 'best_sim_i')."""
        run_id = self._setup_run_v3(db)
        db.reconstruct_checkpoints(run_id=run_id, interval_s=300.0)
        csv_path = str(tmp_path / "checkpoints_v3.csv")
        db.export_checkpoints_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
        assert "best_sim_index" in header, f"Expected best_sim_index in {header}"
        assert "best_sim_i" not in header, f"Legacy best_sim_i found in {header}"


class TestV3AlgorithmIterationsCSVWithAlgoStep:
    """export_algorithm_iterations_csv v3 debe usar 'algo_step' y 'sim_index'."""

    def _setup_v3(self, db):
        run_id = db.insert_run(num_sims=1, num_procs=5, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=1,
            algo_name="GA",
            wall_clock_s=1.0,
            final_makespan=200.0,
            combined_obj=None,
        )
        rows = [
            {
                "algo_step": 1,
                "best_fitness": 250.0,
                "best_makespan": 200.0,
                "iteration_fitness": 260.0,
                "iteration_makespan": 210.0,
            },
            {
                "algo_step": 2,
                "best_fitness": 240.0,
                "best_makespan": 195.0,
                "iteration_fitness": 245.0,
                "iteration_makespan": 198.0,
            },
        ]
        db.save_algorithm_iterations_batch(sim_id, rows)
        return run_id

    def test_csv_has_algo_step_not_generation(self, db, tmp_path):
        """CSV de algorithm_iterations debe tener 'algo_step' (no 'generation')."""
        run_id = self._setup_v3(db)
        csv_path = str(tmp_path / "algo_iter_v3.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = csv.DictReader(f).fieldnames
        assert "algo_step" in header, f"algo_step not in {header}"
        assert "generation" not in header, f"Legacy 'generation' found in {header}"

    def test_csv_has_sim_index_not_sim_i(self, db, tmp_path):
        """CSV de algorithm_iterations debe tener 'sim_index' (no 'sim_i')."""
        run_id = self._setup_v3(db)
        csv_path = str(tmp_path / "algo_iter_v3b.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = csv.DictReader(f).fieldnames
        assert "sim_index" in header, f"sim_index not in {header}"
        assert "sim_i" not in header, f"Legacy 'sim_i' found in {header}"

    def test_csv_has_algo_iter_id_not_iter_id(self, db, tmp_path):
        """CSV de algorithm_iterations debe tener 'algo_iter_id' (no 'iter_id')."""
        run_id = self._setup_v3(db)
        csv_path = str(tmp_path / "algo_iter_v3c.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = csv.DictReader(f).fieldnames
        assert "algo_iter_id" in header, f"algo_iter_id not in {header}"
        assert "iter_id" not in header, f"Legacy 'iter_id' found in {header}"

    def test_csv_breakdown_has_sim_index_not_sim_i(self, db, tmp_path):
        """CSV de breakdown debe tener 'sim_index' (no 'sim_i')."""
        run_id = db.insert_run(num_sims=1, num_procs=3, config={})
        sim_id = db.insert_simulation(
            run_id=run_id,
            sim_index=0,
            algo_name="GA",
            wall_clock_s=5.0,
            final_makespan=400.0,
            combined_obj=None,
        )
        db.save_breakdowns_batch(
            sim_id=sim_id,
            breakdowns=[
                {
                    "job_id": 1,
                    "codigo_cie10": "C50",
                    "grupo": "Oncología",
                    "setup_min": 10.0,
                    "proc_time_min": 60.0,
                    "transition_min": 5.0,
                    "cleanup_min": 8.0,
                }
            ],
        )
        csv_path = str(tmp_path / "breakdown_v3.csv")
        db.export_breakdown_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = csv.DictReader(f).fieldnames
        assert "sim_index" in header, f"sim_index not in {header}"
        assert "sim_i" not in header, f"Legacy 'sim_i' found in {header}"


class TestV3LegacyDBRejectedNotMigrated:
    """Una DB v2 antigua debe ser reconocida como legacy — sin migración, sin compat."""

    def _create_v2_db_in_memory(self):
        """Crea una DB con schema v2 (gen/sim_i/iter_id/best_sim_i)."""
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
            INSERT INTO schema_version VALUES (2);
            CREATE TABLE runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                num_simulations INTEGER NOT NULL,
                num_procedures INTEGER NOT NULL,
                config_snapshot TEXT
            );
            CREATE TABLE simulations (
                sim_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                sim_i INTEGER NOT NULL,
                algo_name TEXT NOT NULL,
                wall_clock_elapsed_s REAL NOT NULL,
                final_makespan REAL NOT NULL,
                combined_obj REAL,
                algo_time_s REAL NOT NULL DEFAULT 0.0
            );
            CREATE TABLE algorithm_iterations (
                iter_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sim_id INTEGER NOT NULL REFERENCES simulations(sim_id),
                generation INTEGER NOT NULL,
                best_fitness REAL NOT NULL,
                best_makespan REAL NOT NULL,
                iteration_fitness REAL NOT NULL,
                iteration_makespan REAL NOT NULL
            );
            CREATE TABLE cie10_breakdown (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sim_id INTEGER NOT NULL REFERENCES simulations(sim_id),
                job_id INTEGER NOT NULL,
                codigo_cie10 TEXT,
                grupo TEXT,
                setup_min REAL NOT NULL,
                proc_time_min REAL NOT NULL,
                transition_min REAL NOT NULL,
                cleanup_min REAL NOT NULL
            );
            CREATE TABLE checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                algo_name TEXT NOT NULL,
                checkpoint_wall_s REAL NOT NULL,
                best_makespan REAL NOT NULL,
                best_sim_i INTEGER NOT NULL
            );
        """)
        conn.commit()
        return conn

    def test_v2_db_raises_on_init_db(self):
        """init_db sobre una DB v2 debe lanzar RuntimeError (DB legacy, regenerar)."""
        v2_conn = self._create_v2_db_in_memory()
        persistence = AnalysisPersistence.__new__(AnalysisPersistence)
        persistence._db_path = ":memory:"
        persistence._conn = v2_conn
        with pytest.raises(RuntimeError, match="legacy"):
            persistence.init_db()
