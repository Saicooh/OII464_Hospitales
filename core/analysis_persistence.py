"""
core/analysis_persistence.py

Capa de persistencia SQLite para el modo de análisis temporal y variabilidad
CIE-10. Todas las operaciones son opt-in y no se invocan en modo normal.

Schema v6 (schedule_quality_metrics + v_simulation_summary):
- schema_version: version=6
- runs, simulations, algorithm_iterations, iteration_schedules: sin cambios
- cie10_breakdown, checkpoints, patient_wait_metrics: sin cambios
- schedule_quality_metrics: overtime, workload balance, idle gaps, VA ratio (NEW)
- v_simulation_summary: VIEW que aplana todo en una fila por simulación (NEW)

Nota: No existe ruta de migración desde v5 ni anterior. Cualquier DB anterior
se considera legacy regenerable (eliminar y recrear).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Schema DDL v6
# ---------------------------------------------------------------------------

_SCHEMA_DDL_V6 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT    NOT NULL,
    num_simulations INTEGER NOT NULL,
    num_procedures  INTEGER NOT NULL,
    config_snapshot TEXT
);

CREATE TABLE IF NOT EXISTS simulations (
    sim_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL REFERENCES runs(run_id),
    sim_index            INTEGER NOT NULL,
    algo_name            TEXT    NOT NULL,
    wall_clock_elapsed_s REAL    NOT NULL,
    final_makespan       REAL    NOT NULL,
    combined_obj         REAL,
    algo_time_s          REAL    NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS algorithm_iterations (
    algo_iter_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_id             INTEGER NOT NULL REFERENCES simulations(sim_id),
    algo_step          INTEGER NOT NULL,
    best_fitness       REAL    NOT NULL,
    best_makespan      REAL    NOT NULL,
    iteration_fitness  REAL    NOT NULL,
    iteration_makespan REAL    NOT NULL,
    UNIQUE(sim_id, algo_step)
);

CREATE TABLE IF NOT EXISTS iteration_schedules (
    iteration_schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    algo_iter_id          INTEGER NOT NULL UNIQUE REFERENCES algorithm_iterations(algo_iter_id) ON DELETE CASCADE,
    solution_json         TEXT    NOT NULL,
    solution_format       TEXT    NOT NULL DEFAULT 'scheduler_solution_v1',
    solution_sha256       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS cie10_breakdown (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_id         INTEGER NOT NULL REFERENCES simulations(sim_id),
    job_id         INTEGER NOT NULL,
    codigo_cie10   TEXT,
    grupo          TEXT,
    setup_min      REAL    NOT NULL,
    proc_time_min  REAL    NOT NULL,
    transition_min REAL    NOT NULL,
    cleanup_min    REAL    NOT NULL,
    source_record_id TEXT,
    estrategia_muestreo TEXT,
    setup_op1      REAL,
    setup_op2      REAL,
    cleanup_op1    REAL,
    cleanup_op2    REAL,
    tiempos_dinamicos_en_simulacion BOOLEAN
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                INTEGER NOT NULL REFERENCES runs(run_id),
    algo_name             TEXT    NOT NULL,
    checkpoint_wall_s     REAL    NOT NULL,
    best_makespan         REAL    NOT NULL,
    best_sim_index        INTEGER NOT NULL,
    simulations_completed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS patient_wait_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_id          INTEGER NOT NULL REFERENCES simulations(sim_id),
    job_id          INTEGER NOT NULL,
    op1_room        TEXT,
    op2_room        TEXT,
    op1_finish      REAL    NOT NULL,
    op2_start       REAL    NOT NULL,
    transition_used REAL    NOT NULL DEFAULT 0.0,
    extra_wait_min  REAL    NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS schedule_quality_metrics (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_id                INTEGER NOT NULL UNIQUE REFERENCES simulations(sim_id),
    rooms_used            INTEGER NOT NULL DEFAULT 0,
    total_overtime_min    REAL    NOT NULL DEFAULT 0.0,
    max_room_overtime_min REAL    NOT NULL DEFAULT 0.0,
    personnel_count       INTEGER NOT NULL DEFAULT 0,
    workload_std_min      REAL    NOT NULL DEFAULT 0.0,
    workload_max_min      REAL    NOT NULL DEFAULT 0.0,
    workload_min_min      REAL    NOT NULL DEFAULT 0.0,
    idle_gap_count        INTEGER NOT NULL DEFAULT 0,
    idle_gap_total_min    REAL    NOT NULL DEFAULT 0.0,
    avg_idle_gap_min      REAL    NOT NULL DEFAULT 0.0,
    value_added_ratio     REAL    NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_sim_run          ON simulations(run_id, algo_name);
CREATE INDEX IF NOT EXISTS idx_algo_iter_sim    ON algorithm_iterations(sim_id);
CREATE INDEX IF NOT EXISTS idx_iter_sched       ON iteration_schedules(algo_iter_id);
CREATE INDEX IF NOT EXISTS idx_breakdown_sim    ON cie10_breakdown(sim_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_run   ON checkpoints(run_id, algo_name);
CREATE INDEX IF NOT EXISTS idx_patient_wait_sim ON patient_wait_metrics(sim_id);
CREATE INDEX IF NOT EXISTS idx_quality_sim      ON schedule_quality_metrics(sim_id);

-- Vista unificada: una fila por simulación con todos los KPIs
CREATE VIEW IF NOT EXISTS v_simulation_summary AS
SELECT
    s.sim_id,
    s.run_id,
    r.num_procedures,
    s.sim_index,
    s.algo_name,
    s.wall_clock_elapsed_s,
    s.final_makespan,
    s.combined_obj,
    s.algo_time_s,
    q.rooms_used,
    q.total_overtime_min,
    q.max_room_overtime_min,
    q.personnel_count,
    q.workload_std_min,
    q.workload_max_min,
    q.workload_min_min,
    q.idle_gap_count,
    q.idle_gap_total_min,
    q.avg_idle_gap_min,
    q.value_added_ratio,
    pw.total_patients,
    pw.patients_with_extra_wait,
    pw.avg_extra_wait_min,
    pw.max_extra_wait_min
FROM simulations s
JOIN runs r ON s.run_id = r.run_id
LEFT JOIN schedule_quality_metrics q ON s.sim_id = q.sim_id
LEFT JOIN (
    SELECT
        sim_id,
        COUNT(*)                                                AS total_patients,
        SUM(CASE WHEN extra_wait_min > 0.01 THEN 1 ELSE 0 END) AS patients_with_extra_wait,
        AVG(CASE WHEN extra_wait_min > 0.01 THEN extra_wait_min END) AS avg_extra_wait_min,
        MAX(extra_wait_min)                                     AS max_extra_wait_min
    FROM patient_wait_metrics
    GROUP BY sim_id
) pw ON s.sim_id = pw.sim_id;
"""

_SCHEMA_VERSION = 6


class AnalysisPersistence:
    """Persistencia SQLite para el modo de análisis temporal.

    Parameters
    ----------
    db_path:
        Ruta al archivo SQLite. Usa ":memory:" para pruebas en memoria.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    @property
    def db_path(self) -> str:
        """Path to the SQLite database file."""
        return self._db_path

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Inicializa el schema v6 limpio.

        Si la DB ya tiene un schema anterior (v1, v2, v3, v4 o v5), lanza RuntimeError
        indicando que es una DB legacy que debe eliminarse y regenerarse.
        """
        current_version = self._detect_schema_version()

        if current_version == 0:
            # DB vacía → crear schema v4 desde cero
            self._conn.executescript(_SCHEMA_DDL_V6)
            self._set_schema_version(_SCHEMA_VERSION)
        elif current_version == _SCHEMA_VERSION:
            # Ya está en v6 → nada que hacer (idempotente)
            pass
        else:
            # Versión legacy → no hay migración; el usuario debe regenerar
            raise RuntimeError(
                f"DB legacy detectada (schema_version={current_version}). "
                "Elimina el archivo analysis.db y vuelve a ejecutar para regenerar "
                f"con el schema v{_SCHEMA_VERSION}. No existe ruta de migración automática."
            )

        self._conn.commit()

    def _detect_schema_version(self) -> int:
        """Retorna la versión del schema actual.

        Returns
        -------
        0: DB vacía (sin schema)
        1: Schema v1 (tiene tabla 'iterations', sin 'schema_version')
        N: Schema vN (tiene 'schema_version' con version=N)
        """
        tables = {
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        if "schema_version" in tables:
            version = self._conn.execute(
                "SELECT version FROM schema_version"
            ).fetchone()
            return version[0] if version else 0

        if "iterations" in tables:
            return 1  # v1: tiene iterations pero no schema_version

        return 0  # DB vacía

    def _set_schema_version(self, version: int) -> None:
        """Upsert the schema version."""
        self._conn.execute("DELETE FROM schema_version")
        self._conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (version,)
        )

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def insert_run(self, num_sims: int, num_procs: int, config: dict) -> int:
        """Inserta un nuevo registro de corrida y devuelve su run_id."""
        started_at = datetime.now(timezone.utc).isoformat()
        config_json = json.dumps(config, ensure_ascii=False)
        cursor = self._conn.execute(
            "INSERT INTO runs (started_at, num_simulations, num_procedures, config_snapshot)"
            " VALUES (?, ?, ?, ?)",
            (started_at, num_sims, num_procs, config_json),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Simulations
    # ------------------------------------------------------------------

    def insert_simulation(
        self,
        run_id: int,
        sim_index: int,
        algo_name: str,
        wall_clock_s: float,
        final_makespan: float,
        combined_obj: Optional[float],
        algo_time_s: float = 0.0,
    ) -> int:
        """Inserta una simulación completada y devuelve su sim_id.

        Parameters
        ----------
        run_id: ID de la corrida padre.
        sim_index: índice de la simulación dentro de la corrida.
        algo_name: nombre del algoritmo.
        wall_clock_s: tiempo de pared transcurrido al finalizar la simulación.
        final_makespan: mejor makespan al final de la simulación.
        combined_obj: objetivo combinado final (None si no disponible).
        algo_time_s: tiempo de CPU del algoritmo (default 0.0).

        Returns
        -------
        sim_id: entero > 0.
        """
        cursor = self._conn.execute(
            "INSERT INTO simulations"
            " (run_id, sim_index, algo_name, wall_clock_elapsed_s,"
            "  final_makespan, combined_obj, algo_time_s)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                sim_index,
                algo_name,
                wall_clock_s,
                final_makespan,
                combined_obj,
                algo_time_s,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Algorithm iterations batch
    # ------------------------------------------------------------------

    def save_algorithm_iterations_batch(
        self, sim_id: int, rows: list[dict]
    ) -> list[int]:
        """Inserta un lote de iteraciones internas del algoritmo.

        Parameters
        ----------
        sim_id: ID de la simulación padre.
        rows: lista de dicts con claves:
            algo_step, best_fitness, best_makespan,
            iteration_fitness, iteration_makespan

        Returns
        -------
        Lista de algo_iter_ids creados (en el mismo orden que rows).
        """
        algo_iter_ids: list[int] = []
        with self._conn:
            for row in rows:
                cursor = self._conn.execute(
                    "INSERT INTO algorithm_iterations"
                    " (sim_id, algo_step, best_fitness, best_makespan,"
                    "  iteration_fitness, iteration_makespan)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        sim_id,
                        row["algo_step"],
                        row["best_fitness"],
                        row["best_makespan"],
                        row["iteration_fitness"],
                        row["iteration_makespan"],
                    ),
                )
                algo_iter_ids.append(cursor.lastrowid)  # type: ignore[arg-type]
        return algo_iter_ids

    # ------------------------------------------------------------------
    # Iteration schedules batch (v4)
    # ------------------------------------------------------------------

    def save_iteration_schedules_batch(self, schedules: list[dict]) -> None:
        """Inserta un lote de schedules serializados por iteración.

        Parameters
        ----------
        schedules: lista de dicts con claves:
            algo_iter_id, solution_json (string JSON)
        """
        with self._conn:
            for item in schedules:
                solution_json = item["solution_json"]
                sha256 = hashlib.sha256(
                    solution_json.encode("utf-8")
                ).hexdigest()
                self._conn.execute(
                    "INSERT INTO iteration_schedules"
                    " (algo_iter_id, solution_json, solution_format, solution_sha256)"
                    " VALUES (?, ?, ?, ?)",
                    (
                        item["algo_iter_id"],
                        solution_json,
                        "scheduler_solution_v1",
                        sha256,
                    ),
                )

    def get_iteration_schedules_for_sim(self, sim_id: int) -> list[dict]:
        """Retorna las filas de iteration_schedules para un sim_id dado.

        Returns
        -------
        Lista de dicts con claves: algo_iter_id, algo_step, solution_json.
        Ordenados por algo_step ASC.
        """
        rows = self._conn.execute(
            "SELECT isch.algo_iter_id, ai.algo_step, isch.solution_json"
            " FROM iteration_schedules isch"
            " JOIN algorithm_iterations ai ON ai.algo_iter_id = isch.algo_iter_id"
            " WHERE ai.sim_id = ?"
            " ORDER BY ai.algo_step ASC",
            (sim_id,),
        ).fetchall()
        return [
            {"algo_iter_id": r[0], "algo_step": r[1], "solution_json": r[2]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # CIE-10 breakdown batch
    # ------------------------------------------------------------------

    def save_breakdowns_batch(self, sim_id: int, breakdowns: list[dict]) -> None:
        """Inserta un lote de desglose CIE-10 vinculado a una simulación.

        Parameters
        ----------
        sim_id: ID de la simulación a la que pertenecen los breakdowns.
        breakdowns: lista de dicts con claves:
            job_id, codigo_cie10, grupo,
            setup_min, proc_time_min, transition_min, cleanup_min
        """
        with self._conn:
            for bd in breakdowns:
                self._conn.execute(
                    "INSERT INTO cie10_breakdown"
                    " (sim_id, job_id, codigo_cie10, grupo,"
                    "  setup_min, proc_time_min, transition_min, cleanup_min,"
                    "  source_record_id, estrategia_muestreo, setup_op1,"
                    "  setup_op2, cleanup_op1, cleanup_op2, tiempos_dinamicos_en_simulacion)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        sim_id,
                        bd["job_id"],
                        bd.get("codigo_cie10"),
                        bd.get("grupo"),
                        bd["setup_min"],
                        bd["proc_time_min"],
                        bd["transition_min"],
                        bd["cleanup_min"],
                        bd.get("source_record_id"),
                        bd.get("estrategia_muestreo"),
                        bd.get("setup_op1"),
                        bd.get("setup_op2"),
                        bd.get("cleanup_op1"),
                        bd.get("cleanup_op2"),
                        bd.get("tiempos_dinamicos_en_simulacion"),
                    ),
                )

    # ------------------------------------------------------------------
    # Checkpoint reconstruction
    # ------------------------------------------------------------------

    def reconstruct_checkpoints(self, run_id: int, interval_s: float) -> None:
        """Reconstruye checkpoints temporales post-hoc desde la tabla 'simulations'."""
        self._conn.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
        self._conn.commit()

        algos = [
            row[0]
            for row in self._conn.execute(
                "SELECT DISTINCT algo_name FROM simulations WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        ]

        for algo_name in algos:
            rows = self._conn.execute(
                "SELECT wall_clock_elapsed_s, final_makespan, sim_index "
                "FROM simulations "
                "WHERE run_id = ? AND algo_name = ? "
                "ORDER BY wall_clock_elapsed_s ASC",
                (run_id, algo_name),
            ).fetchall()

            if not rows:
                continue

            max_elapsed = rows[-1][0]
            num_windows = int(max_elapsed / interval_s) + 1

            checkpoint_rows = []
            for window_idx in range(num_windows):
                cutoff = (window_idx + 1) * interval_s
                candidates = [r for r in rows if r[0] <= cutoff]
                if not candidates:
                    continue
                best_row = min(candidates, key=lambda r: r[1])
                checkpoint_rows.append(
                    (run_id, algo_name, cutoff, best_row[1], best_row[2], len(candidates))
                )

            if checkpoint_rows:
                self._conn.executemany(
                    "INSERT INTO checkpoints"
                    " (run_id, algo_name, checkpoint_wall_s, best_makespan, best_sim_index, simulations_completed)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    checkpoint_rows,
                )
                self._conn.commit()

    # ------------------------------------------------------------------
    # CSV exports
    # ------------------------------------------------------------------

    def export_algorithm_iterations_csv(self, run_id: int, path: str) -> None:
        """Exporta las iteraciones internas de algoritmos de una corrida a CSV."""
        rows = self._conn.execute(
            "SELECT ai.algo_iter_id, s.sim_index, s.algo_name,"
            "       ai.algo_step, ai.best_fitness, ai.best_makespan,"
            "       ai.iteration_fitness, ai.iteration_makespan"
            " FROM algorithm_iterations ai"
            " JOIN simulations s ON ai.sim_id = s.sim_id"
            " WHERE s.run_id = ?"
            " ORDER BY s.sim_index, s.algo_name, ai.algo_step",
            (run_id,),
        ).fetchall()

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "algo_iter_id",
                    "sim_index",
                    "algo_name",
                    "algo_step",
                    "best_fitness",
                    "makespan_of_best_fitness",
                    "iteration_fitness",
                    "iteration_makespan",
                ]
            )
            writer.writerows(rows)

    def export_checkpoints_csv(self, run_id: int, path: str) -> None:
        """Exporta los checkpoints de una corrida a un archivo CSV."""
        rows = self._conn.execute(
            "SELECT run_id, algo_name, checkpoint_wall_s, best_makespan, best_sim_index, simulations_completed"
            " FROM checkpoints WHERE run_id = ?"
            " ORDER BY algo_name, checkpoint_wall_s",
            (run_id,),
        ).fetchall()

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "run_id",
                    "algo_name",
                    "checkpoint_wall_s",
                    "makespan_of_best_fitness",
                    "best_sim_index",
                    "simulations_completed",
                ]
            )
            writer.writerows(rows)

    def export_breakdown_csv(self, run_id: int, path: str) -> None:
        """Exporta el desglose CIE-10 de una corrida a un archivo CSV."""
        rows = self._conn.execute(
            "SELECT s.sim_index, s.algo_name, b.job_id,"
            " b.codigo_cie10, b.grupo, b.setup_min, b.proc_time_min,"
            " b.transition_min, b.cleanup_min, b.source_record_id,"
            " b.estrategia_muestreo, b.setup_op1, b.setup_op2,"
            " b.cleanup_op1, b.cleanup_op2, b.tiempos_dinamicos_en_simulacion"
            " FROM cie10_breakdown b"
            " JOIN simulations s ON b.sim_id = s.sim_id"
            " WHERE s.run_id = ?"
            " ORDER BY s.sim_index, s.algo_name, b.job_id",
            (run_id,),
        ).fetchall()

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sim_index",
                    "algo_name",
                    "job_id",
                    "codigo_cie10",
                    "grupo",
                    "setup_min",
                    "proc_time_min",
                    "transition_min",
                    "cleanup_min",
                    "source_record_id",
                    "estrategia_muestreo",
                    "setup_op1",
                    "setup_op2",
                    "cleanup_op1",
                    "cleanup_op2",
                    "tiempos_dinamicos_en_simulacion",
                ]
            )
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Patient wait metrics
    # ------------------------------------------------------------------

    def save_patient_wait_batch(self, sim_id: int, metrics: list[dict]) -> None:
        """Inserta métricas de espera de paciente en batch.

        Args:
            sim_id: ID de la simulación.
            metrics: Lista de dicts con job_id, op1_room, op2_room,
                     op1_finish, op2_start, transition_used, extra_wait_min.
        """
        if not metrics:
            return
        self._conn.executemany(
            "INSERT INTO patient_wait_metrics "
            "(sim_id, job_id, op1_room, op2_room, op1_finish, op2_start, "
            " transition_used, extra_wait_min) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    sim_id,
                    m["job_id"],
                    m.get("op1_room", ""),
                    m.get("op2_room", ""),
                    m["op1_finish"],
                    m["op2_start"],
                    m.get("transition_used", 0),
                    m["extra_wait_min"],
                )
                for m in metrics
            ],
        )
        self._conn.commit()

    def export_patient_wait_csv(self, run_id: int, path: str) -> None:
        """Exporta las métricas de espera de paciente de un run a CSV."""
        rows = self._conn.execute(
            "SELECT s.sim_index, s.algo_name, pw.job_id, "
            "       pw.op1_room, pw.op2_room, pw.op1_finish, pw.op2_start, "
            "       pw.transition_used, pw.extra_wait_min "
            "FROM patient_wait_metrics pw "
            "JOIN simulations s ON pw.sim_id = s.sim_id "
            "WHERE s.run_id = ? "
            "ORDER BY s.sim_index, s.algo_name, pw.job_id",
            (run_id,),
        ).fetchall()

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sim_index",
                    "algo_name",
                    "job_id",
                    "op1_room",
                    "op2_room",
                    "op1_finish",
                    "op2_start",
                    "transition_used",
                    "extra_wait_min",
                ]
            )
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Schedule quality metrics
    # ------------------------------------------------------------------

    def save_quality_metrics(self, sim_id: int, metrics: dict) -> None:
        """Inserta métricas de calidad de schedule para una simulación.

        Semántica de upsert: usa ``INSERT OR REPLACE`` sobre la clave única ``sim_id``,
        por lo que llamar dos veces con el mismo ``sim_id`` sobreescribe los valores
        anteriores sin duplicar filas.

        Args:
            sim_id: ID de la simulación.
            metrics: dict con rooms_used, total_overtime_min, max_room_overtime_min,
                     personnel_count, workload_std_min, workload_max_min, workload_min_min,
                     idle_gap_count, idle_gap_total_min, avg_idle_gap_min, value_added_ratio.
        """
        if not metrics:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO schedule_quality_metrics "
            "(sim_id, rooms_used, total_overtime_min, max_room_overtime_min, "
            " personnel_count, workload_std_min, workload_max_min, workload_min_min, "
            " idle_gap_count, idle_gap_total_min, avg_idle_gap_min, value_added_ratio) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sim_id,
                metrics.get("rooms_used", 0),
                metrics.get("total_overtime_min", 0),
                metrics.get("max_room_overtime_min", 0),
                metrics.get("personnel_count", 0),
                metrics.get("workload_std_min", 0),
                metrics.get("workload_max_min", 0),
                metrics.get("workload_min_min", 0),
                metrics.get("idle_gap_count", 0),
                metrics.get("idle_gap_total_min", 0),
                metrics.get("avg_idle_gap_min", 0),
                metrics.get("value_added_ratio", 0),
            ),
        )
        self._conn.commit()

    def export_quality_metrics_csv(self, run_id: int, path: str) -> None:
        """Exporta las métricas de calidad de schedule de un run a CSV."""
        rows = self._conn.execute(
            "SELECT s.sim_index, s.algo_name, "
            "       q.rooms_used, q.total_overtime_min, q.max_room_overtime_min, "
            "       q.personnel_count, q.workload_std_min, "
            "       q.workload_max_min, q.workload_min_min, "
            "       q.idle_gap_count, q.idle_gap_total_min, q.avg_idle_gap_min, "
            "       q.value_added_ratio "
            "FROM schedule_quality_metrics q "
            "JOIN simulations s ON q.sim_id = s.sim_id "
            "WHERE s.run_id = ? "
            "ORDER BY s.sim_index, s.algo_name",
            (run_id,),
        ).fetchall()

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sim_index",
                    "algo_name",
                    "rooms_used",
                    "total_overtime_min",
                    "max_room_overtime_min",
                    "personnel_count",
                    "workload_std_min",
                    "workload_max_min",
                    "workload_min_min",
                    "idle_gap_count",
                    "idle_gap_total_min",
                    "avg_idle_gap_min",
                    "value_added_ratio",
                ]
            )
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Unified simulation summary (from SQL View)
    # ------------------------------------------------------------------

    def export_simulation_summary_csv(self, run_id: int, path: str) -> None:
        """Exporta la vista unificada v_simulation_summary a CSV para un run."""
        rows = self._conn.execute(
            "SELECT "
            "    run_id, "
            "    num_procedures, "
            "    sim_index, "
            "    algo_name, "
            "    final_makespan, "
            "    algo_time_s, "
            "    rooms_used, "
            "    total_overtime_min, "
            "    max_room_overtime_min, "
            "    personnel_count, "
            "    workload_std_min, "
            "    workload_max_min, "
            "    workload_min_min, "
            "    idle_gap_count, "
            "    idle_gap_total_min, "
            "    avg_idle_gap_min, "
            "    value_added_ratio, "
            "    total_patients, "
            "    patients_with_extra_wait, "
            "    avg_extra_wait_min, "
            "    max_extra_wait_min "
            "FROM v_simulation_summary "
            "WHERE run_id = ? "
            "ORDER BY sim_index, algo_name",
            (run_id,),
        ).fetchall()

        if not rows:
            return

        col_names = [
            "run_id",
            "num_procedures",
            "sim_index",
            "algo_name",
            "final_makespan",
            "algo_time_s",
            "rooms_used",
            "total_overtime_min",
            "max_room_overtime_min",
            "personnel_count",
            "workload_std_min",
            "workload_max_min",
            "workload_min_min",
            "idle_gap_count",
            "idle_gap_total_min",
            "avg_idle_gap_min",
            "value_added_ratio",
            "total_patients",
            "patients_with_extra_wait",
            "avg_extra_wait_min",
            "max_extra_wait_min",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
            writer.writerows(rows)

    def export_best_runs_by_mh_csv(self, run_id: int, path: str) -> None:
        """Exporta las mejores corridas de cada metaheurística para un run (según combined_obj)."""
        rows = self._conn.execute(
            "WITH RankedSimulations AS ("
            "    SELECT *, "
            "           ROW_NUMBER() OVER ("
            "               PARTITION BY run_id, algo_name "
            "               ORDER BY COALESCE(combined_obj, final_makespan) ASC, final_makespan ASC"
            "           ) as rank "
            "    FROM v_simulation_summary"
            "    WHERE run_id = ?"
            ") "
            "SELECT "
            "    run_id, "
            "    num_procedures, "
            "    sim_index, "
            "    algo_name, "
            "    final_makespan, "
            "    combined_obj, "
            "    algo_time_s, "
            "    rooms_used, "
            "    total_overtime_min, "
            "    max_room_overtime_min, "
            "    personnel_count, "
            "    workload_std_min, "
            "    workload_max_min, "
            "    workload_min_min, "
            "    idle_gap_count, "
            "    idle_gap_total_min, "
            "    avg_idle_gap_min, "
            "    value_added_ratio, "
            "    total_patients, "
            "    patients_with_extra_wait, "
            "    avg_extra_wait_min, "
            "    max_extra_wait_min "
            "FROM RankedSimulations "
            "WHERE rank = 1 "
            "ORDER BY algo_name",
            (run_id,),
        ).fetchall()

        if not rows:
            return

        col_names = [
            "run_id",
            "num_procedures",
            "sim_index",
            "algo_name",
            "final_makespan",
            "combined_obj",
            "algo_time_s",
            "rooms_used",
            "total_overtime_min",
            "max_room_overtime_min",
            "personnel_count",
            "workload_std_min",
            "workload_max_min",
            "workload_min_min",
            "idle_gap_count",
            "idle_gap_total_min",
            "avg_idle_gap_min",
            "value_added_ratio",
            "total_patients",
            "patients_with_extra_wait",
            "avg_extra_wait_min",
            "max_extra_wait_min",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
            writer.writerows(rows)


    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Cierra la conexión SQLite."""
        self._conn.close()
