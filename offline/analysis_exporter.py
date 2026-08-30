"""
AnalysisExporter — export offline de schedules por iteración.

Lee iteration_schedules desde SQLite (schema v6), reconstruye surgeries_data
desde la información persistida de la campaña, llama
calculate_schedule_fitness(..., return_details=True) y exporta:
  - schedule_by_iteration.csv   — Gantt detallado: una fila por operación por iteración
  - strategy_by_iteration.csv   — Estrategia de asignación de pabellones por iteración

Nota: breakdown_by_iteration fue eliminado por ser derivable de schedule_by_iteration.
"""

import csv
import json
import logging
import os
import sys
import sqlite3
from collections import defaultdict


logger = logging.getLogger(__name__)


class IterationExportError(RuntimeError):
    """Una iteración no pudo recalcularse desde la evidencia persistida."""

# Asegurar que el directorio raíz del proyecto está en el PYTHONPATH
# Esto es crítico para que los workers paralelos de joblib (en Windows)
# puedan encontrar los módulos como 'simulation' y 'data'.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class AnalysisExporter:
    """Exporta schedules por iteración desde una DB de análisis v6."""

    # ------------------------------------------------------------------
    # Columnas de cada CSV
    # ------------------------------------------------------------------

    SCHEDULE_HEADERS = [
        "run_id", "sim_index", "algo_name", "algo_step",
        "job_id", "operation_num", "room", "personnel",
        "start_time", "processing_end", "end_time",
        "setup_used", "transition_used", "cleanup_used",
    ]

    STRATEGY_HEADERS = [
        "run_id", "sim_index", "algo_name", "algo_step",
        "room", "operation_sequence",
    ]



    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def export_iteration_csvs(self, db_path: str, output_dir: str, n_jobs: int = -1) -> None:
        """
        Lee iteration_schedules desde db_path y escribe los 2 CSVs en output_dir:
          - schedule_by_iteration.csv
          - strategy_by_iteration.csv

        Args:
            db_path:    Ruta al archivo SQLite generado por analysis mode.
            output_dir: Directorio donde se escriben los CSVs.
            n_jobs:     Número de procesos concurrentes (por defecto -1).
        """
        from joblib import Parallel, delayed
        from tqdm import tqdm

        os.makedirs(output_dir, exist_ok=True)

        schedule_path = os.path.join(output_dir, "schedule_by_iteration.csv")
        strategy_path = os.path.join(output_dir, "strategy_by_iteration.csv")

        print("Leyendo registros desde SQLite...")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = self._fetch_iteration_data(conn)
            # Convertimos las filas a diccionarios nativos para que sean picklable por joblib
            rows_data = [dict(row) for row in rows]
        finally:
            conn.close()

        # Reconstruct the campaign instance once per run.  This is deliberately
        # done before dispatching rows to joblib so workers receive only data
        # recovered from the read-only analysis database.
        from reproducibility.instance_reconstruction import (
            reconstruct_instance,
            to_surgeries_data,
        )

        day_data_by_run = {}
        for row in rows_data:
            run_id = row["run_id"]
            if run_id not in day_data_by_run:
                day_data_by_run[run_id] = to_surgeries_data(
                    reconstruct_instance(db_path, run_id)
                )
            row["day_data"] = day_data_by_run[run_id]

        print(f"Calculando fitness y rehidratando {len(rows_data)} iteraciones en paralelo...")
        # Procesamos en paralelo devolviendo listas de filas a insertar
        # The scheduler resolves configuration at import time. Threads share
        # the active module state; reused process workers can retain a stale
        # configuration profile from an earlier export.
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_process_row_data)(row_dict) for row_dict in tqdm(rows_data, desc="Procesando Iteraciones")
        )

        print("Escribiendo resultados en CSVs...")
        with (
            open(schedule_path, "w", newline="", encoding="utf-8") as sched_f,
            open(strategy_path, "w", newline="", encoding="utf-8") as strat_f,
        ):
            sched_w = csv.DictWriter(sched_f, fieldnames=self.SCHEDULE_HEADERS)
            strat_w = csv.DictWriter(strat_f, fieldnames=self.STRATEGY_HEADERS)

            sched_w.writeheader()
            strat_w.writeheader()

            for sched_rows, strat_rows, _brkd_rows in results:
                sched_w.writerows(sched_rows)
                strat_w.writerows(strat_rows)

        # Exportar la caracterización de instancias en el mismo lote de CSVs
        self.export_instance_characterization(db_path, output_dir)

        print(f"Exportación de CSVs finalizada exitosamente en {output_dir}")

    def export_instance_characterization(self, db_path: str, output_dir: str) -> None:
        """
        Extrae la caracterización de instancias (cirugías nominales de cada run)
        desde la tabla cie10_breakdown de la base de datos y la exporta a CSV.
        """
        csv_path = os.path.join(output_dir, "instance_characterization.csv")
        print(f"Exportando caracterización de instancias a: {csv_path}...")
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT run_id FROM simulations")
            runs = [r[0] for r in cursor.fetchall()]
            if not runs:
                print("Advertencia: No se encontraron runs en simulations. No se puede exportar la caracterización.")
                return

            rows_to_write = []
            for run_id in sorted(runs):
                cursor.execute("""
                    SELECT sim_id FROM simulations 
                    WHERE run_id = ? 
                    LIMIT 1
                """, (run_id,))
                res = cursor.fetchone()
                if not res:
                    continue
                sim_id = res[0]

                cursor.execute("""
                    SELECT job_id, codigo_cie10, grupo, setup_min, proc_time_min, transition_min, cleanup_min
                    FROM cie10_breakdown
                    WHERE sim_id = ?
                    ORDER BY job_id
                """, (sim_id,))
                
                for row in cursor.fetchall():
                    job_id, cie10, grupo, setup, proc_time, transition, cleanup = row
                    rows_to_write.append({
                        "run_id": run_id,
                        "job_id": job_id,
                        "codigo_cie10": cie10,
                        "grupo": grupo,
                        "setup_min": round(setup, 4),
                        "proc_time_min": round(proc_time, 4),
                        "transition_min": round(transition, 4),
                        "cleanup_min": round(cleanup, 4)
                    })

            if rows_to_write:
                headers = ["run_id", "job_id", "codigo_cie10", "grupo", "setup_min", "proc_time_min", "transition_min", "cleanup_min"]
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows_to_write)
                print("Caracterización de instancias exportada exitosamente.")
            else:
                print("Advertencia: No se encontraron filas de cie10_breakdown para exportar.")
        except Exception as e:
            message = f"Failed to export instance characterization from {db_path!r}"
            logger.exception(message)
            raise IterationExportError(message) from e
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_iteration_data(self, conn):
        """
        Recupera todas las filas de iteration_schedules JOIN algorithm_iterations JOIN simulations.
        Ordena por algo_name, sim_id, algo_step para procesamiento secuencial.
        """
        sql = """
            SELECT
                sim.run_id,
                sim.sim_id,
                sim.sim_index,
                sim.algo_name,
                ai.algo_step,
                ai.best_fitness,
                ai.best_makespan,
                isch.solution_json
            FROM iteration_schedules isch
            JOIN algorithm_iterations ai  ON isch.algo_iter_id = ai.algo_iter_id
            JOIN simulations sim           ON ai.sim_id = sim.sim_id
            JOIN runs r                    ON sim.run_id = r.run_id
            ORDER BY sim.algo_name ASC, sim.sim_id ASC, ai.algo_step ASC
        """
        return conn.execute(sql).fetchall()


# =====================================================================
# Funciones puras (fuera de la clase para pickling con joblib)
# =====================================================================

def _process_row_data(row_dict: dict):
    """Recalcula una iteración con la instancia persistida y devuelve sus filas."""
    from simulation.scheduler import calculate_schedule_fitness

    run_id = row_dict["run_id"]
    sim_index = row_dict["sim_index"]
    algo_name = row_dict["algo_name"]
    algo_step = row_dict["algo_step"]
    solution = json.loads(row_dict["solution_json"])

    day_data = row_dict["day_data"]
    normalized = _normalize_solution_static(solution)

    try:
        result = calculate_schedule_fitness(normalized, day_data, return_details=True)
    except Exception as exc:
        message = (
            "Failed to export persisted iteration "
            f"run_id={run_id}, sim_index={sim_index}, "
            f"algo_name={algo_name!r}, algo_step={algo_step}"
        )
        logger.exception(message)
        raise IterationExportError(message) from exc

    if result is None or result[2] is None:
        return [], [], []

    combined_obj, makespan, schedule_details = result

    if not schedule_details:
        return [], [], []

    # Contexto común para todas las filas de esta iteración
    ctx = {
        "run_id": run_id,
        "sim_index": sim_index,
        "algo_name": algo_name,
        "algo_step": algo_step,
    }

    # --- schedule_by_iteration.csv ---
    # Acumulamos por room para reconstruir la estrategia
    # y por job para el breakdown
    room_ops = defaultdict(list)
    job_times = defaultdict(lambda: {"start": float("inf"), "finish": float("-inf")})
    sched_rows = []

    for detail in schedule_details:
        job = detail.get("Job")
        op = detail.get("Operation")
        room = detail.get("Resource", "")
        personnel = detail.get("Personnel", "")
        start = detail.get("Start", 0.0)
        proc_end = detail.get("ProcessingEnd", start)
        finish = detail.get("Finish", start)
        setup_used = detail.get("SetupUsed", 0.0)
        transition_used = detail.get("TransitionUsed")
        cleanup_used = detail.get("CleanupUsed", 0.0)

        sched_rows.append({
            **ctx,
            "job_id": job,
            "operation_num": op,
            "room": room,
            "personnel": personnel,
            "start_time": round(start, 4),
            "processing_end": round(proc_end, 4),
            "end_time": round(finish, 4),
            "setup_used": round(setup_used or 0.0, 4),
            "transition_used": round(transition_used, 4) if transition_used is not None else "",
            "cleanup_used": round(cleanup_used or 0.0, 4),
        })

        # Registrar operación en su room para la estrategia
        room_ops[room].append((start, job, op))

        # Registrar tiempos por job para el breakdown
        if job is not None:
            job_times[job]["start"] = min(job_times[job]["start"], start)
            job_times[job]["finish"] = max(job_times[job]["finish"], finish)

    # --- strategy_by_iteration.csv ---
    # Reconstruir la secuencia de operaciones por pabellón, ordenada por start_time
    strat_rows = []
    for room in sorted(room_ops.keys()):
        ops_sorted = sorted(room_ops[room], key=lambda x: x[0])
        sequence_str = " -> ".join(f"{job}(Op{op})" for _, job, op in ops_sorted)
        strat_rows.append({
            **ctx,
            "room": room,
            "operation_sequence": sequence_str,
        })

    # --- breakdown_by_iteration.csv ---
    # Una fila por job: total_wait = makespan_global - span del job; makespan = makespan global
    brkd_rows = []
    for job_id, times in job_times.items():
        job_span = times["finish"] - times["start"] if times["finish"] > times["start"] else 0.0
        total_wait = round(max(0.0, makespan - job_span), 4)
        brkd_rows.append({
            **ctx,
            "job_id": job_id,
            "total_wait": total_wait,
            "makespan": round(makespan, 4),
        })

    return sched_rows, strat_rows, brkd_rows


def _normalize_solution_static(solution: dict) -> dict:
    """Convierte json string keys a enteros para el scheduler."""
    raw_room = solution.get("room_assignment", {})
    room_assignment = {}
    for job_key, ops in raw_room.items():
        job_int = int(job_key)
        room_assignment[job_int] = {int(op_key): v for op_key, v in ops.items()}

    raw_seq = solution.get("job_sequence_base", [])
    job_sequence_base = [int(j) for j in raw_seq]

    return {
        "job_sequence_base": job_sequence_base,
        "room_assignment": room_assignment,
    }


def _find_latest_analysis_db(base_dir: str = "results") -> str | None:
    """Busca recursivamente todos los archivos 'analysis.db' y retorna el más reciente."""
    candidate_dbs = []
    if not os.path.exists(base_dir):
        return None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "analysis.db":
                path = os.path.join(root, file)
                candidate_dbs.append((path, os.path.getmtime(path)))
    if not candidate_dbs:
        return None
    # Ordenar por mtime descendente (el más nuevo primero)
    candidate_dbs.sort(key=lambda x: x[1], reverse=True)
    return candidate_dbs[0][0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exportador Offline de SQLite a CSV")
    parser.add_argument(
        "--db",
        type=str,
        required=False,
        default=None,
        help="Ruta al archivo analysis.db. Si no se especifica, usa el más reciente en 'results/'."
    )

    args = parser.parse_args()

    db_path = args.db
    if not db_path:
        db_path = _find_latest_analysis_db()
        if not db_path:
            print("Error: No se especificó --db y no se encontró ningún archivo 'analysis.db' en 'results/'.")
            sys.exit(1)
        print(f"No se especificó --db. Usando la base de datos más reciente: {db_path}")

    if not os.path.exists(db_path):
        print(f"Error: No se encontró la base de datos en '{db_path}'")
        sys.exit(1)

    # Usar la misma carpeta donde está el DB como directorio de salida
    output_directory = os.path.dirname(db_path)

    print(f"Iniciando exportación para: {db_path}")
    exporter = AnalysisExporter()
    exporter.export_iteration_csvs(db_path=db_path, output_dir=output_directory)
