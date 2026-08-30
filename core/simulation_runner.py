"""
High-level simulation runner that orchestrates the execution flow.
"""

from joblib import Parallel, delayed
from core.file_manager import FileManager
from core.report_generator import ReportGenerator
from simulation.workers.elective_worker import ElectiveWorker
from config.config import (
    JOB_TYPES,
    get_algorithms,
    NUM_SIMULATIONS,
    STD_FACTOR,
    ALPHA_TEST,
    ALL_ROOMS,
    VERBOSE_MODE,
    USE_REAL_DATA,
    TRACE_CSV_PATH,
    RESET_TRACE_ON_START,
    NUM_PROCEDURES,
    ANALYSIS_MODE_ENABLED,
    ANALYSIS_NUM_RUNS,
    ANALYSIS_SIMS_PER_RUN,
    ANALYSIS_SQLITE_PATH,
    ANALYSIS_CHECKPOINT_INTERVAL,
    ANALYSIS_EXPORT_CSV,
    ANALYSIS_TEMPORAL_ENABLED,
    ANALYSIS_SWEEP_ENABLED,
    ANALYSIS_SWEEP_VALUES,
    ANALYSIS_SWEEP_SIMS,
    ANALYSIS_CHECKPOINTS_CSV_PATH,
    ANALYSIS_BREAKDOWN_CSV_PATH,
    ANALYSIS_ITERATIONS_CSV_PATH,
    ANALYSIS_FULL_REPORTS_ENABLED,
    ANALYSIS_ARTIFACT_SAVE_MODE,
    ANALYSIS_FIXED_POOL,
    N_JOBS,
)
from utils.logger import logger
import os
import time


class SimulationRunner:
    """
    Orchestrates elective simulation execution.
    """

    def __init__(self):
        self.file_manager = FileManager()
        self.report_generator = ReportGenerator()
        self.job_ids = list(range(1, NUM_PROCEDURES + 1))
        self.n_jobs = N_JOBS
        self.algorithms = get_algorithms()
        self._setup_real_data()

    def _setup_real_data(self):
        """
        Prepara el entorno para datos reales si USE_REAL_DATA está habilitado.

        Acciones:
        - Imprime banner informativo.
        - Limpia el CSV de trazabilidad cruda si RESET_TRACE_ON_START=True.
        - Pre-carga el dataset PKL en el proceso principal para que el singleton
          esté listo antes de que los workers paralelos arranquen.
          (Evita condición de carrera en la carga del PKL desde joblib.)
        """
        if not USE_REAL_DATA:
            return

        logger.info(f"\n{'=' * 70}")
        logger.info("REAL DATA MODE ENABLED - Using PKL dataset (CIE10Dataset)")
        logger.info(f"  Trace CSV: {TRACE_CSV_PATH}")
        logger.info(f"{'=' * 70}")

        if RESET_TRACE_ON_START:
            from data.raw_trace_writer import reset_trace_file

            reset_trace_file(TRACE_CSV_PATH)
            logger.info("  -> Trace CSV reset for this run.")

        # Pre-carga del dataset para que el singleton esté disponible.
        # joblib con loky usa procesos separados; el singleton NO se comparte
        # automáticamente. El singleton se cargará una vez por proceso worker
        # (lazy). Aquí solo validamos que el PKL es accesible.
        try:
            from data.real_batch_generator import _get_dataset

            ds = _get_dataset()
            logger.info(
                f"  -> Dataset loaded: {ds.metadata['validos_finales']} records, "
                f"{ds.metadata['n_top20_codigos']} top20 codes, "
                f"{ds.metadata['n_otros_codigos']} otros codes."
            )
        except Exception as e:
            logger.error(f"  -> FATAL: Could not load PKL dataset: {e}")
            raise

    def run_elective_mode(self):
        """
        Executes elective simulation mode.
        """
        output_dirs = self.file_manager.setup_elective_directories()

        data_mode = "REAL PKL DATA" if USE_REAL_DATA else "SYNTHETIC DATA"
        logger.info(f"\n{'=' * 70}")
        logger.info(
            f"ELECTIVE SIMULATION MODE [{data_mode}] - PARALLEL with {self.n_jobs} workers"
        )
        logger.info(f"{'=' * 70}")

        start_time = time.time()

        # Parallel execution
        verbose_level = 10 if not VERBOSE_MODE else 0
        worker = ElectiveWorker(self.job_ids, self.algorithms, STD_FACTOR)

        results = Parallel(n_jobs=self.n_jobs, verbose=verbose_level)(
            delayed(worker.run)(sim_i) for sim_i in range(NUM_SIMULATIONS)
        )

        # Aggregate results
        all_results, best_overall = self._aggregate_results(
            results, self.algorithms
        )  # TODO: review/rename if needed

        elapsed = time.time() - start_time
        logger.info(f"\nAll {NUM_SIMULATIONS} elective simulations completed!")

        # Generate reports
        self.report_generator.generate_elective_reports(
            all_results, best_overall, output_dirs, ALL_ROOMS, ALPHA_TEST
        )

        logger.info(
            f"\nProcess completed! (Total time: {elapsed:.2f}s). Check the 'results' folder."
        )

    def _aggregate_results(self, results, algorithms):
        """Aggregates results from parallel workers (elective mode)."""
        all_results = {
            spec["name"]: {
                "makespan": [],
                "solution": [],
                "time": [],
                "best_hist": [],
                "avg_hist": [],
            }
            for spec in algorithms
        }

        best_overall = {
            spec["name"]: {
                "makespan": float("inf"),
                "schedule": None,
                "sim_num": -1,
                "job_label_map": None,
            }
            for spec in algorithms
        }

        for sim_i, sim_results in results:
            for algo_name, result in sim_results.items():
                all_results[algo_name]["makespan"].append(result["makespan"])
                all_results[algo_name]["solution"].append(result["solution"])
                all_results[algo_name]["time"].append(result["time"])
                all_results[algo_name]["best_hist"].append(result["best_hist"])
                all_results[algo_name]["avg_hist"].append(result["avg_hist"])

                if result["makespan"] < best_overall[algo_name]["makespan"]:
                    best_overall[algo_name] = {
                        "makespan": result["makespan"],
                        "schedule": result["solution"],
                        "sim_num": sim_i,
                        "job_label_map": result.get("job_label_map"),
                    }

        return all_results, best_overall

    # -------------------------------------------------------------------------
    # Analysis Mode (opt-in) — Lote 2
    # -------------------------------------------------------------------------

    def run_elective_analysis_mode(self):
        """
        Executes the elective analysis mode: N independent runs of M simulations each.
        Collects per-iteration wall-clock elapsed time and CIE-10 breakdown, then
        batch-inserts into SQLite. No hot exports during the run.

        Called only when ANALYSIS_MODE_ENABLED=True.
        Normal elective mode is completely unchanged.
        """
        from core.analysis_persistence import AnalysisPersistence
        import json as _json

        # Fail-fast: iteration_schedules export requires artifact_save_mode == "all".
        # "best_only" and "sampled" do not capture every iteration, making per-iteration
        # schedule export impossible. Raise early rather than producing incomplete data.
        if ANALYSIS_ARTIFACT_SAVE_MODE != "all":
            raise ValueError(
                f"run_elective_analysis_mode requires artifact_save_mode='all', "
                f"got '{ANALYSIS_ARTIFACT_SAVE_MODE}'. "
                "Set analysis_mode.artifact_save_mode: all in your config."
            )

        # Generate a single timestamp for this analysis session BEFORE building
        # any path. All outputs (DB, CSVs, checkpoint reports) land under
        # results/<timestamp>/... to isolate each analysis execution.
        analysis_timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Build DB path under the timestamped directory so it is isolated.
        # The base_dir comes from FileManager to honour testability overrides.
        _ts_base = os.path.join(self.file_manager._base_dir, analysis_timestamp)
        os.makedirs(_ts_base, exist_ok=True)
        _db_path = os.path.join(_ts_base, "analysis.db")

        persistence = AnalysisPersistence(_db_path)
        persistence.init_db()

        verbose_level = 10 if not VERBOSE_MODE else 0

        if ANALYSIS_SWEEP_ENABLED:
            runs_to_execute = ANALYSIS_SWEEP_VALUES
            actual_num_runs = len(runs_to_execute)
            logger.info(
                f"ANALYSIS MODE — SWEEP ENABLED: {actual_num_runs} runs × {ANALYSIS_SIMS_PER_RUN} sims each"
            )
            logger.info(f"  Sweep procedure counts: {runs_to_execute}")
        else:
            runs_to_execute = [NUM_PROCEDURES] * ANALYSIS_NUM_RUNS
            actual_num_runs = ANALYSIS_NUM_RUNS
            logger.info(
                f"ANALYSIS MODE — {actual_num_runs} runs × {ANALYSIS_SIMS_PER_RUN} sims each"
            )

        logger.info(f"  SQLite: {_db_path}")
        logger.info(f"{'=' * 70}")

        config_snapshot = {
            "num_runs": actual_num_runs,
            "sims_per_run": ANALYSIS_SIMS_PER_RUN,
            "checkpoint_interval_seconds": ANALYSIS_CHECKPOINT_INTERVAL,
            "std_factor": STD_FACTOR,
            "use_real_data": USE_REAL_DATA,
            "artifact_save_mode": ANALYSIS_ARTIFACT_SAVE_MODE,
            "fixed_pool_per_run": ANALYSIS_FIXED_POOL,
        }

        for run_idx, num_procs in enumerate(runs_to_execute):
            # Update job_ids for this specific run in the sweep
            self.job_ids = list(range(1, num_procs + 1))
            config_snapshot["num_procedures"] = num_procs

            run_id = persistence.insert_run(
                num_sims=ANALYSIS_SIMS_PER_RUN,
                num_procs=num_procs,
                config=config_snapshot,
            )

            wall_clock_start = time.time()
            _data_seed = run_idx if ANALYSIS_FIXED_POOL else None
            worker = ElectiveWorker(
                self.job_ids,
                self.algorithms,
                STD_FACTOR,
                wall_clock_start=wall_clock_start,
                data_seed_override=_data_seed,
            )

            raw_results = Parallel(n_jobs=self.n_jobs, verbose=verbose_level)(
                delayed(worker.run)(sim_i) for sim_i in range(ANALYSIS_SIMS_PER_RUN)
            )

            # raw_results: list of (sim_i, sim_results, wall_clock_elapsed_s)
            breakdown_rows_by_sim: dict = {}
            # schedule_store: {(algo_name, sim_i) -> schedule_details}
            # Used by _generate_checkpoint_reports to inject real schedule into best_run
            schedule_store: dict = {}

            for sim_i, sim_results, batch_trace, elapsed_s in raw_results:
                for algo_name, result in sim_results.items():
                    snapshots = result.get("iteration_snapshots", [])
                    last_snap = snapshots[-1] if snapshots else None
                    combined_obj_val = last_snap.best_fitness if last_snap is not None else None

                    sim_id = persistence.insert_simulation(
                        run_id=run_id,
                        sim_index=sim_i,
                        algo_name=algo_name,
                        wall_clock_s=elapsed_s,
                        final_makespan=result.get("makespan", float("inf")),
                        combined_obj=combined_obj_val,
                        algo_time_s=result.get("time", 0.0),
                    )
                    # Persist algorithm iteration snapshots from the callback
                    if snapshots:
                        algo_iter_rows = [
                            {
                                "algo_step": snap.algo_step,
                                "best_fitness": snap.best_fitness,
                                "best_makespan": snap.best_makespan,
                                "iteration_fitness": snap.iteration_fitness,
                                "iteration_makespan": snap.iteration_makespan,
                            }
                            for snap in snapshots
                        ]
                        algo_iter_ids = persistence.save_algorithm_iterations_batch(
                            sim_id, algo_iter_rows
                        )
                        # Persist iteration_schedules: solo si el snapshot tiene solución
                        import json as _json2

                        schedule_items = []
                        for snap, algo_iter_id in zip(snapshots, algo_iter_ids):
                            if snap.best_solution_snapshot is not None:
                                solution_json = _json2.dumps(
                                    snap.best_solution_snapshot,
                                    ensure_ascii=False,
                                )
                                schedule_items.append(
                                    {
                                        "algo_iter_id": algo_iter_id,
                                        "solution_json": solution_json,
                                    }
                                )
                        if schedule_items:
                            persistence.save_iteration_schedules_batch(schedule_items)

                    # Extract breakdown for every valid schedule
                    schedule = result.get("solution")
                    job_label_map = result.get("job_label_map")
                    job_grupo_map = result.get("job_grupo_map")
                    if schedule:
                        rows = self._extract_cie10_breakdown(
                            schedule, job_label_map, job_grupo_map=job_grupo_map, batch_trace=batch_trace
                        )
                        breakdown_rows_by_sim[sim_id] = rows

                        # Patient wait metrics (clinical traceability)
                        from utils.statistics import (
                            calculate_patient_wait_metrics,
                            calculate_schedule_quality_metrics,
                        )
                        wait_result = calculate_patient_wait_metrics(schedule)
                        if wait_result["per_patient"]:
                            persistence.save_patient_wait_batch(
                                sim_id, wait_result["per_patient"]
                            )

                        # Schedule quality metrics (operational KPIs)
                        quality = calculate_schedule_quality_metrics(schedule)
                        persistence.save_quality_metrics(sim_id, quality)

                    # Store schedule for checkpoint report generation (Fix 2)
                    if schedule:
                        schedule_store[(algo_name, sim_i)] = schedule

            for sim_id, bd_rows in breakdown_rows_by_sim.items():
                if bd_rows:
                    persistence.save_breakdowns_batch(sim_id=sim_id, breakdowns=bd_rows)

            elapsed_run = time.time() - wall_clock_start
            logger.info(
                f"  Run {run_idx + 1}/{actual_num_runs} complete — "
                f"{ANALYSIS_SIMS_PER_RUN} sims in {elapsed_run:.2f}s"
            )

            # Post-hoc checkpoint reconstruction for this run
            persistence.reconstruct_checkpoints(run_id, ANALYSIS_CHECKPOINT_INTERVAL)
            logger.info(f"  Run {run_idx + 1}: checkpoints reconstructed.")

            # Export individual room overtimes for this run
            if ANALYSIS_EXPORT_CSV:
                try:
                    from utils import reporting as _reporting
                    _r_suffix = f"_run{run_id}"
                    _r_overtimes_csv = os.path.join(_ts_base, f"room_overtimes{_r_suffix}.csv")
                    
                    run_all_results = {}
                    for algo_spec in self.algorithms:
                        algo_name = algo_spec["name"]
                        run_all_results[algo_name] = {"solution": []}
                        for sim_i in range(ANALYSIS_SIMS_PER_RUN):
                            schedule = schedule_store.get((algo_name, sim_i))
                            run_all_results[algo_name]["solution"].append(schedule)
                    
                    _reporting.export_room_overtimes_csv(run_all_results, _r_overtimes_csv, ALL_ROOMS)
                except Exception as ex_overtime:
                    logger.error(f"Error exporting room overtimes for run {run_id}: {ex_overtime}")

            # Optional: generate full reports per checkpoint if enabled
            if ANALYSIS_FULL_REPORTS_ENABLED:
                self._generate_checkpoint_reports(
                    persistence,
                    run_id,
                    run_idx,
                    schedule_store=schedule_store,
                    timestamp=analysis_timestamp,
                )

        logger.info("\nAnalysis mode complete.")

        # CSV exports (once, after all runs) — separated by run_id
        if ANALYSIS_EXPORT_CSV:
            import sqlite3 as _sqlite3
            from utils import statistics as _stats, reporting as _reporting
            from config.config import EXP_CONFIG

            # Get all run_ids from the DB
            _conn = _sqlite3.connect(_db_path)
            _conn.row_factory = _sqlite3.Row
            try:
                _run_ids = [
                    row["run_id"]
                    for row in _conn.execute(
                        "SELECT DISTINCT run_id FROM runs ORDER BY run_id"
                    ).fetchall()
                ]
            finally:
                _conn.close()

            for _r_id in _run_ids:
                _suffix = f"_run{_r_id}"
                _chk_csv = os.path.join(_ts_base, f"analysis_checkpoints{_suffix}.csv")
                _iter_csv = os.path.join(_ts_base, f"analysis_algorithm_iterations{_suffix}.csv")
                _pw_csv = os.path.join(_ts_base, f"patient_wait_metrics{_suffix}.csv")
                _sim_sum_csv = os.path.join(_ts_base, f"simulation_summary{_suffix}.csv")
                _best_runs_csv = os.path.join(_ts_base, f"best_runs_by_mh{_suffix}.csv")

                persistence.export_checkpoints_csv(_r_id, _chk_csv)
                persistence.export_algorithm_iterations_csv(_r_id, _iter_csv)
                persistence.export_patient_wait_csv(_r_id, _pw_csv)
                persistence.export_simulation_summary_csv(_r_id, _sim_sum_csv)
                persistence.export_best_runs_by_mh_csv(_r_id, _best_runs_csv)

                logger.info(f"  CSV exported (Run {_r_id}): checkpoints, iterations, patient_wait, simulation_summary, best_runs_by_mh")

            # Statistical analysis — Friedman + Wilcoxon paired tests (per run)
            _conn = _sqlite3.connect(_db_path)
            _conn.row_factory = _sqlite3.Row
            try:
                _sim_rows = _conn.execute(
                    "SELECT run_id, algo_name, final_makespan FROM simulations "
                    "ORDER BY run_id, algo_name, sim_index"
                ).fetchall()
            finally:
                _conn.close()

            _results_by_run = {}
            for _row in _sim_rows:
                _r_id = _row["run_id"]
                _algo = _row["algo_name"]
                if _r_id not in _results_by_run:
                    _results_by_run[_r_id] = {}
                if _algo not in _results_by_run[_r_id]:
                    _results_by_run[_r_id][_algo] = {"makespan": []}
                _results_by_run[_r_id][_algo]["makespan"].append(_row["final_makespan"])

            _alpha = EXP_CONFIG.get("alpha_test", ALPHA_TEST)

            for _r_id, _results_for_stats in _results_by_run.items():
                if len(_results_for_stats) >= 2:
                    _pairwise = _stats.perform_u_test_mannwhitney(
                        _results_for_stats, _alpha, verbose=False
                    )
                    _stats_csv = os.path.join(_ts_base, f"statistical_analysis_run{_r_id}.csv")
                    _reporting.export_statistical_analysis(_pairwise, _stats_csv)
                    logger.info(f"  Statistical analysis CSV (Run {_r_id}): {_stats_csv}")

            # Representative PAIRED operational aggregates + waiting significance.
            # These replace cherry-picked best_runs_by_mh tables for analysis reporting.
            _conn = _sqlite3.connect(_db_path)
            _conn.row_factory = _sqlite3.Row
            try:
                _op_rows = _conn.execute(
                    "SELECT run_id, algo_name, final_makespan, "
                    "       patients_with_extra_wait, avg_extra_wait_min "
                    "FROM v_simulation_summary "
                    "ORDER BY run_id, algo_name, sim_index"
                ).fetchall()
            finally:
                _conn.close()

            _op_by_run = {}
            for _row in _op_rows:
                _r_id = _row["run_id"]
                _algo = _row["algo_name"]
                _algo_dict = _op_by_run.setdefault(_r_id, {}).setdefault(
                    _algo,
                    {
                        "final_makespan": [],
                        "patients_with_extra_wait": [],
                        "avg_extra_wait_min": [],
                    },
                )
                _algo_dict["final_makespan"].append(_row["final_makespan"])
                _algo_dict["patients_with_extra_wait"].append(_row["patients_with_extra_wait"])
                _algo_dict["avg_extra_wait_min"].append(_row["avg_extra_wait_min"])

            for _r_id, _op_for_run in _op_by_run.items():
                if len(_op_for_run) < 2:
                    continue
                _op_summary = _stats.compute_operational_summary(_op_for_run)
                _op_csv = os.path.join(_ts_base, f"operational_paired_run{_r_id}.csv")
                _reporting.export_operational_paired_summary(_op_summary, _op_csv)

                _wait_test = _stats.perform_paired_statistical_test(
                    _op_for_run, _alpha, verbose=False, metric="patients_with_extra_wait"
                )
                _wait_csv = os.path.join(_ts_base, f"waiting_significance_run{_r_id}.csv")
                _reporting.export_statistical_analysis(_wait_test["pairwise"], _wait_csv)
                logger.info(
                    f"  Operational paired + waiting significance CSV (Run {_r_id}): "
                    f"{_op_csv}, {_wait_csv}"
                )

        # NOTE: sweep is no longer called from here.
        # Routing (temporal → sweep) is handled by main.py.
        return _db_path

    def run_sweep_mode(self) -> list:
        """Barrido de num_procedures (X) para aproximar 80% de eficiencia promedio de OR.

        Para cada valor X en ANALYSIS_SWEEP_VALUES, ejecuta ANALYSIS_SWEEP_SIMS
        simulaciones con num_procedures=X, calcula la ocupación promedio de salas
        y devuelve una lista de dicts con las claves ``num_procedures`` y
        ``avg_utilization`` (float en [0, 1], o None si no hay schedule disponible).

        Returns
        -------
        list[dict]:
            Un entry por valor X, ej:
            [{"num_procedures": 5, "avg_utilization": 0.73}, ...]
        """
        from utils.statistics import calculate_room_kpis

        sweep_values = ANALYSIS_SWEEP_VALUES
        n_sims = ANALYSIS_SWEEP_SIMS
        verbose_level = 10 if not VERBOSE_MODE else 0

        logger.info(f"\n{'=' * 70}")
        logger.info(f"SWEEP MODE — values={sweep_values}, sims_per_x={n_sims}")
        logger.info(f"{'=' * 70}")

        results = []
        for x in sweep_values:
            job_ids_x = list(range(1, x + 1))
            worker_x = ElectiveWorker(job_ids_x, self.algorithms, STD_FACTOR)

            raw = Parallel(n_jobs=min(self.n_jobs, n_sims), verbose=verbose_level)(
                delayed(worker_x.run)(sim_i) for sim_i in range(n_sims)
            )

            utilizations = []
            util_by_algo = {algo["name"]: [] for algo in self.algorithms}
            for sim_i, sim_results in raw:
                for algo_name, result in sim_results.items():
                    schedule = result.get("solution")
                    if schedule:
                        kpis = calculate_room_kpis(schedule)
                        avg_pct = kpis.get("Average", {}).get("occupancy_rate", None)
                        if avg_pct is not None:
                            util_val = avg_pct / 100.0
                            utilizations.append(util_val)
                            if algo_name in util_by_algo:
                                util_by_algo[algo_name].append(util_val)

            avg_util = (
                float(sum(utilizations) / len(utilizations)) if utilizations else None
            )
            
            # Calculate per-algorithm average utilization
            algo_avg_utils = {}
            for algo_name, utils in util_by_algo.items():
                algo_avg_utils[algo_name] = float(sum(utils) / len(utils)) if utils else None

            results.append({
                "num_procedures": x, 
                "avg_utilization": avg_util,
                "utilization_by_algorithm": algo_avg_utils
            })
            
            logger.info(
                f"  X={x}: avg_utilization_combined={avg_util:.4f}"
                if avg_util is not None
                else f"  X={x}: avg_utilization_combined=None"
            )
            for algo_name, u_val in algo_avg_utils.items():
                if u_val is not None:
                    logger.info(f"    - {algo_name}: avg_utilization={u_val:.4f}")

        logger.info("\nSweep mode complete.")

        # Identify X closest to 80% OR utilization target (combined)
        target = 0.80
        valid = [r for r in results if r["avg_utilization"] is not None]
        if valid:
            closest = min(valid, key=lambda r: abs(r["avg_utilization"] - target))
            logger.info(
                f"  → [Combined] X más cercano al {target * 100:.0f}% de eficiencia: "
                f"num_procedures={closest['num_procedures']} "
                f"(avg_utilization={closest['avg_utilization']:.4f})"
            )
        else:
            closest = None
            logger.warning(
                "  → No utilization data available to determine closest to 80%."
            )

        # Identify X closest to 80% OR utilization target (per algorithm)
        closest_by_algo = {}
        for algo in self.algorithms:
            algo_name = algo["name"]
            algo_valid = [
                r for r in results 
                if r["utilization_by_algorithm"].get(algo_name) is not None
            ]
            if algo_valid:
                closest_algo = min(
                    algo_valid, 
                    key=lambda r: abs(r["utilization_by_algorithm"][algo_name] - target)
                )
                closest_by_algo[algo_name] = {
                    "num_procedures": closest_algo["num_procedures"],
                    "avg_utilization": closest_algo["utilization_by_algorithm"][algo_name]
                }
                logger.info(
                    f"  → [{algo_name}] X más cercano al {target * 100:.0f}% de eficiencia: "
                    f"num_procedures={closest_algo['num_procedures']} "
                    f"(avg_utilization={closest_algo['utilization_by_algorithm'][algo_name]:.4f})"
                )

        return {
            "sweep": results, 
            "closest_to_80pct": closest,
            "closest_to_80pct_by_algorithm": closest_by_algo
        }

    def _generate_checkpoint_reports(
        self,
        persistence,
        run_id: int,
        run_idx: int,
        schedule_store: dict | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Generates CSV reports for each checkpoint of a given run.

        Queries the best simulation per (algo_name, checkpoint_wall_s) from the
        checkpoints table and delegates to ``ReportGenerator.generate_checkpoint_report()``.

        The schedule for the best simulation is looked up from ``schedule_store``
        (a dict {(algo_name, sim_i) -> schedule_details}) when available. This
        allows checkpoint reports to include real schedule data rather than an
        empty placeholder.

        Output directories are created under results/<timestamp>/run<N>/checkpoints/<id>/
        when ``timestamp`` is provided, otherwise falls back to the run-based path.

        Args:
            persistence: AnalysisPersistence instance.
            run_id: Run ID in the DB.
            run_idx: Zero-based run index (for display).
            schedule_store: Optional dict mapping (algo_name, sim_i) → schedule_details.
            timestamp: Optional timestamp string (e.g. "20260416_120000") used for
                       timestamped output directory structure.
        """
        import sqlite3

        convergence_histories = {}
        conn = sqlite3.connect(persistence.db_path)
        conn.row_factory = sqlite3.Row
        try:
            checkpoints = conn.execute(
                "SELECT algo_name, checkpoint_wall_s, best_makespan, best_sim_index, simulations_completed "
                "FROM checkpoints WHERE run_id = ? "
                "ORDER BY algo_name, checkpoint_wall_s",
                (run_id,),
            ).fetchall()

            # Retrieve convergence histories for all simulations in this run
            iter_rows = conn.execute(
                "SELECT s.algo_name, s.sim_index, ai.best_fitness, ai.iteration_fitness, ai.best_makespan, ai.iteration_makespan "
                "FROM algorithm_iterations ai "
                "JOIN simulations s ON ai.sim_id = s.sim_id "
                "WHERE s.run_id = ? "
                "ORDER BY s.algo_name, s.sim_index, ai.algo_step",
                (run_id,),
            ).fetchall()
            for r in iter_rows:
                key = (r["algo_name"], r["sim_index"])
                if key not in convergence_histories:
                    convergence_histories[key] = {
                        "best_hist": [], "avg_hist": [],
                        "best_makespan_hist": [], "avg_makespan_hist": []
                    }
                convergence_histories[key]["best_hist"].append(r["best_fitness"])
                convergence_histories[key]["avg_hist"].append(r["iteration_fitness"])
                convergence_histories[key]["best_makespan_hist"].append(r["best_makespan"])
                convergence_histories[key]["avg_makespan_hist"].append(r["iteration_makespan"])
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not load algorithm iterations from DB: {e}")
            try:
                checkpoints = conn.execute(
                    "SELECT algo_name, checkpoint_wall_s, best_makespan, best_sim_index, simulations_completed "
                    "FROM checkpoints WHERE run_id = ? "
                    "ORDER BY algo_name, checkpoint_wall_s",
                    (run_id,),
                ).fetchall()
            except Exception:
                checkpoints = []
        finally:
            conn.close()

        if not checkpoints:
            return

        logger.info(
            f"  Run {run_idx + 1}: generating checkpoint reports for "
            f"{len(checkpoints)} checkpoint(s)…"
        )

        for chk in checkpoints:
            algo_name = chk["algo_name"]
            chk_wall = chk["checkpoint_wall_s"]
            best_makespan = chk["best_makespan"]
            best_sim_index = chk["best_sim_index"]

            # Look up real schedule from in-memory store when available
            schedule = []
            if schedule_store is not None:
                schedule = schedule_store.get((algo_name, best_sim_index), [])

            hist_data = convergence_histories.get((algo_name, best_sim_index), {})
            best_hist = hist_data.get("best_hist", [])
            avg_hist = hist_data.get("avg_hist", [])
            best_makespan_hist = hist_data.get("best_makespan_hist", [])
            avg_makespan_hist = hist_data.get("avg_makespan_hist", [])

            best_run = {
                "algo_name": algo_name,
                "makespan": best_makespan,
                "schedule": schedule,
                "sim_num": best_sim_index,
                "job_label_map": None,
                "best_hist": best_hist,
                "avg_hist": avg_hist,
                "best_makespan_hist": best_makespan_hist,
                "avg_makespan_hist": avg_makespan_hist,
                "simulations_completed": chk["simulations_completed"],
            }

            # Use timestamped output_dirs when timestamp is provided (Fix 3 / Lote 7)
            # Spec: results/<timestamp>/checkpoints/run<N>/<checkpoint_id>/
            checkpoint_id = f"{algo_name}_wall{int(chk_wall)}"
            run_label = f"run{run_idx + 1}"
            _chk_subdirs = FileManager.CHECKPOINT_PLOT_SUBDIRS
            if timestamp is not None:
                output_dirs = self.file_manager.setup_analysis_directories(
                    timestamp, f"checkpoints/{run_label}/{checkpoint_id}",
                    plot_subdirs=_chk_subdirs,
                )
            else:
                output_dirs = self.file_manager.setup_analysis_directories(
                    "unversioned", f"checkpoints/{run_label}/{checkpoint_id}",
                    plot_subdirs=_chk_subdirs,
                )

            self.report_generator.generate_checkpoint_report(
                best_run, output_dirs, ALL_ROOMS
            )

    def _extract_cie10_breakdown(
        self, schedule_details, job_label_map, job_grupo_map=None, batch_trace=None
    ):
        """
        Extracts per-job CIE-10 breakdown from schedule_details and batch_trace.

        schedule_details is a flat list of operation-level dicts with keys:
          Job, Operation, Start, Finish, ProcessingEnd, SetupUsed, TransitionUsed, CleanupUsed

        Each job has 2 operations:
          op1 (anesthesia): SetupUsed = setup_min
          op2 (surgery):    TransitionUsed = transition_min (op1→op2 transition),
                            CleanupUsed = cleanup_min,
                            (Finish - Start) - TransitionUsed - CleanupUsed = proc_time_min

        Args:
            schedule_details (list[dict]): Flat list from the static scheduler.
            job_label_map (dict | None): {job_id: codigo_cie10} from batch_trace.
            job_grupo_map (dict | None): {job_id: grupo} from batch_trace (e.g. 'top20'/'otros').
            batch_trace (list[dict] | None): Full raw data trace for extra columns.

        Returns:
            list[dict]: One row per job with breakdown fields, including 'grupo' and trace info.
        """
        if not schedule_details:
            return []

        # Convert batch_trace to dict mapped by job_id for quick lookup
        trace_map = {}
        if batch_trace:
            for row in batch_trace:
                trace_map[row["job_id"]] = row

        # Group operations by job_id
        from collections import defaultdict

        ops_by_job = defaultdict(dict)
        for entry in schedule_details:
            job_id = entry.get("Job")
            op_num = entry.get("Operation")
            if job_id is not None and op_num is not None:
                ops_by_job[job_id][op_num] = entry

        results = []
        for job_id, ops in ops_by_job.items():
            op1 = ops.get(1, {})
            op2 = ops.get(2, {})

            setup_min = op1.get("SetupUsed") or 0.0
            # In the PKL model, the actual patient transition (tiempo_transicion) is registered
            # on op1 (Anesthesia), while op2 (Surgery) has TransitionUsed=0.0.
            # In legacy test cases or synthetic mode, it may be registered on op2.
            # We read from both dynamically to be 100% robust and compatible.
            transition_min = (op1.get("TransitionUsed") or 0.0) or (op2.get("TransitionUsed") or 0.0)
            cleanup_min = op2.get("CleanupUsed") or 0.0
            
            # For proc_time_min, only subtract TransitionUsed if it was actually tracked on op2
            # (since op2_duration is purely surgery + cleanup + op2_transition).
            op2_transition = op2.get("TransitionUsed") or 0.0
            op2_duration = (op2.get("Finish") or 0.0) - (op2.get("Start") or 0.0)
            proc_time_min = op2_duration - op2_transition - cleanup_min

            # Find matching trace record if any
            trace_record = trace_map.get(job_id, {})

            results.append(
                {
                    "job_id": job_id,
                    "codigo_cie10": job_label_map.get(job_id) if job_label_map is not None else None,
                    "grupo": job_grupo_map.get(job_id) if job_grupo_map is not None else None,
                    "setup_min": setup_min,
                    "proc_time_min": proc_time_min,
                    "transition_min": transition_min,
                    "cleanup_min": cleanup_min,
                    "source_record_id": trace_record.get("source_record_id"),
                    "estrategia_muestreo": trace_record.get("estrategia_muestreo"),
                    "setup_op1": trace_record.get("setup_op1"),
                    "setup_op2": trace_record.get("setup_op2"),
                    "cleanup_op1": trace_record.get("cleanup_op1"),
                    "cleanup_op2": trace_record.get("cleanup_op2"),
                    "tiempos_dinamicos_en_simulacion": trace_record.get("tiempos_dinamicos_en_simulacion"),
                }
            )

        return results
