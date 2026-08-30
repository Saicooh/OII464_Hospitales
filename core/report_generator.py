"""
Centralizes all report and plot generation logic.
"""

import os
from utils import plotting, reporting, statistics
from config.config import get_algorithms, VERBOSE_MODE
from utils.logger import logger


class ReportGenerator:
    """
    Generates all reports and plots for simulation results.
    """

    def __init__(self):
        self.algorithms = get_algorithms()

    def generate_elective_reports(
        self, all_results, best_overall, output_dirs, all_rooms, alpha_test
    ):
        """Generates reports for elective simulation mode."""
        logger.info(f"\n{'=' * 70}")
        logger.info("GENERATING ELECTIVE SIMULATION REPORTS")
        logger.info(f"{'=' * 70}")

        # CSV summary
        reporting.export_montecarlo_summary(
            all_results, os.path.join(output_dirs["csv"], "summary_results.csv")
        )
        reporting.export_room_overtimes_csv(
            all_results, os.path.join(output_dirs["csv"], "elective_room_overtimes.csv"), all_rooms
        )

        # Statistical analysis
        pairwise_stats = statistics.perform_u_test_mannwhitney(
            all_results, alpha_test, verbose=VERBOSE_MODE
        )
        reporting.export_statistical_analysis(
            pairwise_stats, os.path.join(output_dirs["csv"], "statistical_analysis.csv")
        )

        # Summary plots
        plotting.generate_summary_plots(all_results, output_dirs["plots"])

        # Detailed reports per algorithm
        for algo_name, best_run in best_overall.items():
            if best_run["schedule"]:
                self._generate_algorithm_reports(
                    algo_name,
                    best_run,
                    all_results[algo_name],
                    output_dirs,
                    all_rooms,
                )

        logger.info(f"\n{'=' * 70}")
        logger.info("ELECTIVE SIMULATION REPORTS COMPLETE")
        logger.info(f"{'=' * 70}")

    def _generate_algorithm_reports(
        self, algo_name, best_run, algo_results, output_dirs, all_rooms
    ):
        """Generate detailed reports for one algorithm's best elective run."""
        logger.info(f"\n> Processing elective results for: {algo_name}")
        logger.info(
            f"  -> Best run found in Sim #{best_run['sim_num'] + 1} (Makespan: {best_run['makespan']:.2f})"
        )

        # Verify whether we have a detailed schedule
        schedule_data = best_run.get("schedule", [])

        if isinstance(schedule_data, list) and schedule_data:
            # CSV exports (only if we have a detailed schedule)
            reporting.export_full_schedule_to_csv(
                schedule_data,
                os.path.join(output_dirs["csv"], f"elective_best_schedule_{algo_name.lower()}.csv"),
            )

            reporting.export_sequencing_strategy_to_csv(
                schedule_data,
                os.path.join(output_dirs["csv"], f"elective_best_strategy_{algo_name.lower()}.csv"),
            )

            reporting.export_routing_explanation_csv(
                schedule_data,
                os.path.join(output_dirs["csv"], f"elective_routing_explanation_{algo_name.lower()}.csv"),
            )

            # Gantt chart
            sim_num = best_run.get("sim_num", 0) + 1
            job_label_map = best_run.get("job_label_map")

            plotting.plot_gantt_chart(
                schedule_data,
                all_rooms,
                f"{algo_name} - Best Elective Schedule",
                algo_name,
                output_dirs["plots"],
                sim_num=sim_num,
                job_label_map=job_label_map,
            )

            # --- New Detailed Plots (Workload, KPI, CIE10) ---
            try:
                # 1. Personnel workload stats
                personnel_stats = statistics.analyze_personnel_workload(schedule_data)
                if personnel_stats:
                    plotting.plot_personnel_workload(
                        personnel_stats,
                        output_dirs["plots"],
                        algo_name,
                        sim_num=sim_num,
                    )
                    plotting.plot_personnel_gantt(
                        personnel_stats, schedule_data, output_dirs["plots"], algo_name
                    )

                # 2. Room KPIs
                kpis = statistics.calculate_room_kpis(schedule_data)
                if kpis:
                    plotting.plot_kpi_histogram(kpis, output_dirs["plots"], algo_name)

                # 3. Job / CIE10 usage
                plotting.plot_cie10_histogram(
                    schedule_data, output_dirs["plots"], algo_name
                )

                # 4. Personnel Usage %
                plotting.plot_personnel_usage_histogram(
                    schedule_data, output_dirs["plots"], algo_name
                )

            except Exception as e:
                logger.error(f"  -> [Error] Generating new detailed plots failed: {e}")

        else:
            logger.warning(
                f"  -> No detailed schedule available for {algo_name} (only makespan and convergence)"
            )

        # Convergence plot (always available)
        if (
            algo_results["best_hist"]
            and len(algo_results["best_hist"]) > best_run["sim_num"]
        ):
            if algo_results["best_hist"][best_run["sim_num"]]:
                plotting.plot_convergence_history(
                    algo_results["best_hist"][best_run["sim_num"]],
                    algo_results["avg_hist"][best_run["sim_num"]],
                    len(algo_results["best_hist"][best_run["sim_num"]]),
                    algo_name,
                    best_run["sim_num"] + 1,
                    output_dirs["plots"],
                    metric_name="Fitness",
                )

        if (
            "best_makespan_hist" in algo_results
            and algo_results["best_makespan_hist"]
            and len(algo_results["best_makespan_hist"]) > best_run["sim_num"]
        ):
            if algo_results["best_makespan_hist"][best_run["sim_num"]]:
                plotting.plot_convergence_history(
                    algo_results["best_makespan_hist"][best_run["sim_num"]],
                    algo_results["avg_makespan_hist"][best_run["sim_num"]],
                    len(algo_results["best_makespan_hist"][best_run["sim_num"]]),
                    algo_name,
                    best_run["sim_num"] + 1,
                    output_dirs["plots"],
                    metric_name="Makespan",
                )

    def generate_checkpoint_report(
        self, best_run: dict, output_dirs: dict, all_rooms: list
    ) -> None:
        """Generates a minimal report for the best accumulated snapshot at a checkpoint.

        This is called per-checkpoint during analysis mode when
        ``ANALYSIS_FULL_REPORTS_ENABLED=True``.  It writes the schedule CSV
        (including ``TransitionUsed``) and the sequencing strategy CSV for the
        best run seen so far.

        Plots are intentionally skipped here to keep checkpoint generation fast.
        A full plot pass can be added later without breaking the contract.

        Parameters
        ----------
        best_run:
            Dict with keys ``algo_name``, ``makespan``, ``schedule``, ``sim_num``,
            and optionally ``job_label_map``.
        output_dirs:
            Dict with ``"csv"`` and ``"plots"`` path keys (from ``FileManager``).
        all_rooms:
            List of room names (used by plotting utilities — reserved for future use).
        """
        algo_name = best_run.get("algo_name", "unknown")
        schedule_data = best_run.get("schedule") or []

        if not schedule_data:
            logger.info(
                f"  -> [Checkpoint] No schedule for {algo_name}; skipping CSV export."
            )
            return

        csv_dir = output_dirs.get("csv", "results/csv")
        os.makedirs(csv_dir, exist_ok=True)

        reporting.export_full_schedule_to_csv(
            schedule_data,
            os.path.join(csv_dir, f"checkpoint_schedule_{algo_name.lower()}.csv"),
        )
        reporting.export_sequencing_strategy_to_csv(
            schedule_data,
            os.path.join(csv_dir, f"checkpoint_strategy_{algo_name.lower()}.csv"),
        )
        reporting.export_routing_explanation_csv(
            schedule_data,
            os.path.join(csv_dir, f"checkpoint_routing_explanation_{algo_name.lower()}.csv"),
        )
        logger.info(
            f"  -> [Checkpoint] {algo_name} CSV exported to {csv_dir} "
            f"(simulations completed: {best_run.get('simulations_completed', 0)})"
        )

        # Gantt chart — mirrors normal-mode output for this checkpoint
        plots_dir = output_dirs.get("plots", "")
        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            sim_num = best_run.get("sim_num", 0) + 1
            job_label_map = best_run.get("job_label_map")
            try:
                plotting.plot_gantt_chart(
                    schedule_data,
                    all_rooms,
                    f"{algo_name} - Checkpoint Best Schedule",
                    algo_name,
                    plots_dir,
                    sim_num=sim_num,
                    job_label_map=job_label_map,
                )
                logger.info(
                    f"  -> [Checkpoint] {algo_name} Gantt plot saved to {plots_dir}"
                )

                # Personnel workload stats
                personnel_stats = statistics.analyze_personnel_workload(schedule_data)
                if personnel_stats:
                    plotting.plot_personnel_workload(
                        personnel_stats,
                        plots_dir,
                        algo_name,
                        sim_num=sim_num,
                    )
                    plotting.plot_personnel_gantt(
                        personnel_stats, schedule_data, plots_dir, algo_name
                    )

                # Room KPIs
                kpis = statistics.calculate_room_kpis(schedule_data)
                if kpis:
                    plotting.plot_kpi_histogram(kpis, plots_dir, algo_name)

                # Job / CIE10 usage
                plotting.plot_cie10_histogram(
                    schedule_data, plots_dir, algo_name
                )

                # Convergence plot (Fitness)
                best_hist = best_run.get("best_hist")
                avg_hist = best_run.get("avg_hist")
                if best_hist:
                    plotting.plot_convergence_history(
                        best_hist,
                        avg_hist or [float("inf")] * len(best_hist),
                        len(best_hist),
                        algo_name,
                        sim_num,
                        plots_dir,
                        metric_name="Fitness",
                    )

                # Convergence plot (Makespan)
                best_makespan_hist = best_run.get("best_makespan_hist")
                avg_makespan_hist = best_run.get("avg_makespan_hist")
                if best_makespan_hist:
                    plotting.plot_convergence_history(
                        best_makespan_hist,
                        avg_makespan_hist or [float("inf")] * len(best_makespan_hist),
                        len(best_makespan_hist),
                        algo_name,
                        sim_num,
                        plots_dir,
                        metric_name="Makespan",
                    )

                logger.info(
                    f"  -> [Checkpoint] {algo_name} detailed workload, KPI, CIE10, and convergence plots saved to {plots_dir}"
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    f"  -> [Checkpoint] Plots generation failed for {algo_name}: {exc}"
                )

    def generate_analysis_reports(
        self, sqlite_path: str, output_dir: str = "results/analysis"
    ) -> None:
        """
        Genera reportes derivados del modo analítico a partir del SQLite de análisis.

        Exporta CSVs para cada run presente en la base. Para DBs en schema v6
        también exporta patient_wait_metrics.csv, schedule_quality_metrics.csv
        y simulation_summary.csv usando la vista v_simulation_summary.
        Crea ``output_dir`` si no existe.

        Args:
            sqlite_path: Ruta al archivo SQLite generado por ``run_elective_analysis_mode()``.
            output_dir:  Directorio de salida para los archivos CSV exportados.
                         Por defecto ``'results/analysis'``.
        """
        import sqlite3
        import csv

        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(sqlite_path):
            logger.warning(
                f"generate_analysis_reports: SQLite not found at '{sqlite_path}'. Nothing exported."
            )
            return

        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row

        try:
            # Export checkpoints CSV (all runs combined) — schema v6-ready
            chk_path = os.path.join(output_dir, "analysis_checkpoints.csv")
            cursor = conn.execute("PRAGMA table_info(checkpoints)")
            columns = [row["name"] for row in cursor.fetchall()]
            has_sim_completed = "simulations_completed" in columns

            if has_sim_completed:
                chk_rows = conn.execute(
                    "SELECT run_id, algo_name, checkpoint_wall_s, best_makespan, best_sim_index, simulations_completed "
                    "FROM checkpoints ORDER BY run_id, algo_name, checkpoint_wall_s"
                ).fetchall()
            else:
                chk_rows = conn.execute(
                    "SELECT run_id, algo_name, checkpoint_wall_s, best_makespan, best_sim_index "
                    "FROM checkpoints ORDER BY run_id, algo_name, checkpoint_wall_s"
                ).fetchall()

            if chk_rows:
                with open(chk_path, "w", newline="", encoding="utf-8") as f:
                    fieldnames = [
                        "run_id",
                        "algo_name",
                        "checkpoint_wall_s",
                        "best_makespan",
                        "best_sim_index",
                    ]
                    if has_sim_completed:
                        fieldnames.append("simulations_completed")

                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in chk_rows:
                        writer.writerow(dict(row))
                logger.info(f"  Analysis checkpoints CSV: {chk_path}")



            # Export algorithm_iterations CSV — solo si la tabla existe (schema v4+)
            ai_path = os.path.join(output_dir, "analysis_algorithm_iterations.csv")
            _has_ai_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='algorithm_iterations'"
            ).fetchone() is not None
            ai_rows = []
            if _has_ai_table:
                ai_rows = conn.execute(
                    "SELECT ai.algo_iter_id, s.algo_name, s.sim_index, "
                    "ai.algo_step, ai.best_fitness, ai.best_makespan, "
                    "ai.iteration_fitness, ai.iteration_makespan "
                    "FROM algorithm_iterations ai "
                    "JOIN simulations s ON ai.sim_id = s.sim_id "
                    "ORDER BY s.algo_name, s.sim_index, ai.algo_step"
                ).fetchall()
            if ai_rows:
                with open(ai_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "algo_iter_id", "algo_name", "sim_index",
                            "algo_step", "best_fitness", "best_makespan",
                            "iteration_fitness", "iteration_makespan",
                        ],
                    )
                    writer.writeheader()
                    for row in ai_rows:
                        writer.writerow(dict(row))
                logger.info(f"  Analysis algorithm iterations CSV: {ai_path}")

            # Export patient_wait_metrics CSV — schema v6
            _has_pw_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='patient_wait_metrics'"
            ).fetchone() is not None
            if _has_pw_table:
                pw_rows = conn.execute(
                    "SELECT s.sim_index, s.algo_name, pw.job_id, "
                    "       pw.op1_room, pw.op2_room, pw.op1_finish, pw.op2_start, "
                    "       pw.transition_used, pw.extra_wait_min "
                    "FROM patient_wait_metrics pw "
                    "JOIN simulations s ON pw.sim_id = s.sim_id "
                    "ORDER BY s.sim_index, s.algo_name, pw.job_id"
                ).fetchall()
                if pw_rows:
                    pw_path = os.path.join(output_dir, "patient_wait_metrics.csv")
                    with open(pw_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=[
                                "sim_index", "algo_name", "job_id",
                                "op1_room", "op2_room", "op1_finish", "op2_start",
                                "transition_used", "extra_wait_min",
                            ],
                        )
                        writer.writeheader()
                        for row in pw_rows:
                            writer.writerow(dict(row))
                    logger.info(f"  Patient wait metrics CSV: {pw_path}")

            # Export schedule_quality_metrics CSV — schema v6
            _has_sq_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schedule_quality_metrics'"
            ).fetchone() is not None
            if _has_sq_table:
                sq_rows = conn.execute(
                    "SELECT s.sim_index, s.algo_name, "
                    "       q.rooms_used, q.total_overtime_min, q.max_room_overtime_min, "
                    "       q.personnel_count, q.workload_std_min, "
                    "       q.workload_max_min, q.workload_min_min, "
                    "       q.idle_gap_count, q.idle_gap_total_min, q.avg_idle_gap_min, "
                    "       q.value_added_ratio "
                    "FROM schedule_quality_metrics q "
                    "JOIN simulations s ON q.sim_id = s.sim_id "
                    "ORDER BY s.sim_index, s.algo_name"
                ).fetchall()
                if sq_rows:
                    sq_path = os.path.join(output_dir, "schedule_quality_metrics.csv")
                    with open(sq_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=[
                                "sim_index", "algo_name",
                                "rooms_used", "total_overtime_min", "max_room_overtime_min",
                                "personnel_count", "workload_std_min",
                                "workload_max_min", "workload_min_min",
                                "idle_gap_count", "idle_gap_total_min", "avg_idle_gap_min",
                                "value_added_ratio",
                            ],
                        )
                        writer.writeheader()
                        for row in sq_rows:
                            writer.writerow(dict(row))
                    logger.info(f"  Schedule quality metrics CSV: {sq_path}")

            # Export simulation_summary CSV from v_simulation_summary view — schema v6
            _has_summary_view = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='view' AND name='v_simulation_summary'"
            ).fetchone() is not None
            if _has_summary_view:
                ss_rows = conn.execute(
                    "SELECT run_id, num_procedures, sim_index, algo_name, final_makespan, "
                    "       algo_time_s, rooms_used, total_overtime_min, max_room_overtime_min, "
                    "       personnel_count, workload_std_min, workload_max_min, workload_min_min, "
                    "       idle_gap_count, idle_gap_total_min, avg_idle_gap_min, value_added_ratio, "
                    "       total_patients, patients_with_extra_wait, avg_extra_wait_min, max_extra_wait_min "
                    "FROM v_simulation_summary ORDER BY run_id, sim_index, algo_name"
                ).fetchall()
                if ss_rows:
                    ss_path = os.path.join(output_dir, "simulation_summary.csv")
                    col_names = [
                        "run_id", "num_procedures", "sim_index", "algo_name", "final_makespan",
                        "algo_time_s", "rooms_used", "total_overtime_min", "max_room_overtime_min",
                        "personnel_count", "workload_std_min", "workload_max_min", "workload_min_min",
                        "idle_gap_count", "idle_gap_total_min", "avg_idle_gap_min", "value_added_ratio",
                        "total_patients", "patients_with_extra_wait", "avg_extra_wait_min", "max_extra_wait_min",
                    ]
                    with open(ss_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(col_names)
                        writer.writerows([tuple(row) for row in ss_rows])
                    logger.info(f"  Simulation summary CSV: {ss_path}")

                # Export best_runs_by_mh CSV — filters the best simulation per (run_id, algo_name) by combined_obj
                best_runs_rows = conn.execute(
                    "WITH RankedSimulations AS ("
                    "    SELECT *, "
                    "           ROW_NUMBER() OVER ("
                    "               PARTITION BY run_id, algo_name "
                    "               ORDER BY COALESCE(combined_obj, final_makespan) ASC, final_makespan ASC"
                    "           ) as rank "
                    "    FROM v_simulation_summary"
                    ") "
                    "SELECT run_id, num_procedures, sim_index, algo_name, final_makespan, combined_obj, "
                    "       algo_time_s, rooms_used, total_overtime_min, max_room_overtime_min, "
                    "       personnel_count, workload_std_min, workload_max_min, workload_min_min, "
                    "       idle_gap_count, idle_gap_total_min, avg_idle_gap_min, value_added_ratio, "
                    "       total_patients, patients_with_extra_wait, avg_extra_wait_min, max_extra_wait_min "
                    "FROM RankedSimulations "
                    "WHERE rank = 1 "
                    "ORDER BY run_id, algo_name"
                ).fetchall()
                if best_runs_rows:
                    best_runs_path = os.path.join(output_dir, "best_runs_by_mh.csv")
                    col_names_best = [
                        "run_id", "num_procedures", "sim_index", "algo_name", "final_makespan", "combined_obj",
                        "algo_time_s", "rooms_used", "total_overtime_min", "max_room_overtime_min",
                        "personnel_count", "workload_std_min", "workload_max_min", "workload_min_min",
                        "idle_gap_count", "idle_gap_total_min", "avg_idle_gap_min", "value_added_ratio",
                        "total_patients", "patients_with_extra_wait", "avg_extra_wait_min", "max_extra_wait_min",
                    ]
                    with open(best_runs_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(col_names_best)
                        writer.writerows([tuple(row) for row in best_runs_rows])
                    logger.info(f"  Best runs by MH CSV: {best_runs_path}")

                # Representative PAIRED operational aggregates + waiting significance,
                # per run_id. These replace cherry-picked best_runs_by_mh tables for
                # analysis reporting (patient waiting + representative best-solution rows).
                op_rows = conn.execute(
                    "SELECT run_id, algo_name, final_makespan, "
                    "       patients_with_extra_wait, avg_extra_wait_min "
                    "FROM v_simulation_summary "
                    "ORDER BY run_id, algo_name, sim_index"
                ).fetchall()
                op_by_run = {}
                for row in op_rows:
                    r_id = row["run_id"]
                    algo = row["algo_name"]
                    algo_dict = op_by_run.setdefault(r_id, {}).setdefault(
                        algo,
                        {
                            "final_makespan": [],
                            "patients_with_extra_wait": [],
                            "avg_extra_wait_min": [],
                        },
                    )
                    algo_dict["final_makespan"].append(row["final_makespan"])
                    algo_dict["patients_with_extra_wait"].append(row["patients_with_extra_wait"])
                    algo_dict["avg_extra_wait_min"].append(row["avg_extra_wait_min"])

                from config.config import EXP_CONFIG as _EXP_CONFIG
                alpha_op = _EXP_CONFIG.get("alpha_test", 0.05)
                for r_id, op_for_run in op_by_run.items():
                    if len(op_for_run) < 2:
                        continue
                    op_summary = statistics.compute_operational_summary(op_for_run)
                    op_path = os.path.join(output_dir, f"operational_paired_run{r_id}.csv")
                    reporting.export_operational_paired_summary(op_summary, op_path)

                    wait_test = statistics.perform_paired_statistical_test(
                        op_for_run, alpha_op, verbose=False,
                        metric="patients_with_extra_wait",
                    )
                    wait_path = os.path.join(output_dir, f"waiting_significance_run{r_id}.csv")
                    reporting.export_statistical_analysis(wait_test["pairwise"], wait_path)
                    logger.info(
                        f"  Operational paired + waiting significance CSV (Run {r_id}): "
                        f"{op_path}, {wait_path}"
                    )

            # Statistical analysis — Mann-Whitney U pairwise comparison
            _sim_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(simulations)").fetchall()
            }
            sim_rows = []
            if "final_makespan" in _sim_cols:
                sim_rows = conn.execute(
                    "SELECT algo_name, final_makespan FROM simulations "
                    "ORDER BY algo_name, sim_index"
                ).fetchall()

            all_results_for_stats = {}
            for row in sim_rows:
                algo = row["algo_name"]
                if algo not in all_results_for_stats:
                    all_results_for_stats[algo] = {"makespan": []}
                all_results_for_stats[algo]["makespan"].append(row["final_makespan"])

            if len(all_results_for_stats) >= 2:
                from config.config import EXP_CONFIG
                alpha_test = EXP_CONFIG.get("alpha_test", 0.05)
                pairwise_stats = statistics.perform_u_test_mannwhitney(
                    all_results_for_stats, alpha_test, verbose=False
                )
                stats_path = os.path.join(output_dir, "statistical_analysis.csv")
                reporting.export_statistical_analysis(pairwise_stats, stats_path)
                logger.info(f"  Statistical analysis CSV: {stats_path}")

            logger.info(
                f"generate_analysis_reports complete — output_dir='{output_dir}'"
            )
        finally:
            conn.close()

        # Si la DB tiene iteration_schedules, exportar CSVs por iteración
        self._export_iteration_csvs_if_available(sqlite_path, output_dir)

    def _export_iteration_csvs_if_available(self, sqlite_path: str, output_dir: str) -> None:
        """Exporta los 3 CSVs de iteración si iteration_schedules tiene filas."""
        import sqlite3

        if not os.path.exists(sqlite_path):
            return

        try:
            conn = sqlite3.connect(sqlite_path)
            count = conn.execute(
                "SELECT COUNT(*) FROM iteration_schedules"
            ).fetchone()[0]
            conn.close()
        except Exception:
            return

        if count == 0:
            return

        from offline.analysis_exporter import AnalysisExporter

        exporter = AnalysisExporter()
        exporter.export_iteration_csvs(sqlite_path, output_dir)
        logger.info(f"  Iteration CSVs exported to: {output_dir}")
