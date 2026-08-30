# /utils/reporting.py
"""
Module for reporting functions: printing summaries to the console and
exporting schedule results to CSV files.
"""

import csv
import numpy as np
import os

# Import necessary constants from the configuration file
from config.config import ALL_ROOMS

from utils.logger import logger


# ==== Internal Generic Helpers ====

def _format_two_decimals(value):
    """Returns a value formatted to two decimal places if numeric (and not a boolean), otherwise returns it as is."""
    if isinstance(value, bool):
        return value
    return f"{value:.2f}" if isinstance(value, (int, float, np.floating)) else value

def _build_room_schedules(schedule_details):
    """Groups operations by room to facilitate reporting."""
    room_schedules = {room: [] for room in ALL_ROOMS}
    for t in schedule_details:
        room = t.get("Resource")
        if room:
            room_schedules.setdefault(room, []).append(
                (t.get("Start", -1), t.get("Job"), t.get("Operation"))
            )
    return room_schedules

def _build_job_timetables(schedule_details):
    """Structures start times by surgery and operation."""
    job_timetables = {}
    for t in schedule_details:
        job = t.get("Job")
        if job is not None:
            job_timetables.setdefault(job, {})[t.get("Operation")] = t.get("Start", -1)
    return job_timetables

def _safe_open_csv(filename):
    """Context manager to safely open CSV files."""
    return open(filename, "w", newline="", encoding="utf-8")


# ==== Console Printing Functions ====

def print_schedule_summary(schedule_details, title="Schedule"):
    """Prints a detailed summary of the schedule, ordered by time."""
    if not schedule_details:
        return
    print(f"\n--- Schedule Summary: {title} ---")
    header = f"{'Job':<12} | {'Operation':^10} | {'Room':<12} | {'Personnel':<12} | {'Start':>10} | {'Finish':>10} | {'Duration':>10}"
    print(header)
    print("-" * len(header))

    for task in sorted(schedule_details, key=lambda x: x.get("Start", float("inf"))):
        job_label = str(task.get("Job"))
        start, finish = task.get("Start", 0.0), task.get("Finish", 0.0)
        duration = finish - start
        print(
            f"{job_label:<12} | {int(task.get('Operation', 0)):^10d} | {str(task.get('Resource', '')):<12} | "
            f"{str(task.get('Personnel', 'N/A')):<12} | {start:>10.2f} | {finish:>10.2f} | {duration:>10.2f}"
        )
    print("-" * len(header))

def print_sequencing_strategy(schedule_details):
    """Displays the sequence of surgeries assigned to each room."""
    if not schedule_details:
        return
    print("\n--- Sequencing Strategy by Room ---")
    room_schedules = _build_room_schedules(schedule_details)
    print(f"{'Room':<12} | {'Operation Sequence (Surgery(Op))'}")
    print("-" * 80)
    for room_name in sorted(room_schedules.keys()):
        schedule = sorted(room_schedules[room_name], key=lambda x: x[0])
        seq = [f"{j}(Op{o})" for _, j, o in schedule]
        print(f"{room_name:<12} | {' -> '.join(seq) if seq else 'No assignments'}")
    print("-" * 80)


# ==== High-Level Report Generation Functions ====

def generate_summary_reports(all_results, pairwise_stats, output_dir):
    """Generates and saves all summary CSV files."""
    logger.info("  -> Generating summary CSV files...")

    summary_path = export_montecarlo_summary(
        all_results, os.path.join(output_dir, "summary_results.csv")
    )
    if summary_path:
        logger.info(f"    - Monte Carlo summary saved to: {summary_path}")

    stats_path = export_statistical_analysis(
        pairwise_stats, os.path.join(output_dir, "statistical_analysis.csv")
    )
    if stats_path:
        logger.info(f"    - Statistical analysis saved to: {stats_path}")


def generate_detailed_reports_for_algo(details, name, output_dir):
    """Generates and saves all detailed CSV files for a single algorithm's best run."""
    schedule_csv = export_full_schedule_to_csv(
        details, os.path.join(output_dir, f"best_schedule_{name.lower()}.csv")
    )
    strategy_csv = export_sequencing_strategy_to_csv(
        details, os.path.join(output_dir, f"best_strategy_{name.lower()}.csv")
    )

    if schedule_csv:
        logger.info(f"    - Full schedule: {schedule_csv}")
    if strategy_csv:
        logger.info(f"    - Sequencing strategy: {strategy_csv}")


# ==== Low-Level CSV Export Functions ====


def export_full_schedule_to_csv(schedule_details, filename):
    """Exports the full detailed schedule to a CSV file, returning the filename on success.

    Always emits a ``TransitionUsed`` column even when the schedule dicts do not
    contain the key (legacy synthetic mode).  Missing values are written as an
    empty string to preserve backward-compatible CSV shape.
    """
    if not schedule_details:
        print(f"  -> [Warning] No data to export to {filename}.")
        return None
    try:
        with _safe_open_csv(filename) as csvfile:
            fieldnames = list(schedule_details[0].keys())
            # Guarantee TransitionUsed column is always present.
            if "TransitionUsed" not in fieldnames:
                fieldnames = fieldnames + ["TransitionUsed"]
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            for task in schedule_details:
                row = {k: _format_two_decimals(v) for k, v in task.items()}
                if "TransitionUsed" not in row:
                    row["TransitionUsed"] = ""
                writer.writerow(row)

        # NEW: Calculate and print the TRUE makespan from the CSV data
        max_finish = max(task.get("Finish", 0) for task in schedule_details)
        # print(f"  -> [Debug] TRUE MAKESPAN from schedule_details: {max_finish:.2f}")

        return filename
    except (IOError, IndexError) as e:
        print(f"  -> [Error] Exporting schedule to CSV failed: {e}")
        return None


def export_sequencing_strategy_to_csv(schedule_details, filename):
    """Exports the operation sequence per room to a CSV file, returning the filename on success."""
    if not schedule_details:
        return
    room_schedules = _build_room_schedules(schedule_details)
    try:
        with _safe_open_csv(filename) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Room", "Operation_Sequence"])
            for room_name in sorted(room_schedules.keys()):
                schedule = sorted(room_schedules[room_name], key=lambda x: x[0])
                seq = " -> ".join(
                    [
                        f"{j}(Op{o})"
                        for _, j, o in schedule
                    ]
                )
                writer.writerow([room_name, seq])
        return filename
    except IOError as e:
        print(f"  -> [Error] Exporting strategy to CSV failed: {e}")
        return None


def export_montecarlo_summary(all_results, filename):
    """Exports the Monte Carlo summary, returning the filename on success."""
    try:
        with _safe_open_csv(filename) as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "algorithm",
                    "valid_simulations",
                    "makespan_min",
                    "makespan_median",
                    "makespan_avg",
                    "makespan_std",
                    "time_avg_s",
                ]
            )

            for name, results in all_results.items():
                makespans = [m for m in results["makespan"] if m != float("inf")]
                times = results["time"]

                if not makespans:
                    continue

                writer.writerow(
                    [
                        name,
                        len(makespans),
                        _format_two_decimals(np.min(makespans)),
                        _format_two_decimals(np.median(makespans)),
                        _format_two_decimals(np.mean(makespans)),
                        _format_two_decimals(np.std(makespans, ddof=1))
                        if len(makespans) > 1
                        else 0.0,
                        _format_two_decimals(np.mean(times)),
                    ]
                )

        logger.info(f"    - Monte Carlo summary saved to: {filename}")
        return filename
    except IOError as e:
        logger.error(f"  -> [Error] Exporting summary failed: {e}")
        return None


def export_statistical_analysis(comparison_results, filename):
    """Exports the detailed results of the pairwise comparison, returning the filename on success.

    Handles both legacy Mann-Whitney fields (u_stat, rank_biserial_r) and
    the current Wilcoxon paired fields (w_stat, z_stat, effect_r, p_adjusted).
    The p_value and p_adjusted columns are formatted with special threshold display.
    """
    if not comparison_results:
        return
    try:
        with _safe_open_csv(filename) as f:
            fieldnames = next((res.keys() for res in comparison_results if res), [])
            if not fieldnames:
                return

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in comparison_results:
                formatted_row = {}
                for k, v in row.items():
                    if k in ("p_value", "p_adjusted"):
                        if isinstance(v, (int, float, np.floating)):
                            formatted_row[k] = "≥0.05" if v >= 0.05 else f"{v:.4f}"
                        else:
                            formatted_row[k] = v
                    else:
                        formatted_row[k] = _format_two_decimals(v)

                writer.writerow(formatted_row)
        return filename
    except (IOError, IndexError) as e:
        print(f"  -> [Error] Exporting analysis failed: {e}")
        return None


def export_operational_paired_summary(operational_summary, filename):
    """Exports representative PAIRED operational aggregates, one row per algorithm.

    Consumes the dict returned by ``statistics.compute_operational_summary``:
    mean±sd of patients_with_extra_wait and avg_extra_wait_min, plus the paired
    makespan win-rate (%) and mean rank. This replaces cherry-picked best-run
    tables (``best_runs_by_mh``) for analysis reporting.

    Returns the filename on success, None on failure.
    """
    if not operational_summary:
        return None
    try:
        with _safe_open_csv(filename) as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "algo_name",
                    "n",
                    "patients_with_extra_wait_mean",
                    "patients_with_extra_wait_sd",
                    "avg_extra_wait_min_mean",
                    "avg_extra_wait_min_sd",
                    "makespan_win_rate_pct",
                    "makespan_mean_rank",
                ]
            )
            for algo_name in sorted(operational_summary.keys()):
                stats_row = operational_summary[algo_name]
                writer.writerow(
                    [
                        algo_name,
                        stats_row.get("n", 0),
                        _format_two_decimals(stats_row.get("patients_with_extra_wait_mean")),
                        _format_two_decimals(stats_row.get("patients_with_extra_wait_sd")),
                        _format_two_decimals(stats_row.get("avg_extra_wait_min_mean")),
                        _format_two_decimals(stats_row.get("avg_extra_wait_min_sd")),
                        _format_two_decimals(stats_row.get("win_rate_pct")),
                        _format_two_decimals(stats_row.get("mean_rank")),
                    ]
                )
        logger.info(f"    - Operational paired summary saved to: {filename}")
        return filename
    except IOError as e:
        logger.error(f"  -> [Error] Exporting operational paired summary failed: {e}")
        return None



def export_room_overtimes_csv(all_results, filename, all_rooms):
    """Exports individual room overtimes for all simulation runs to a CSV file."""
    try:
        from collections import defaultdict
        
        # Sort rooms numerically if they end with a number (e.g., Pabellon_1)
        def sort_key(room_name):
            try:
                return int(room_name.split("_")[-1])
            except ValueError:
                return room_name
        
        sorted_rooms = sorted(all_rooms, key=sort_key)
        room_headers = [f"{room}_overtime" for room in sorted_rooms]
        headers = ["algorithm", "simulation_id"] + room_headers
        
        STANDARD_SHIFT_MIN = 480.0  # 8-hour shift
        
        with _safe_open_csv(filename) as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for name, results in all_results.items():
                schedules = results.get("solution", [])
                for sim_i, schedule in enumerate(schedules):
                    if not schedule:
                        continue
                    
                    # Group operations by room
                    ops_by_room = defaultdict(list)
                    for t in schedule:
                        resource = t.get("Resource", "")
                        start = t.get("Start", 0)
                        finish = t.get("Finish", 0)
                        if resource:
                            ops_by_room[resource].append((start, finish))
                    
                    row = [name, sim_i]
                    for room in sorted_rooms:
                        ops = ops_by_room.get(room, [])
                        if not ops:
                            overtime = 0.0
                        else:
                            room_end = max(finish for _, finish in ops)
                            overtime = max(0.0, room_end - STANDARD_SHIFT_MIN)
                        row.append(_format_two_decimals(overtime))
                    writer.writerow(row)
                    
        logger.info(f"    - Room overtimes summary saved to: {filename}")
        return filename
    except Exception as e:
        logger.error(f"  -> [Error] Exporting room overtimes failed: {e}")
        return None


def export_routing_explanation_csv(schedule_details, filename):
    """
    Exports a detailed analytical report explaining room routing and delays.
    For each task, it determines when the assigned room, personnel, and patient
    became available, and identifies the main driver/reason for the start time.
    """
    try:
        if not schedule_details:
            return None

        # Sort tasks by Start time to process chronologically
        sorted_tasks = sorted(schedule_details, key=lambda x: x.get("Start", 0))

        # Track release times
        room_last_finish = {}
        personnel_last_finish = {}
        op1_finish = {}  # job -> op1 finish time

        # First pass: map op1 finish times for job ready time
        for t in sorted_tasks:
            job = t.get("Job")
            op = t.get("Operation")
            if op == 1:
                op1_finish[job] = t.get("Finish", 0.0)

        headers = [
            "job_id",
            "operation",
            "room_assigned",
            "personnel_assigned",
            "start_time",
            "processing_start",
            "finish_time",
            "room_free_before_start",
            "personnel_free_before_start",
            "patient_ready_before_start",
            "primary_delay_reason"
        ]

        with _safe_open_csv(filename) as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for t in sorted_tasks:
                job = t.get("Job")
                op = t.get("Operation")
                room = t.get("Resource")
                pers = t.get("Personnel")
                start = t.get("Start", 0.0)
                setup_used = t.get("SetupUsed") or 0.0
                trans_used = t.get("TransitionUsed") or 0.0
                time_before_proc = setup_used + trans_used
                proc_start = start + time_before_proc
                finish = t.get("Finish", 0.0)

                # Fetch previous status
                room_free = room_last_finish.get(room, 0.0)
                pers_free = personnel_last_finish.get(pers, 0.0)
                patient_ready = op1_finish.get(job, 0.0) if op > 1 else 0.0

                # Determine the primary constraint / delay reason
                if start < 1e-3:
                    reason = "Immediate / First Block"
                else:
                    reasons = []
                    if room_free > 0 and abs(start - room_free) < 1e-3:
                        reasons.append("Waiting for Room Release")
                    if pers_free > 0 and abs(start - pers_free) < 1e-3:
                        reasons.append("Waiting for Staff Availability")
                    if op > 1 and patient_ready > 0 and abs(start - patient_ready) < 1e-3:
                        reasons.append("Waiting for Op1 Completion")

                    if not reasons:
                        max_c = max(room_free, pers_free, patient_ready if op > 1 else 0.0)
                        if max_c <= 0:
                            reason = "Immediate / First Block"
                        elif abs(max_c - room_free) < 1e-3:
                            reason = "Waiting for Room Release"
                        elif abs(max_c - pers_free) < 1e-3:
                            reason = "Waiting for Staff Availability"
                        elif op > 1 and abs(max_c - patient_ready) < 1e-3:
                            reason = "Waiting for Op1 Completion"
                        else:
                            reason = "Optimizer Assignment Sequence"
                    else:
                        reason = " & ".join(reasons)

                writer.writerow([
                    job,
                    op,
                    room,
                    pers,
                    _format_two_decimals(start),
                    _format_two_decimals(proc_start),
                    _format_two_decimals(finish),
                    _format_two_decimals(room_free),
                    _format_two_decimals(pers_free),
                    _format_two_decimals(patient_ready),
                    reason
                ])

                # Update states
                room_last_finish[room] = finish
                personnel_last_finish[pers] = finish

        logger.info(f"    - Routing explanation saved to: {filename}")
        return filename
    except Exception as e:
        logger.error(f"  -> [Error] Exporting routing explanation failed: {e}")
        return None

