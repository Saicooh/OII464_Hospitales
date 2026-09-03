"""Run-scoped persistence and reporting for the supported execution path."""

from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from core.iteration_callback import IterationSnapshot
from simulation.result_model import ScheduleEntry, SimulationResult


PLOT_SUBDIRECTORIES = (
    "gantt", "personnel", "histograms", "boxplot", "advanced", "convergence",
)
CSV_FILES = (
    "summary_results.csv",
    "best_schedule_dmshoa.csv",
    "sequencing_strategy_dmshoa.csv",
    "routing_explanation_dmshoa.csv",
    "room_overtimes.csv",
)

_SCHEMA_DDL_V7 = """
CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY,
    run_identity TEXT NOT NULL UNIQUE,
    started_at TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    instance_digest TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    num_simulations INTEGER NOT NULL,
    num_procedures INTEGER NOT NULL
);
CREATE TABLE simulations (
    sim_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    simulation_index INTEGER NOT NULL,
    solver_seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    combined_objective REAL NOT NULL,
    makespan REAL NOT NULL,
    algorithm_seconds REAL NOT NULL,
    wall_clock_seconds REAL NOT NULL,
    UNIQUE(run_id, simulation_index)
);
CREATE TABLE schedule_entries (
    schedule_entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_id INTEGER NOT NULL REFERENCES simulations(sim_id),
    entry_index INTEGER NOT NULL,
    job_id INTEGER NOT NULL,
    operation INTEGER NOT NULL,
    room TEXT NOT NULL,
    personnel TEXT NOT NULL,
    start REAL NOT NULL,
    processing_end REAL NOT NULL,
    finish REAL NOT NULL,
    setup REAL NOT NULL,
    transition REAL NOT NULL,
    cleanup REAL NOT NULL
);
CREATE TABLE patient_wait_metrics (
    wait_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_id INTEGER NOT NULL REFERENCES simulations(sim_id),
    job_id INTEGER NOT NULL,
    op1_room TEXT NOT NULL,
    op2_room TEXT NOT NULL,
    op1_finish REAL NOT NULL,
    op2_start REAL NOT NULL,
    transition_used REAL NOT NULL,
    extra_wait_min REAL NOT NULL
);
CREATE TABLE schedule_quality_metrics (
    quality_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_id INTEGER NOT NULL UNIQUE REFERENCES simulations(sim_id),
    rooms_used INTEGER NOT NULL,
    total_overtime_min REAL NOT NULL,
    max_room_overtime_min REAL NOT NULL,
    personnel_count INTEGER NOT NULL,
    workload_std_min REAL NOT NULL,
    workload_max_min REAL NOT NULL,
    workload_min_min REAL NOT NULL,
    idle_gap_count INTEGER NOT NULL,
    idle_gap_total_min REAL NOT NULL,
    avg_idle_gap_min REAL NOT NULL,
    value_added_ratio REAL NOT NULL
);
CREATE TABLE algorithm_iterations (
    iteration_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_id INTEGER NOT NULL REFERENCES simulations(sim_id),
    algo_step INTEGER NOT NULL,
    best_fitness REAL NOT NULL,
    best_makespan REAL NOT NULL,
    iteration_fitness REAL NOT NULL,
    iteration_makespan REAL NOT NULL
);
CREATE TABLE statistical_summaries (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    metric TEXT NOT NULL,
    sample_size INTEGER NOT NULL,
    minimum REAL NOT NULL,
    median REAL NOT NULL,
    average REAL NOT NULL,
    maximum REAL NOT NULL
);
CREATE INDEX idx_v7_sim_run ON simulations(run_id, simulation_index);
CREATE INDEX idx_v7_schedule_sim ON schedule_entries(sim_id);
CREATE INDEX idx_v7_wait_sim ON patient_wait_metrics(sim_id);
CREATE INDEX idx_v7_iter_sim ON algorithm_iterations(sim_id);
"""


def _wait_rows(schedule: tuple[ScheduleEntry, ...]) -> list[dict[str, object]]:
    by_job: dict[int, dict[int, ScheduleEntry]] = defaultdict(dict)
    for entry in schedule:
        by_job[entry.job_id][entry.operation] = entry
    rows = []
    for job_id, operations in sorted(by_job.items()):
        if 1 not in operations or 2 not in operations:
            continue
        first, second = operations[1], operations[2]
        transition = second.transition
        rows.append(
            {
                "job_id": job_id,
                "op1_room": first.room,
                "op2_room": second.room,
                "op1_finish": first.finish,
                "op2_start": second.start,
                "transition_used": transition,
                "extra_wait_min": max(0.0, second.start - first.finish - transition),
            }
        )
    return rows


def _quality_metrics(schedule: tuple[ScheduleEntry, ...]) -> dict[str, float | int]:
    if not schedule:
        return {
            "rooms_used": 0, "total_overtime_min": 0.0,
            "max_room_overtime_min": 0.0, "personnel_count": 0,
            "workload_std_min": 0.0, "workload_max_min": 0.0,
            "workload_min_min": 0.0, "idle_gap_count": 0,
            "idle_gap_total_min": 0.0, "avg_idle_gap_min": 0.0,
            "value_added_ratio": 0.0,
        }
    by_room: dict[str, list[ScheduleEntry]] = defaultdict(list)
    by_person: dict[str, list[ScheduleEntry]] = defaultdict(list)
    for entry in schedule:
        by_room[entry.room].append(entry)
        by_person[entry.personnel].append(entry)
    room_overtimes = [
        max(0.0, max(item.finish for item in entries) - 480.0)
        for entries in by_room.values()
    ]
    workloads = [sum(item.finish - item.start for item in entries) for entries in by_person.values()]
    if len(workloads) > 1:
        average = sum(workloads) / len(workloads)
        workload_std = (sum((value - average) ** 2 for value in workloads) / (len(workloads) - 1)) ** 0.5
    else:
        workload_std = 0.0
    gap_count = 0
    gap_total = 0.0
    for entries in by_room.values():
        ordered = sorted(entries, key=lambda item: item.start)
        for previous, current in zip(ordered, ordered[1:]):
            gap = current.start - previous.finish
            if gap > 5.0:
                gap_count += 1
                gap_total += gap
    processing = sum(max(0.0, item.processing_end - item.start) for item in schedule)
    all_time = processing + sum(item.cleanup for item in schedule)
    return {
        "rooms_used": len(by_room),
        "total_overtime_min": sum(room_overtimes),
        "max_room_overtime_min": max(room_overtimes, default=0.0),
        "personnel_count": len(by_person),
        "workload_std_min": workload_std,
        "workload_max_min": max(workloads, default=0.0),
        "workload_min_min": min(workloads, default=0.0),
        "idle_gap_count": gap_count,
        "idle_gap_total_min": gap_total,
        "avg_idle_gap_min": gap_total / gap_count if gap_count else 0.0,
        "value_added_ratio": processing / all_time if all_time else 0.0,
    }


def _iteration_snapshots(result: SimulationResult) -> tuple[IterationSnapshot, ...]:
    """Use typed telemetry when present and preserve uneven history lengths."""
    if result.iterations:
        return result.iterations
    count = max(len(result.best_fitness_history), len(result.average_fitness_history))
    return tuple(
        IterationSnapshot(
            algo_step=step,
            best_fitness=(
                result.best_fitness_history[step - 1]
                if step <= len(result.best_fitness_history)
                else result.combined_objective
            ),
            best_makespan=result.makespan,
            iteration_fitness=(
                result.average_fitness_history[step - 1]
                if step <= len(result.average_fitness_history)
                else result.combined_objective
            ),
            iteration_makespan=result.makespan,
        )
        for step in range(1, count + 1)
    )


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_number(timestamp_dir: Path) -> int:
    existing = []
    for candidate in timestamp_dir.glob("run*"):
        if candidate.is_dir() and candidate.name[3:].isdigit():
            existing.append(int(candidate.name[3:]))
    return max(existing, default=0) + 1


def _plot_outputs(run_root: Path, results: tuple[SimulationResult, ...]) -> None:
    for subdirectory in PLOT_SUBDIRECTORIES:
        (run_root / "plots" / subdirectory).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    completed = [result for result in results if result.is_completed]
    makespans = [result.makespan for result in completed] or [0.0]

    figure, axis = plt.subplots()
    axis.hist(makespans, bins=max(1, min(10, len(makespans))))
    axis.set(title="Makespan", xlabel="minutes", ylabel="simulations")
    figure.savefig(run_root / "plots" / "histograms" / "makespan.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.boxplot(makespans)
    axis.set(title="Makespan distribution", ylabel="minutes")
    figure.savefig(run_root / "plots" / "boxplot" / "makespan.png", bbox_inches="tight")
    plt.close(figure)

    best = min(completed, key=lambda result: result.makespan, default=None)
    figure, axis = plt.subplots()
    if best is not None:
        for entry in best.schedule:
            axis.barh(f"J{entry.job_id}/O{entry.operation}", entry.finish - entry.start, left=entry.start)
    axis.set(title="Best schedule", xlabel="minutes")
    figure.savefig(run_root / "plots" / "gantt" / "best_schedule.png", bbox_inches="tight")
    plt.close(figure)

    workloads: dict[str, float] = defaultdict(float)
    if best is not None:
        for entry in best.schedule:
            workloads[entry.personnel] += entry.finish - entry.start
    figure, axis = plt.subplots()
    if workloads:
        axis.bar(list(workloads), list(workloads.values()))
    axis.set(title="Personnel workload", ylabel="minutes")
    figure.savefig(run_root / "plots" / "personnel" / "workload.png", bbox_inches="tight")
    plt.close(figure)

    quality = _quality_metrics(best.schedule if best is not None else ())
    figure, axis = plt.subplots()
    keys = ("rooms_used", "personnel_count", "idle_gap_count")
    axis.bar(list(keys), [float(quality[key]) for key in keys])
    axis.set(title="Operational KPIs")
    figure.savefig(run_root / "plots" / "advanced" / "operational_kpis.png", bbox_inches="tight")
    plt.close(figure)

    for result in completed:
        history = result.best_fitness_history or (result.makespan,)
        figure, axis = plt.subplots()
        axis.plot(range(1, len(history) + 1), history)
        axis.set(title=f"Convergence simulation {result.simulation_index}", xlabel="step", ylabel="objective")
        figure.savefig(
            run_root / "plots" / "convergence" / f"simulation_{result.simulation_index}.png",
            bbox_inches="tight",
        )
        plt.close(figure)


def write_run_outputs(
    results: Iterable[SimulationResult],
    output_root: str | Path = "results",
    run_timestamp: str | None = None,
) -> Path:
    """Persist one supported run under ``results/<timestamp>/run<N>/``."""
    materialized = tuple(results)
    if not materialized:
        raise ValueError("at least one SimulationResult is required")
    if len({result.instance_digest for result in materialized}) != 1:
        raise ValueError("all results in one run must share an instance digest")
    root = Path(output_root)
    timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamp_dir = root / timestamp
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    run_root = timestamp_dir / f"run{_run_number(timestamp_dir)}"
    csv_root = run_root / "csv"
    csv_root.mkdir(parents=True)
    for subdirectory in PLOT_SUBDIRECTORIES:
        (run_root / "plots" / subdirectory).mkdir(parents=True, exist_ok=True)

    instance_id = materialized[0].instance_id
    digest = materialized[0].instance_digest
    completed = [result for result in materialized if result.is_completed]
    job_ids = {entry.job_id for result in completed for entry in result.schedule}
    started_at = datetime.now(timezone.utc).isoformat()
    db_path = run_root / "analysis.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(_SCHEMA_DDL_V7)
        connection.execute("INSERT INTO schema_version(version) VALUES (7)")
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (int(run_root.name[3:]), run_root.name, started_at, instance_id, digest,
             "dmshoa", len(materialized), len(job_ids)),
        )
        summary_rows: list[dict[str, object]] = []
        routing_rows: list[dict[str, object]] = []
        sequencing_rows: list[dict[str, object]] = []
        overtime_rows: list[dict[str, object]] = []
        for result in materialized:
            cursor = connection.execute(
                "INSERT INTO simulations(run_id, simulation_index, solver_seed, status, error, combined_objective, makespan, algorithm_seconds, wall_clock_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (int(run_root.name[3:]), result.simulation_index, result.solver_seed,
                 result.status, result.error, result.combined_objective, result.makespan,
                 result.algorithm_seconds, result.wall_clock_seconds),
            )
            sim_id = cursor.lastrowid
            for entry_index, entry in enumerate(result.schedule):
                connection.execute(
                    "INSERT INTO schedule_entries(sim_id, entry_index, job_id, operation, room, personnel, start, processing_end, finish, setup, transition, cleanup) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (sim_id, entry_index, entry.job_id, entry.operation, entry.room, entry.personnel,
                     entry.start, entry.processing_end, entry.finish, entry.setup,
                     entry.transition, entry.cleanup),
                )
                routing_rows.append({
                    "simulation_index": result.simulation_index, "job_id": entry.job_id,
                    "operation": entry.operation, "room": entry.room, "personnel": entry.personnel,
                    "start": entry.start, "processing_end": entry.processing_end,
                    "finish": entry.finish,
                })
            waits = _wait_rows(result.schedule)
            for wait in waits:
                connection.execute(
                    "INSERT INTO patient_wait_metrics(sim_id, job_id, op1_room, op2_room, op1_finish, op2_start, transition_used, extra_wait_min) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (sim_id, wait["job_id"], wait["op1_room"], wait["op2_room"],
                     wait["op1_finish"], wait["op2_start"], wait["transition_used"],
                     wait["extra_wait_min"]),
                )
            quality = _quality_metrics(result.schedule)
            connection.execute(
                "INSERT INTO schedule_quality_metrics VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sim_id, *(quality[key] for key in (
                    "rooms_used", "total_overtime_min", "max_room_overtime_min",
                    "personnel_count", "workload_std_min", "workload_max_min",
                    "workload_min_min", "idle_gap_count", "idle_gap_total_min",
                    "avg_idle_gap_min", "value_added_ratio",
                ))),
            )
            for snapshot in _iteration_snapshots(result):
                connection.execute(
                    "INSERT INTO algorithm_iterations(sim_id, algo_step, best_fitness, best_makespan, iteration_fitness, iteration_makespan) VALUES (?, ?, ?, ?, ?, ?)",
                    (sim_id, snapshot.algo_step, snapshot.best_fitness, snapshot.best_makespan,
                     snapshot.iteration_fitness, snapshot.iteration_makespan),
                )
            summary_rows.append({
                "simulation_index": result.simulation_index, "solver_seed": result.solver_seed,
                "instance_id": result.instance_id, "instance_digest": result.instance_digest,
                "status": result.status, "combined_objective": result.combined_objective,
                "makespan": result.makespan, "algorithm_seconds": result.algorithm_seconds,
                "wall_clock_seconds": result.wall_clock_seconds,
            })
            for room in sorted({entry.room for entry in result.schedule}):
                room_entries = [entry for entry in result.schedule if entry.room == room]
                overtime_rows.append({
                    "simulation_index": result.simulation_index, "room": room,
                    "room_end": max(entry.finish for entry in room_entries),
                    "overtime_min": max(0.0, max(entry.finish for entry in room_entries) - 480.0),
                })
            for rank, job_id in enumerate(result.schedule[::2], start=1):
                sequencing_rows.append({
                    "simulation_index": result.simulation_index, "sequence_rank": rank,
                    "job_id": job_id.job_id,
                })
        values = [result.makespan for result in completed]
        if values:
            for metric, metric_values in (("makespan", values), ("combined_objective", [r.combined_objective for r in completed])):
                connection.execute(
                    "INSERT INTO statistical_summaries(run_id, metric, sample_size, minimum, median, average, maximum) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (int(run_root.name[3:]), metric, len(metric_values), min(metric_values),
                     median(metric_values), mean(metric_values), max(metric_values)),
                )
        connection.commit()
    finally:
        connection.close()

    best = min(completed, key=lambda result: result.makespan, default=None)
    _write_csv(
        csv_root / "summary_results.csv",
        list(summary_rows[0]) if summary_rows else ["simulation_index"],
        summary_rows,
    )
    _write_csv(
        csv_root / "best_schedule_dmshoa.csv",
        ["job_id", "operation", "room", "personnel", "start", "processing_end", "finish"],
        ({"job_id": entry.job_id, "operation": entry.operation, "room": entry.room,
          "personnel": entry.personnel, "start": entry.start,
          "processing_end": entry.processing_end, "finish": entry.finish}
         for entry in (best.schedule if best is not None else ())),
    )
    _write_csv(csv_root / "sequencing_strategy_dmshoa.csv", ["simulation_index", "sequence_rank", "job_id"], sequencing_rows)
    _write_csv(
        csv_root / "routing_explanation_dmshoa.csv",
        ["simulation_index", "job_id", "operation", "room", "personnel", "start", "processing_end", "finish"],
        routing_rows,
    )
    _write_csv(csv_root / "room_overtimes.csv", ["simulation_index", "room", "room_end", "overtime_min"], overtime_rows)
    _plot_outputs(run_root, materialized)
    return run_root


__all__ = ["CSV_FILES", "PLOT_SUBDIRECTORIES", "write_run_outputs"]
