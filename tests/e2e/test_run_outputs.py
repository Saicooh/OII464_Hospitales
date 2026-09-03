import csv
import sqlite3

from core.run_outputs import CSV_FILES, PLOT_SUBDIRECTORIES, write_run_outputs
from core.iteration_callback import IterationSnapshot
from simulation.result_model import ScheduleEntry, SimulationResult


def _result(simulation_index: int, seed: int) -> SimulationResult:
    entries = (
        ScheduleEntry(1, 1, "OR-1", "AN-1", 0.0, 10.0, 11.0, 1.0, 0.0, 1.0),
        ScheduleEntry(1, 2, "OR-1", "SU-1", 11.0, 31.0, 33.0, 0.0, 0.0, 2.0),
        ScheduleEntry(2, 1, "OR-2", "AN-2", 0.0, 12.0, 13.0, 1.0, 0.0, 1.0),
        ScheduleEntry(2, 2, "OR-2", "SU-2", 13.0, 35.0, 37.0, 0.0, 0.0, 2.0),
    )
    return SimulationResult(
        simulation_index=simulation_index,
        solver_seed=seed,
        instance_id="HOSP-DIDACT-03-01",
        instance_digest="a" * 64,
        status="completed",
        error=None,
        combined_objective=float(40 + simulation_index),
        makespan=float(37 + simulation_index),
        schedule=entries,
        algorithm_seconds=0.1,
        wall_clock_seconds=0.2,
        best_fitness_history=(40.0, 39.0),
        average_fitness_history=(42.0, 41.0),
        iterations=(IterationSnapshot(1, 40.0, 37.0, 40.0, 37.0),),
    )


def test_supported_run_writes_v7_outputs_without_obsolete_artifacts(tmp_path):
    run_root = write_run_outputs((_result(0, 11), _result(1, 22)), tmp_path / "results", run_timestamp="20260902_120000")

    assert run_root == tmp_path / "results" / "20260902_120000" / "run1"
    assert (run_root / "analysis.db").exists()
    assert {path.name for path in (run_root / "csv").glob("*.csv")} == set(CSV_FILES)
    assert {path.name for path in (run_root / "plots").iterdir()} == set(PLOT_SUBDIRECTORIES)
    assert all(any((run_root / "plots" / name).iterdir()) for name in PLOT_SUBDIRECTORIES)

    with sqlite3.connect(run_root / "analysis.db") as connection:
        assert connection.execute("SELECT version FROM schema_version").fetchone() == (7,)
        assert connection.execute("SELECT COUNT(*) FROM simulations").fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM patient_wait_metrics").fetchone() == (4,)
        assert connection.execute("SELECT COUNT(*) FROM statistical_summaries").fetchone() == (2,)
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not any("checkpoint" in table or "replay" in table for table in tables)

    with (run_root / "csv" / "summary_results.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    prohibited = ("analysis_algorithm_iterations_run", "convergence_by_time_run", "convergence_combined_run")
    assert not any(any(term in str(path) for term in prohibited) for path in run_root.rglob("*"))


def test_two_supported_runs_share_no_output_root(tmp_path):
    output_root = tmp_path / "results"
    first = write_run_outputs((_result(0, 11),), output_root, run_timestamp="20260902_120000")
    second = write_run_outputs((_result(0, 22),), output_root, run_timestamp="20260902_120000")
    assert first != second
    assert first.name == "run1"
    assert second.name == "run2"
    assert first.exists() and second.exists()
