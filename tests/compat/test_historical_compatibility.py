import csv
from pathlib import Path

from core.legacy_runner import LegacyExperimentRunner
from data.instance_loader import load_instance


INSTANCE = Path(__file__).parents[2] / "instances/didactic/HOSP-DIDACT-03-01.yaml"
LARGE_INSTANCE = Path(__file__).parents[2] / "instances/hospital_12rooms/HOSP-12R-30-03.yaml"


def test_four_algorithm_runner_preserves_historical_tables_and_plot_paths(
    monkeypatch, tmp_path
):
    import config.config as config
    import algorithms.dpso as dpso
    import algorithms.ga as ga
    import algorithms.sboa as sboa

    monkeypatch.setattr(config, "GA_ENABLED", True)
    monkeypatch.setattr(config, "DPSO_ENABLED", True)
    monkeypatch.setattr(config, "SBOA_ENABLED", True)
    monkeypatch.setattr(config, "MSHOA_ENABLED", True)
    monkeypatch.setattr(config, "MAX_GENERATIONS", 2)
    monkeypatch.setattr(config, "MAX_ITERATIONS_DPSO", 2)
    monkeypatch.setattr(config, "SBOA_MAX_ITER", 2)
    monkeypatch.setattr(config, "MAX_ITERATIONS_MSHOA", 2)
    monkeypatch.setattr(config, "POPULATION_SIZE_GA", 3)
    monkeypatch.setattr(config, "SWARM_SIZE_DPSO", 3)
    monkeypatch.setattr(config, "SBOA_POP_SIZE", 3)
    monkeypatch.setattr(config, "MSHOA_POP_SIZE", 3)
    monkeypatch.setattr(ga, "POPULATION_SIZE_GA", 3)
    monkeypatch.setattr(ga, "MAX_GENERATIONS", 2)
    monkeypatch.setattr(dpso, "SWARM_SIZE_DPSO", 3)
    monkeypatch.setattr(dpso, "MAX_ITERATIONS_DPSO", 2)
    monkeypatch.setattr(sboa, "SBOA_POP_SIZE", 3)
    monkeypatch.setattr(sboa, "SBOA_MAX_ITER", 2)
    monkeypatch.setattr(config, "_ALGORITHMS_CACHE", None)

    runner = LegacyExperimentRunner(
        instance_path=INSTANCE, num_simulations=1, n_jobs=1, base_dir=tmp_path
    )
    assert [spec["name"] for spec in runner.algorithm_specs] == [
        "GA", "dPSO", "SBOA", "dMShOA"
    ]
    output = runner.run_elective_mode()
    csv_root = Path(output["csv"])
    plots_root = Path(output["plots"])

    with (csv_root / "summary_results.csv").open(newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))
    assert [row["algorithm"] for row in summary_rows] == [
        "GA", "dPSO", "SBOA", "dMShOA"
    ]
    assert all(int(row["valid_simulations"]) == 1 for row in summary_rows)

    with (csv_root / "summary_results.csv").open(newline="", encoding="utf-8") as stream:
        assert next(csv.reader(stream)) == [
            "algorithm", "valid_simulations", "makespan_min",
            "makespan_median", "makespan_avg", "makespan_std", "time_avg_s",
        ]

    expected_headers = {
        "schedule": [
            "Job", "Operation", "Resource", "Personnel", "Start",
            "ProcessingEnd", "Finish", "SetupUsed", "TransitionUsed",
            "CleanupUsed",
        ],
        "strategy": ["Room", "Operation_Sequence"],
        "routing": [
            "job_id", "operation", "room_assigned", "personnel_assigned",
            "start_time", "processing_start", "finish_time",
            "room_free_before_start", "personnel_free_before_start",
            "patient_ready_before_start", "primary_delay_reason",
        ],
    }
    for algorithm in ("ga", "dpso", "sboa", "dmshoa"):
        csv_files = {
            "schedule": csv_root / f"elective_best_schedule_{algorithm}.csv",
            "strategy": csv_root / f"elective_best_strategy_{algorithm}.csv",
            "routing": csv_root / f"elective_routing_explanation_{algorithm}.csv",
        }
        for kind, path in csv_files.items():
            assert path.exists()
            with path.open(newline="", encoding="utf-8") as stream:
                assert next(csv.reader(stream)) == expected_headers[kind]

        assert (plots_root / "gantt" / f"best_gantt_{algorithm}.png").exists()
        gantt_svg = plots_root / "gantt" / "svg" / f"best_gantt_{algorithm}.svg"
        assert gantt_svg.exists()
        assert "Twelve-room case" not in gantt_svg.read_text(encoding="utf-8")
        assert (plots_root / "convergence" / f"{algorithm}_convergence_sim_1.png").exists()
        assert (
            plots_root / "convergence" / "svg" / f"{algorithm}_convergence_sim_1.svg"
        ).exists()

    assert (plots_root / "boxplot" / "elective_makespan_comparison.png").exists()
    assert (
        plots_root / "boxplot" / "svg" / "elective_makespan_comparison.svg"
    ).exists()
    assert (plots_root / "barplot" / "elective_execution_time.png").exists()
    assert (
        plots_root / "barplot" / "svg" / "elective_execution_time.svg"
    ).exists()


def test_compatibility_runner_uses_selected_instance_resources():
    context = load_instance(INSTANCE)
    assert context.rooms == ("OR-1", "OR-2")
    assert context.personnel_by_operation == (
        (1, ("AN-1", "AN-2", "AN-3")),
        (2, ("SU-1", "SU-2", "SU-3", "SU-4")),
    )


def test_large_synthetic_instance_returns_finite_historical_schedules(
    monkeypatch, tmp_path
):
    import config.config as config
    import algorithms.dpso as dpso
    import algorithms.ga as ga
    import algorithms.sboa as sboa

    monkeypatch.setattr(config, "GA_ENABLED", True)
    monkeypatch.setattr(config, "DPSO_ENABLED", True)
    monkeypatch.setattr(config, "SBOA_ENABLED", True)
    monkeypatch.setattr(config, "MSHOA_ENABLED", True)
    monkeypatch.setattr(config, "MAX_WAIT_TIMES", {1: 0.0, 2: 500.0})
    monkeypatch.setattr(config, "MAX_GENERATIONS", 3)
    monkeypatch.setattr(config, "MAX_ITERATIONS_DPSO", 3)
    monkeypatch.setattr(config, "SBOA_MAX_ITER", 3)
    monkeypatch.setattr(config, "MAX_ITERATIONS_MSHOA", 3)
    monkeypatch.setattr(config, "POPULATION_SIZE_GA", 5)
    monkeypatch.setattr(config, "SWARM_SIZE_DPSO", 5)
    monkeypatch.setattr(config, "SBOA_POP_SIZE", 5)
    monkeypatch.setattr(config, "MSHOA_POP_SIZE", 5)
    monkeypatch.setattr(ga, "POPULATION_SIZE_GA", 5)
    monkeypatch.setattr(ga, "MAX_GENERATIONS", 3)
    monkeypatch.setattr(dpso, "SWARM_SIZE_DPSO", 5)
    monkeypatch.setattr(dpso, "MAX_ITERATIONS_DPSO", 3)
    monkeypatch.setattr(sboa, "SBOA_POP_SIZE", 5)
    monkeypatch.setattr(sboa, "SBOA_MAX_ITER", 3)
    monkeypatch.setattr(config, "_ALGORITHMS_CACHE", None)

    runner = LegacyExperimentRunner(
        instance_path=LARGE_INSTANCE, num_simulations=1, n_jobs=1, base_dir=tmp_path
    )
    output = runner.run_elective_mode()
    csv_root = Path(output["csv"])

    with (csv_root / "summary_results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["algorithm"] for row in rows] == [
        "GA", "dPSO", "SBOA", "dMShOA"
    ]
    assert all(int(row["valid_simulations"]) == 1 for row in rows)
    assert all(float(row["makespan_min"]) < float("inf") for row in rows)

    for algorithm in ("ga", "dpso", "sboa", "dmshoa"):
        schedule_path = csv_root / f"elective_best_schedule_{algorithm}.csv"
        with schedule_path.open(newline="", encoding="utf-8") as stream:
            assert len(list(csv.DictReader(stream))) == 60
