"""
Tests para AnalysisExporter — Task 4.1 RED.

El exporter lee iteration_schedules desde SQLite, rehidrata surgeries_data
(determinístico por sim_index + std_factor), llama calculate_schedule_fitness
con return_details=True y exporta tres CSVs:
  - schedule_by_iteration.csv
  - strategy_by_iteration.csv
  - breakdown_by_iteration.csv
"""
import os
import sys
import sqlite3
import json
import csv
import hashlib
import inspect
import yaml
import pytest


# ---------------------------------------------------------------------------
# Helpers: minimal config + análisis DB poblada
# ---------------------------------------------------------------------------


def _make_minimal_config(num_procedures=5):
    return {
        "experiment": {
            "num_simulations": 1,
            "std_factor_times": 0.0,
            "alpha_test": 0.05,
            "num_procedures": num_procedures,
            "output_dirs": {"plots": "results/plots", "csv": "results/csv"},
        },
        "real_data": {"enabled": False},
        "logging": {"verbose_mode": False},
        "times": {
            "setup": {"1": 10, "2": 10, "3": 10},
            "cleanup": {"1": 5, "2": 5, "3": 5},
            "max_wait": {"1": 100, "2": 100},
        },
        "jobs": {
            "types": {
                "1": 1, "2": 1, "3": 1, "4": 3, "5": 3,
            }
        },
        "resources": {"num_pabellones": 4},
        "personnel": {"num_anesthesiologists": 1, "num_surgeons": 1},
        "algorithms": {
            "alpha": 1e-6,
            "beta": 0.7,
            "gamma": 1.4,
            "delta": 100.0,
            "ga": {
                "enabled": True,
                "population_size": 2,
                "max_generations": 2,
                "crossover_probability": 0.8,
                "mutation_probability": 0.3,
                "elitism_count": 1,
            },
            "dpso": {"enabled": False, "swarm_size": 2, "max_iterations": 1,
                     "w": 0.7, "c1": 1.5, "c2": 1.5, "vel_high": 4.0, "vel_low": -4.0},
            "sboa": {"enabled": False, "population_size": 2, "max_iterations": 1,
                     "lower_bound": -5.0, "upper_bound": 5.0},
            "dmshoa": {"enabled": False, "population_size": 2, "max_iterations": 1,
                       "lower_bound": -5.0, "upper_bound": 5.0, "k": 3},
        },
    }


def _setup_analysis_db(tmp_path):
    """Build a small campaign-shaped DB with persisted iteration schedules."""
    from tests.reproducibility.conftest import HAND_COMPUTED_JOBS, build_synthetic_db

    cfg = _make_minimal_config(num_procedures=5)
    cfg_file = tmp_path / "cfg_exporter.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if any(
            mod_name.startswith(p)
            for p in ["config.", "core.", "algorithms.", "simulation.", "simulation.workers.", "data."]
        ):
            del sys.modules[mod_name]

    from core.analysis_persistence import AnalysisPersistence

    db_path = build_synthetic_db(tmp_path / "analysis.db", HAND_COMPUTED_JOBS)
    persistence = AnalysisPersistence(str(db_path))
    persistence.init_db()
    sim_ids = [
        row[0]
        for row in persistence._conn.execute(
            "SELECT sim_id FROM simulations WHERE sim_index = 0"
        ).fetchall()
    ]
    solution = json.dumps(
        {
            "job_sequence_base": [1, 2],
            "room_assignment": {
                "1": {"1": "Pabellon_1", "2": "Pabellon_2"},
                "2": {"1": "Pabellon_3", "2": "Pabellon_4"},
            },
        }
    )
    for sim_id in sim_ids:
        algo_iter_id = persistence.save_algorithm_iterations_batch(
            sim_id,
            [
                {
                    "algo_step": 1,
                    "best_fitness": 1.0,
                    "best_makespan": 160.0,
                    "iteration_fitness": 1.0,
                    "iteration_makespan": 160.0,
                }
            ],
        )[0]
        persistence.save_iteration_schedules_batch(
            [{"algo_iter_id": algo_iter_id, "solution_json": solution}]
        )
    persistence.close()
    return str(db_path)


# ---------------------------------------------------------------------------
# Task 4.1 — AnalysisExporter exists and is importable
# ---------------------------------------------------------------------------


class TestAnalysisExporterImport:
    def test_analysis_exporter_is_importable(self):
        """AnalysisExporter debe ser importable desde offline.analysis_exporter."""
        from offline.analysis_exporter import AnalysisExporter  # noqa: F401

    def test_analysis_exporter_has_export_method(self):
        """AnalysisExporter debe tener método export_iteration_csvs."""
        from offline.analysis_exporter import AnalysisExporter

        exporter = AnalysisExporter.__new__(AnalysisExporter)
        assert hasattr(exporter, "export_iteration_csvs"), (
            "AnalysisExporter debe tener método export_iteration_csvs(db_path, output_dir)"
        )


# ---------------------------------------------------------------------------
# Task 4.2 — export produce los 3 CSVs
# ---------------------------------------------------------------------------


class TestAnalysisExporterOutputFiles:
    def test_export_creates_schedule_by_iteration_csv(self, tmp_path):
        """export_iteration_csvs debe crear schedule_by_iteration.csv."""
        db_path = _setup_analysis_db(tmp_path)
        output_dir = str(tmp_path / "export")
        os.makedirs(output_dir, exist_ok=True)

        from offline.analysis_exporter import AnalysisExporter

        exporter = AnalysisExporter()
        exporter.export_iteration_csvs(db_path, output_dir, n_jobs=1)

        assert os.path.exists(os.path.join(output_dir, "schedule_by_iteration.csv")), (
            "schedule_by_iteration.csv debe existir tras export"
        )

    def test_export_creates_strategy_by_iteration_csv(self, tmp_path):
        """export_iteration_csvs debe crear strategy_by_iteration.csv."""
        db_path = _setup_analysis_db(tmp_path)
        output_dir = str(tmp_path / "export")
        os.makedirs(output_dir, exist_ok=True)

        from offline.analysis_exporter import AnalysisExporter

        exporter = AnalysisExporter()
        exporter.export_iteration_csvs(db_path, output_dir, n_jobs=1)

        assert os.path.exists(os.path.join(output_dir, "strategy_by_iteration.csv")), (
            "strategy_by_iteration.csv debe existir tras export"
        )

    def test_export_does_not_create_breakdown_by_iteration_csv(self, tmp_path):
        """export_iteration_csvs no debe crear breakdown_by_iteration.csv por ser redundante."""
        db_path = _setup_analysis_db(tmp_path)
        output_dir = str(tmp_path / "export")
        os.makedirs(output_dir, exist_ok=True)

        from offline.analysis_exporter import AnalysisExporter

        exporter = AnalysisExporter()
        exporter.export_iteration_csvs(db_path, output_dir, n_jobs=1)

        assert not os.path.exists(os.path.join(output_dir, "breakdown_by_iteration.csv")), (
            "breakdown_by_iteration.csv no debe crearse tras export"
        )


# ---------------------------------------------------------------------------
# Task 4.2 — CSVs tienen filas reales y columnas esperadas
# ---------------------------------------------------------------------------


class TestAnalysisExporterCSVContent:
    def _run_export(self, tmp_path):
        db_path = _setup_analysis_db(tmp_path)
        output_dir = str(tmp_path / "export")
        os.makedirs(output_dir, exist_ok=True)
        from offline.analysis_exporter import AnalysisExporter

        exporter = AnalysisExporter()
        exporter.export_iteration_csvs(db_path, output_dir, n_jobs=1)
        return output_dir

    def test_schedule_csv_has_expected_columns(self, tmp_path):
        """schedule_by_iteration.csv debe tener: sim_index, algo_step, job_id, operation_num, room, start_time, end_time."""
        output_dir = self._run_export(tmp_path)
        path = os.path.join(output_dir, "schedule_by_iteration.csv")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        expected = {"sim_index", "algo_step", "job_id", "operation_num", "room", "start_time", "end_time"}
        assert expected.issubset(set(headers)), (
            f"Faltan columnas en schedule_by_iteration.csv: {expected - set(headers)}"
        )

    def test_schedule_csv_has_at_least_one_row(self, tmp_path):
        """schedule_by_iteration.csv debe tener al menos 1 fila de datos."""
        output_dir = self._run_export(tmp_path)
        path = os.path.join(output_dir, "schedule_by_iteration.csv")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0, "schedule_by_iteration.csv debe tener al menos 1 fila"

    def test_strategy_csv_has_expected_columns(self, tmp_path):
        """strategy_by_iteration.csv debe tener: sim_index, algo_step, room, operation_sequence."""
        output_dir = self._run_export(tmp_path)
        path = os.path.join(output_dir, "strategy_by_iteration.csv")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        expected = {"sim_index", "algo_step", "room", "operation_sequence"}
        assert expected.issubset(set(headers)), (
            f"Faltan columnas en strategy_by_iteration.csv: {expected - set(headers)}"
        )



    def test_algo_step_values_are_positive_integers(self, tmp_path):
        """El campo algo_step en schedule_by_iteration.csv debe ser entero >= 1."""
        output_dir = self._run_export(tmp_path)
        path = os.path.join(output_dir, "schedule_by_iteration.csv")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            steps = [int(row["algo_step"]) for row in reader]
        assert all(s >= 1 for s in steps), "Todos los algo_step deben ser >= 1"


class TestAnalysisExporterPersistenceContract:
    def test_exporter_has_no_sampling_path(self):
        """The exporter must not recreate campaign instances at analysis time."""
        from offline import analysis_exporter

        source = inspect.getsource(analysis_exporter)
        assert "generate_day_surgeries_from_pkl" not in source
        assert "generate_day_surgeries_data" not in source

    def test_process_row_uses_persisted_day_data(self, monkeypatch):
        """The scheduler receives the instance reconstructed before dispatch."""
        from offline import analysis_exporter

        captured = {}

        def fake_fitness(solution, day_data, return_details=False):
            captured["day_data"] = day_data
            return 0.0, 20.0, [
                {
                    "Job": 1,
                    "Operation": 1,
                    "Resource": "R1",
                    "Personnel": "A1",
                    "Start": 0.0,
                    "ProcessingEnd": 10.0,
                    "Finish": 10.0,
                }
            ]

        monkeypatch.setattr(
            "simulation.scheduler.calculate_schedule_fitness", fake_fitness
        )
        persisted_day_data = {
            1: {
                1: 30.0,
                2: 40.0,
                "setup_by_op": {1: 5.0, 2: 0.0},
                "transition_by_op": {1: 2.0, 2: 0.0},
                "cleanup_by_op": {1: 0.0, 2: 3.0},
            }
        }
        row = {
            "run_id": 7,
            "sim_index": 2,
            "algo_name": "ga",
            "algo_step": 1,
            "solution_json": json.dumps(
                {"job_sequence_base": [1], "room_assignment": {"1": {"1": "R1"}}}
            ),
            "day_data": persisted_day_data,
        }

        analysis_exporter._process_row_data(row)

        assert captured["day_data"] is persisted_day_data

    def test_scheduler_error_is_logged_and_propagated(self, monkeypatch, caplog):
        """A failed iteration must not silently become an empty CSV result."""
        from offline.analysis_exporter import IterationExportError, _process_row_data

        def failing_fitness(*args, **kwargs):
            raise ValueError("scheduler exploded")

        monkeypatch.setattr(
            "simulation.scheduler.calculate_schedule_fitness", failing_fitness
        )
        row = {
            "run_id": 7,
            "sim_index": 2,
            "algo_name": "ga",
            "algo_step": 3,
            "solution_json": json.dumps(
                {"job_sequence_base": [1], "room_assignment": {"1": {"1": "R1"}}}
            ),
            "day_data": {1: {1: 30.0, 2: 40.0}},
        }

        with pytest.raises(IterationExportError, match="run_id=7.*algo_step=3"):
            _process_row_data(row)

        assert "scheduler exploded" in caplog.text
        assert "sim_index=2" in caplog.text
