"""
Tests for Lote 4 — ReportGenerator and checkpoint reporting.

Tasks covered:
  4.3 - export_full_schedule_to_csv always includes TransitionUsed column
  4.3 - ReportGenerator.generate_checkpoint_report() generates reports from best_run snapshot
  4.4 - run_elective_analysis_mode calls checkpoint reports when ANALYSIS_FULL_REPORTS_ENABLED=True
"""

import csv
import os
import yaml
import sys
import tempfile
import pytest


# ---------------------------------------------------------------------------
# Config helpers (reuse pattern from existing tests)
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
            "types": {str(i): (i % 3 + 1) for i in range(1, 6)},
        },
        "resources": {"num_pabellones": 2},
        "personnel": {
            "num_anesthesiologists": 1,
            "num_surgeons": 1,
        },
        "algorithms": {
            "alpha": 1e-6,
            "beta": 0.7,
            "gamma": 1.4,
            "delta": 100.0,
            "ga": {
                "enabled": True,
                "population_size": 2,
                "max_generations": 1,
                "crossover_probability": 0.8,
                "mutation_probability": 0.3,
                "elitism_count": 1,
            },
            "dpso": {
                "enabled": False,
                "swarm_size": 2,
                "max_iterations": 1,
                "w": 0.7,
                "c1": 1.5,
                "c2": 1.5,
                "vel_high": 4.0,
                "vel_low": -4.0,
            },
            "sboa": {
                "enabled": False,
                "population_size": 3,
                "max_iterations": 1,
                "lower_bound": -5.0,
                "upper_bound": 5.0,
            },
            "dmshoa": {
                "enabled": False,
                "population_size": 2,
                "max_iterations": 1,
                "k": 0.3,
                "lower_bound": -5.0,
                "upper_bound": 5.0,
            },
        },
    }


def _setup_config(tmp_path, extra=None):
    cfg = _make_minimal_config()
    if extra:
        cfg.update(extra)
    cfg_file = tmp_path / "config_l4.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if any(
            mod_name.startswith(p)
            for p in [
                "config.",
                "core.simulation_runner",
                "core.analysis_persistence",
                "core.report_generator",
                "simulation.workers.",
            ]
        ):
            del sys.modules[mod_name]


def _make_schedule_with_transition(job_id=1, transition_used=5.0):
    """Schedule with TransitionUsed present on op2."""
    return [
        {
            "Job": job_id,
            "Operation": 1,
            "Resource": "Pabellon_1",
            "Personnel": "A1",
            "Start": 0.0,
            "ProcessingEnd": 10.0,
            "Finish": 10.0,
            "SetupUsed": 10.0,
            "TransitionUsed": None,
            "CleanupUsed": 0.0,
        },
        {
            "Job": job_id,
            "Operation": 2,
            "Resource": "Pabellon_1",
            "Personnel": "S1",
            "Start": 10.0,
            "ProcessingEnd": 75.0,
            "Finish": 80.0,
            "SetupUsed": 0.0,
            "TransitionUsed": transition_used,
            "CleanupUsed": 8.0,
        },
    ]


def _make_schedule_without_transition(job_id=1):
    """Schedule without TransitionUsed key (legacy synthetic mode)."""
    return [
        {
            "Job": job_id,
            "Operation": 1,
            "Resource": "Pabellon_1",
            "Personnel": "A1",
            "Start": 0.0,
            "Finish": 10.0,
            "SetupUsed": 5.0,
            "CleanupUsed": 2.0,
        },
        {
            "Job": job_id,
            "Operation": 2,
            "Resource": "Pabellon_1",
            "Personnel": "S1",
            "Start": 12.0,
            "Finish": 70.0,
            "SetupUsed": 3.0,
            "CleanupUsed": 5.0,
        },
    ]


# ---------------------------------------------------------------------------
# Task 4.3a — export_full_schedule_to_csv includes TransitionUsed
# ---------------------------------------------------------------------------


class TestExportScheduleTransitionUsedColumn:
    """export_full_schedule_to_csv must always emit TransitionUsed column."""

    def test_transition_used_column_present_when_in_schedule(self, tmp_path):
        """When schedule dicts have TransitionUsed, the CSV includes that column."""
        _setup_config(tmp_path)
        from utils.reporting import export_full_schedule_to_csv

        schedule = _make_schedule_with_transition(job_id=1, transition_used=5.0)
        out_path = str(tmp_path / "schedule.csv")
        export_full_schedule_to_csv(schedule, out_path)

        assert os.path.exists(out_path), "CSV file must be created"
        with open(out_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        assert "TransitionUsed" in headers, (
            f"CSV must contain 'TransitionUsed' column. Got: {headers}"
        )

    def test_transition_used_values_written_correctly(self, tmp_path):
        """TransitionUsed values in CSV match those in schedule_details."""
        _setup_config(tmp_path)
        from utils.reporting import export_full_schedule_to_csv

        schedule = _make_schedule_with_transition(job_id=2, transition_used=7.5)
        out_path = str(tmp_path / "schedule2.csv")
        export_full_schedule_to_csv(schedule, out_path)

        with open(out_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # op2 row should have TransitionUsed = 7.5
        # Note: numeric values are formatted to 2 decimals, so Operation == "2.00"
        op2_rows = [r for r in rows if str(r.get("Operation", "")).startswith("2")]
        assert len(op2_rows) == 1
        assert float(op2_rows[0]["TransitionUsed"]) == pytest.approx(7.5)

    def test_transition_used_column_present_even_when_missing_from_schedule(
        self, tmp_path
    ):
        """Even if schedule dicts lack TransitionUsed key, column must be present (None/empty)."""
        _setup_config(tmp_path)
        from utils.reporting import export_full_schedule_to_csv

        schedule = _make_schedule_without_transition(job_id=3)
        out_path = str(tmp_path / "schedule3.csv")
        # This is a behavioral spec: the CSV writer must guarantee the column
        # If the existing code uses fieldnames=schedule_details[0].keys(),
        # TransitionUsed would be missing when the key is absent.
        # We want it to be present regardless.
        export_full_schedule_to_csv(schedule, out_path)

        with open(out_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        assert "TransitionUsed" in headers, (
            f"'TransitionUsed' must be present even when missing from schedule dicts. Got: {headers}"
        )


# ---------------------------------------------------------------------------
# Task 4.3b — ReportGenerator.generate_checkpoint_report()
# ---------------------------------------------------------------------------


class TestGenerateCheckpointReport:
    """ReportGenerator.generate_checkpoint_report() produces CSV + plots for best_run."""

    def test_method_exists_and_callable(self, tmp_path):
        """ReportGenerator must expose generate_checkpoint_report()."""
        _setup_config(tmp_path)
        from core.report_generator import ReportGenerator

        rg = ReportGenerator()
        assert hasattr(rg, "generate_checkpoint_report"), (
            "ReportGenerator must have generate_checkpoint_report()"
        )
        assert callable(rg.generate_checkpoint_report)

    def test_checkpoint_report_creates_schedule_csv(self, tmp_path):
        """generate_checkpoint_report() creates at least one CSV file in output_dirs['csv']."""
        _setup_config(tmp_path)
        from core.report_generator import ReportGenerator
        from core.file_manager import FileManager

        rg = ReportGenerator()
        fm = FileManager(base_dir=str(tmp_path / "results"))
        output_dirs = fm.setup_analysis_directories("20260416_120000", "elective")

        schedule = _make_schedule_with_transition(job_id=1, transition_used=5.0)
        best_run = {
            "makespan": 80.0,
            "schedule": schedule,
            "sim_num": 0,
            "algo_name": "GA",
            "job_label_map": None,
        }
        all_rooms = ["Pabellon_1"]

        rg.generate_checkpoint_report(best_run, output_dirs, all_rooms)

        csv_dir = output_dirs["csv"]
        csv_files = (
            [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
            if os.path.exists(csv_dir)
            else []
        )
        assert len(csv_files) >= 1, (
            f"generate_checkpoint_report must produce at least 1 CSV. Found: {csv_files}"
        )

    def test_checkpoint_report_csv_contains_transition_used(self, tmp_path):
        """The schedule CSV produced by generate_checkpoint_report() includes TransitionUsed."""
        _setup_config(tmp_path)
        from core.report_generator import ReportGenerator
        from core.file_manager import FileManager

        rg = ReportGenerator()
        fm = FileManager(base_dir=str(tmp_path / "results"))
        output_dirs = fm.setup_analysis_directories("20260416_120001", "elective")

        schedule = _make_schedule_with_transition(job_id=2, transition_used=9.0)
        best_run = {
            "makespan": 90.0,
            "schedule": schedule,
            "sim_num": 1,
            "algo_name": "GA",
            "job_label_map": None,
        }
        rg.generate_checkpoint_report(best_run, output_dirs, all_rooms=["Pabellon_1"])

        csv_dir = output_dirs["csv"]
        schedule_csvs = [
            os.path.join(csv_dir, f)
            for f in os.listdir(csv_dir)
            if "schedule" in f.lower() and f.endswith(".csv")
        ]
        assert len(schedule_csvs) >= 1, (
            f"Expected at least 1 schedule CSV, got: {schedule_csvs}"
        )

        with open(schedule_csvs[0], newline="", encoding="utf-8") as fh:
            headers = csv.DictReader(fh).fieldnames or []
        assert "TransitionUsed" in headers, (
            f"Schedule CSV must have TransitionUsed. Got: {headers}"
        )

    def test_checkpoint_report_graceful_when_empty_schedule(self, tmp_path):
        """generate_checkpoint_report() does not crash when schedule is empty/None."""
        _setup_config(tmp_path)
        from core.report_generator import ReportGenerator
        from core.file_manager import FileManager

        rg = ReportGenerator()
        fm = FileManager(base_dir=str(tmp_path / "results"))
        output_dirs = fm.setup_analysis_directories("20260416_120002", "elective")

        best_run = {
            "makespan": float("inf"),
            "schedule": [],
            "sim_num": 0,
            "algo_name": "GA",
            "job_label_map": None,
        }
        # Must not raise
        try:
            rg.generate_checkpoint_report(
                best_run, output_dirs, all_rooms=["Pabellon_1"]
            )
        except Exception as exc:
            raise AssertionError(
                f"generate_checkpoint_report must handle empty schedule gracefully. Error: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Task 4.4 — checkpoint reports generated when ANALYSIS_FULL_REPORTS_ENABLED=True
# ---------------------------------------------------------------------------


def _make_full_reports_config(tmp_path, db_path):
    """Config with full_reports_enabled=True."""
    cfg = _make_minimal_config()
    cfg["analysis_mode"] = {
        "enabled": True,
        "full_reports_enabled": True,
        "num_runs": 1,
        "sims_per_run": 2,
        "checkpoint_interval_seconds": 1,
        "sqlite_path": db_path,
        "sweep_enabled": False,
        "sweep_num_procedures": [],
        "sweep_sims_per_x": 2,
        "export_csv_after_run": False,
        "artifact_save_mode": "all",
    }
    cfg_file = tmp_path / "config_full_reports.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if any(
            mod_name.startswith(p)
            for p in [
                "config.",
                "core.simulation_runner",
                "core.analysis_persistence",
                "core.report_generator",
                "simulation.workers.",
            ]
        ):
            del sys.modules[mod_name]


class TestCheckpointReportsIntegration:
    """Task 4.4 — run_elective_analysis_mode() generates checkpoint reports when enabled."""

    def test_generate_checkpoint_report_called_when_full_reports_enabled(
        self, tmp_path
    ):
        """When ANALYSIS_FULL_REPORTS_ENABLED=True, generate_checkpoint_report is invoked at least once."""
        db_path = str(tmp_path / "full_rpts.db")
        _make_full_reports_config(tmp_path, db_path)

        from core.simulation_runner import SimulationRunner
        import core.report_generator as rg_module

        runner = SimulationRunner()
        calls = []
        original_method = rg_module.ReportGenerator.generate_checkpoint_report

        def _spy(self_rg, *args, **kwargs):
            calls.append(True)
            return original_method(self_rg, *args, **kwargs)

        rg_module.ReportGenerator.generate_checkpoint_report = _spy
        try:
            runner.run_elective_analysis_mode()
        finally:
            rg_module.ReportGenerator.generate_checkpoint_report = original_method

        assert len(calls) >= 1, (
            "generate_checkpoint_report must be called at least once when "
            "ANALYSIS_FULL_REPORTS_ENABLED=True. "
            f"Called {len(calls)} times."
        )

    def test_no_checkpoint_reports_when_full_reports_disabled(self, tmp_path):
        """When ANALYSIS_FULL_REPORTS_ENABLED=False (default), generate_checkpoint_report NOT called."""
        cfg = _make_minimal_config()
        cfg["analysis_mode"] = {
            "enabled": True,
            "full_reports_enabled": False,
            "num_runs": 1,
            "sims_per_run": 2,
            "checkpoint_interval_seconds": 1,
            "sqlite_path": str(tmp_path / "no_rpts.db"),
            "sweep_enabled": False,
            "sweep_num_procedures": [],
            "sweep_sims_per_x": 2,
            "export_csv_after_run": False,
            "artifact_save_mode": "all",
        }
        cfg_file = tmp_path / "config_no_rpts.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod_name in list(sys.modules.keys()):
            if any(
                mod_name.startswith(p)
                for p in [
                    "config.",
                    "core.simulation_runner",
                    "core.analysis_persistence",
                    "core.report_generator",
                    "simulation.workers.",
                ]
            ):
                del sys.modules[mod_name]

        from core.simulation_runner import SimulationRunner
        import core.report_generator as rg_module

        runner = SimulationRunner()
        calls = []
        original_method = rg_module.ReportGenerator.generate_checkpoint_report

        def _spy(self_rg, *args, **kwargs):
            calls.append(True)
            return original_method(self_rg, *args, **kwargs)

        rg_module.ReportGenerator.generate_checkpoint_report = _spy
        try:
            runner.run_elective_analysis_mode()
        finally:
            rg_module.ReportGenerator.generate_checkpoint_report = original_method

        assert len(calls) == 0, (
            f"generate_checkpoint_report must NOT be called when full_reports_enabled=False. "
            f"Called {len(calls)} times."
        )


# ---------------------------------------------------------------------------
# Task 5.1/Lote5 — generate_checkpoint_report() must create Gantt plot PNG
# ---------------------------------------------------------------------------


class TestCheckpointReportGanttPlot:
    """Task Lote5 — generate_checkpoint_report() should generate a Gantt PNG plot."""

    def test_checkpoint_report_creates_gantt_plot(self, tmp_path):
        """generate_checkpoint_report() debe crear al menos un archivo PNG (Gantt) en plots_dir."""
        _setup_config(tmp_path)
        from core.report_generator import ReportGenerator
        from core.file_manager import FileManager

        rg = ReportGenerator()
        fm = FileManager(base_dir=str(tmp_path / "results"))
        output_dirs = fm.setup_analysis_directories("20260416_200000", "elective")

        schedule = _make_schedule_with_transition(job_id=1, transition_used=3.0)
        best_run = {
            "makespan": 75.0,
            "schedule": schedule,
            "sim_num": 0,
            "algo_name": "GA",
            "job_label_map": None,
        }
        all_rooms = ["Pabellon_1"]

        rg.generate_checkpoint_report(best_run, output_dirs, all_rooms)

        plots_dir = output_dirs.get("plots", "")
        # PNG may be in a subdirectory (e.g. plots_dir/gantt/)
        plot_files = []
        if plots_dir and os.path.exists(plots_dir):
            for root, dirs, files in os.walk(plots_dir):
                plot_files.extend(f for f in files if f.endswith(".png"))

        assert len(plot_files) >= 1, (
            f"generate_checkpoint_report() debe crear al menos 1 PNG en plots_dir='{plots_dir}'. "
            f"Archivos encontrados: {plot_files}. "
            "Añadir plot_gantt_chart() al checkpoint report para acercarlo al modo normal."
        )

    def test_checkpoint_report_gantt_filename_contains_algo_name(self, tmp_path):
        """El PNG del Gantt debe contener el nombre del algoritmo en el nombre de archivo."""
        _setup_config(tmp_path)
        from core.report_generator import ReportGenerator
        from core.file_manager import FileManager

        rg = ReportGenerator()
        fm = FileManager(base_dir=str(tmp_path / "results"))
        output_dirs = fm.setup_analysis_directories("20260416_200001", "elective")

        schedule = _make_schedule_with_transition(job_id=2, transition_used=7.0)
        best_run = {
            "makespan": 88.0,
            "schedule": schedule,
            "sim_num": 1,
            "algo_name": "GA",
            "job_label_map": None,
        }
        all_rooms = ["Pabellon_1"]

        rg.generate_checkpoint_report(best_run, output_dirs, all_rooms)

        plots_dir = output_dirs.get("plots", "")
        plot_files = []
        if plots_dir and os.path.exists(plots_dir):
            for root, dirs, files in os.walk(plots_dir):
                plot_files.extend(f for f in files if f.endswith(".png"))

        algo_plots = [f for f in plot_files if "ga" in f.lower()]
        assert len(algo_plots) >= 1, (
            f"El PNG del Gantt debe contener 'ga' en su nombre. Archivos PNG: {plot_files}"
        )

    def test_checkpoint_report_creates_convergence_plot(self, tmp_path):
        """generate_checkpoint_report() debe crear un archivo PNG de convergencia si best_hist está presente."""
        _setup_config(tmp_path)
        from core.report_generator import ReportGenerator
        from core.file_manager import FileManager

        rg = ReportGenerator()
        fm = FileManager(base_dir=str(tmp_path / "results"))
        output_dirs = fm.setup_analysis_directories(
            "20260416_200002", "elective", plot_subdirs=fm.CHECKPOINT_PLOT_SUBDIRS
        )

        schedule = _make_schedule_with_transition(job_id=1, transition_used=3.0)
        best_run = {
            "makespan": 75.0,
            "schedule": schedule,
            "sim_num": 0,
            "algo_name": "GA",
            "job_label_map": None,
            "best_hist": [100.0, 90.0, 80.0, 75.0],
            "avg_hist": [110.0, 100.0, 95.0, 90.0],
        }
        all_rooms = ["Pabellon_1"]

        rg.generate_checkpoint_report(best_run, output_dirs, all_rooms)

        plots_dir = output_dirs.get("plots", "")
        convergence_dir = os.path.join(plots_dir, "convergence")
        assert os.path.isdir(convergence_dir), f"El directorio de convergencia {convergence_dir} no fue creado."

        plot_files = [f for f in os.listdir(convergence_dir) if f.endswith(".png")]
        assert len(plot_files) == 1, (
            f"Se esperaba exactamente 1 archivo PNG de convergencia, pero se encontraron: {plot_files}"
        )
        assert "ga" in plot_files[0].lower()

        # Validate that the image file size is > 0 (it was actually created and is not empty)
        img_path = os.path.join(convergence_dir, plot_files[0])
        assert os.path.getsize(img_path) > 0, f"El archivo de imagen {img_path} está vacío."


class TestRoomOvertimeExport:
    """Verifies that the individual room overtime CSV is generated and formatted correctly."""

    def test_export_room_overtimes_csv(self, tmp_path):
        _setup_config(tmp_path)
        from utils.reporting import export_room_overtimes_csv

        all_results = {
            "GA": {
                "makespan": [80.0],
                "solution": [_make_schedule_with_transition(job_id=1, transition_used=5.0)],
                "time": [0.5],
            }
        }
        out_path = str(tmp_path / "room_overtimes.csv")
        rooms = ["Pabellon_1", "Pabellon_2"]

        export_room_overtimes_csv(all_results, out_path, rooms)

        assert os.path.exists(out_path)
        with open(out_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)

        assert headers == ["algorithm", "simulation_id", "Pabellon_1_overtime", "Pabellon_2_overtime"]
        assert len(rows) == 1
        assert rows[0]["algorithm"] == "GA"
        assert rows[0]["simulation_id"] == "0"
        # Makespan is 80, standard shift is 480, so overtime is 0.00
        assert float(rows[0]["Pabellon_1_overtime"]) == 0.0
        assert float(rows[0]["Pabellon_2_overtime"]) == 0.0


class TestRoutingExplanationExport:
    """Verifies that the routing explanation CSV is generated and formatted correctly."""

    def test_export_routing_explanation_csv(self, tmp_path):
        from utils.reporting import export_routing_explanation_csv

        schedule = [
            {
                "Job": 1,
                "Operation": 1,
                "Resource": "Pabellon_1",
                "Personnel": "Anestesiologo_1",
                "Start": 0.0,
                "ProcessingEnd": 30.0,
                "Finish": 30.0,
                "SetupUsed": 5.0,
                "TransitionUsed": 0.0,
                "CleanupUsed": 0.0,
            },
            {
                "Job": 1,
                "Operation": 2,
                "Resource": "Pabellon_1",
                "Personnel": "Cirujano_1",
                "Start": 35.0,
                "ProcessingEnd": 95.0,
                "Finish": 110.0,
                "SetupUsed": 0.0,
                "TransitionUsed": 5.0,
                "CleanupUsed": 15.0,
            }
        ]

        out_path = str(tmp_path / "routing_explanation.csv")
        export_routing_explanation_csv(schedule, out_path)

        assert os.path.exists(out_path)
        with open(out_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)

        expected_headers = [
            "job_id", "operation", "room_assigned", "personnel_assigned",
            "start_time", "processing_start", "finish_time",
            "room_free_before_start", "personnel_free_before_start",
            "patient_ready_before_start", "primary_delay_reason"
        ]
        assert headers == expected_headers
        assert len(rows) == 2
        assert rows[0]["job_id"] == "1"
        assert rows[0]["operation"] == "1"
        assert rows[0]["primary_delay_reason"] == "Immediate / First Block"
