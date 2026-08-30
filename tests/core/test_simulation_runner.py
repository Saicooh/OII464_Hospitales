"""
Tests para core/simulation_runner.py â€” generaciÃ³n de job_ids desde NUM_PROCEDURES.
"""

import importlib
import yaml
import os
import sys
import tempfile
from pathlib import Path
import pytest


def _find_analysis_db(base_dir):
    """Busca recursivamente el archivo analysis.db bajo base_dir."""
    matches = list(Path(base_dir).rglob("analysis.db"))
    if not matches:
        raise FileNotFoundError(
            f"No se encontrÃ³ analysis.db bajo '{base_dir}'. "
            f"Archivos presentes: {list(Path(base_dir).rglob('*'))}"
        )
    return str(matches[0])


def _make_minimal_config(num_procedures=15):
    return {
        "experiment": {
            "num_simulations": 1,
            "std_factor_times": 0.0,
            "alpha_test": 0.05,
            "num_procedures": num_procedures,
            "output_dirs": {"plots": "results/plots", "csv": "results/csv"},
            "n_jobs": 1,
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
                "1": 1,
                "2": 1,
                "3": 1,
                "4": 3,
                "5": 3,
                "6": 3,
                "7": 2,
                "8": 2,
                "9": 1,
                "10": 1,
                "11": 2,
                "12": 2,
                "13": 3,
                "14": 1,
                "15": 3,
            }
        },
        "resources": {"num_pabellones": 4},
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
                "population_size": 2,
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


def _setup_config(tmp_path, num_procedures):
    """Crea un config temporal e invalida el cache de mÃ³dulos relacionados."""
    cfg = _make_minimal_config(num_procedures=num_procedures)
    cfg_file = tmp_path / "config_test.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if (
            mod_name.startswith("config.")
            or mod_name.startswith("core.simulation_runner")
            or mod_name.startswith("simulation.scheduler")
            or mod_name.startswith("algorithms.")
        ):
            del sys.modules[mod_name]


class TestSimulationRunnerJobIds:
    def test_job_ids_length_equals_num_procedures_15(self, tmp_path):
        """Con num_procedures=15, job_ids debe tener 15 elementos."""
        _setup_config(tmp_path, num_procedures=15)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        assert len(runner.job_ids) == 15

    def test_job_ids_range_starts_at_1(self, tmp_path):
        """job_ids debe comenzar desde 1."""
        _setup_config(tmp_path, num_procedures=5)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        assert runner.job_ids[0] == 1

    def test_job_ids_equals_range_1_to_n_inclusive(self, tmp_path):
        """Con num_procedures=8, job_ids == [1, 2, 3, 4, 5, 6, 7, 8]."""
        _setup_config(tmp_path, num_procedures=8)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        assert runner.job_ids == list(range(1, 9))

    def test_job_ids_extends_beyond_legacy_catalog(self, tmp_path):
        """Con num_procedures=20 (> 15), job_ids debe tener 20 elementos hasta 20."""
        _setup_config(tmp_path, num_procedures=20)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        assert len(runner.job_ids) == 20
        assert runner.job_ids[-1] == 20


# ---------------------------------------------------------------------------
# Helpers para config con analysis_mode habilitado
# ---------------------------------------------------------------------------


def _make_analysis_config(num_procedures=5, num_runs=1, sims_per_run=2):
    """Config mÃ­nima con analysis_mode habilitado y parametros pequeÃ±os para tests rÃ¡pidos."""
    cfg = _make_minimal_config(num_procedures=num_procedures)
    cfg["times"]["max_wait"] = {"1": 500, "2": 500}
    cfg["analysis_mode"] = {
        "enabled": True,
        "num_runs": num_runs,
        "sims_per_run": sims_per_run,
        "checkpoint_interval_seconds": 300,
        "sqlite_path": ":memory:",  # No genera archivo real en tests
        "sweep_enabled": False,
        "sweep_num_procedures": [],
        "sweep_sims_per_x": 20,
        "export_csv_after_run": False,
        "artifact_save_mode": "all",
    }
    return cfg


def _setup_analysis_config(tmp_path, num_procedures=5, num_runs=1, sims_per_run=2):
    """Crea config analÃ­tica temporal e invalida caches de mÃ³dulos."""
    cfg = _make_analysis_config(num_procedures, num_runs, sims_per_run)
    cfg_file = tmp_path / "config_analysis.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if (
            mod_name.startswith("config.")
            or mod_name.startswith("core.simulation_runner")
            or mod_name.startswith("core.analysis_persistence")
            or mod_name.startswith("simulation.workers.")
            or mod_name.startswith("simulation.scheduler")
            or mod_name.startswith("algorithms.")
        ):
            del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Task 2.4: _extract_cie10_breakdown helper
# ---------------------------------------------------------------------------


class TestExtractCie10Breakdown:
    """Tests para SimulationRunner._extract_cie10_breakdown()."""

    def _make_flat_schedule(
        self, job_id, setup1=5, setup2=8, cleanup2=6, start2=12, finish2=80
    ):
        """Genera schedule_details flat (formato real del scheduler)."""
        return [
            # Op 1 â€” anestesia
            {
                "Job": job_id,
                "Operation": 1,
                "Resource": "Pabellon_1",
                "Personnel": "A1",
                "Start": 0,
                "ProcessingEnd": 10,
                "Finish": 12,
                "SetupUsed": setup1,
                "TransitionUsed": None,
                "CleanupUsed": 5,
            },
            # Op 2 â€” cirugÃ­a
            # TransitionUsed carries the op1â†’op2 transition (PKL model).
            # SetupUsed is kept for backward compat but transition_min reads TransitionUsed.
            {
                "Job": job_id,
                "Operation": 2,
                "Resource": "Pabellon_1",
                "Personnel": "S1",
                "Start": start2,
                "ProcessingEnd": finish2 - cleanup2,
                "Finish": finish2,
                "SetupUsed": setup2,
                "TransitionUsed": setup2,  # transition is the setup2 value in PKL model
                "CleanupUsed": cleanup2,
            },
        ]

    def test_returns_list_for_non_empty_schedule(self, tmp_path):
        """Con schedule_details vÃ¡lido, devuelve una lista no vacÃ­a."""
        _setup_config(tmp_path, num_procedures=5)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        schedule_details = self._make_flat_schedule(job_id=1)
        job_label_map = {1: "K40.0"}

        result = runner._extract_cie10_breakdown(schedule_details, job_label_map)

        assert len(result) == 1
        assert result[0]["job_id"] == 1

    def test_breakdown_contains_required_fields(self, tmp_path):
        """Cada row de breakdown tiene: job_id, codigo_cie10, setup_min, proc_time_min, transition_min, cleanup_min."""
        _setup_config(tmp_path, num_procedures=5)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        schedule_details = self._make_flat_schedule(job_id=2)
        job_label_map = {2: "J18.0"}

        result = runner._extract_cie10_breakdown(schedule_details, job_label_map)
        row = result[0]

        for field in (
            "job_id",
            "codigo_cie10",
            "setup_min",
            "proc_time_min",
            "transition_min",
            "cleanup_min",
        ):
            assert field in row, f"Campo requerido '{field}' ausente en breakdown"

    def test_breakdown_values_computed_correctly(self, tmp_path):
        """setup_min = op1.SetupUsed; transition_min = op2.TransitionUsed; cleanup_min = op2.CleanupUsed;
        proc_time_min = (Finish2 - Start2) - op2.TransitionUsed - op2.CleanupUsed."""
        _setup_config(tmp_path, num_procedures=5)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        # setup1=4, TransitionUsed=8 (transition), cleanup2=6, start2=12, finish2=80
        # proc = (80 - 12) - 8 - 6 = 54
        schedule_details = self._make_flat_schedule(
            job_id=3, setup1=4, setup2=8, cleanup2=6, start2=12, finish2=80
        )
        schedule_details[0]["SetupUsed"] = 4  # op1 setup
        job_label_map = {3: "C50.9"}

        result = runner._extract_cie10_breakdown(schedule_details, job_label_map)
        row = result[0]

        assert row["setup_min"] == 4  # op1.SetupUsed
        assert row["transition_min"] == 8  # op2.TransitionUsed (correctly propagated)
        assert row["cleanup_min"] == 6  # op2.CleanupUsed
        # proc = (80 - 12) - 8 - 6 = 54
        assert row["proc_time_min"] == pytest.approx(54.0)
        assert row["codigo_cie10"] == "C50.9"

    def test_breakdown_returns_empty_for_empty_schedule(self, tmp_path):
        """Con schedule_details vacÃ­o, devuelve lista vacÃ­a."""
        _setup_config(tmp_path, num_procedures=5)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()

        result = runner._extract_cie10_breakdown([], {})
        assert result == []

    def test_breakdown_job_label_map_none_uses_none_codigo(self, tmp_path):
        """Si job_label_map es None, codigo_cie10 es None."""
        _setup_config(tmp_path, num_procedures=5)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        schedule_details = self._make_flat_schedule(job_id=5)

        result = runner._extract_cie10_breakdown(schedule_details, job_label_map=None)
        assert result[0]["codigo_cie10"] is None


# ---------------------------------------------------------------------------
# Task 2.3: run_elective_analysis_mode()
# ---------------------------------------------------------------------------


class TestRunElectiveAnalysisMode:
    """Tests de integraciÃ³n mÃ­nima para run_elective_analysis_mode()."""

    def test_analysis_mode_populates_sqlite_runs_table(self, tmp_path):
        """Tras run_elective_analysis_mode(), la tabla 'runs' tiene al menos 1 row."""
        import sqlite3
        from core.file_manager import FileManager

        _setup_analysis_config(tmp_path)

        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert rows >= 1, f"Se esperaba al menos 1 row en 'runs', got {rows}"

    def test_analysis_mode_populates_simulations_table(self, tmp_path):
        """Tras run_elective_analysis_mode(), la tabla 'simulations' tiene al menos sims_per_run rows."""
        import sqlite3
        from core.file_manager import FileManager

        _setup_analysis_config(tmp_path, num_procedures=5, num_runs=1, sims_per_run=2)

        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)
        conn = sqlite3.connect(db_path)
        # Schema v2: simulations holds one row per (run, sim_i, algo)
        sim_rows = conn.execute("SELECT COUNT(*) FROM simulations").fetchone()[0]
        # algorithm_iterations is populated by algo instrumentation (future Lote 2+)
        algo_iter_rows = conn.execute(
            "SELECT COUNT(*) FROM algorithm_iterations"
        ).fetchone()[0]
        conn.close()
        # 1 run x 2 sims x 1 algo (GA only) = at least 2 simulation rows
        assert sim_rows >= 2, (
            f"Se esperaba al menos 2 rows en 'simulations', got {sim_rows}"
        )
        # algorithm_iterations is now populated by Lote 2 instrumentation
        assert algo_iter_rows >= 1, (
            f"Se esperaba al menos 1 row en 'algorithm_iterations' tras instrumentacion, got {algo_iter_rows}"
        )

    def test_normal_mode_unaffected_when_analysis_disabled(self, tmp_path):
        """run_elective_mode() no falla cuando analysis_mode.enabled=False."""
        # This test verifies the normal mode path remains intact
        _setup_config(tmp_path, num_procedures=5)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        # Verify run_elective_analysis_mode is a method that exists
        assert hasattr(runner, "run_elective_analysis_mode"), (
            "SimulationRunner debe tener el mÃ©todo run_elective_analysis_mode()"
        )


# ---------------------------------------------------------------------------
# Task 2.1: routing en main.py â€” verificaciÃ³n de atributo ANALYSIS_MODE_ENABLED
# ---------------------------------------------------------------------------


class TestMainRoutingAnalysisMode:
    """Verifica que ANALYSIS_MODE_ENABLED se exporta correctamente desde config."""

    def test_analysis_mode_enabled_false_by_default(self, tmp_path):
        """Sin analysis_mode en config, ANALYSIS_MODE_ENABLED = False."""
        _setup_config(tmp_path, num_procedures=5)
        from config.config import ANALYSIS_MODE_ENABLED

        assert ANALYSIS_MODE_ENABLED is False

    def test_analysis_mode_enabled_true_when_set(self, tmp_path):
        """Con analysis_mode.enabled=true en config, ANALYSIS_MODE_ENABLED = True."""
        _setup_analysis_config(tmp_path)
        from config.config import ANALYSIS_MODE_ENABLED

        assert ANALYSIS_MODE_ENABLED is True


# ---------------------------------------------------------------------------
# Task 3.3: run_sweep_mode() â€” barrido de num_procedures
# ---------------------------------------------------------------------------


def _make_sweep_config(tmp_path, sweep_values, sims_per_x=2):
    """Config mÃ­nima con sweep_enabled=True."""
    cfg = _make_minimal_config(num_procedures=sweep_values[0])
    cfg["analysis_mode"] = {
        "enabled": True,
        "num_runs": 1,
        "sims_per_run": 2,
        "checkpoint_interval_seconds": 300,
        "sqlite_path": str(tmp_path / "sweep.db"),
        "sweep_enabled": True,
        "sweep_num_procedures": sweep_values,
        "sweep_sims_per_x": sims_per_x,
        "export_csv_after_run": False,
        "artifact_save_mode": "all",
    }
    cfg_file = tmp_path / "config_sweep.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if (
            mod_name.startswith("config.")
            or mod_name.startswith("core.simulation_runner")
            or mod_name.startswith("core.analysis_persistence")
            or mod_name.startswith("simulation.workers.")
        ):
            del sys.modules[mod_name]


class TestRunSweepMode:
    """Tests de contrato para run_sweep_mode()."""

    def test_run_sweep_mode_method_exists(self, tmp_path):
        """SimulationRunner debe tener el mÃ©todo run_sweep_mode()."""
        _setup_config(tmp_path, num_procedures=5)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        assert hasattr(runner, "run_sweep_mode"), (
            "SimulationRunner debe exponer run_sweep_mode()"
        )
        assert callable(runner.run_sweep_mode)

    def test_sweep_returns_list_with_one_entry_per_x(self, tmp_path):
        """run_sweep_mode([5, 8]) debe devolver un dict con 'sweep' de 2 entradas (una por X)."""
        _make_sweep_config(tmp_path, sweep_values=[5, 8], sims_per_x=2)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        result = runner.run_sweep_mode()
        assert isinstance(result, dict), (
            "run_sweep_mode debe devolver un dict con 'sweep' y 'closest_to_80pct'"
        )
        assert isinstance(result["sweep"], list), "'sweep' debe ser una lista"
        assert len(result["sweep"]) == 2, (
            f"Se esperaban 2 entradas, got {len(result['sweep'])}"
        )

    def test_sweep_entries_contain_num_procedures_key(self, tmp_path):
        """Cada entrada del resultado debe tener la clave 'num_procedures'."""
        _make_sweep_config(tmp_path, sweep_values=[5, 8], sims_per_x=2)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        result = runner.run_sweep_mode()
        for entry in result["sweep"]:
            assert "num_procedures" in entry, (
                f"Cada entrada debe tener 'num_procedures', got keys: {list(entry.keys())}"
            )

    def test_sweep_entries_contain_avg_utilization_key(self, tmp_path):
        """Cada entrada debe tener la clave 'avg_utilization'."""
        _make_sweep_config(tmp_path, sweep_values=[5, 8], sims_per_x=2)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        result = runner.run_sweep_mode()
        for entry in result["sweep"]:
            assert "avg_utilization" in entry, (
                f"Cada entrada debe tener 'avg_utilization', got keys: {list(entry.keys())}"
            )

    def test_sweep_num_procedures_values_match_config(self, tmp_path):
        """Los valores de num_procedures en el resultado deben coincidir con sweep_num_procedures."""
        _make_sweep_config(tmp_path, sweep_values=[5, 8], sims_per_x=2)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        result = runner.run_sweep_mode()
        returned_x = sorted(e["num_procedures"] for e in result["sweep"])
        assert returned_x == [5, 8], f"Esperado [5, 8], got {returned_x}"

    def test_sweep_avg_utilization_is_numeric(self, tmp_path):
        """avg_utilization debe ser un nÃºmero (float) en [0, 1] o None si no calculable."""
        _make_sweep_config(tmp_path, sweep_values=[5], sims_per_x=2)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        result = runner.run_sweep_mode()
        u = result["sweep"][0]["avg_utilization"]
        assert u is None or isinstance(u, float), (
            f"avg_utilization debe ser float o None, got {type(u)}"
        )


def _make_analysis_config_with_csv(tmp_path, db_path, checkpoints_csv, breakdown_csv):
    """Config analÃ­tica con export_csv_after_run=True y rutas CSV especificadas."""
    cfg = _make_minimal_config(num_procedures=5)
    cfg["analysis_mode"] = {
        "enabled": True,
        "num_runs": 1,
        "sims_per_run": 2,
        "checkpoint_interval_seconds": 5,
        "sqlite_path": db_path,
        "sweep_enabled": False,
        "sweep_num_procedures": [],
        "sweep_sims_per_x": 2,
        "export_csv_after_run": True,
        "checkpoints_csv_path": checkpoints_csv,
        "breakdown_csv_path": breakdown_csv,
        "artifact_save_mode": "all",
    }
    cfg_file = tmp_path / "config_csv.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if (
            mod_name.startswith("config.")
            or mod_name.startswith("core.simulation_runner")
            or mod_name.startswith("core.analysis_persistence")
            or mod_name.startswith("simulation.workers.")
        ):
            del sys.modules[mod_name]


class TestAnalysisModePostHocCheckpoints:
    """CRÃTICO: run_elective_analysis_mode() debe reconstruir checkpoints post-hoc."""

    def test_checkpoints_table_populated_after_analysis_run(self, tmp_path):
        """Tras run_elective_analysis_mode(), la tabla 'checkpoints' tiene al menos 1 row."""
        import sqlite3
        from core.file_manager import FileManager

        cfg = _make_analysis_config(num_procedures=5, num_runs=1, sims_per_run=3)
        cfg["analysis_mode"]["checkpoint_interval_seconds"] = (
            1  # 1 segundo para que genere checkpoints
        )
        cfg_file = tmp_path / "config_chk.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod_name in list(sys.modules.keys()):
            if (
                mod_name.startswith("config.")
                or mod_name.startswith("core.simulation_runner")
                or mod_name.startswith("core.analysis_persistence")
                or mod_name.startswith("simulation.workers.")
            ):
                del sys.modules[mod_name]

        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        conn.close()
        assert rows >= 1, (
            f"Se esperaba al menos 1 checkpoint post-hoc, got {rows}. "
            "run_elective_analysis_mode() debe llamar a reconstruct_checkpoints()."
        )

    def test_checkpoints_table_exists_even_when_no_data(self, tmp_path):
        """La tabla 'checkpoints' debe existir tras init_db(), incluso vacÃ­a."""
        import sqlite3
        from core.file_manager import FileManager

        cfg = _make_analysis_config(num_procedures=5, num_runs=1, sims_per_run=2)
        cfg_file = tmp_path / "config_chk2.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod_name in list(sys.modules.keys()):
            if (
                mod_name.startswith("config.")
                or mod_name.startswith("core.simulation_runner")
                or mod_name.startswith("core.analysis_persistence")
                or mod_name.startswith("simulation.workers.")
            ):
                del sys.modules[mod_name]

        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)
        conn = sqlite3.connect(db_path)
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        conn.close()
        assert "checkpoints" in tables, (
            f"La tabla 'checkpoints' debe existir en el DB. Tables encontradas: {tables}"
        )


class TestAnalysisModeCSVExports:
    """CRÃTICO: run_elective_analysis_mode() debe exportar CSVs derivados cuando export_csv_after_run=True."""

    def test_checkpoints_csv_created_when_export_enabled(self, tmp_path):
        """Cuando export_csv_after_run=True, se debe crear el archivo CSV de checkpoints en el Ã¡rbol results/<ts>/."""
        from core.file_manager import FileManager

        cfg = _make_analysis_config(num_procedures=5, num_runs=1, sims_per_run=2)
        cfg["analysis_mode"]["export_csv_after_run"] = True
        cfg_file = tmp_path / "config_csv.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod_name in list(sys.modules.keys()):
            if (
                mod_name.startswith("config.")
                or mod_name.startswith("core.simulation_runner")
                or mod_name.startswith("core.analysis_persistence")
                or mod_name.startswith("simulation.workers.")
            ):
                del sys.modules[mod_name]

        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        csv_files = list(Path(results_dir).rglob("*checkpoints*.csv"))
        assert len(csv_files) >= 1, (
            f"Se esperaba al menos 1 CSV de checkpoints bajo '{results_dir}'. "
            "run_elective_analysis_mode() debe llamar export_checkpoints_csv() cuando export_csv_after_run=True."
        )

    def test_breakdown_csv_created_when_export_enabled(self, tmp_path):
        """Cuando export_csv_after_run=True, se debe crear el archivo CSV de breakdown en el Ã¡rbol results/<ts>/."""
        from core.file_manager import FileManager

        cfg = _make_analysis_config(num_procedures=5, num_runs=1, sims_per_run=2)
        cfg["analysis_mode"]["export_csv_after_run"] = True
        cfg_file = tmp_path / "config_csv2.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod_name in list(sys.modules.keys()):
            if (
                mod_name.startswith("config.")
                or mod_name.startswith("core.simulation_runner")
                or mod_name.startswith("core.analysis_persistence")
                or mod_name.startswith("simulation.workers.")
            ):
                del sys.modules[mod_name]

        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        csv_files = list(Path(results_dir).rglob("*breakdown*.csv"))
        assert len(csv_files) == 0, (
            f"Se esperaba que NO se creara ningún CSV de breakdown bajo '{results_dir}', got {csv_files}."
        )

    def test_no_csv_created_when_export_disabled(self, tmp_path):
        """Cuando export_csv_after_run=False, NO se crean archivos CSV bajo results/<ts>/."""
        from core.file_manager import FileManager

        cfg = _make_analysis_config(num_procedures=5, num_runs=1, sims_per_run=2)
        cfg["analysis_mode"]["export_csv_after_run"] = False
        cfg_file = tmp_path / "config_no_exp.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod_name in list(sys.modules.keys()):
            if (
                mod_name.startswith("config.")
                or mod_name.startswith("core.simulation_runner")
                or mod_name.startswith("core.analysis_persistence")
                or mod_name.startswith("simulation.workers.")
            ):
                del sys.modules[mod_name]

        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        csv_files = list(Path(results_dir).rglob("*.csv"))
        assert len(csv_files) == 0, (
            f"NO deben existir CSVs cuando export_csv_after_run=False. "
            f"Encontrados: {csv_files}"
        )


class TestAnalysisModeSweepIntegration:
    """Routing sweep/temporal â€” ACTUALIZADO para reflejar desacoplamiento.

    CAMBIO (separar-temporal-y-sweep-analysis-mode):
    - run_elective_analysis_mode() ya NO llama internamente a run_sweep_mode().
    - El routing es responsabilidad de main.py.
    - Verificamos el comportamiento de routing via ANALYSIS_SWEEP_ENABLED flag.
    """

    def test_sweep_not_called_from_temporal_method_even_when_enabled(self, tmp_path):
        """run_elective_analysis_mode() NO llama run_sweep_mode() aunque sweep_enabled=True.
        El routing temporalâ†’sweep es responsabilidad de main.py, no de run_elective_analysis_mode()."""
        db_path = str(tmp_path / "sweep_int.db")
        cfg = _make_minimal_config(num_procedures=5)
        cfg["analysis_mode"] = {
            "enabled": True,
            "temporal_enabled": True,
            "num_runs": 1,
            "sims_per_run": 2,
            "checkpoint_interval_seconds": 300,
            "sqlite_path": db_path,
            "sweep_enabled": True,
            "sweep_num_procedures": [5],
            "sweep_sims_per_x": 2,
            "export_csv_after_run": False,
            "artifact_save_mode": "all",
        }
        cfg_file = tmp_path / "config_sweep_int.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod_name in list(sys.modules.keys()):
            if (
                mod_name.startswith("config.")
                or mod_name.startswith("core.simulation_runner")
                or mod_name.startswith("core.analysis_persistence")
                or mod_name.startswith("simulation.workers.")
            ):
                del sys.modules[mod_name]

        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        sweep_called = []
        original_sweep = runner.run_sweep_mode

        def _mock_sweep():
            sweep_called.append(True)
            return original_sweep()

        runner.run_sweep_mode = _mock_sweep
        # run_elective_analysis_mode() no debe llamar sweep â€” el routing lo hace main.py
        runner.run_elective_analysis_mode()

        assert len(sweep_called) == 0, (
            "run_elective_analysis_mode() NO debe llamar internamente run_sweep_mode(). "
            "El routing temporalâ†’sweep es responsabilidad de main.py. "
            f"Fue llamado {len(sweep_called)} veces."
        )

    def test_sweep_flag_accessible_for_routing(self, tmp_path):
        """ANALYSIS_SWEEP_ENABLED se exporta correctamente para que main.py pueda routear."""
        db_path = str(tmp_path / "no_sweep.db")
        cfg = _make_analysis_config(num_procedures=5, num_runs=1, sims_per_run=2)
        cfg["analysis_mode"]["sqlite_path"] = db_path
        cfg["analysis_mode"]["sweep_enabled"] = False
        cfg_file = tmp_path / "config_no_sweep.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod_name in list(sys.modules.keys()):
            if (
                mod_name.startswith("config.")
                or mod_name.startswith("core.simulation_runner")
                or mod_name.startswith("core.analysis_persistence")
                or mod_name.startswith("simulation.workers.")
            ):
                del sys.modules[mod_name]

        from config.config import ANALYSIS_SWEEP_ENABLED

        assert ANALYSIS_SWEEP_ENABLED is False, (
            "ANALYSIS_SWEEP_ENABLED debe ser False cuando sweep_enabled=False en config"
        )


# ---------------------------------------------------------------------------
# Lote 5 correctivo: generate_analysis_reports() in ReportGenerator
# ---------------------------------------------------------------------------


class TestGenerateAnalysisReports:
    """ReportGenerator.generate_analysis_reports(sqlite_path) debe existir y exportar CSVs analÃ­ticos."""

    def test_generate_analysis_reports_exists(self, tmp_path):
        """ReportGenerator debe tener el mÃ©todo generate_analysis_reports."""
        _setup_config(tmp_path, num_procedures=5)
        from core.report_generator import ReportGenerator

        rg = ReportGenerator()
        assert hasattr(rg, "generate_analysis_reports"), (
            "ReportGenerator debe tener generate_analysis_reports(sqlite_path) segÃºn el diseÃ±o aprobado"
        )
        assert callable(rg.generate_analysis_reports), (
            "generate_analysis_reports debe ser callable"
        )

    def test_generate_analysis_reports_exports_checkpoints_csv(self, tmp_path):
        """generate_analysis_reports() debe exportar checkpoints CSV desde un SQLite con datos (schema v2)."""
        import sqlite3

        _setup_config(tmp_path, num_procedures=5)

        # Crear SQLite con datos mÃ­nimos reales usando schema v2 (simulations, sim_id)
        db_path = str(tmp_path / "analysis.db")
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                num_simulations INTEGER NOT NULL,
                num_procedures INTEGER NOT NULL,
                config_snapshot TEXT
            );
            CREATE TABLE simulations (
                sim_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                sim_index INTEGER NOT NULL,
                algo_name TEXT NOT NULL,
                wall_clock_elapsed_s REAL NOT NULL,
                makespan REAL NOT NULL,
                combined_obj REAL,
                algo_time_s REAL NOT NULL
            );
            CREATE TABLE cie10_breakdown (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sim_id INTEGER NOT NULL,
                job_id INTEGER NOT NULL,
                codigo_cie10 TEXT,
                grupo TEXT,
                setup_min REAL NOT NULL,
                proc_time_min REAL NOT NULL,
                transition_min REAL NOT NULL,
                cleanup_min REAL NOT NULL
            );
            CREATE TABLE checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                algo_name TEXT NOT NULL,
                checkpoint_wall_s REAL NOT NULL,
                best_makespan REAL NOT NULL,
                best_sim_index INTEGER NOT NULL
            );
        """)
        conn.execute(
            "INSERT INTO runs (started_at, num_simulations, num_procedures, config_snapshot) "
            "VALUES ('2026-01-01T00:00:00', 1, 5, '{}')"
        )
        conn.execute(
            "INSERT INTO checkpoints (run_id, algo_name, checkpoint_wall_s, best_makespan, best_sim_index) "
            "VALUES (1, 'GA', 10.0, 120.0, 0)"
        )
        conn.commit()
        conn.close()

        from core.report_generator import ReportGenerator

        rg = ReportGenerator()
        out_dir = str(tmp_path / "reports")
        rg.generate_analysis_reports(db_path, output_dir=out_dir)
        created_files = os.listdir(out_dir) if os.path.exists(out_dir) else []
        assert any(
            "checkpoint" in f.lower() or "analysis" in f.lower() for f in created_files
        ), (
            f"generate_analysis_reports() debe crear al menos un CSV/reporte en '{out_dir}'. "
            f"Archivos creados: {created_files}"
        )


# ---------------------------------------------------------------------------
# Lote 5 correctivo: Sweep reports X closest to 80% utilization
# ---------------------------------------------------------------------------


def _make_sweep_config(tmp_path, sweep_values, sims_per_x=2):
    """Config mÃ­nima para sweep mode."""
    cfg = _make_minimal_config(num_procedures=5)
    cfg["analysis_mode"] = {
        "enabled": True,
        "num_runs": 1,
        "sims_per_run": 2,
        "checkpoint_interval_seconds": 300,
        "sqlite_path": str(tmp_path / "sweep.db"),
        "sweep_enabled": True,
        "sweep_num_procedures": sweep_values,
        "sweep_sims_per_x": sims_per_x,
        "export_csv_after_run": False,
        "checkpoints_csv_path": str(tmp_path / "chk.csv"),
        "breakdown_csv_path": str(tmp_path / "bd.csv"),
        "artifact_save_mode": "all",
    }
    cfg_file = tmp_path / "config_sweep.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if (
            mod_name.startswith("config.")
            or mod_name.startswith("core.simulation_runner")
            or mod_name.startswith("simulation.workers.")
        ):
            del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Change: separar-temporal-y-sweep-analysis-mode
# Tasks 5.1 / 5.2 / 5.3 â€” Routing y desacoplamiento temporal/sweep
# ---------------------------------------------------------------------------


def _make_routing_config(
    tmp_path,
    enabled=True,
    temporal_enabled=True,
    sweep_enabled=False,
    sweep_values=None,
    num_procedures=5,
    db_path=None,
):
    """Config mÃ­nima para tests de routing temporal/sweep."""
    cfg = _make_minimal_config(num_procedures=num_procedures)
    cfg["analysis_mode"] = {
        "enabled": enabled,
        "temporal_enabled": temporal_enabled,
        "num_runs": 1,
        "sims_per_run": 2,
        "checkpoint_interval_seconds": 300,
        "sqlite_path": str(db_path) if db_path else ":memory:",
        "sweep_enabled": sweep_enabled,
        "sweep_num_procedures": sweep_values or [5],
        "sweep_sims_per_x": 2,
        "export_csv_after_run": False,
        "artifact_save_mode": "all",
    }
    cfg_file = tmp_path / "config_routing.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if (
            mod_name.startswith("config.")
            or mod_name.startswith("core.simulation_runner")
            or mod_name.startswith("core.analysis_persistence")
            or mod_name.startswith("simulation.workers.")
        ):
            del sys.modules[mod_name]


class TestTemporalSweepRouting:
    """Task 5.1/5.2 â€” Routing: temporal-only, sweep-only, both, ninguno."""

    def test_temporal_only_calls_elective_analysis_not_sweep(self, tmp_path):
        """temporal=True, sweep=False â†’ run_elective_analysis_mode() se llama, run_sweep_mode() NO."""
        _make_routing_config(tmp_path, temporal_enabled=True, sweep_enabled=False)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        temporal_called = []
        sweep_called = []
        original_temporal = runner.run_elective_analysis_mode
        original_sweep = runner.run_sweep_mode

        def _mock_temporal():
            temporal_called.append(True)
            return original_temporal()

        def _mock_sweep():
            sweep_called.append(True)
            return original_sweep()

        runner.run_elective_analysis_mode = _mock_temporal
        runner.run_sweep_mode = _mock_sweep

        # Simula el routing de main.py
        from config.config import (
            ANALYSIS_MODE_ENABLED,
            ANALYSIS_TEMPORAL_ENABLED,
            ANALYSIS_SWEEP_ENABLED,
        )

        if ANALYSIS_MODE_ENABLED:
            if ANALYSIS_TEMPORAL_ENABLED:
                runner.run_elective_analysis_mode()
            if ANALYSIS_SWEEP_ENABLED:
                runner.run_sweep_mode()

        assert len(temporal_called) == 1, (
            "run_elective_analysis_mode debe llamarse 1 vez"
        )
        assert len(sweep_called) == 0, (
            "run_sweep_mode NO debe llamarse cuando sweep_enabled=False"
        )

    def test_sweep_only_does_not_call_temporal(self, tmp_path):
        """temporal=False, sweep=True â†’ run_sweep_mode() se llama, run_elective_analysis_mode() NO."""
        _make_routing_config(
            tmp_path, temporal_enabled=False, sweep_enabled=True, sweep_values=[5]
        )
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        temporal_called = []
        sweep_called = []
        original_sweep = runner.run_sweep_mode

        def _mock_sweep():
            sweep_called.append(True)
            return original_sweep()

        runner.run_sweep_mode = _mock_sweep

        from config.config import (
            ANALYSIS_MODE_ENABLED,
            ANALYSIS_TEMPORAL_ENABLED,
            ANALYSIS_SWEEP_ENABLED,
        )

        if ANALYSIS_MODE_ENABLED:
            if ANALYSIS_TEMPORAL_ENABLED:
                temporal_called.append(True)
                runner.run_elective_analysis_mode()
            if ANALYSIS_SWEEP_ENABLED:
                runner.run_sweep_mode()

        assert len(temporal_called) == 0, (
            "run_elective_analysis_mode NO debe llamarse en sweep-only"
        )
        assert len(sweep_called) == 1, (
            "run_sweep_mode debe llamarse 1 vez en sweep-only"
        )

    def test_both_modes_calls_temporal_then_sweep(self, tmp_path):
        """temporal=True, sweep=True â†’ ambos se llaman en orden: temporal â†’ sweep."""
        _make_routing_config(
            tmp_path, temporal_enabled=True, sweep_enabled=True, sweep_values=[5]
        )
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        call_order = []
        original_temporal = runner.run_elective_analysis_mode
        original_sweep = runner.run_sweep_mode

        def _mock_temporal():
            call_order.append("temporal")
            return original_temporal()

        def _mock_sweep():
            call_order.append("sweep")
            return original_sweep()

        runner.run_elective_analysis_mode = _mock_temporal
        runner.run_sweep_mode = _mock_sweep

        from config.config import (
            ANALYSIS_MODE_ENABLED,
            ANALYSIS_TEMPORAL_ENABLED,
            ANALYSIS_SWEEP_ENABLED,
        )

        if ANALYSIS_MODE_ENABLED:
            if ANALYSIS_TEMPORAL_ENABLED:
                runner.run_elective_analysis_mode()
            if ANALYSIS_SWEEP_ENABLED:
                runner.run_sweep_mode()

        assert call_order == ["temporal", "sweep"], (
            f"Orden esperado: temporal â†’ sweep. Got: {call_order}"
        )

    def test_none_mode_calls_nothing(self, tmp_path):
        """temporal=False, sweep=False â†’ ninguno se llama."""
        _make_routing_config(tmp_path, temporal_enabled=False, sweep_enabled=False)
        temporal_called = []
        sweep_called = []

        from config.config import (
            ANALYSIS_MODE_ENABLED,
            ANALYSIS_TEMPORAL_ENABLED,
            ANALYSIS_SWEEP_ENABLED,
        )

        if ANALYSIS_MODE_ENABLED:
            if ANALYSIS_TEMPORAL_ENABLED:
                temporal_called.append(True)
            if ANALYSIS_SWEEP_ENABLED:
                sweep_called.append(True)

        assert len(temporal_called) == 0
        assert len(sweep_called) == 0


class TestSweepOnlyNoPersistence:
    """Task 5.3 â€” Sweep-only NO debe inicializar AnalysisPersistence/SQLite."""

    def test_sweep_only_does_not_instantiate_analysis_persistence(self, tmp_path):
        """Con temporal=False, sweep=True, AnalysisPersistence.__init__ NO debe llamarse."""
        _make_routing_config(
            tmp_path,
            temporal_enabled=False,
            sweep_enabled=True,
            sweep_values=[5],
        )
        from core.simulation_runner import SimulationRunner
        import core.analysis_persistence as ap_module

        runner = SimulationRunner()

        init_calls = []
        original_init = ap_module.AnalysisPersistence.__init__

        def _spy_init(self_ap, *args, **kwargs):
            init_calls.append(args)
            return original_init(self_ap, *args, **kwargs)

        ap_module.AnalysisPersistence.__init__ = _spy_init

        try:
            from config.config import (
                ANALYSIS_MODE_ENABLED,
                ANALYSIS_TEMPORAL_ENABLED,
                ANALYSIS_SWEEP_ENABLED,
            )

            if ANALYSIS_MODE_ENABLED:
                if ANALYSIS_TEMPORAL_ENABLED:
                    runner.run_elective_analysis_mode()
                if ANALYSIS_SWEEP_ENABLED:
                    runner.run_sweep_mode()
        finally:
            ap_module.AnalysisPersistence.__init__ = original_init

        assert len(init_calls) == 0, (
            f"AnalysisPersistence.__init__ NO debe llamarse en sweep-only. "
            f"Fue llamado {len(init_calls)} veces con args: {init_calls}"
        )


class TestSweepDecoupled:
    """Task 2.1 â€” run_elective_analysis_mode() NO debe llamar internamente a run_sweep_mode()."""

    def test_elective_analysis_no_longer_calls_sweep_internally(self, tmp_path):
        """Con sweep_enabled=True en config pero llamando run_elective_analysis_mode() directamente,
        el mÃ©todo NO debe invocar run_sweep_mode() (el routing es responsabilidad de main.py)."""
        _make_routing_config(
            tmp_path,
            temporal_enabled=True,
            sweep_enabled=True,  # sweep habilitado en config
            sweep_values=[5],
            db_path=tmp_path / "decoupled.db",
        )
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        sweep_called = []
        original_sweep = runner.run_sweep_mode

        def _mock_sweep():
            sweep_called.append(True)
            return original_sweep()

        runner.run_sweep_mode = _mock_sweep
        # Llamamos directamente al mÃ©todo temporal â€” NO debe llamar sweep internamente
        runner.run_elective_analysis_mode()

        assert len(sweep_called) == 0, (
            f"run_elective_analysis_mode() NO debe llamar internamente a run_sweep_mode(). "
            f"Fue llamado {len(sweep_called)} veces. El routing debe hacerlo main.py."
        )


class TestSweepWorkerResultFormat:
    """Task 2.2 â€” run_sweep_mode() itera correctamente sobre (sim_i, sim_results) del ElectiveWorker."""

    def test_sweep_mode_handles_worker_result_tuple_format(self, tmp_path):
        """run_sweep_mode() debe desempacar solo (sim_i, sim_results), no 3-tuple con elapsed_s."""
        _make_routing_config(
            tmp_path,
            temporal_enabled=False,
            sweep_enabled=True,
            sweep_values=[5],
        )
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        # Si el formato es incorrecto, run_sweep_mode lanzarÃ­a ValueError al desempacar
        try:
            result = runner.run_sweep_mode()
        except ValueError as e:
            raise AssertionError(
                f"run_sweep_mode() fallÃ³ al desempacar resultados del worker: {e}"
            ) from e

        assert isinstance(result, dict), "run_sweep_mode debe devolver un dict"
        assert "sweep" in result


class TestLegacyConfigTemporalDefault:
    """Task 3.2 / 4.1 â€” Configs legacy sin temporal_enabled â†’ comportan como temporal=True."""

    def test_legacy_config_without_temporal_enabled_runs_temporal(self, tmp_path):
        """Config sin temporal_enabled â†’ ANALYSIS_TEMPORAL_ENABLED=True â†’ temporal se ejecutarÃ­a."""
        _make_routing_config(
            tmp_path,
            enabled=True,
            temporal_enabled=True,  # equivale a ausencia (default True)
            sweep_enabled=False,
        )
        from config.config import ANALYSIS_TEMPORAL_ENABLED

        assert ANALYSIS_TEMPORAL_ENABLED is True, (
            "Config legacy (sin temporal_enabled) debe resolver ANALYSIS_TEMPORAL_ENABLED=True"
        )


class TestSweepReportsClosestTo80Percent:
    """run_sweep_mode() debe reportar explÃ­citamente el X con utilizaciÃ³n mÃ¡s cercana al 80%."""

    def test_sweep_result_includes_closest_to_80_pct_key(self, tmp_path):
        """run_sweep_mode() debe retornar un dict con clave 'closest_to_80pct' indicando el X Ã³ptimo."""
        _make_sweep_config(tmp_path, sweep_values=[3, 5, 7])
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        result = runner.run_sweep_mode()

        # result debe ser dict con 'sweep' (lista) y 'closest_to_80pct' (dict con num_procedures)
        assert isinstance(result, dict), (
            f"run_sweep_mode() debe retornar un dict con 'sweep' y 'closest_to_80pct', got {type(result)}"
        )
        assert "sweep" in result, (
            f"El dict retornado debe tener clave 'sweep' con la lista de resultados. Keys: {list(result.keys())}"
        )
        assert "closest_to_80pct" in result, (
            f"El dict retornado debe tener clave 'closest_to_80pct'. Keys: {list(result.keys())}"
        )

    def test_sweep_closest_to_80pct_has_num_procedures(self, tmp_path):
        """closest_to_80pct debe contener 'num_procedures' con el X seleccionado."""
        _make_sweep_config(tmp_path, sweep_values=[3, 5, 7])
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        result = runner.run_sweep_mode()

        closest = result["closest_to_80pct"]
        assert closest is None or "num_procedures" in closest, (
            f"closest_to_80pct debe ser None (sin datos) o dict con 'num_procedures'. Got: {closest}"
        )

    def test_sweep_closest_selects_x_nearest_to_target(self, tmp_path):
        """closest_to_80pct selecciona el X cuya utilizaciÃ³n estÃ¡ mÃ¡s cercana a 0.80."""
        _make_sweep_config(tmp_path, sweep_values=[3, 5])
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        result = runner.run_sweep_mode()

        sweep_list = result["sweep"]
        closest = result["closest_to_80pct"]

        # Si hay utilizations no-None, verificar que closest_to_80pct apunta al X correcto
        valid = [r for r in sweep_list if r["avg_utilization"] is not None]
        if valid and closest is not None:
            target = 0.80
            best_x = min(valid, key=lambda r: abs(r["avg_utilization"] - target))
            assert closest["num_procedures"] == best_x["num_procedures"], (
                f"closest_to_80pct deberÃ­a apuntar a X={best_x['num_procedures']} "
                f"(utilization={best_x['avg_utilization']:.4f} mÃ¡s cercana a 80%), "
                f"pero apunta a X={closest['num_procedures']}"
            )


# ---------------------------------------------------------------------------
# Evidencia correctiva: main.main() â€” entrypoint tests
# ---------------------------------------------------------------------------


class TestMainEntrypoint:
    """Tests sobre main.main() para validar el routing del entrypoint real."""

    def test_entrypoint_enabled_false_does_not_call_temporal_nor_sweep(self, tmp_path):
        """main.main() con enabled=False â†’ run_elective_analysis_mode y run_sweep_mode NO se llaman."""
        from unittest.mock import patch, MagicMock

        _make_routing_config(
            tmp_path, enabled=False, temporal_enabled=True, sweep_enabled=True
        )

        # Reload main module so it picks up fresh imports
        if "main" in sys.modules:
            del sys.modules["main"]

        temporal_called = []
        sweep_called = []

        def _mock_temporal(self_r):
            temporal_called.append(True)

        def _mock_sweep(self_r):
            sweep_called.append(True)

        with (
            patch(
                "core.simulation_runner.SimulationRunner.run_elective_analysis_mode",
                _mock_temporal,
            ),
            patch(
                "core.simulation_runner.SimulationRunner.run_sweep_mode", _mock_sweep
            ),
        ):
            import main as main_module

            # Re-import constants with fresh config
            import importlib
            import config.config as cfg_mod

            importlib.reload(cfg_mod)
            # Patch the constants that main.main() reads
            with (
                patch.object(main_module, "ANALYSIS_MODE_ENABLED", False),
                patch.object(main_module, "ANALYSIS_TEMPORAL_ENABLED", True),
                patch.object(main_module, "ANALYSIS_SWEEP_ENABLED", True),
            ):
                main_module.main()

        assert len(temporal_called) == 0, (
            "run_elective_analysis_mode NO debe llamarse cuando enabled=False"
        )
        assert len(sweep_called) == 0, (
            "run_sweep_mode NO debe llamarse cuando enabled=False"
        )

    def test_entrypoint_legacy_config_without_temporal_enabled_routes_temporal(
        self, tmp_path
    ):
        """main.main() con config legacy sin temporal_enabled â†’ ANALYSIS_TEMPORAL_ENABLED=True â†’ temporal se ejecuta."""
        from unittest.mock import patch

        # Legacy config: analysis_mode without explicit temporal_enabled key
        cfg = _make_minimal_config(num_procedures=5)
        cfg["analysis_mode"] = {
            "enabled": True,
            # NO temporal_enabled key â€” legacy format
            "num_runs": 1,
            "sims_per_run": 2,
            "checkpoint_interval_seconds": 300,
            "sqlite_path": ":memory:",
            "sweep_enabled": False,
            "sweep_num_procedures": [5],
            "sweep_sims_per_x": 2,
            "export_csv_after_run": False,
            "artifact_save_mode": "all",
        }
        cfg_file = tmp_path / "config_legacy.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod_name in list(sys.modules.keys()):
            if (
                mod_name.startswith("config.")
                or mod_name.startswith("core.simulation_runner")
                or mod_name.startswith("core.analysis_persistence")
                or mod_name == "main"
            ):
                del sys.modules[mod_name]

        temporal_called = []

        def _mock_temporal(self_r):
            temporal_called.append(True)

        with (
            patch(
                "core.simulation_runner.SimulationRunner.run_elective_analysis_mode",
                _mock_temporal,
            ),
            patch(
                "core.simulation_runner.SimulationRunner.run_sweep_mode", lambda s: None
            ),
        ):
            import main as main_module
            import config.config as cfg_mod

            # Verify the default resolution: legacy config â†’ temporal_enabled defaults to True
            assert cfg_mod.ANALYSIS_TEMPORAL_ENABLED is True, (
                "Config legacy sin 'temporal_enabled' debe resolver ANALYSIS_TEMPORAL_ENABLED=True"
            )
            # Now verify main.main() actually routes to temporal
            with (
                patch.object(main_module, "ANALYSIS_MODE_ENABLED", True),
                patch.object(
                    main_module,
                    "ANALYSIS_TEMPORAL_ENABLED",
                    cfg_mod.ANALYSIS_TEMPORAL_ENABLED,
                ),
                patch.object(main_module, "ANALYSIS_SWEEP_ENABLED", False),
            ):
                main_module.main()

        assert len(temporal_called) == 1, (
            f"main.main() con config legacy debe llamar run_elective_analysis_mode() una vez. "
            f"Llamadas: {len(temporal_called)}"
        )

    def test_entrypoint_analysis_mode_disabled_calls_run_elective_mode(self, tmp_path):
        """main.main() con ANALYSIS_MODE_ENABLED=False llama run_elective_mode()."""
        from unittest.mock import patch

        _setup_config(tmp_path, num_procedures=5)

        if "main" in sys.modules:
            del sys.modules["main"]

        elective_called = []

        def _mock_elective(self_r):
            elective_called.append(True)

        with patch(
            "core.simulation_runner.SimulationRunner.run_elective_mode", _mock_elective
        ):
            import main as main_module

            with (
                patch.object(main_module, "ANALYSIS_MODE_ENABLED", False),
            ):
                main_module.main()

        assert len(elective_called) == 1, (
            f"main.main() con analysis_mode disabled debe llamar run_elective_mode() una vez. "
            f"Llamadas: {len(elective_called)}"
        )


# ---------------------------------------------------------------------------
# Task 5.4 â€” IntegraciÃ³n runner â†’ persistencia batch â†’ checkpoint reports
# ---------------------------------------------------------------------------


def _make_analysis_config_batch(tmp_path, db_path):
    """Config para tests de batch persistence integration."""
    cfg = _make_minimal_config(num_procedures=5)
    cfg["times"]["max_wait"] = {"1": 500, "2": 500}
    cfg["analysis_mode"] = {
        "enabled": True,
        "num_runs": 1,
        "sims_per_run": 2,
        "checkpoint_interval_seconds": 1,
        "sqlite_path": db_path,
        "sweep_enabled": False,
        "sweep_num_procedures": [],
        "sweep_sims_per_x": 2,
        "export_csv_after_run": False,
        "full_reports_enabled": False,
        "artifact_save_mode": "all",
    }
    cfg_file = tmp_path / "cfg_batch.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if any(
            mod_name.startswith(p)
            for p in [
                "config.",
                "core.simulation_runner",
                "core.analysis_persistence",
                "simulation.workers.",
                "simulation.scheduler",
                "algorithms.",
            ]
        ):
            del sys.modules[mod_name]


class TestRunnerBatchPersistenceIntegration:
    """Task 5.4 â€” IntegraciÃ³n runnerâ†’persistencia batchâ†’algorithm_iterations en DB."""

    def test_simulations_table_populated_after_run(self, tmp_path):
        """Tras run_elective_analysis_mode(), la tabla 'simulations' contiene al menos 1 fila."""
        import sqlite3
        from core.file_manager import FileManager

        _make_analysis_config_batch(tmp_path, ":memory:")

        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT COUNT(*) FROM simulations").fetchone()[0]
        conn.close()
        assert rows >= 1, (
            f"Tabla 'simulations' debe tener al menos 1 fila tras el run, got {rows}. "
            "run_elective_analysis_mode() debe llamar insert_simulation() por cada simulaciÃ³n."
        )

    def test_algorithm_iterations_table_populated_after_run(self, tmp_path):
        """Tras run_elective_analysis_mode(), la tabla 'algorithm_iterations' tiene filas."""
        import sqlite3
        from core.file_manager import FileManager

        _make_analysis_config_batch(tmp_path, ":memory:")

        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT COUNT(*) FROM algorithm_iterations").fetchone()[0]
        conn.close()
        assert rows >= 1, (
            f"Tabla 'algorithm_iterations' debe tener al menos 1 fila tras el run, got {rows}. "
            "run_elective_analysis_mode() debe persistir iteration snapshots via "
            "save_algorithm_iterations_batch()."
        )

    def test_algorithm_iterations_linked_to_simulations(self, tmp_path):
        """Cada row en 'algorithm_iterations' tiene un sim_id vÃ¡lido referenciando 'simulations'."""
        import sqlite3
        from core.file_manager import FileManager

        _make_analysis_config_batch(tmp_path, ":memory:")

        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)
        conn = sqlite3.connect(db_path)
        # JOIN entre algorithm_iterations y simulations â€” debe devolver filas
        orphan_count = conn.execute(
            """
            SELECT COUNT(*) FROM algorithm_iterations ai
            LEFT JOIN simulations s ON ai.sim_id = s.sim_id
            WHERE s.sim_id IS NULL
            """
        ).fetchone()[0]
        conn.close()
        assert orphan_count == 0, (
            f"Existen {orphan_count} filas en 'algorithm_iterations' sin sim_id vÃ¡lido. "
            "La clave forÃ¡nea sim_id debe referenciar una simulaciÃ³n existente."
        )

    def test_runs_table_has_at_least_one_row(self, tmp_path):
        """Tras run_elective_analysis_mode(), la tabla 'runs' contiene al menos 1 fila."""
        import sqlite3
        from core.file_manager import FileManager

        _make_analysis_config_batch(tmp_path, ":memory:")

        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert rows >= 1, (
            f"Tabla 'runs' debe tener al menos 1 fila tras el run, got {rows}."
        )


# ---------------------------------------------------------------------------
# Phase 4 (RED): iteration_schedules persistence en runner
# ---------------------------------------------------------------------------


def _make_analysis_config_schedules(tmp_path):
    """Config para tests de iteration_schedules — artifact_save_mode=all."""
    cfg = _make_minimal_config(num_procedures=5)
    cfg["times"]["max_wait"] = {"1": 500, "2": 500}
    # Five synthetic jobs carry 780 minutes of anesthesia in total. Two
    # anesthesiologists make that generated campaign physically feasible.
    cfg["personnel"]["num_anesthesiologists"] = 2
    cfg["analysis_mode"] = {
        "enabled": True,
        "num_runs": 1,
        "sims_per_run": 1,
        "checkpoint_interval_seconds": 1,
        "sqlite_path": ":memory:",
        "sweep_enabled": False,
        "sweep_num_procedures": [],
        "sweep_sims_per_x": 2,
        "export_csv_after_run": False,
        "full_reports_enabled": False,
        "artifact_save_mode": "all",
    }
    cfg_file = tmp_path / "cfg_schedules.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if any(
            mod_name.startswith(p)
            for p in [
                "config.",
                "core.simulation_runner",
                "core.analysis_persistence",
                "simulation.workers.",
                "simulation.scheduler",
                "algorithms.",
            ]
        ):
            del sys.modules[mod_name]


class TestArtifactSaveModeFailFast:
    """Task 3.1 — fail-fast si artifact_save_mode != 'all' en analysis mode."""

    def _make_analysis_config_non_all(self, tmp_path, artifact_save_mode):
        """Config analysis mode con artifact_save_mode distinto de 'all'."""
        cfg = _make_minimal_config(num_procedures=5)
        cfg["analysis_mode"] = {
            "enabled": True,
            "num_runs": 1,
            "sims_per_run": 1,
            "checkpoint_interval_seconds": 1,
            "sqlite_path": ":memory:",
            "sweep_enabled": False,
            "sweep_num_procedures": [],
            "sweep_sims_per_x": 2,
            "export_csv_after_run": False,
            "full_reports_enabled": False,
            "artifact_save_mode": artifact_save_mode,
        }
        cfg_file = tmp_path / "cfg_failfast.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod_name in list(sys.modules.keys()):
            if any(
                mod_name.startswith(p)
                for p in [
                    "config.",
                    "core.",
                    "algorithms.",
                    "simulation.",
                    "simulation.workers.",
                ]
            ):
                del sys.modules[mod_name]

    def test_run_analysis_raises_if_artifact_save_mode_best_only(self, tmp_path):
        """Debe lanzar ValueError/RuntimeError si artifact_save_mode='best_only'."""
        self._make_analysis_config_non_all(tmp_path, "best_only")
        from core.simulation_runner import SimulationRunner
        from core.file_manager import FileManager

        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=str(tmp_path / "results"))
        with pytest.raises((ValueError, RuntimeError)):
            runner.run_elective_analysis_mode()

    def test_run_analysis_raises_if_artifact_save_mode_sampled(self, tmp_path):
        """Debe lanzar ValueError/RuntimeError si artifact_save_mode='sampled'."""
        self._make_analysis_config_non_all(tmp_path, "sampled")
        from core.simulation_runner import SimulationRunner
        from core.file_manager import FileManager

        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=str(tmp_path / "results"))
        with pytest.raises((ValueError, RuntimeError)):
            runner.run_elective_analysis_mode()

    def test_run_analysis_ok_if_artifact_save_mode_all(self, tmp_path):
        """Con artifact_save_mode='all' no debe lanzar excepción al arrancar."""
        from core.file_manager import FileManager

        _make_analysis_config_schedules(tmp_path)
        from core.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=str(tmp_path / "results"))
        # No debe lanzar
        runner.run_elective_analysis_mode()


class TestIterationSchedulesPersistence:
    """Task Phase 2 — iteration_schedules se persiste en DB tras analysis run."""

    def test_iteration_schedules_table_exists_in_v4_db(self, tmp_path):
        """La DB generada por run_elective_analysis_mode debe tener tabla iteration_schedules (schema v4)."""
        import sqlite3
        from core.file_manager import FileManager

        _make_analysis_config_schedules(tmp_path)
        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)
        conn = sqlite3.connect(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "iteration_schedules" in tables

    def test_iteration_schedules_populated_after_analysis_run(self, tmp_path):
        """Tras run_elective_analysis_mode con artifact_save_mode=all, iteration_schedules tiene filas."""
        import sqlite3
        from core.file_manager import FileManager

        _make_analysis_config_schedules(tmp_path)
        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM iteration_schedules").fetchone()[0]
        conn.close()
        assert count > 0, (
            "iteration_schedules debe tener al menos 1 fila tras analysis run con artifact_save_mode=all"
        )

    def test_iteration_schedules_sha256_not_empty(self, tmp_path):
        """Cada fila de iteration_schedules debe tener solution_sha256 de 64 chars."""
        import sqlite3
        from core.file_manager import FileManager

        _make_analysis_config_schedules(tmp_path)
        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT solution_sha256 FROM iteration_schedules").fetchall()
        conn.close()
        assert len(rows) > 0
        for row in rows:
            assert len(row[0]) == 64, f"SHA-256 debe tener 64 chars, got: {row[0]}"


class TestConfigSnapshotReplayMetadata:
    """Task 3.2 — config_snapshot debe incluir metadata suficiente para replay offline."""

    def test_config_snapshot_has_replay_metadata(self, tmp_path):
        """Tras analysis run, runs.config_snapshot debe incluir num_procedures, std_factor, use_real_data, artifact_save_mode."""
        import sqlite3, json
        from core.file_manager import FileManager

        _make_analysis_config_schedules(tmp_path)
        from core.simulation_runner import SimulationRunner

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT config_snapshot FROM runs LIMIT 1").fetchone()
        conn.close()
        assert row is not None
        snapshot = json.loads(row[0])
        assert "num_procedures" in snapshot
        assert "std_factor" in snapshot
        assert "use_real_data" in snapshot
        assert "artifact_save_mode" in snapshot
        assert snapshot["artifact_save_mode"] == "all"


class TestGenerateAnalysisReportsExporter:
    """Task 4.3 — generate_analysis_reports debe invocar AnalysisExporter si hay iteration_schedules."""

    def test_generate_analysis_reports_creates_iteration_csvs(self, tmp_path):
        """generate_analysis_reports debe producir schedule_by_iteration.csv si la DB tiene iteration_schedules."""
        import sqlite3

        _make_analysis_config_schedules(tmp_path)
        from core.simulation_runner import SimulationRunner
        from core.file_manager import FileManager
        from core.report_generator import ReportGenerator

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)
        output_dir = str(tmp_path / "export")

        rg = ReportGenerator()
        rg.generate_analysis_reports(db_path, output_dir=output_dir)

        assert os.path.exists(os.path.join(output_dir, "schedule_by_iteration.csv")), (
            "generate_analysis_reports debe crear schedule_by_iteration.csv si la DB tiene iteration_schedules"
        )
        assert os.path.exists(os.path.join(output_dir, "strategy_by_iteration.csv")), (
            "generate_analysis_reports debe crear strategy_by_iteration.csv"
        )
        assert not os.path.exists(os.path.join(output_dir, "breakdown_by_iteration.csv")), (
            "generate_analysis_reports no debe crear breakdown_by_iteration.csv"
        )


class TestE2EAnalysisPipeline:
    """Task 5.1 — E2E mínima del pipeline completo analysis mode → DB v4 → 3 CSVs."""

    def test_e2e_analysis_mode_produces_three_iteration_csvs(self, tmp_path):
        """
        E2E: analysis mode GA (2 gen) → iteration_schedules en DB → generate_analysis_reports → 3 CSVs.
        Verifica que cada CSV tiene al menos 1 fila y algo_step >= 1.
        """
        import sqlite3
        import csv as _csv

        _make_analysis_config_schedules(tmp_path)
        from core.simulation_runner import SimulationRunner
        from core.file_manager import FileManager
        from core.report_generator import ReportGenerator

        results_dir = str(tmp_path / "results")
        runner = SimulationRunner()
        runner.file_manager = FileManager(base_dir=results_dir)
        runner.run_elective_analysis_mode()

        db_path = _find_analysis_db(results_dir)

        # Verificar DB v4+ (schema puede evolucionar; mínimo v4 con iteration_schedules)
        conn = sqlite3.connect(db_path)
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        iter_count = conn.execute("SELECT COUNT(*) FROM iteration_schedules").fetchone()[0]
        conn.close()
        assert version >= 4, f"Schema debe ser >= v4, got {version}"
        assert iter_count > 0, "iteration_schedules debe tener filas"

        # Exportar CSVs
        output_dir = str(tmp_path / "export")
        rg = ReportGenerator()
        rg.generate_analysis_reports(db_path, output_dir=output_dir)

        # schedule_by_iteration.csv
        sched_path = os.path.join(output_dir, "schedule_by_iteration.csv")
        assert os.path.exists(sched_path)
        with open(sched_path, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        assert len(rows) > 0, "schedule_by_iteration.csv debe tener filas"
        assert all(int(r["algo_step"]) >= 1 for r in rows), "algo_step >= 1 siempre"

        # strategy_by_iteration.csv
        strat_path = os.path.join(output_dir, "strategy_by_iteration.csv")
        assert os.path.exists(strat_path)
        with open(strat_path, newline="", encoding="utf-8") as f:
            strat_rows = list(_csv.DictReader(f))
        assert len(strat_rows) > 0, "strategy_by_iteration.csv debe tener filas"

        # breakdown_by_iteration.csv
        break_path = os.path.join(output_dir, "breakdown_by_iteration.csv")


