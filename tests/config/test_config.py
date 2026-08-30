"""
Tests para config/config.py — parametrización de procedimientos y helper get_job_type.

Estrategia: se recargan los módulos usando una config YAML temporal inyectada
vía la variable de entorno HOSPITAL_CONFIG_PATH, para aislar cada test del
estado global del módulo real.
"""

import importlib
import yaml
import os
import sys
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_JOBS = {
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

_BASE_ALGORITHMS = {
    "alpha": 1e-6,
    "beta": 0.7,
    "gamma": 1.4,
    "delta": 100.0,
    "ga": {
        "enabled": False,
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
}

def _make_config(num_procedures=None, num_anesthesiologists=1, num_surgeons=1):
    """Devuelve un dict de config mínimo con los campos requeridos por config.py.

    Usa el esquema nuevo: personnel.num_anesthesiologists / personnel.num_surgeons.
    """
    cfg = {
        "experiment": {
            "num_simulations": 1,
            "std_factor_times": 0.0,
            "alpha_test": 0.05,
            "output_dirs": {"plots": "results/plots", "csv": "results/csv"},
        },
        "real_data": {"enabled": False},
        "logging": {"verbose_mode": False},
        "times": {
            "setup": {"1": 10, "2": 10, "3": 10},
            "cleanup": {"1": 5, "2": 5, "3": 5},
            "max_wait": {"1": 100, "2": 100},
        },
        "jobs": {"types": _BASE_JOBS},
        "resources": {"num_pabellones": 4},
        "personnel": {
            "num_anesthesiologists": num_anesthesiologists,
            "num_surgeons": num_surgeons,
        },
        "algorithms": _BASE_ALGORITHMS,
    }
    if num_procedures is not None:
        cfg["experiment"]["num_procedures"] = num_procedures
    return cfg


def _reload_config(
    tmp_path, num_procedures=None, num_anesthesiologists=1, num_surgeons=1
):
    """Escribe un YAML temporal y recarga config.config desde cero."""
    cfg = _make_config(
        num_procedures=num_procedures,
        num_anesthesiologists=num_anesthesiologists,
        num_surgeons=num_surgeons,
    )
    cfg_file = tmp_path / "config_test.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)

    # Limpiar módulo cacheado para forzar recarga completa
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("config."):
            del sys.modules[mod_name]

    import config.config as cc

    importlib.reload(cc)
    return cc


# ---------------------------------------------------------------------------
# Task 1.1 — NUM_PROCEDURES: fallback y aceptación de valores
# ---------------------------------------------------------------------------


class TestNumProcedures:
    def test_fallback_when_absent(self, tmp_path):
        """Sin num_procedures en config, NUM_PROCEDURES debe ser len(JOB_TYPES) = 15."""
        cc = _reload_config(tmp_path, num_procedures=None)
        assert cc.NUM_PROCEDURES == len(cc.JOB_TYPES)

    def test_accepts_value_greater_than_15(self, tmp_path):
        """num_procedures = 50 debe almacenarse como 50, sin clamp."""
        cc = _reload_config(tmp_path, num_procedures=50)
        assert cc.NUM_PROCEDURES == 50

    def test_accepts_exact_15(self, tmp_path):
        """num_procedures = 15 es el valor legacy; debe ser aceptado tal cual."""
        cc = _reload_config(tmp_path, num_procedures=15)
        assert cc.NUM_PROCEDURES == 15

    def test_accepts_small_valid_value(self, tmp_path):
        """num_procedures = 3 es válido (>= 1); debe almacenarse como 3."""
        cc = _reload_config(tmp_path, num_procedures=3)
        assert cc.NUM_PROCEDURES == 3

    def test_invalid_zero_falls_back_to_legacy(self, tmp_path):
        """num_procedures = 0 no es válido (< 1); debe caer al fallback len(JOB_TYPES)."""
        cc = _reload_config(tmp_path, num_procedures=0)
        assert cc.NUM_PROCEDURES == len(cc.JOB_TYPES)


# ---------------------------------------------------------------------------
# Task 1.2 — get_job_type: catálogo legacy e IDs fuera de catálogo (cíclico)
# ---------------------------------------------------------------------------


class TestGetJobType:
    def test_known_id_returns_catalog_value(self, tmp_path):
        """job_id 1 está en JOB_TYPES legacy y debe retornar el valor del catálogo (1)."""
        cc = _reload_config(tmp_path)
        assert cc.get_job_type(1) == cc.JOB_TYPES[1]

    def test_known_id_7_returns_type_2(self, tmp_path):
        """job_id 7 tiene job_type 2 en el catálogo legacy."""
        cc = _reload_config(tmp_path)
        assert cc.get_job_type(7) == 2

    def test_known_id_15_boundary(self, tmp_path):
        """job_id 15 es el último en el catálogo; debe retornar su tipo (3)."""
        cc = _reload_config(tmp_path)
        assert cc.get_job_type(15) == cc.JOB_TYPES[15]

    def test_id_16_cyclic_returns_1(self, tmp_path):
        """job_id 16 está fuera del catálogo; (16-1) % 3 = 0 → tipo 1."""
        cc = _reload_config(tmp_path)
        assert cc.get_job_type(16) == 1

    def test_id_17_cyclic_returns_2(self, tmp_path):
        """job_id 17 está fuera del catálogo; (17-1) % 3 = 1 → tipo 2."""
        cc = _reload_config(tmp_path)
        assert cc.get_job_type(17) == 2

    def test_id_18_cyclic_returns_3(self, tmp_path):
        """job_id 18 está fuera del catálogo; (18-1) % 3 = 2 → tipo 3."""
        cc = _reload_config(tmp_path)
        assert cc.get_job_type(18) == 3

    def test_id_19_wraps_back_to_1(self, tmp_path):
        """job_id 19: (19-1) % 3 = 0 → tipo 1 (wrap del ciclo)."""
        cc = _reload_config(tmp_path)
        assert cc.get_job_type(19) == 1


# ---------------------------------------------------------------------------
# Task 4.1 — Tests de regresión: compatibilidad N=15, ciclos N=18/20
# ---------------------------------------------------------------------------


class TestRegressionAndIntegration:
    def test_n15_produces_same_job_ids_as_legacy(self, tmp_path):
        """N=15 genera job_ids idénticos al rango legacy [1..15]."""
        cc = _reload_config(tmp_path, num_procedures=15)
        expected_legacy = list(range(1, 16))
        actual = list(range(1, cc.NUM_PROCEDURES + 1))
        assert actual == expected_legacy

    def test_n18_extra_jobs_use_cyclic_types(self, tmp_path):
        """N=18: jobs 16, 17, 18 deben usar tipos 1, 2, 3 respectivamente."""
        cc = _reload_config(tmp_path, num_procedures=18)
        assert cc.get_job_type(16) == 1
        assert cc.get_job_type(17) == 2
        assert cc.get_job_type(18) == 3

    def test_all_job_types_are_valid_1_2_or_3(self, tmp_path):
        """Para ids 1..30, get_job_type siempre retorna un valor en {1, 2, 3}."""
        cc = _reload_config(tmp_path, num_procedures=30)
        for jid in range(1, 31):
            t = cc.get_job_type(jid)
            assert t in {1, 2, 3}, f"job_id={jid} retornó tipo inválido: {t}"


# ---------------------------------------------------------------------------
# Tarea 1.1 — Personnel: generación de IDs desde num_anesthesiologists/num_surgeons
# ---------------------------------------------------------------------------


def _reload_config_raw(tmp_path, personnel_block):
    """Recarga config con un bloque personnel arbitrario para tests de validación."""
    cfg = _make_config()
    cfg["personnel"] = personnel_block
    cfg_file = tmp_path / "config_raw.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("config."):
            del sys.modules[mod_name]
    import config.config as cc

    importlib.reload(cc)
    return cc


class TestPersonnelIdGeneration:
    def test_two_anesthesiologists_generates_A1_A2(self, tmp_path):
        """num_anesthesiologists=2 debe generar ['A1', 'A2']."""
        cc = _reload_config(tmp_path, num_anesthesiologists=2, num_surgeons=1)
        assert cc._ANESTHESIOLOGISTS == ["A1", "A2"]

    def test_three_surgeons_generates_S1_S2_S3(self, tmp_path):
        """num_surgeons=3 debe generar ['S1', 'S2', 'S3']."""
        cc = _reload_config(tmp_path, num_anesthesiologists=1, num_surgeons=3)
        assert cc._SURGEONS == ["S1", "S2", "S3"]

    def test_six_each_generates_correct_lists(self, tmp_path):
        """num_anesthesiologists=6, num_surgeons=6 debe generar listas de 6 elementos cada una."""
        cc = _reload_config(tmp_path, num_anesthesiologists=6, num_surgeons=6)
        assert cc._ANESTHESIOLOGISTS == [f"A{i + 1}" for i in range(6)]
        assert cc._SURGEONS == [f"S{i + 1}" for i in range(6)]

    def test_zero_anesthesiologists_generates_empty_list(self, tmp_path):
        """num_anesthesiologists=0 debe generar [] (pool vacío es válido si surgeons > 0)."""
        cc = _reload_config(tmp_path, num_anesthesiologists=0, num_surgeons=2)
        assert cc._ANESTHESIOLOGISTS == []

    def test_zero_surgeons_generates_empty_list(self, tmp_path):
        """num_surgeons=0 debe generar [] (pool vacío es válido si anesthesiologists > 0)."""
        cc = _reload_config(tmp_path, num_anesthesiologists=2, num_surgeons=0)
        assert cc._SURGEONS == []

    def test_all_anesthesiologist_ids_start_with_A(self, tmp_path):
        """Todos los IDs de anestesistas deben comenzar con 'A'."""
        cc = _reload_config(tmp_path, num_anesthesiologists=4, num_surgeons=1)
        assert all(aid.startswith("A") for aid in cc._ANESTHESIOLOGISTS)

    def test_all_surgeon_ids_start_with_S(self, tmp_path):
        """Todos los IDs de cirujanos deben comenzar con 'S'."""
        cc = _reload_config(tmp_path, num_anesthesiologists=1, num_surgeons=4)
        assert all(sid.startswith("S") for sid in cc._SURGEONS)


# ---------------------------------------------------------------------------
# Tarea 1.2 — Personnel: exports ALL_PERSONNEL y PERSONNEL_BY_OPERATION
# ---------------------------------------------------------------------------


class TestPersonnelExports:
    def test_all_personnel_combines_both_lists(self, tmp_path):
        """ALL_PERSONNEL debe ser la concatenación de anestesistas + cirujanos."""
        cc = _reload_config(tmp_path, num_anesthesiologists=2, num_surgeons=3)
        assert cc.ALL_PERSONNEL == ["A1", "A2", "S1", "S2", "S3"]

    def test_personnel_by_operation_key1_is_anesthesiologists(self, tmp_path):
        """PERSONNEL_BY_OPERATION[1] debe contener los IDs de anestesistas."""
        cc = _reload_config(tmp_path, num_anesthesiologists=3, num_surgeons=1)
        assert cc.PERSONNEL_BY_OPERATION[1] == ["A1", "A2", "A3"]

    def test_personnel_by_operation_key2_is_surgeons(self, tmp_path):
        """PERSONNEL_BY_OPERATION[2] debe contener los IDs de cirujanos."""
        cc = _reload_config(tmp_path, num_anesthesiologists=1, num_surgeons=2)
        assert cc.PERSONNEL_BY_OPERATION[2] == ["S1", "S2"]

    def test_all_personnel_length_equals_sum(self, tmp_path):
        """len(ALL_PERSONNEL) debe ser num_anesthesiologists + num_surgeons."""
        cc = _reload_config(tmp_path, num_anesthesiologists=4, num_surgeons=5)
        assert len(cc.ALL_PERSONNEL) == 9


# ---------------------------------------------------------------------------
# Tarea 1.1 — Personnel: validaciones (TypeError / ValueError)
# ---------------------------------------------------------------------------


class TestPersonnelValidation:
    def test_rejects_negative_anesthesiologists(self, tmp_path):
        """num_anesthesiologists=-1 debe lanzar ValueError."""
        with pytest.raises(ValueError):
            _reload_config_raw(
                tmp_path, {"num_anesthesiologists": -1, "num_surgeons": 1}
            )

    def test_rejects_negative_surgeons(self, tmp_path):
        """num_surgeons=-1 debe lanzar ValueError."""
        with pytest.raises(ValueError):
            _reload_config_raw(
                tmp_path, {"num_anesthesiologists": 1, "num_surgeons": -1}
            )

    def test_rejects_string_anesthesiologists(self, tmp_path):
        """num_anesthesiologists='six' debe lanzar TypeError."""
        with pytest.raises(TypeError):
            _reload_config_raw(
                tmp_path, {"num_anesthesiologists": "six", "num_surgeons": 1}
            )

    def test_rejects_float_surgeons(self, tmp_path):
        """num_surgeons=2.5 debe lanzar TypeError."""
        with pytest.raises(TypeError):
            _reload_config_raw(
                tmp_path, {"num_anesthesiologists": 1, "num_surgeons": 2.5}
            )

    def test_rejects_bool_anesthesiologists(self, tmp_path):
        """num_anesthesiologists=True debe lanzar TypeError (bool es subclase de int)."""
        with pytest.raises(TypeError):
            _reload_config_raw(
                tmp_path, {"num_anesthesiologists": True, "num_surgeons": 1}
            )

    def test_rejects_bool_surgeons(self, tmp_path):
        """num_surgeons=False debe lanzar TypeError."""
        with pytest.raises(TypeError):
            _reload_config_raw(
                tmp_path, {"num_anesthesiologists": 1, "num_surgeons": False}
            )

    def test_rejects_both_zero(self, tmp_path):
        """num_anesthesiologists=0 AND num_surgeons=0 debe lanzar ValueError."""
        with pytest.raises(ValueError):
            _reload_config_raw(
                tmp_path, {"num_anesthesiologists": 0, "num_surgeons": 0}
            )

    def test_allows_zero_anesthesiologists_with_nonzero_surgeons(self, tmp_path):
        """num_anesthesiologists=0 con num_surgeons=1 debe cargarse sin error."""
        cc = _reload_config_raw(
            tmp_path, {"num_anesthesiologists": 0, "num_surgeons": 1}
        )
        assert cc._ANESTHESIOLOGISTS == []
        assert cc._SURGEONS == ["S1"]

    def test_allows_zero_surgeons_with_nonzero_anesthesiologists(self, tmp_path):
        """num_surgeons=0 con num_anesthesiologists=1 debe cargarse sin error."""
        cc = _reload_config_raw(
            tmp_path, {"num_anesthesiologists": 1, "num_surgeons": 0}
        )
        assert cc._ANESTHESIOLOGISTS == ["A1"]
        assert cc._SURGEONS == []


# ---------------------------------------------------------------------------
# Task 1.1 / 1.2 — analysis_mode config opt-in (Lote 1)
# ---------------------------------------------------------------------------


def _make_config_with_analysis(analysis_mode_block=None, **kwargs):
    """Devuelve config mínimo con bloque analysis_mode opcional."""
    cfg = _make_config(**kwargs)
    if analysis_mode_block is not None:
        cfg["analysis_mode"] = analysis_mode_block
    return cfg


def _reload_config_with_analysis(tmp_path, analysis_mode_block=None, **kwargs):
    """Escribe config con analysis_mode opcional y recarga config.config."""
    cfg = _make_config_with_analysis(analysis_mode_block=analysis_mode_block, **kwargs)
    cfg_file = tmp_path / "config_analysis_test.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("config."):
            del sys.modules[mod_name]
    import config.config as cc

    importlib.reload(cc)
    return cc


class TestAnalysisModeConfig:
    """Tests backward-compatible parsing of analysis_mode config section."""

    def test_absent_analysis_mode_defaults_to_disabled(self, tmp_path):
        """Sin analysis_mode en config, ANALYSIS_MODE_ENABLED debe ser False."""
        cc = _reload_config_with_analysis(tmp_path, analysis_mode_block=None)
        assert cc.ANALYSIS_MODE_ENABLED is False

    def test_explicit_false_keeps_disabled(self, tmp_path):
        """analysis_mode.enabled=false debe resultar en ANALYSIS_MODE_ENABLED=False."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": False}
        )
        assert cc.ANALYSIS_MODE_ENABLED is False

    def test_enabled_true_activates_analysis_mode(self, tmp_path):
        """analysis_mode.enabled=true debe resultar en ANALYSIS_MODE_ENABLED=True."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True}
        )
        assert cc.ANALYSIS_MODE_ENABLED is True

    def test_num_runs_default_is_4(self, tmp_path):
        """Sin num_runs explícito, ANALYSIS_NUM_RUNS debe ser 4."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True}
        )
        assert cc.ANALYSIS_NUM_RUNS == 4

    def test_num_runs_override(self, tmp_path):
        """num_runs=2 debe persistir en ANALYSIS_NUM_RUNS."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True, "num_runs": 2}
        )
        assert cc.ANALYSIS_NUM_RUNS == 2

    def test_sims_per_run_default_is_300(self, tmp_path):
        """Sin sims_per_run explícito, ANALYSIS_SIMS_PER_RUN debe ser 300."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True}
        )
        assert cc.ANALYSIS_SIMS_PER_RUN == 300

    def test_sims_per_run_override(self, tmp_path):
        """sims_per_run=10 debe persistir en ANALYSIS_SIMS_PER_RUN."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={"enabled": True, "sims_per_run": 10},
        )
        assert cc.ANALYSIS_SIMS_PER_RUN == 10

    def test_checkpoint_interval_default_is_300(self, tmp_path):
        """Sin checkpoint_interval_seconds, ANALYSIS_CHECKPOINT_INTERVAL debe ser 300."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True}
        )
        assert cc.ANALYSIS_CHECKPOINT_INTERVAL == 300

    def test_sqlite_path_default(self, tmp_path):
        """Sin sqlite_path, ANALYSIS_SQLITE_PATH debe ser 'results/analysis.db'."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True}
        )
        assert cc.ANALYSIS_SQLITE_PATH == "results/analysis.db"

    def test_sqlite_path_override(self, tmp_path):
        """sqlite_path explícito debe almacenarse en ANALYSIS_SQLITE_PATH."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={"enabled": True, "sqlite_path": "/tmp/test.db"},
        )
        assert cc.ANALYSIS_SQLITE_PATH == "/tmp/test.db"

    def test_sweep_disabled_by_default(self, tmp_path):
        """Sin sweep_enabled, ANALYSIS_SWEEP_ENABLED debe ser False."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True}
        )
        assert cc.ANALYSIS_SWEEP_ENABLED is False

    def test_sweep_values_default_is_empty(self, tmp_path):
        """Sin sweep_num_procedures, ANALYSIS_SWEEP_VALUES debe ser []."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True}
        )
        assert cc.ANALYSIS_SWEEP_VALUES == []

    def test_sweep_values_override(self, tmp_path):
        """sweep_num_procedures=[10,20,30] debe almacenarse en ANALYSIS_SWEEP_VALUES."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={
                "enabled": True,
                "sweep_enabled": True,
                "sweep_num_procedures": [10, 20, 30],
            },
        )
        assert cc.ANALYSIS_SWEEP_VALUES == [10, 20, 30]

    def test_sweep_sims_default_is_20(self, tmp_path):
        """Sin sweep_sims_per_x, ANALYSIS_SWEEP_SIMS debe ser 20."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True}
        )
        assert cc.ANALYSIS_SWEEP_SIMS == 20

    def test_export_csv_after_run_default_is_true(self, tmp_path):
        """Sin export_csv_after_run, ANALYSIS_EXPORT_CSV debe ser True."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True}
        )
        assert cc.ANALYSIS_EXPORT_CSV is True

    def test_normal_mode_constants_unchanged_when_analysis_absent(self, tmp_path):
        """Al no haber analysis_mode, las constantes legacy no cambian."""
        cc = _reload_config_with_analysis(tmp_path, analysis_mode_block=None)
        # Verifica constantes legacy clave
        assert cc.NUM_SIMULATIONS == 1
        assert cc.NUM_PROCEDURES == len(cc.JOB_TYPES)

    def test_normal_mode_constants_unchanged_when_analysis_enabled(self, tmp_path):
        """Con analysis_mode.enabled=true, las constantes legacy no se alteran."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={"enabled": True, "num_runs": 2, "sims_per_run": 10},
        )
        # NUM_SIMULATIONS del bloque experiment sigue intacto
        assert cc.NUM_SIMULATIONS == 1


# ---------------------------------------------------------------------------
# Task 4.1 / 4.2 — ANALYSIS_TEMPORAL_ENABLED: nuevo flag, default true, legacy
# ---------------------------------------------------------------------------


class TestAnalysisTemporalEnabledConfig:
    """Tests para el nuevo flag temporal_enabled en analysis_mode."""

    def test_temporal_enabled_default_true_when_key_absent(self, tmp_path):
        """Sin temporal_enabled en config, ANALYSIS_TEMPORAL_ENABLED debe ser True (backward compat)."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={"enabled": True},  # sin temporal_enabled
        )
        assert cc.ANALYSIS_TEMPORAL_ENABLED is True

    def test_temporal_enabled_explicit_true(self, tmp_path):
        """Con temporal_enabled=true explícito, ANALYSIS_TEMPORAL_ENABLED debe ser True."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={"enabled": True, "temporal_enabled": True},
        )
        assert cc.ANALYSIS_TEMPORAL_ENABLED is True

    def test_temporal_enabled_explicit_false(self, tmp_path):
        """Con temporal_enabled=false, ANALYSIS_TEMPORAL_ENABLED debe ser False."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={"enabled": True, "temporal_enabled": False},
        )
        assert cc.ANALYSIS_TEMPORAL_ENABLED is False

    def test_temporal_enabled_false_when_master_switch_off(self, tmp_path):
        """Cuando enabled=false, ANALYSIS_TEMPORAL_ENABLED es irrelevante pero accesible."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={"enabled": False, "temporal_enabled": True},
        )
        # ANALYSIS_MODE_ENABLED=False → el master switch anula todo,
        # pero ANALYSIS_TEMPORAL_ENABLED sigue siendo parseable como True
        assert cc.ANALYSIS_MODE_ENABLED is False
        assert cc.ANALYSIS_TEMPORAL_ENABLED is True

    def test_temporal_enabled_absent_with_no_analysis_block(self, tmp_path):
        """Sin bloque analysis_mode, ANALYSIS_TEMPORAL_ENABLED debe ser True (default)."""
        cc = _reload_config_with_analysis(tmp_path, analysis_mode_block=None)
        assert cc.ANALYSIS_TEMPORAL_ENABLED is True


# ---------------------------------------------------------------------------
# Task 1.1 — ANALYSIS_ARTIFACT_SAVE_MODE y ANALYSIS_FULL_REPORTS_ENABLED
# ---------------------------------------------------------------------------


class TestArtifactSaveModeConfig:
    """Nuevas constantes de política de artefactos para analysis mode (Lote 1)."""

    def test_artifact_save_mode_default_is_best_only(self, tmp_path):
        """Sin artifact_save_mode, ANALYSIS_ARTIFACT_SAVE_MODE debe ser 'best_only'."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True}
        )
        assert cc.ANALYSIS_ARTIFACT_SAVE_MODE == "best_only"

    def test_artifact_save_mode_override_to_all(self, tmp_path):
        """artifact_save_mode='all' debe almacenarse en ANALYSIS_ARTIFACT_SAVE_MODE."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={"enabled": True, "artifact_save_mode": "all"},
        )
        assert cc.ANALYSIS_ARTIFACT_SAVE_MODE == "all"

    def test_artifact_save_mode_override_to_sampled(self, tmp_path):
        """artifact_save_mode='sampled' debe almacenarse en ANALYSIS_ARTIFACT_SAVE_MODE."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={"enabled": True, "artifact_save_mode": "sampled"},
        )
        assert cc.ANALYSIS_ARTIFACT_SAVE_MODE == "sampled"

    def test_artifact_save_mode_absent_with_no_analysis_block(self, tmp_path):
        """Sin bloque analysis_mode, ANALYSIS_ARTIFACT_SAVE_MODE debe ser 'best_only'."""
        cc = _reload_config_with_analysis(tmp_path, analysis_mode_block=None)
        assert cc.ANALYSIS_ARTIFACT_SAVE_MODE == "best_only"

    def test_full_reports_enabled_default_is_false(self, tmp_path):
        """Sin full_reports_enabled, ANALYSIS_FULL_REPORTS_ENABLED debe ser False (opt-in)."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True}
        )
        assert cc.ANALYSIS_FULL_REPORTS_ENABLED is False

    def test_full_reports_enabled_override_to_true(self, tmp_path):
        """full_reports_enabled=true debe resultar en ANALYSIS_FULL_REPORTS_ENABLED=True."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={"enabled": True, "full_reports_enabled": True},
        )
        assert cc.ANALYSIS_FULL_REPORTS_ENABLED is True

    def test_full_reports_enabled_absent_with_no_analysis_block(self, tmp_path):
        """Sin bloque analysis_mode, ANALYSIS_FULL_REPORTS_ENABLED debe ser False."""
        cc = _reload_config_with_analysis(tmp_path, analysis_mode_block=None)
        assert cc.ANALYSIS_FULL_REPORTS_ENABLED is False

    def test_normal_mode_constants_not_affected_by_new_keys(self, tmp_path):
        """Las constantes legacy (NUM_SIMULATIONS, etc.) no deben verse afectadas."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={
                "enabled": True,
                "artifact_save_mode": "all",
                "full_reports_enabled": True,
            },
        )
        assert cc.NUM_SIMULATIONS == 1
        assert cc.NUM_PROCEDURES == len(cc.JOB_TYPES)


# ---------------------------------------------------------------------------
# Task 5.5 — ANALYSIS_ITERATIONS_CSV_PATH: nueva key de config (Lote 5)
# ---------------------------------------------------------------------------


class TestAnalysisIterationsCsvPathConfig:
    """Tests para ANALYSIS_ITERATIONS_CSV_PATH en config (Task 5.5)."""

    def test_iterations_csv_path_default(self, tmp_path):
        """Sin iterations_csv_path explícito, debe tener valor default estándar."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True}
        )
        assert (
            cc.ANALYSIS_ITERATIONS_CSV_PATH
            == "results/analysis_algorithm_iterations.csv"
        )

    def test_iterations_csv_path_override(self, tmp_path):
        """iterations_csv_path explícito debe almacenarse en ANALYSIS_ITERATIONS_CSV_PATH."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={
                "enabled": True,
                "iterations_csv_path": "/tmp/custom_iterations.csv",
            },
        )
        assert cc.ANALYSIS_ITERATIONS_CSV_PATH == "/tmp/custom_iterations.csv"

    def test_iterations_csv_path_absent_with_no_analysis_block(self, tmp_path):
        """Sin bloque analysis_mode, ANALYSIS_ITERATIONS_CSV_PATH debe retornar el default."""
        cc = _reload_config_with_analysis(tmp_path, analysis_mode_block=None)
        assert (
            cc.ANALYSIS_ITERATIONS_CSV_PATH
            == "results/analysis_algorithm_iterations.csv"
        )

    def test_iterations_csv_path_is_string(self, tmp_path):
        """ANALYSIS_ITERATIONS_CSV_PATH debe ser un string."""
        cc = _reload_config_with_analysis(
            tmp_path, analysis_mode_block={"enabled": True}
        )
        assert isinstance(cc.ANALYSIS_ITERATIONS_CSV_PATH, str)

    def test_normal_mode_not_affected_by_iterations_csv_path(self, tmp_path):
        """Configurar iterations_csv_path no debe alterar constantes del modo normal."""
        cc = _reload_config_with_analysis(
            tmp_path,
            analysis_mode_block={
                "enabled": False,
                "iterations_csv_path": "/tmp/iters.csv",
            },
        )
        assert cc.ANALYSIS_MODE_ENABLED is False
        assert cc.NUM_SIMULATIONS == 1


# ---------------------------------------------------------------------------
# Task 1.1 (YAML) — loader por defecto usa config/config.yaml
# Task 1.2 (YAML) — override por HOSPITAL_CONFIG_PATH con YAML y override inexistente
# ---------------------------------------------------------------------------


class TestYamlLoader:
    """Tests del contrato del loader YAML: default path y override por env var."""

    def test_loader_reads_yaml_via_override(self, tmp_path):
        """HOSPITAL_CONFIG_PATH apuntando a un .yaml es leído correctamente."""
        cc = _reload_config(tmp_path, num_procedures=5)
        assert cc.NUM_PROCEDURES == 5

    def test_loader_override_nonexistent_raises_file_not_found(self, tmp_path):
        """HOSPITAL_CONFIG_PATH apuntando a ruta inexistente lanza FileNotFoundError."""
        os.environ["HOSPITAL_CONFIG_PATH"] = str(tmp_path / "no_existe.yaml")
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("config."):
                del sys.modules[mod_name]
        with pytest.raises(FileNotFoundError):
            import config.config  # noqa: F401  — la excepción ocurre en module-level

    def test_loader_yaml_produces_correct_dict_structure(self, tmp_path):
        """El YAML cargado produce el mismo dict que el JSON equivalente."""
        cc = _reload_config(tmp_path, num_procedures=15, num_anesthesiologists=2, num_surgeons=3)
        assert cc.NUM_PROCEDURES == 15
        assert cc._ANESTHESIOLOGISTS == ["A1", "A2"]
        assert cc._SURGEONS == ["S1", "S2", "S3"]

    def test_default_path_loads_config_yaml_when_env_unset(self, monkeypatch):
        """Sin HOSPITAL_CONFIG_PATH, el loader carga config/config.yaml y produce estructura válida.

        Spec: YAML Configuration Loading — Default configuration loading.
        Valida el contrato del loader para el default path sin ningún override de entorno.
        """
        # Eliminar HOSPITAL_CONFIG_PATH del entorno para garantizar default path
        monkeypatch.delenv("HOSPITAL_CONFIG_PATH", raising=False)

        # Limpiar módulos config cacheados para forzar recarga desde cero
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("config."):
                del sys.modules[mod_name]

        import config.config as cc
        importlib.reload(cc)

        # El loader debe haber cargado config/config.yaml sin excepción.
        # Validamos las claves estructurales mínimas requeridas por el spec.
        assert isinstance(cc.JOB_TYPES, dict), "JOB_TYPES debe ser dict"
        assert len(cc.JOB_TYPES) > 0, "JOB_TYPES no puede estar vacío"
        assert isinstance(cc.NUM_PROCEDURES, int), "NUM_PROCEDURES debe ser int"
        assert cc.NUM_PROCEDURES > 0, "NUM_PROCEDURES debe ser positivo"
        assert isinstance(cc.NUM_SIMULATIONS, int), "NUM_SIMULATIONS debe ser int"
        assert isinstance(cc._ANESTHESIOLOGISTS, list), "_ANESTHESIOLOGISTS debe ser list"
        assert isinstance(cc._SURGEONS, list), "_SURGEONS debe ser list"


# ---------------------------------------------------------------------------
# E2E — equivalencia estructural de YAMLs versionados migrados
# Spec: YAML Configuration Format and Structure — File format and comment migration
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL_KEYS = {
    "experiment", "jobs", "resources", "personnel", "algorithms",
    "times", "logging",
}

_VERSIONED_YAML_FILES = [
    "config/config.yaml",
    "config/config.quick.yaml",
    "config/config.validation.yaml",
]

_LEGACY_JSON_FILES = [
    "config/config.json",
    "config/config.quick.json",
    "config/config.validation.json",
]


class TestVersionedYamlMigration:
    """Tests end-to-end de equivalencia estructural post-migración.

    Spec: YAML Configuration Format and Structure — File format and comment migration.
    Garantizan que:
    1. Los YAMLs versionados existen y son parseables.
    2. Contienen las claves estructurales del contrato interno.
    3. Los JSON legacy han sido eliminados (YAML como única fuente de verdad).
    """

    def _project_root(self):
        return Path(__file__).resolve().parent.parent.parent

    def test_config_yaml_exists_and_is_parseable(self):
        """config/config.yaml existe y se parsea como dict no vacío."""
        root = self._project_root()
        cfg_path = root / "config" / "config.yaml"
        assert cfg_path.exists(), f"Archivo no encontrado: {cfg_path}"
        with cfg_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict) and len(data) > 0, "config.yaml debe ser un dict no vacío"

    def test_config_quick_yaml_exists_and_is_parseable(self):
        """config/config.quick.yaml existe y se parsea como dict no vacío."""
        root = self._project_root()
        cfg_path = root / "config" / "config.quick.yaml"
        assert cfg_path.exists(), f"Archivo no encontrado: {cfg_path}"
        with cfg_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict) and len(data) > 0, "config.quick.yaml debe ser un dict no vacío"

    def test_config_validation_yaml_exists_and_is_parseable(self):
        """config/config.validation.yaml existe y se parsea como dict no vacío."""
        root = self._project_root()
        cfg_path = root / "config" / "config.validation.yaml"
        assert cfg_path.exists(), f"Archivo no encontrado: {cfg_path}"
        with cfg_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict) and len(data) > 0, "config.validation.yaml debe ser un dict no vacío"

    @pytest.mark.parametrize("yaml_rel_path", _VERSIONED_YAML_FILES)
    def test_versioned_yaml_has_required_structural_keys(self, yaml_rel_path):
        """Cada YAML versionado contiene todas las claves estructurales del contrato."""
        root = self._project_root()
        cfg_path = root / yaml_rel_path
        assert cfg_path.exists(), f"YAML versionado no encontrado: {yaml_rel_path}"
        with cfg_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        missing = _REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
        assert not missing, (
            f"{yaml_rel_path} le faltan claves requeridas: {missing}"
        )

    @pytest.mark.parametrize("json_rel_path", _LEGACY_JSON_FILES)
    def test_legacy_json_config_does_not_exist(self, json_rel_path):
        """Los JSON de configuración legacy han sido eliminados (YAML es la única fuente de verdad)."""
        root = self._project_root()
        json_path = root / json_rel_path
        assert not json_path.exists(), (
            f"JSON legacy encontrado — debe eliminarse: {json_path}\n"
            "Spec: YAML es la única fuente de verdad para configuración."
        )



