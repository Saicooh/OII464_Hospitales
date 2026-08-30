"""
tests/data/test_pkl_loader.py — Tests formales del cargador y limpiador de PKL.

Valida que el filtro de anomalías de transición (traspaso excesivo y cirugía corta)
funcione correctamente sobre el dataset real.
"""

import pandas as pd
import pytest
from data.pkl_loader import build_timing_quality_report, load_and_prepare


def test_build_timing_quality_report_counts_and_examples():
    df = pd.DataFrame(
        {
            "record_id": [101, 102, 103],
            "tiempo_transicion": [-1.0, 0.0, 61.0],
            "setup_qx_anestesia": [0.0, 61.0, 5.0],
            "tiempo_anestesia": [0.0, 121.0, 10.0],
            "tiempo_cirugia": [0.0, 361.0, 50.0],
            "tiempo_limpieza": [0.0, -5.0, 61.0],
        }
    )

    report = build_timing_quality_report(df, example_limit=1)

    assert report["rows_evaluated"] == 3
    assert report["summary"] == {
        "total_negative_values": 2,
        "total_zero_values": 5,
        "invalid_surgery_zero_count": 1,
        "total_extreme_outliers": 5,
    }
    assert report["stages"]["transition"]["negative_count"] == 1
    assert report["stages"]["transition"]["zero_count"] == 1
    assert report["stages"]["transition"]["extreme_outlier_count"] == 1
    assert report["stages"]["surgery"]["examples"]["zero"] == [
        {
            "record_id": 101,
            "stage": "surgery",
            "column": "tiempo_cirugia",
            "kind": "zero",
            "value_minutes": 0.0,
        }
    ]


def test_build_timing_quality_report_does_not_mutate_input():
    df = pd.DataFrame(
        {
            "record_id": [1],
            "tiempo_transicion": [0.0],
            "setup_qx_anestesia": [0.0],
            "tiempo_anestesia": [0.0],
            "tiempo_cirugia": [1.0],
            "tiempo_limpieza": [0.0],
        }
    )
    original = df.copy(deep=True)

    build_timing_quality_report(df)

    pd.testing.assert_frame_equal(df, original)


def test_load_and_prepare_reports_timing_quality_without_excluding_cleanup_zero(tmp_path):
    pkl_path = tmp_path / "timing_quality_fixture.pkl"
    pd.DataFrame(
        {
            "Diag. Post. 1 (CIE10)": ["A00", "B00"],
            "Tiempo Anestesia": ["00:20:00", "00:10:00"],
            "Ingreso Pabellón": ["2026-01-01 08:00:00", "2026-01-01 08:00:00"],
            "Ingreso Quirófano": ["2026-01-01 08:00:00", "2026-01-01 08:05:00"],
            "Inicio Anestesia": ["2026-01-01 08:00:00", "2026-01-01 08:10:00"],
            "Inicio Intervención": ["2026-01-01 08:10:00", "2026-01-01 08:15:00"],
            "Término Intervención": ["2026-01-01 09:00:00", "2026-01-01 08:15:00"],
            "Salida Quirófano": ["2026-01-01 09:00:00", "2026-01-01 08:20:00"],
        }
    ).to_pickle(pkl_path)

    df_clean, metadata = load_and_prepare(pkl_path)
    report = metadata["timing_quality_report"]

    assert len(df_clean) == 1
    assert df_clean.iloc[0]["record_id"] == 0
    assert df_clean.iloc[0]["tiempo_limpieza"] == 0.0
    assert report["stages"]["cleanup"]["zero_count"] == 1
    assert report["stages"]["surgery"]["zero_count"] == 1
    assert report["summary"]["invalid_surgery_zero_count"] == 1


def test_load_and_prepare_excludes_transition_anomalies():
    """
    Verifica que la función load_and_prepare excluya correctamente del dataset real
    los registros quirúrgicos anómalos (tiempo_anestesia > 30 min y tiempo_cirugia <= 10 min).
    """
    # 1. Cargar el dataset real
    df_clean, metadata = load_and_prepare()

    # 2. Verificar que se hayan detectado y excluido anomalías de transición en el PKL real
    excluidos = metadata.get("excluidos_anomalia_transicion", 0)
    assert excluidos > 0, (
        f"Se esperaba encontrar al menos una anomalía de transición excluida en el dataset real, "
        f"pero se reportaron {excluidos} excluidos."
    )

    # 3. Validar que en el DataFrame limpio no quede ningún registro que cumpla la anomalía
    anomalous_records = df_clean[
        (df_clean["tiempo_anestesia"] > 30.0)
        & (df_clean["tiempo_cirugia"] <= 10.0)
    ]
    assert len(anomalous_records) == 0, (
        f"Se encontraron {len(anomalous_records)} registros anómalos en el DataFrame limpio: "
        f"{anomalous_records[['tiempo_anestesia', 'tiempo_cirugia']]}"
    )


# ---------------------------------------------------------------------------
# Tests para las nuevas columnas semánticas
# ---------------------------------------------------------------------------


def test_load_and_prepare_has_semantic_columns():
    """
    Verifica que el dataset normalizado exponga las columnas semánticas
    requeridas según el modelo corregido.
    """
    df_clean, metadata = load_and_prepare()

    required_columns = [
        "tiempo_transicion",       # Tiempo Pabellón a Quirófano
        "tiempo_anestesia",        # Tiempo Anestesia a Intervención (standalone)
        "tiempo_cirugia",          # Tiempo Intervención
        "tiempo_limpieza",         # Tiempo Intervención a Salida Quirófano
        "setup_qx_anestesia",      # Tiempo Quirófano a Anestesia
        "tiempo_anestesia_legacy_total",  # Tiempo Anestesia legacy (composite)
        "tiempo_preparacion",      # tiempo_transicion + setup_qx_anestesia
    ]
    for col in required_columns:
        assert col in df_clean.columns, (
            f"Columna requerida '{col}' no encontrada en el dataset limpio. "
            f"Columnas disponibles: {list(df_clean.columns)}"
        )


def test_load_and_prepare_no_legacy_setup_anestesia_interv():
    """
    Verifica que la columna 'setup_anestesia_interv' NO existe en el nuevo modelo.
    Fue reemplazada por 'tiempo_anestesia' (standalone anesthesia phase).
    """
    df_clean, metadata = load_and_prepare()
    assert "setup_anestesia_interv" not in df_clean.columns, (
        "La columna obsoleta 'setup_anestesia_interv' no debe existir en el nuevo modelo. "
        "Fue reemplazada por 'tiempo_anestesia'."
    )


def test_tiempo_transicion_has_valid_values():
    """
    Verifica que tiempo_transicion (Tiempo Pabellón a Quirófano) tenga valores
    no negativos en el dataset limpio.
    """
    df_clean, metadata = load_and_prepare()
    assert df_clean["tiempo_transicion"].notna().any(), (
        "Se esperaba al menos algún valor no nulo en tiempo_transicion."
    )
    assert (df_clean["tiempo_transicion"] >= 0).all(), (
        "tiempo_transicion no debe contener valores negativos."
    )


def test_tiempo_preparacion_uses_transicion_plus_setup():
    """
    Verifica que tiempo_preparacion = tiempo_transicion + setup_qx_anestesia
    (no la suma anterior con setup_anestesia_interv).
    """
    df_clean, metadata = load_and_prepare()
    expected = df_clean["tiempo_transicion"] + df_clean["setup_qx_anestesia"]
    diff = (df_clean["tiempo_preparacion"] - expected).abs().max()
    assert diff < 1e-6, (
        f"tiempo_preparacion debe ser tiempo_transicion + setup_qx_anestesia. "
        f"Diferencia máxima: {diff}"
    )


def test_tiempo_anestesia_is_independent_phase():
    """
    Verifica que tiempo_anestesia (standalone) sea consistentemente menor
    que tiempo_anestesia_legacy_total (que es compuesto y debe incluir más).
    """
    df_clean, metadata = load_and_prepare()
    valid = df_clean.dropna(subset=["tiempo_anestesia", "tiempo_anestesia_legacy_total"])
    assert (valid["tiempo_anestesia"] <= valid["tiempo_anestesia_legacy_total"]).all(), (
        "tiempo_anestesia (standalone) debe ser <= tiempo_anestesia_legacy_total (compuesto)."
    )
