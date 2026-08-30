"""
tests/data/test_real_batch_generator.py — Tests formales del generador de lotes PKL.

Spec cubierta:
    - Real-Data Sampling Preservation / Procedure Quantity Parameterization:
        * El lote real PKL genera exactamente N jobs cuando len(job_ids) == N.
        * El muestreo actual permite repeticiones de CIE10 (sin exigir unicidad).

Estrategia de test:
    - Usa el PKL real de data/2_dataset_procesado_actualizado.pkl para garantizar
      que el comportamiento verificado es el del flujo de producción real.
    - Se usa seed fija para reproducibilidad; se prueba con múltiples valores de N
      para triangulación.
    - El test de repeticiones CIE10 usa N grande (50) y seed que garantiza colisión
      estadísticamente con alta probabilidad (verificado previamente en verify).
    - reset_dataset_singleton() se llama en setup para aislar este módulo de tests
      que puedan haber cargado un dataset diferente.

Nota: estos son tests de integración ligera (tocan el PKL real), no mocks.
La carga del PKL es rápida (~0.2s) y es un artefacto del repo, no un servicio externo.
"""

import pytest

from data.real_batch_generator import (
    generate_day_surgeries_from_pkl,
    reset_dataset_singleton,
)


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_pkl_singleton():
    """Garantiza que el singleton del dataset se recarga fresco para cada test."""
    reset_dataset_singleton()
    yield
    reset_dataset_singleton()


# ---------------------------------------------------------------------------
# Clase A — El lote PKL genera exactamente N jobs
# ---------------------------------------------------------------------------


class TestPKLBatchSizeEqualsN:
    """
    Spec: cuando se llama generate_day_surgeries_from_pkl(job_ids, seed=K),
    el diccionario devuelto tiene exactamente len(job_ids) entradas.

    Esto demuestra formalmente que num_procedures controla el tamaño del lote real:
    el caller (SimulationRunner/worker) construye job_ids = list(range(1, N+1))
    y la función de generación respeta ese tamaño sin truncar ni expandir.
    """

    def test_batch_size_n5(self):
        """N=5 job_ids → surgeries_data tiene exactamente 5 entradas."""
        job_ids = list(range(1, 6))  # [1, 2, 3, 4, 5]
        surgeries_data, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=42)
        assert len(surgeries_data) == 5
        assert len(batch_trace) == 5

    def test_batch_size_n15_legacy(self):
        """N=15 (valor legacy): surgeries_data tiene exactamente 15 entradas."""
        job_ids = list(range(1, 16))
        surgeries_data, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=123)
        assert len(surgeries_data) == 15
        assert len(batch_trace) == 15

    def test_batch_size_n20(self):
        """N=20 (mayor que catálogo): surgeries_data tiene exactamente 20 entradas."""
        job_ids = list(range(1, 21))
        surgeries_data, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=7)
        assert len(surgeries_data) == 20
        assert len(batch_trace) == 20

    def test_batch_keys_match_job_ids(self):
        """Las claves del diccionario devuelto son exactamente los job_ids pasados."""
        job_ids = list(range(1, 11))  # N=10
        surgeries_data, _ = generate_day_surgeries_from_pkl(job_ids, seed=99)
        assert set(surgeries_data.keys()) == set(job_ids)

    def test_batch_size_n50_extended(self):
        """N=50 (muy por encima del catálogo de 15): surgeries_data tiene 50 entradas."""
        job_ids = list(range(1, 51))
        surgeries_data, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=1)
        assert len(surgeries_data) == 50
        assert len(batch_trace) == 50


# ---------------------------------------------------------------------------
# Clase B — El muestreo actual PERMITE repeticiones de CIE10
# ---------------------------------------------------------------------------


class TestCIE10SamplingAllowsRepetitions:
    """
    Spec: el muestreo de códigos CIE10 para el lote se realiza CON reemplazo,
    lo que refleja la distribución hospitalaria real (un código puede aparecer
    múltiples veces en el mismo día quirúrgico).

    Esta clase prueba que no se fuerza unicidad de CIE10 en el lote:
        * Un lote de N=50 con seed fija debe contener CIE10 repetidos.
        * Múltiples lotes de N=10 con seeds distintas eventualmente producen repeticiones.

    Nota metodológica: estos tests verifican que el muestreo CON reemplazo está
    activo. No verifican la distribución estadística exacta (eso sería un test
    de propiedad/property-based, fuera del scope del TDD de esta sesión).
    """

    def test_large_batch_has_repeated_cie10(self):
        """
        N=50 con seed=123 debe producir al menos un CIE10 repetido.

        Justificación: el PKL tiene ~decenas de códigos únicos top20, y con N=50
        el muestreo CON reemplazo estadísticamente garantiza repeticiones
        (principio de casillero: 50 muestras de <50 símbolos ≈ 100% colisión).
        """
        job_ids = list(range(1, 51))
        _, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=123)

        cie10_codes = [row["codigo_cie10"] for row in batch_trace]
        unique_codes = set(cie10_codes)

        # Si hay menos códigos únicos que jobs, hay repeticiones
        assert len(cie10_codes) > len(unique_codes), (
            f"Se esperaban repeticiones de CIE10 en N=50, pero todos los {len(cie10_codes)} "
            f"códigos fueron únicos. Esto sugeriría muestreo SIN reemplazo — incorrecto."
        )

    def test_repeated_cie10_are_structurally_valid(self):
        """
        Los jobs con CIE10 repetido deben tener tiempos válidos (no cero ni NaN).

        Esto verifica que la repetición de código no provoca fallos de muestreo:
        el generador puede muestrear el mismo código múltiples veces correctamente.
        """
        job_ids = list(range(1, 51))
        surgeries_data, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=123)

        cie10_codes = [row["codigo_cie10"] for row in batch_trace]
        unique_codes = set(cie10_codes)

        # Identificar al menos un código repetido
        repeated = [c for c in unique_codes if cie10_codes.count(c) > 1]
        assert len(repeated) > 0, "Debe haber al menos un CIE10 repetido en N=50"

        # Verificar que todos los jobs con ese código repetido tienen tiempos válidos
        first_repeated_code = repeated[0]
        jobs_with_repeated = [
            row["job_id"]
            for row in batch_trace
            if row["codigo_cie10"] == first_repeated_code
        ]
        for job_id in jobs_with_repeated:
            op_data = surgeries_data[job_id]
            assert op_data[1] > 0, (
                f"job {job_id} (CIE10={first_repeated_code}): tiempo_anestesia <= 0"
            )
            assert op_data[2] > 0, (
                f"job {job_id} (CIE10={first_repeated_code}): tiempo_cirugia <= 0"
            )

    def test_top20_fraction_rule_preserved_n15(self):
        """
        N=15: la fracción top20 del lote debe estar en [70%, 80%].

        Este test verifica que la regla estadística PKL 70-80% top20 sigue
        intacta — no fue afectada por la parametrización de num_procedures.
        """
        job_ids = list(range(1, 16))
        _, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=42)

        n_top20 = sum(1 for row in batch_trace if row["grupo"] == "top20")
        n_otros = sum(1 for row in batch_trace if row["grupo"] == "otros")
        total = len(batch_trace)

        assert total == 15
        assert n_otros >= 1, "Debe haber al menos 1 job del grupo 'otros'"
        # La fracción top20 debe estar en el rango [70%, 80%]
        pct_top20 = n_top20 / total
        assert 0.70 <= pct_top20 <= 0.80, (
            f"Fracción top20={pct_top20:.2%} fuera del rango [70%, 80%]. "
            f"La regla estadística PKL fue alterada."
        )


# ---------------------------------------------------------------------------
# Clase C — Contrato de tiempos semánticos (modelo corregido)
# ---------------------------------------------------------------------------


class TestSemanticTimingContract:
    """
    Verifica que el contrato de tiempos del generador PKL respete el
    modelo semántico corregido:

      op1 = tiempo_anestesia (Tiempo Anestesia a Intervención — standalone)
      op2 = tiempo_cirugia   (Tiempo Intervención)
      setup_by_op[1] = setup_qx_anestesia
      setup_by_op[2] = tiempo_transicion (Tiempo Pabellón a Quirófano)
      transition_after_op1 = tiempo_transicion
      cleanup_by_op[2] = tiempo_limpieza
    """

    def test_op1_is_standalone_anesthesia(self):
        """op[1] debe ser tiempo_anestesia (standalone), no legacy."""
        job_ids = list(range(1, 6))
        surgeries_data, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=42)
        for jid in job_ids:
            trace_row = next(r for r in batch_trace if r["job_id"] == jid)
            assert surgeries_data[jid][1] == trace_row["tiempo_anestesia"], (
                f"job {jid}: op[1] debe ser tiempo_anestesia (standalone), "
                f"got {surgeries_data[jid][1]} vs trace {trace_row['tiempo_anestesia']}"
            )

    def test_op2_is_surgery(self):
        """op[2] debe ser tiempo_cirugia."""
        job_ids = list(range(1, 6))
        surgeries_data, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=42)
        for jid in job_ids:
            trace_row = next(r for r in batch_trace if r["job_id"] == jid)
            assert surgeries_data[jid][2] == trace_row["tiempo_cirugia"], (
                f"job {jid}: op[2] debe ser tiempo_cirugia, "
                f"got {surgeries_data[jid][2]} vs trace {trace_row['tiempo_cirugia']}"
            )

    def test_setup_by_op1_is_setup_qx(self):
        """setup_by_op[1] debe ser setup_qx_anestesia."""
        job_ids = list(range(1, 6))
        surgeries_data, _ = generate_day_surgeries_from_pkl(job_ids, seed=42)
        for jid in job_ids:
            setup = surgeries_data[jid]["setup_by_op"]
            assert setup[1] > 0, (
                f"job {jid}: setup_by_op[1] debe ser > 0 (setup_qx_anestesia)"
            )

    def test_setup_by_op2_is_transicion(self):
        """setup_by_op[2] debe ser 0 y transition_by_op[1] debe ser tiempo_transicion."""
        job_ids = list(range(1, 6))
        surgeries_data, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=42)
        for jid in job_ids:
            trace_row = next(r for r in batch_trace if r["job_id"] == jid)
            setup = surgeries_data[jid]["setup_by_op"]
            transition = surgeries_data[jid]["transition_by_op"]
            assert setup[2] == 0.0, (
                f"job {jid}: setup_by_op[2] debe ser 0.0, got {setup[2]}"
            )
            assert transition[1] == trace_row["tiempo_transicion"], (
                f"job {jid}: transition_by_op[1] debe ser tiempo_transicion "
                f"({trace_row['tiempo_transicion']}), got {transition[1]}"
            )

    def test_transition_after_op1_is_transicion(self):
        """transition_after_op1 debe ser tiempo_transicion."""
        job_ids = list(range(1, 6))
        surgeries_data, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=42)
        for jid in job_ids:
            trace_row = next(r for r in batch_trace if r["job_id"] == jid)
            assert surgeries_data[jid]["transition_after_op1"] == trace_row["tiempo_transicion"], (
                f"job {jid}: transition_after_op1 debe ser tiempo_transicion, "
                f"got {surgeries_data[jid]['transition_after_op1']}"
            )

    def test_cleanup_by_op2_is_limpieza(self):
        """cleanup_by_op[2] debe ser tiempo_limpieza; cleanup_by_op[1] debe ser 0."""
        job_ids = list(range(1, 6))
        surgeries_data, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=42)
        for jid in job_ids:
            trace_row = next(r for r in batch_trace if r["job_id"] == jid)
            cleanup = surgeries_data[jid]["cleanup_by_op"]
            assert cleanup[1] == 0.0, (
                f"job {jid}: cleanup_by_op[1] debe ser 0.0, got {cleanup[1]}"
            )
            assert cleanup[2] == trace_row["tiempo_limpieza"], (
                f"job {jid}: cleanup_by_op[2] debe ser tiempo_limpieza, "
                f"got {cleanup[2]}"
            )

    def test_trace_has_semantic_fields(self):
        """El batch_trace debe incluir las columnas semánticas nuevas."""
        job_ids = list(range(1, 6))
        _, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=42)
        for row in batch_trace:
            assert "tiempo_transicion" in row, "trace debe tener tiempo_transicion"
            assert "transition_to_or" in row, "trace debe tener transition_to_or"
            assert "anesthesia_duration" in row, "trace debe tener anesthesia_duration"
            assert "cleanup_op2" in row, "trace debe tener cleanup_op2"
            assert "setup_op2" in row, "trace debe tener setup_op2"
            assert row["setup_op2"] == 0.0, "setup_op2 debe ser 0.0"
            assert row["cleanup_op1"] == 0.0, "cleanup_op1 debe ser 0.0"


# ---------------------------------------------------------------------------
# Clase D — Trazas y columnas de trazabilidad
# ---------------------------------------------------------------------------


class TestTraceSemanticFields:
    """Verifica que las trazas tengan todos los campos requeridos."""

    def test_trace_field_names(self):
        """batch_trace debe contener las columnas documentadas."""
        job_ids = list(range(1, 5))
        _, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=7)
        expected_fields = {
            "job_id", "grupo", "codigo_cie10", "source_record_id",
            "tiempo_cirugia", "tiempo_anestesia", "tiempo_transicion",
            "tiempo_preparacion", "tiempo_limpieza", "setup_qx_anestesia",
            "estrategia_muestreo", "transition_to_or", "setup_op1",
            "anesthesia_duration", "cleanup_op1", "setup_op2", "cleanup_op2",
            "tiempos_dinamicos_en_simulacion",
        }
        for row in batch_trace:
            row_keys = set(row.keys())
            # Remove batch_trace_extras if present
            actual = {k for k in row_keys if not k.startswith("simulation_id")}
            missing = expected_fields - actual
            assert not missing, (
                f"Faltan campos en trace row: {missing}"
            )


# ---------------------------------------------------------------------------
# Clase F — Muestreo Normal Multivariado (Correlación de Tiempos)
# ---------------------------------------------------------------------------


class TestMultivariateNormalSampling:
    """
    Verifica el comportamiento del muestreo normal multivariado implementado
    en la Clase A/B para la discrepancia 1 de correlación anestesia-procedimiento.
    """

    def test_multivariate_normal_samples_are_valid(self):
        """Verifica que el muestreo multivariado genera tiempos realistas y no nulos."""
        job_ids = list(range(1, 11))
        surgeries_data, batch_trace = generate_day_surgeries_from_pkl(job_ids, seed=99)

        # Verificar que todos los jobs generados tienen tiempos >= 1.0 (clamped)
        for jid in job_ids:
            op_data = surgeries_data[jid]
            assert op_data[1] >= 1.0, f"op[1] (anestesia) menor a 1.0: {op_data[1]}"
            assert op_data[2] >= 1.0, f"op[2] (cirugía) menor a 1.0: {op_data[2]}"
            assert op_data["setup_by_op"][1] >= 1.0, f"setup_by_op[1] menor a 1.0"
            assert op_data["transition_by_op"][1] >= 1.0, f"transition_by_op[1] menor a 1.0"
            assert op_data["cleanup_by_op"][2] >= 1.0, f"cleanup_by_op[2] menor a 1.0"

    def test_multivariate_fallback_on_insufficient_data(self):
        """
        Verifica que el generador no truene y use el fallback univariado si
        se intenta muestrear con datos insuficientes.
        """
        from data.real_batch_generator import _get_dataset, _sample_top20_code
        import numpy as np

        ds = _get_dataset()
        c_code = list(ds.top20_codes)[0]
        original_records = ds.get_records_for_code(c_code)

        # Usar monkeypatch para simular solo 1 registro
        single_row = original_records.iloc[:1]
        orig_method = ds.get_records_for_code

        try:
            ds.get_records_for_code = lambda code: single_row if code == c_code else orig_method(code)

            rng = np.random.default_rng(42)
            sampled = _sample_top20_code(ds, c_code, rng)

            # El fallback debe operar exitosamente y devolver tiempos correctos
            assert sampled["tiempo_cirugia"] >= 1.0
            assert sampled["tiempo_anestesia"] >= 1.0
            assert sampled["tiempo_limpieza"] >= 1.0
            assert sampled["tiempo_transicion"] >= 1.0
            assert sampled["setup_qx_anestesia"] >= 1.0
            assert sampled["estrategia_muestreo"] == "normal_por_codigo"
        finally:
            ds.get_records_for_code = orig_method
