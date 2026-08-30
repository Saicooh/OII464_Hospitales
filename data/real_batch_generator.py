"""
real_batch_generator.py — Generador de lotes de cirugías desde datos reales PKL.

Propósito:
    Reemplaza la función `generate_day_surgeries_data()` de data_generator.py
    cuando USE_REAL_DATA=True en config.yaml. Devuelve el MISMO contrato de
    datos que el generador sintético, garantizando compatibilidad con el scheduler estático,
    scheduler.py y todos los workers existentes.

Contrato de salida (igual que generate_day_surgeries_data):
    {job_id: {1: float, 2: float}}
      - operación 1: tiempo_anestesia (minutos)
      - operación 2: tiempo_cirugia  (minutos)

Reglas implementadas (según requerimientos del profesor):
    1. Top20: distribución normal propia por código CIE10.
    2. Otros:  muestreo empírico directo de un registro real, dentro del
               historial del código efectivamente elegido para ese slot.
    3. Por lote: 70–80% desde top20, al menos 1 desde 'otros'.
    4. Sin tiempos negativos/imposibles tras muestreo.
    5. Toda la lógica interna usa codigos_cie10, nunca nombres reales.
    6. La selección de códigos dentro de cada grupo se pondera por frecuencia
       histórica del código en ese grupo, reflejando la distribución real.

Trazabilidad:
    Cada lote devuelve adicionalmente un `batch_trace` — lista de dicts con
    los datos crudos usados, lista para exportar al CSV de trazabilidad.
    El caller (worker) es responsable de escribir ese CSV.

    Nota de paralelismo:
    Como joblib/loky usa procesos separados, el CSV de trazabilidad se escribe
    por simulación (un archivo por sim_i) para evitar corrupción por escrituras
    concurrentes entre procesos. Ver raw_trace_writer.py para la API de escritura
    y consolidación.

Autor: Fase 2 — integración datos reales.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from data.pkl_loader import CIE10Dataset

# ---------------------------------------------------------------------------
# Constantes y fallbacks documentados
# ---------------------------------------------------------------------------

# Fallback de desviación estándar cuando std=0 o NaN para un código.
# Se usa el 15% de la media (mismo factor que std_factor actual del proyecto).
_FALLBACK_STD_FACTOR: float = 0.15

# Tiempo mínimo absoluto para cualquier componente temporal (minutos).
# Físicamente imposible que una operación tome menos de este valor.
_MIN_TIME_MINUTES: float = 1.0

# Fracción mínima y máxima del lote asignada al grupo top20.
_TOP20_FRACTION_MIN: float = 0.70
_TOP20_FRACTION_MAX: float = 0.80

# Singleton del dataset cargado — se inicializa una sola vez por proceso.
_DATASET_SINGLETON: CIE10Dataset | None = None


# ---------------------------------------------------------------------------
# Singleton del dataset (carga perezosa y única)
# ---------------------------------------------------------------------------


def _get_dataset() -> CIE10Dataset:
    """
    Devuelve el singleton de CIE10Dataset, cargándolo si aún no existe.

    Estrategia singleton: evita re-leer el PKL en cada iteración de
    simulación. El PKL es pesado; una sola carga por proceso es correcto.
    """
    global _DATASET_SINGLETON
    if _DATASET_SINGLETON is None:
        _DATASET_SINGLETON = CIE10Dataset.from_pkl()
    return _DATASET_SINGLETON


def reset_dataset_singleton() -> None:
    """
    Fuerza la recarga del dataset en el próximo acceso.
    Útil para tests o cuando se cambia el PKL en tiempo de ejecución.
    """
    global _DATASET_SINGLETON
    _DATASET_SINGLETON = None


# ---------------------------------------------------------------------------
# Muestreo top20: distribución normal por código
# ---------------------------------------------------------------------------


def _sample_top20_code(
    ds: CIE10Dataset,
    codigo_cie10: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Muestrea los tiempos de un procedimiento top20 usando distribución normal
    propia del código (media y std calculadas sobre el historial real).

    Parámetros
    ----------
    ds : CIE10Dataset
        Dataset cargado con registros reales.
    codigo_cie10 : str
        Código CIE10 a muestrear (debe estar en top20).
    rng : np.random.Generator
        Generador de aleatoriedad ya inicializado (para reproducibilidad).

    Retorna
    -------
    dict con claves:
        tiempo_cirugia, tiempo_anestesia, tiempo_anestesia_legacy_total,
        tiempo_preparacion, tiempo_limpieza, tiempo_transicion,
        setup_qx_anestesia,
        source_record_id (None para top20 — se usa distribución, no registro)
        estrategia_muestreo = 'normal_por_codigo'
        codigo_cie10
    """
    records = ds.get_records_for_code(codigo_cie10)

    # Si por alguna razón el código no tiene registros, fallback al grupo top20 completo.
    if len(records) == 0:
        records = ds.df_top20

    def _safe_sample(series: pd.Series, rng: np.random.Generator) -> float:
        """
        Muestrea de N(mu, sigma). Si sigma=0 o NaN, usa _FALLBACK_STD_FACTOR * mu.
        Protege contra tiempos negativos/imposibles.
        """
        mu = float(series.mean())
        sigma = float(series.std())

        if math.isnan(sigma) or sigma <= 0.0:
            sigma = _FALLBACK_STD_FACTOR * abs(mu)

        # Protección adicional contra mu=0 (no debería ocurrir tras filtros de pkl_loader)
        if mu <= 0.0:
            mu = _MIN_TIME_MINUTES
            sigma = _FALLBACK_STD_FACTOR * mu

        value = float(rng.normal(mu, sigma))
        return max(_MIN_TIME_MINUTES, value)

    cols = [
        "tiempo_transicion",
        "setup_qx_anestesia",
        "tiempo_anestesia",
        "tiempo_cirugia",
        "tiempo_limpieza",
    ]

    # Intentar muestreo normal multivariado si hay suficientes registros no nulos
    df_sub = records[cols].dropna()
    use_multivariate = False

    if len(df_sub) >= 2:
        try:
            means = df_sub.mean().to_numpy()
            cov_matrix = df_sub.cov().to_numpy()
            if not np.any(np.isnan(cov_matrix)) and not np.any(np.isnan(means)):
                use_multivariate = True
        except Exception:
            use_multivariate = False

    if use_multivariate:
        # Muestreo multivariado usando descomposición SVD para robustez ante matrices singulares
        sample = rng.multivariate_normal(means, cov_matrix, method="svd")
        t_transicion, t_setup_qx, t_anestesia, t_cirugia, t_limpieza = sample

        # Clamping de seguridad al igual que el método univariante
        t_transicion = max(_MIN_TIME_MINUTES, t_transicion)
        t_setup_qx = max(_MIN_TIME_MINUTES, t_setup_qx)
        t_anestesia = max(_MIN_TIME_MINUTES, t_anestesia)
        t_cirugia = max(_MIN_TIME_MINUTES, t_cirugia)
        t_limpieza = max(_MIN_TIME_MINUTES, t_limpieza)
    else:
        # Fallback al muestreo univariante independiente actual
        t_cirugia = _safe_sample(records["tiempo_cirugia"], rng)
        t_anestesia = _safe_sample(records["tiempo_anestesia"], rng)
        t_limpieza = _safe_sample(records["tiempo_limpieza"], rng)
        t_transicion = _safe_sample(records["tiempo_transicion"], rng)
        t_setup_qx = _safe_sample(records["setup_qx_anestesia"], rng)

    t_preparacion = t_transicion + t_setup_qx

    # Legacy composite (trazabilidad únicamente)
    t_anestesia_legacy = _safe_sample(records["tiempo_anestesia_legacy_total"], rng)

    return {
        "codigo_cie10": codigo_cie10,
        "grupo": "top20",
        "source_record_id": None,
        "tiempo_cirugia": round(t_cirugia, 4),
        "tiempo_anestesia": round(t_anestesia, 4),
        "tiempo_anestesia_legacy_total": round(t_anestesia_legacy, 4),
        "tiempo_transicion": round(t_transicion, 4),
        "tiempo_preparacion": round(t_preparacion, 4),
        "tiempo_limpieza": round(t_limpieza, 4),
        "setup_qx_anestesia": round(t_setup_qx, 4),
        "estrategia_muestreo": "normal_por_codigo",
    }


# ---------------------------------------------------------------------------
# Muestreo otros: muestreo empírico directo
# ---------------------------------------------------------------------------


def _sample_otros(
    ds: CIE10Dataset,
    codigo_cie10: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Muestrea un procedimiento del grupo 'otros' eligiendo un registro real
    aleatorio DENTRO DEL HISTORIAL del código CIE10 efectivamente elegido,
    arrastrando TODOS sus tiempos del mismo registro.

    Esto es DISTINTO a la normal: NO se calcula media ni distribución.
    Se elige una fila completa del dataframe, restringida al código elegido.

    El código elegido gobierna el muestreo empírico: si el lote eligió el
    código X del grupo 'otros', se muestrea solo entre los registros de X.
    Así el source_record_id siempre pertenece al código efectivamente elegido.

    Fallbacks documentados:
    - Si el código no tiene registros en df_otros (situación anómala), se
      cae al grupo 'otros' completo (df_otros), que sigue siendo empírico.
    - Si df_otros está vacío (situación extrema), se usa df_clean completo.

    Parámetros
    ----------
    ds : CIE10Dataset
    codigo_cie10 : str
        Código CIE10 del grupo 'otros' elegido en _select_batch_codes().
    rng : np.random.Generator

    Retorna
    -------
    dict con claves:
        tiempo_cirugia, tiempo_anestesia, tiempo_anestesia_legacy_total,
        tiempo_preparacion, tiempo_limpieza, tiempo_transicion,
        setup_qx_anestesia,
        source_record_id, codigo_cie10,
        estrategia_muestreo = 'empirico_por_registro'
        grupo = 'otros'
    """
    # Filtrar dentro del historial del código elegido
    df_codigo = ds.df_otros[ds.df_otros["codigo_cie10"] == codigo_cie10]

    if len(df_codigo) == 0:
        # Fallback nivel 1: el código no tiene registros en df_otros (anómalo).
        # Se cae al grupo 'otros' completo — sigue siendo muestreo empírico.
        df_codigo = ds.df_otros

    if len(df_codigo) == 0:
        # Fallback nivel 2: df_otros vacío (situación extrema).
        df_codigo = ds.df_clean

    # Elegir índice aleatorio usando rng (reproducible)
    idx = int(rng.integers(0, len(df_codigo)))
    row = df_codigo.iloc[idx]

    t_cirugia = float(row["tiempo_cirugia"])
    # tiempo_anestesia = standalone anesthesia phase (Tiempo Anestesia a Intervención)
    t_anestesia = float(row["tiempo_anestesia"])
    t_anestesia_legacy = float(row["tiempo_anestesia_legacy_total"])
    t_transicion = float(row["tiempo_transicion"])
    t_setup_qx = float(row["setup_qx_anestesia"])
    t_preparacion = t_transicion + t_setup_qx
    t_limpieza = float(row["tiempo_limpieza"])

    # Protección mínima: aunque los filtros del pkl_loader ya deberían garantizarlo,
    # cubrimos el caso de NaN en setup (no filtrado explícitamente).
    t_cirugia = (
        max(_MIN_TIME_MINUTES, t_cirugia)
        if not math.isnan(t_cirugia)
        else _MIN_TIME_MINUTES
    )
    t_anestesia = (
        max(_MIN_TIME_MINUTES, t_anestesia)
        if not math.isnan(t_anestesia)
        else _MIN_TIME_MINUTES
    )
    t_limpieza = (
        max(_MIN_TIME_MINUTES, t_limpieza)
        if not math.isnan(t_limpieza)
        else _MIN_TIME_MINUTES
    )
    t_transicion = max(0.0, t_transicion) if not math.isnan(t_transicion) else 0.0
    t_setup_qx = max(0.0, t_setup_qx) if not math.isnan(t_setup_qx) else 0.0
    t_preparacion = t_transicion + t_setup_qx
    t_anestesia_legacy = (
        max(_MIN_TIME_MINUTES, t_anestesia_legacy)
        if not math.isnan(t_anestesia_legacy)
        else _MIN_TIME_MINUTES
    )

    return {
        "codigo_cie10": str(row["codigo_cie10"]),
        "grupo": "otros",
        "source_record_id": int(row["record_id"]),
        "tiempo_cirugia": round(t_cirugia, 4),
        "tiempo_anestesia": round(t_anestesia, 4),
        "tiempo_anestesia_legacy_total": round(t_anestesia_legacy, 4),
        "tiempo_transicion": round(t_transicion, 4),
        "tiempo_preparacion": round(t_preparacion, 4),
        "tiempo_limpieza": round(t_limpieza, 4),
        "setup_qx_anestesia": round(t_setup_qx, 4),
        "estrategia_muestreo": "empirico_por_registro",
    }


# ---------------------------------------------------------------------------
# Selección de códigos para el lote
# ---------------------------------------------------------------------------


def _select_batch_codes(
    ds: CIE10Dataset,
    n_jobs: int,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """
    Selecciona los códigos CIE10 para el lote siguiendo la regla 70–80%/otros.

    Garantías:
        - Al menos 1 código del grupo 'otros'.
        - Entre 70% y 80% del lote desde top20.
        - El resto desde 'otros'.
        - La selección dentro de cada grupo se pondera por frecuencia histórica
          del código en ese grupo, reflejando la distribución real del PKL.
          Un código que aparece 200 veces tiene el doble de probabilidad que uno
          que aparece 100, dentro del mismo grupo.

    Parámetros
    ----------
    ds : CIE10Dataset
    n_jobs : int
        Tamaño total del lote.
    rng : np.random.Generator

    Retorna
    -------
    list of (codigo_cie10, grupo) — con repeticiones permitidas.
    """
    if n_jobs < 2:
        # Con 1 solo job no se puede cumplir la regla de al menos 1 de cada grupo.
        # Elegimos top20 si hay, sino otros.
        if len(ds.top20_codes) > 0:
            freq_top20 = ds.df_top20["codigo_cie10"].value_counts()
            codes_top20 = list(freq_top20.index)
            weights_top20 = freq_top20.values.astype(float)
            weights_top20 /= weights_top20.sum()
            chosen = codes_top20[int(rng.choice(len(codes_top20), p=weights_top20))]
            return [(chosen, "top20")]
        else:
            return [("_otros_", "otros")]

    # Calcular n_top20 en rango [70%, 80%] del lote total
    # con la restricción de dejar al menos 1 para 'otros'
    n_top20_min = math.ceil(n_jobs * _TOP20_FRACTION_MIN)
    n_top20_max = math.floor(n_jobs * _TOP20_FRACTION_MAX)

    # Asegurar al menos 1 para 'otros'
    n_top20_max = min(n_top20_max, n_jobs - 1)
    n_top20_min = min(n_top20_min, n_top20_max)

    # Elegir n_top20 aleatorio dentro del rango
    n_top20 = int(rng.integers(n_top20_min, n_top20_max + 1))
    n_otros = n_jobs - n_top20

    # --- Pesos por frecuencia histórica dentro de cada grupo ---
    # value_counts() devuelve índice ordenado de mayor a menor frecuencia.
    freq_top20 = ds.df_top20["codigo_cie10"].value_counts()
    freq_otros = ds.df_otros["codigo_cie10"].value_counts()

    codes_top20 = list(freq_top20.index)
    weights_top20 = freq_top20.values.astype(float)
    if weights_top20.sum() > 0:
        weights_top20 /= weights_top20.sum()

    codes_otros = list(freq_otros.index)
    weights_otros = freq_otros.values.astype(float)
    if weights_otros.sum() > 0:
        weights_otros /= weights_otros.sum()

    # Si no hay suficientes códigos top20, usar todos y ajustar
    if len(codes_top20) == 0:
        n_otros = n_jobs
        n_top20 = 0

    # Muestreo ponderado con reemplazo — un día puede tener múltiples
    # cirugías del mismo código, y los más frecuentes históricamente tienen
    # mayor probabilidad de ser elegidos (refleja la distribución real).
    selected: list[tuple[str, str]] = []

    if n_top20 > 0 and len(codes_top20) > 0:
        chosen_indices = rng.choice(len(codes_top20), size=n_top20, p=weights_top20)
        for ci in chosen_indices:
            selected.append((codes_top20[int(ci)], "top20"))

    for _ in range(n_otros):
        if len(codes_otros) > 0:
            ci = int(rng.choice(len(codes_otros), p=weights_otros))
            selected.append((codes_otros[ci], "otros"))
        else:
            # Si no hay códigos 'otros', fallback a top20
            if len(codes_top20) > 0:
                ci = int(rng.choice(len(codes_top20), p=weights_top20))
                selected.append((codes_top20[ci], "top20"))

    # Mezclar para evitar que todos los 'otros' queden al final
    indices = list(range(len(selected)))
    rng.shuffle(indices)
    selected = [selected[i] for i in indices]

    return selected


# ---------------------------------------------------------------------------
# Generador principal de lotes
# ---------------------------------------------------------------------------


def generate_day_surgeries_from_pkl(
    job_ids: list,
    *,
    seed: int | None = None,
    batch_trace_extras: dict | None = None,
) -> tuple[dict, list[dict]]:
    """
    Genera los datos de tiempos de un día de cirugías usando el PKL real.

    Contrato de salida compatible con generate_day_surgeries_data():
        {job_id: {1: float, 2: float}}
          - operación 1 → tiempo_anestesia
          - operación 2 → tiempo_cirugia

    Adicionalmente retorna `batch_trace`: lista de dicts con metadata cruda
    para el CSV de trazabilidad.

    Parámetros
    ----------
    job_ids : list
        Lista de integer job IDs for the day's surgeries.
    seed : int | None
        Semilla para reproducibilidad. Si None, no se fija.
    batch_trace_extras : dict | None
        Campos extra a agregar a cada fila del trace (ej. simulation_id).

    Retorna
    -------
    (surgeries_data, batch_trace)
        surgeries_data : dict {job_id: {1: float, 2: float}}
        batch_trace    : list[dict] — una fila por job, lista para CSV
    """
    ds = _get_dataset()
    rng = np.random.default_rng(seed)

    n_jobs = len(job_ids)
    batch_codes = _select_batch_codes(ds, n_jobs, rng)

    surgeries_data: dict = {}
    batch_trace: list[dict] = []

    for job_id, (codigo_cie10, grupo) in zip(job_ids, batch_codes):
        if grupo == "top20":
            sampled = _sample_top20_code(ds, codigo_cie10, rng)
        else:
            # Para 'otros', el código elegido en _select_batch_codes gobierna
            # el muestreo empírico: se toma un registro real DENTRO de ese código.
            sampled = _sample_otros(ds, codigo_cie10, rng)

        # Contrato del scheduler (modelo semántico corregido — aprobado por el profesor):
        #
        #   LÍNEA DE TIEMPO DEL QX (orden cronológico real):
        #
        #     [Paciente en Pabellón]
        #       → tiempo_transicion: Tiempo Pabellón a Quirófano (transporte al QX)
        #         ↓
        #     [Paciente en Quirófano]
        #       → setup_qx_anestesia: Tiempo Quirófano a Anestesia (preparación QX)
        #         ↓
        #     [Inicio Anestesia]
        #       → op1 (tiempo_anestesia = Tiempo Anestesia a Intervención)
        #         ↓
        #     [Inicio Intervención]
        #       → op2 (tiempo_cirugia = Tiempo Intervención)
        #         ↓
        #     [Fin Intervención]
        #       → cleanup (tiempo_limpieza = Tiempo Intervención a Salida)
        #         ↓
        #     [Salida Quirófano]
        #
        #   CONTEXTO: Tiempo Anestesia (legacy) es compuesto NO USAR como op.
        #
        #   ASIGNACIÓN AL SCHEDULER:
        #     op[1]               = tiempo_anestesia (standalone anesthesia phase)
        #     op[2]               = tiempo_cirugia
        #     setup_by_op[1]      = setup_qx_anestesia  (room prep)
        #     setup_by_op[2]      = tiempo_transicion   (patient transport — usado como
        #                           transición en el modelo del scheduler)
        #     transition_after_op1 = tiempo_transicion   (clave semántica explícita)
        #     cleanup_by_op[1]    = 0.0   (sin cleanup entre anestesia y cirugía)
        #     cleanup_by_op[2]    = tiempo_limpieza
        #
        #   NOTA: setup_by_op[2] = tiempo_transicion (NO la duración de anestesia
        #   anterior) porque el scheduler lo usa como tiempo entre Op1 y Op2.
        #   El campo 'transition_after_op1' expone el mismo valor semánticamente
        #   para Gantt y reportes.
        #
        #   Compatibilidad: el modo sintético (data_generator.py) no genera
        #   estas claves; el scheduler hace fallback a SETUP_TIMES/CLEANUP_TIMES.
        surgeries_data[job_id] = {
            # Tiempos de operación: op1=anestesia, op2=cirugía
            1: sampled["tiempo_anestesia"],
            2: sampled["tiempo_cirugia"],
            # Setup por operación
            "setup_by_op": {
                1: sampled["setup_qx_anestesia"],      # preparación del QX
                2: 0.0,                                 # no hay setup antes de cirugía
            },
            # Transición por operación (transporte paciente al QX)
            "transition_by_op": {
                1: sampled["tiempo_transicion"],
                2: 0.0,
            },
            # Transición explícita paciente (clave semántica para Gantt/reportes)
            "transition_after_op1": sampled["tiempo_transicion"],
            # Cleanup: solo después de op2
            "cleanup_by_op": {
                1: 0.0,   # no existe cleanup entre anestesia y cirugía
                2: sampled["tiempo_limpieza"],
            },
            # Legacy: suma total para trazabilidad externa (NO se usa en scheduler)
            "prep": sampled["tiempo_preparacion"],
            "cleanup": sampled["tiempo_limpieza"],
        }

        # Fila de trazabilidad cruda — columnas semánticas explícitas
        trace_row: dict = {
            "job_id": job_id,
            "grupo": sampled["grupo"],
            "codigo_cie10": sampled["codigo_cie10"],
            "source_record_id": sampled.get("source_record_id"),
            "tiempo_cirugia": sampled["tiempo_cirugia"],
            "tiempo_anestesia": sampled["tiempo_anestesia"],
            "tiempo_transicion": sampled["tiempo_transicion"],
            "tiempo_preparacion": sampled["tiempo_preparacion"],
            "tiempo_limpieza": sampled["tiempo_limpieza"],
            "setup_qx_anestesia": sampled["setup_qx_anestesia"],
            "estrategia_muestreo": sampled["estrategia_muestreo"],
            # Trazabilidad por operación (modelo semántico corregido):
            #   transition_to_or   = Tiempo Pabellón a Quirófano (transporte paciente)
            #   setup_op1          = Tiempo Quirófano a Anestesia (preparación QX)
            #   anesthesia_duration = Tiempo Anestesia a Intervención (fase anestesia)
            #   cleanup_op2        = Tiempo Intervención a Salida Quirófano
            #   setup_op2          = 0.0 — no existe setup entre anestesia y cirugía
            #   cleanup_op1        = 0.0 — no existe cleanup de op1
            "transition_to_or": sampled["tiempo_transicion"],
            "setup_op1": sampled["setup_qx_anestesia"],
            "anesthesia_duration": sampled["tiempo_anestesia"],
            "cleanup_op1": 0.0,
            "setup_op2": 0.0,
            "cleanup_op2": sampled["tiempo_limpieza"],
            # Confirmación: tiempos dinámicos reales en scheduler (no estáticos)
            "tiempos_dinamicos_en_simulacion": True,
        }

        if batch_trace_extras:
            trace_row.update(batch_trace_extras)

        batch_trace.append(trace_row)

    return surgeries_data, batch_trace


# ---------------------------------------------------------------------------
# Diagnóstico rápido (ejecutar directamente)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    seed_val = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    print(f"=== Real Batch Generator — Diagnóstico ===")
    print(f"n_jobs={n_jobs}, seed={seed_val}\n")

    job_ids_test = list(range(1, n_jobs + 1))
    data, trace = generate_day_surgeries_from_pkl(
        job_ids_test,
        seed=seed_val,
        batch_trace_extras={"simulation_id": "diag_test"},
    )

    print("Surgeries data (contrato para el scheduler):")
    for jid, ops in data.items():
        print(f"  job {jid}: op1(anest)={ops[1]:.1f}min  op2(cir)={ops[2]:.1f}min")

    print(f"\nBatch trace ({len(trace)} filas):")
    for row in trace:
        print(
            f"  job={row['job_id']:>3}  "
            f"grupo={row['grupo']:>5}  "
            f"cie10={row['codigo_cie10']:<12}  "
            f"rec={str(row['source_record_id']):>6}  "
            f"estrategia={row['estrategia_muestreo']}"
        )

    # Verificar distribución
    n_top20 = sum(1 for r in trace if r["grupo"] == "top20")
    n_otros = sum(1 for r in trace if r["grupo"] == "otros")
    pct_top20 = n_top20 / n_jobs * 100
    print(f"\nDistribución del lote:")
    print(f"  top20: {n_top20}/{n_jobs} ({pct_top20:.1f}%)")
    print(f"  otros: {n_otros}/{n_jobs} ({100 - pct_top20:.1f}%)")
    print(
        f"  Regla 70-80% top20: {'✓' if 70.0 <= pct_top20 <= 80.0 else '✗ (lote pequeño — rango normal)'}"
    )
