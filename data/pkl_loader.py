"""
pkl_loader.py — Capa de preparación de datos reales desde PKL hospitalario.

Propósito:
    Cargar, limpiar y clasificar los registros quirúrgicos reales del hospital
    a partir del archivo PKL procesado, exponiendo una API estable para que
    los módulos de simulación consuman datos empíricos.

Reglas funcionales implementadas:
    1. El grupo top20 corresponde al 20% de los códigos CIE10 con mayor
       frecuencia sobre el total de códigos válidos (no sobre registros).
    2. Los registros sin CIE10 se EXCLUYEN del modelado.
    3. Se registra trazabilidad completa: totales, excluidos y válidos.
    4. Toda la lógica opera con identificadores CIE10, no con nombres reales.
    5. Compatibilidad total con la arquitectura actual (data_generator.py intacto).

Estrategia de tiempo_preparacion:
    Se descompone en dos componentes aditivos, ambos disponibles en el PKL:
        - setup_qx_anestesia: 'Tiempo Quirófano a Anestesia'
          (desde ingreso al quirófano hasta inicio de anestesia)
        - setup_anestesia_interv: 'Tiempo Anestesia a Intervención'
          (desde inicio de anestesia hasta inicio de la cirugía)
    La suma de ambas representa el tiempo total de preparación preoperatoria.
    Se exponen ambos componentes y el total para máxima flexibilidad.

Criterios de filtrado de tiempos (conservadores y documentados):
    - Se reportan ceros en transición, setup, anestesia y limpieza sin excluirlos por sí solos.
    - Se excluyen registros donde tiempo_cirugia == 0 exacto (imposible físico).
    - Se reportan duraciones negativas y outliers extremos para trazabilidad.
    - NO se aplica filtro por outliers extremos en esta fase; se delega al
      análisis de distribución de la fase 2 para no perder información.
    - Cirugía mantiene umbral exclusivo > 0; limpieza mantiene umbral inclusivo
      >= 0 para reportar ceros sin descartarlos por sí solos.

Autor: capa de migración a datos reales — Fase 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes internas
# ---------------------------------------------------------------------------

# Ruta por defecto al PKL — relativa al directorio del módulo (data/)
_DEFAULT_PKL_PATH = Path(__file__).parent.parent / "datasets" / "2_dataset_procesado_actualizado.pkl"

# Nombres de columnas en el PKL (strings con caracteres especiales UTF-8 intactos)
# Se acceden por nombre a través de _resolve_col para evitar problemas de encoding,
# pero se documenta aquí el nombre semántico para legibilidad.
_COL_CIE10 = "Diag. Post. 1 (CIE10)"  # col 9 — objeto, 16% nulos
_COL_ANESTESIA_LEGACY = "Tiempo Anestesia"  # col 28 — str HH:MM:SS, 0 nulos — LEGACY composite

# Las seis columnas base (fuente de verdad temporal)
_COL_ING_PABELLON = "Ingreso Pabellón"
_COL_ING_QUIROFANO = "Ingreso Quirófano"
_COL_INI_ANESTESIA = "Inicio Anestesia"
_COL_INI_INTERVENCION = "Inicio Intervención"
_COL_TER_INTERVENCION = "Término Intervención"
_COL_SAL_QUIROFANO = "Salida Quirófano"

# Fracción para definir el grupo "top N%"
_TOP_FRACTION = 0.20


_TIMING_CONTRACT = {
    "transition": {
        "column": "tiempo_transicion",
        "source_column": f"{_COL_ING_QUIROFANO} - {_COL_ING_PABELLON}",
        "outlier_threshold_minutes": 60.0,
    },
    "setup": {
        "column": "setup_qx_anestesia",
        "source_column": f"{_COL_INI_ANESTESIA} - {_COL_ING_QUIROFANO}",
        "outlier_threshold_minutes": 60.0,
    },
    "anesthesia": {
        "column": "tiempo_anestesia",
        "source_column": f"{_COL_INI_INTERVENCION} - {_COL_INI_ANESTESIA}",
        "outlier_threshold_minutes": 120.0,
    },
    "surgery": {
        "column": "tiempo_cirugia",
        "source_column": f"{_COL_TER_INTERVENCION} - {_COL_INI_INTERVENCION}",
        "outlier_threshold_minutes": 360.0,
    },
    "cleanup": {
        "column": "tiempo_limpieza",
        "source_column": f"{_COL_SAL_QUIROFANO} - {_COL_TER_INTERVENCION}",
        "outlier_threshold_minutes": 60.0,
    },
}


# ---------------------------------------------------------------------------
# Funciones auxiliares internas
# ---------------------------------------------------------------------------


def _resolve_col(df: pd.DataFrame, expected: str) -> str:
    """
    Devuelve el nombre real de la columna en el DataFrame que coincide
    semánticamente con `expected`. Necesario porque el encoding de las columnas
    con tildes puede variar según la plataforma.

    Estrategia:
        1. Match exacto (caso ideal).
        2. Match por bytes UTF-8 (por si la terminal recodifica).
        3. Match por posición si se conoce el índice (fallback).

    Lanza KeyError si no puede resolver.
    """
    if expected in df.columns:
        return expected

    # Fallback: comparar representación de bytes UTF-8
    expected_bytes = expected.encode("utf-8")
    for col in df.columns:
        if col.encode("utf-8") == expected_bytes:
            return col

    raise KeyError(
        f"No se encontró la columna '{expected}' en el PKL. "
        f"Columnas disponibles (primeras 15): {list(df.columns[:15])}"
    )


def _hhmmss_to_minutes(s: object) -> float:
    """
    Convierte una cadena 'HH:MM:SS' a minutos decimales.
    Retorna NaN ante valores nulos o malformados.

    Ejemplos:
        '01:49:10' → 109.17
        '00:30:00' → 30.0
        NaN       → NaN
    """
    if pd.isna(s):
        return float("nan")
    try:
        parts = str(s).strip().split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) > 2 else 0
        return hours * 60.0 + minutes + seconds / 60.0
    except (ValueError, IndexError):
        return float("nan")


def _record_examples(
    df: pd.DataFrame,
    mask: pd.Series,
    *,
    stage: str,
    column: str,
    kind: str,
    limit: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if limit <= 0:
        return examples

    matching = (
        df.loc[mask, ["record_id", column]]
        if "record_id" in df.columns
        else df.loc[mask, [column]]
    )
    for idx, row in matching.head(limit).iterrows():
        raw_record_id = row["record_id"] if "record_id" in matching.columns else idx
        raw_value = row[column]
        examples.append(
            {
                "record_id": int(raw_record_id) if not pd.isna(raw_record_id) else None,
                "stage": stage,
                "column": column,
                "kind": kind,
                "value_minutes": float(raw_value) if not pd.isna(raw_value) else None,
            }
        )
    return examples


def build_timing_quality_report(
    df: pd.DataFrame,
    *,
    example_limit: int = 5,
) -> dict[str, Any]:
    """
    Build a non-mutating timing anomaly report for normalized PKL timing rows.

    The report follows the current hospital timing contract and only reports
    anomalies. It does not cap, replace, clamp, or otherwise mutate source
    values.
    """
    stage_reports: dict[str, dict[str, Any]] = {}
    total_negative = 0
    total_zero = 0
    total_extreme_outliers = 0
    invalid_surgery_zero_count = 0

    for stage, config in _TIMING_CONTRACT.items():
        column = config["column"]
        if column not in df.columns:
            raise KeyError(f"Missing timing column '{column}' for stage '{stage}'")

        values = pd.to_numeric(df[column], errors="coerce")
        negative_mask = values < 0
        zero_mask = values == 0
        threshold = float(config["outlier_threshold_minutes"])
        outlier_mask = values > threshold

        negative_count = int(negative_mask.sum())
        zero_count = int(zero_mask.sum())
        outlier_count = int(outlier_mask.sum())

        if stage == "surgery":
            invalid_surgery_zero_count = zero_count

        total_negative += negative_count
        total_zero += zero_count
        total_extreme_outliers += outlier_count

        stage_reports[stage] = {
            "column": column,
            "source_column": config["source_column"],
            "negative_count": negative_count,
            "zero_count": zero_count,
            "extreme_outlier_count": outlier_count,
            "outlier_threshold_minutes": threshold,
            "examples": {
                "negative": _record_examples(
                    df,
                    negative_mask,
                    stage=stage,
                    column=column,
                    kind="negative",
                    limit=example_limit,
                ),
                "zero": _record_examples(
                    df,
                    zero_mask,
                    stage=stage,
                    column=column,
                    kind="zero",
                    limit=example_limit,
                ),
                "extreme_outlier": _record_examples(
                    df,
                    outlier_mask,
                    stage=stage,
                    column=column,
                    kind="extreme_outlier",
                    limit=example_limit,
                ),
            },
        }

    return {
        "rows_evaluated": int(len(df)),
        "example_limit": int(example_limit),
        "policy": {
            "zero_transition_setup_anesthesia_cleanup": "report_only",
            "zero_surgery": "invalid_suspicious",
            "negative_durations": "report",
            "extreme_outliers": "alert_only",
        },
        "summary": {
            "total_negative_values": total_negative,
            "total_zero_values": total_zero,
            "invalid_surgery_zero_count": invalid_surgery_zero_count,
            "total_extreme_outliers": total_extreme_outliers,
        },
        "stages": stage_reports,
    }


# ---------------------------------------------------------------------------
# Carga y preparación principal
# ---------------------------------------------------------------------------


def load_and_prepare(
    pkl_path: Optional[str | Path] = None,
    *,
    min_surgery_minutes: float = 0.0,
    min_cleanup_minutes: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Carga el PKL, aplica limpieza conservadora y devuelve el dataset válido
    junto con metadata de trazabilidad.

    Parámetros
    ----------
    pkl_path : str | Path | None
        Ruta al archivo PKL. Si es None, usa el path por defecto en data/.
    min_surgery_minutes : float
        Umbral mínimo para tiempo_cirugia (exclusivo). Por defecto 0.0:
        solo se excluyen valores exactamente iguales a cero.
    min_cleanup_minutes : float
        Umbral mínimo para tiempo_limpieza (inclusivo). Por defecto 0.0:
        los ceros se reportan, pero no se excluyen por sí solos.

    Retorna
    -------
    df_clean : pd.DataFrame
        DataFrame con solo los registros válidos y columnas normalizadas:
            record_id                     — índice original del PKL (int64, único)
            codigo_cie10                  — código CIE10 (str)
            tiempo_cirugia                — Tiempo Intervención en minutos (float)
            tiempo_anestesia              — Tiempo Anestesia a Intervención (minutos, float)
                                              FASE DE ANESTESIA INDEPENDIENTE.
            tiempo_anestesia_legacy_total — Tiempo Anestesia compuesto legacy (minutos, float)
                                              Inicio Anestesia → Salida Quirófano.
                                              NO usar como operación independiente.
            tiempo_transicion             — Tiempo Pabellón a Quirófano (minutos, float)
                                              Transporte del paciente al quirófano.
            setup_qx_anestesia            — Quirófano→Anestesia en minutos (float)
            tiempo_limpieza               — Tiempo Intervención→Salida Quirófano (float)
            tiempo_preparacion            — tiempo_transicion + setup_qx_anestesia (float)

    metadata : dict
        Diccionario con trazabilidad completa:
            total_original          — filas en el PKL crudo
            excluidos_cie10_nulo    — filas sin CIE10 válido
            excluidos_tiempo_inv    — filas con tiempos inválidos (post-CIE10)
            validos_finales         — filas que pasaron todos los filtros
            n_codigos_unicos        — cantidad de códigos CIE10 únicos válidos
            top20_fraction          — fracción usada para el grupo top (0.20)
            n_top20_codigos         — cantidad de códigos en el grupo top20
            n_otros_codigos         — cantidad de códigos en el grupo otros
            records_top20           — registros pertenecientes al top20
            records_otros           — registros pertenecientes al grupo otros
    """
    path = Path(pkl_path) if pkl_path else _DEFAULT_PKL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el PKL en '{path}'. Verifica que el archivo esté en data/."
        )

    # --- Carga bruta ---
    df_raw: pd.DataFrame = pd.read_pickle(path)
    total_original = len(df_raw)

    # --- Resolución de columnas (resiliente a encoding) ---
    col_cie10 = _resolve_col(df_raw, _COL_CIE10)
    col_anestesia_legacy = _resolve_col(df_raw, _COL_ANESTESIA_LEGACY)
    col_ing_pabellon = _resolve_col(df_raw, _COL_ING_PABELLON)
    col_ing_quirofano = _resolve_col(df_raw, _COL_ING_QUIROFANO)
    col_ini_anestesia = _resolve_col(df_raw, _COL_INI_ANESTESIA)
    col_ini_intervencion = _resolve_col(df_raw, _COL_INI_INTERVENCION)
    col_ter_intervencion = _resolve_col(df_raw, _COL_TER_INTERVENCION)
    col_sal_quirofano = _resolve_col(df_raw, _COL_SAL_QUIROFANO)

    # --- Filtro 1: excluir CIE10 nulo o vacío ---
    mask_cie10_valido = (
        df_raw[col_cie10].notna()
        & (df_raw[col_cie10].astype(str).str.strip() != "")
        & (df_raw[col_cie10].astype(str).str.strip().str.lower() != "nan")
    )
    excluidos_cie10_nulo = int((~mask_cie10_valido).sum())
    df_cie10 = df_raw[mask_cie10_valido].copy()

    # --- Conversión Tiempo Anestesia (legacy) HH:MM:SS → minutos ---
    df_cie10 = df_cie10.copy()
    anestesia_legacy_min = df_cie10[col_anestesia_legacy].apply(_hhmmss_to_minutes)

    # Parse base timestamps as datetimes to ensure correct math
    t_ing_pabellon = pd.to_datetime(df_cie10[col_ing_pabellon], errors="coerce")
    t_ing_quirofano = pd.to_datetime(df_cie10[col_ing_quirofano], errors="coerce")
    t_ini_anestesia = pd.to_datetime(df_cie10[col_ini_anestesia], errors="coerce")
    t_ini_intervencion = pd.to_datetime(df_cie10[col_ini_intervencion], errors="coerce")
    t_ter_intervencion = pd.to_datetime(df_cie10[col_ter_intervencion], errors="coerce")
    t_sal_quirofano = pd.to_datetime(df_cie10[col_sal_quirofano], errors="coerce")

    # --- Construcción del dataset normalizado recalculando desde las bases ---
    # El modelo semántico de tiempos recalculados (fuente contractual):
    #
    #   tiempo_transicion      = Ingreso Quirófano - Ingreso Pabellón
    #   setup_qx_anestesia     = Inicio Anestesia - Ingreso Quirófano
    #   tiempo_anestesia       = Inicio Intervención - Inicio Anestesia
    #   tiempo_cirugia         = Término Intervención - Inicio Intervención
    #   tiempo_limpieza        = Salida Quirófano - Término Intervención
    #
    df_norm = pd.DataFrame(
        {
            "record_id": df_cie10.index,
            "codigo_cie10": df_cie10[col_cie10].str.strip(),
            "tiempo_cirugia": (t_ter_intervencion - t_ini_intervencion).dt.total_seconds() / 60.0,
            "tiempo_anestesia": (t_ini_intervencion - t_ini_anestesia).dt.total_seconds() / 60.0,
            "tiempo_anestesia_legacy_total": anestesia_legacy_min.values,
            "tiempo_transicion": (t_ing_quirofano - t_ing_pabellon).dt.total_seconds() / 60.0,
            "setup_qx_anestesia": (t_ini_anestesia - t_ing_quirofano).dt.total_seconds() / 60.0,
            "tiempo_limpieza": (t_sal_quirofano - t_ter_intervencion).dt.total_seconds() / 60.0,
        },
        index=df_cie10.index,
    )

    # tiempo_preparacion = transporte paciente + preparación del QX
    df_norm["tiempo_preparacion"] = (
        df_norm["tiempo_transicion"] + df_norm["setup_qx_anestesia"]
    )

    timing_quality_report = build_timing_quality_report(df_norm)

    # --- Filtro 2: tiempos inválidos (criterio conservador documentado) ---
    # Solo se excluyen valores estrictamente <= umbral (por defecto 0).
    # La filosofía: no asumir umbrales clínicos que no tenemos confirmados;
    # solo descartar lo que es físicamente imposible (cero o negativo).
    # tiempo_transicion puede ser 0 (paciente ya en QX) o NaN — no se filtra agresivamente.
    mask_tiempos_validos = (
        (df_norm["tiempo_cirugia"] > min_surgery_minutes)
        & (df_norm["tiempo_limpieza"] >= min_cleanup_minutes)
        & df_norm["tiempo_anestesia"].notna()
    )
    excluidos_tiempo_inv = int((~mask_tiempos_validos).sum())
    df_pre_clean = df_norm[mask_tiempos_validos].copy()

    # --- Filtro 3: anomalías de anestesia prolongada con cirugía corta ---
    # Se excluyen registros con tiempo_anestesia > 30 min y tiempo_cirugia <= 10 min
    # (anestesia larga para cirugía muy corta — clínicamente sospechoso).
    mask_anomalia_transicion = (
        (df_pre_clean["tiempo_anestesia"] > 30.0)
        & (df_pre_clean["tiempo_cirugia"] <= 10.0)
    )
    excluidos_anomalia_transicion = int(mask_anomalia_transicion.sum())
    df_clean = df_pre_clean[~mask_anomalia_transicion].copy()
    validos_finales = len(df_clean)

    # --- Construcción de metadata de trazabilidad ---
    freq = df_clean["codigo_cie10"].value_counts()
    n_codigos_unicos = len(freq)
    n_top20_codigos = int(n_codigos_unicos * _TOP_FRACTION)
    n_otros_codigos = n_codigos_unicos - n_top20_codigos

    top20_codes = set(freq.head(n_top20_codigos).index)
    records_top20 = int((df_clean["codigo_cie10"].isin(top20_codes)).sum())
    records_otros = validos_finales - records_top20

    metadata: dict = {
        "total_original": total_original,
        "excluidos_cie10_nulo": excluidos_cie10_nulo,
        "excluidos_tiempo_inv": excluidos_tiempo_inv,
        "excluidos_anomalia_transicion": excluidos_anomalia_transicion,
        "validos_finales": validos_finales,
        "n_codigos_unicos": n_codigos_unicos,
        "top20_fraction": _TOP_FRACTION,
        "n_top20_codigos": n_top20_codigos,
        "n_otros_codigos": n_otros_codigos,
        "records_top20": records_top20,
        "records_otros": records_otros,
        "pkl_path": str(path),
        "timing_quality_report": timing_quality_report,
    }

    return df_clean, metadata


# ---------------------------------------------------------------------------
# Clasificación top20 / otros
# ---------------------------------------------------------------------------


class CIE10Dataset:
    """
    Interfaz de alto nivel sobre el dataset preparado.

    Encapsula la clasificación top20/otros y expone la API que usará
    la capa de simulación en fases posteriores.

    Uso típico
    ----------
    >>> ds = CIE10Dataset.from_pkl()
    >>> print(ds.metadata)
    >>> df_top20 = ds.df_top20
    >>> df_otros = ds.df_otros
    >>> lista_top20 = ds.top20_codes
    """

    def __init__(self, df_clean: pd.DataFrame, metadata: dict) -> None:
        self._df = df_clean
        self.metadata = metadata
        self._top20_codes: frozenset[str] | None = None
        self._otros_codes: frozenset[str] | None = None
        self._df_top20: pd.DataFrame | None = None
        self._df_otros: pd.DataFrame | None = None
        self._build_classification()

    # ------------------------------------------------------------------
    # Constructor de conveniencia
    # ------------------------------------------------------------------

    @classmethod
    def from_pkl(
        cls,
        pkl_path: Optional[str | Path] = None,
        **kwargs,
    ) -> "CIE10Dataset":
        """
        Carga el PKL y construye el dataset clasificado.

        Parámetros
        ----------
        pkl_path : str | Path | None
            Ruta al PKL. Si None, usa el path por defecto.
        **kwargs
            Argumentos adicionales para `load_and_prepare`.
        """
        df_clean, metadata = load_and_prepare(pkl_path, **kwargs)
        return cls(df_clean, metadata)

    # ------------------------------------------------------------------
    # Clasificación interna
    # ------------------------------------------------------------------

    def _build_classification(self) -> None:
        """
        Calcula el top 20% de códigos CIE10 por frecuencia y separa los datos.

        Criterio exacto:
            top20 = los primeros int(total_codigos * 0.20) códigos ordenados
            por frecuencia descendente. En caso de empate en la frontera,
            value_counts() los ordena por primera aparición (comportamiento
            estable de pandas).
        """
        freq = self._df["codigo_cie10"].value_counts()
        n_top = self.metadata["n_top20_codigos"]

        top_codes = frozenset(freq.head(n_top).index)
        otros_codes = frozenset(freq.index) - top_codes

        self._top20_codes = top_codes
        self._otros_codes = otros_codes

        mask_top = self._df["codigo_cie10"].isin(top_codes)
        self._df_top20 = self._df[mask_top].copy()
        self._df_otros = self._df[~mask_top].copy()

    # ------------------------------------------------------------------
    # Propiedades públicas
    # ------------------------------------------------------------------

    @property
    def df_clean(self) -> pd.DataFrame:
        """Dataset completo limpio con todos los registros válidos."""
        return self._df

    @property
    def top20_codes(self) -> frozenset[str]:
        """Conjunto inmutable de códigos CIE10 en el grupo top20."""
        return self._top20_codes  # type: ignore[return-value]

    @property
    def otros_codes(self) -> frozenset[str]:
        """Conjunto inmutable de códigos CIE10 en el grupo 'otros'."""
        return self._otros_codes  # type: ignore[return-value]

    @property
    def df_top20(self) -> pd.DataFrame:
        """DataFrame con solo los registros del grupo top20."""
        return self._df_top20  # type: ignore[return-value]

    @property
    def df_otros(self) -> pd.DataFrame:
        """DataFrame con solo los registros del grupo 'otros'."""
        return self._df_otros  # type: ignore[return-value]

    def is_top20(self, codigo_cie10: str) -> bool:
        """Retorna True si el código pertenece al grupo top20."""
        return codigo_cie10 in self._top20_codes  # type: ignore[operator]

    def get_records_for_code(self, codigo_cie10: str) -> pd.DataFrame:
        """Retorna todos los registros históricos de un código CIE10."""
        return self._df[self._df["codigo_cie10"] == codigo_cie10].copy()

    # ------------------------------------------------------------------
    # Trazabilidad y exportación
    # ------------------------------------------------------------------

    def summary_dict(self) -> dict:
        """
        Devuelve un diccionario plano con los campos de trazabilidad
        mínima para exportar a CSV o log.

        Campos:
            total_registros_originales
            excluidos_cie10_nulo
            excluidos_tiempo_invalido
            validos_finales
            n_codigos_validos
            n_codigos_top20
            n_codigos_otros
            registros_top20
            registros_otros
            top20_fraccion
            pkl_path
        """
        m = self.metadata
        timing_quality_summary = m.get("timing_quality_report", {}).get("summary", {})
        return {
            "total_registros_originales": m["total_original"],
            "excluidos_cie10_nulo": m["excluidos_cie10_nulo"],
            "excluidos_tiempo_invalido": m["excluidos_tiempo_inv"],
            "excluidos_anomalia_transicion": m.get("excluidos_anomalia_transicion", 0),
            "timing_quality_negative_values": timing_quality_summary.get("total_negative_values", 0),
            "timing_quality_zero_values": timing_quality_summary.get("total_zero_values", 0),
            "timing_quality_invalid_surgery_zero": timing_quality_summary.get("invalid_surgery_zero_count", 0),
            "timing_quality_extreme_outliers": timing_quality_summary.get("total_extreme_outliers", 0),
            "validos_finales": m["validos_finales"],
            "n_codigos_validos": m["n_codigos_unicos"],
            "n_codigos_top20": m["n_top20_codigos"],
            "n_codigos_otros": m["n_otros_codigos"],
            "registros_top20": m["records_top20"],
            "registros_otros": m["records_otros"],
            "top20_fraccion": m["top20_fraction"],
            "pkl_path": m["pkl_path"],
        }

    def export_summary_csv(self, output_path: str | Path) -> Path:
        """
        Exporta el resumen de preparación de datos a un CSV de una sola fila.

        Útil para trazabilidad del trabajo de investigación / ablation study.

        Parámetros
        ----------
        output_path : str | Path
            Ruta del archivo CSV de destino.

        Retorna
        -------
        Path
            Ruta al archivo creado.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = self.summary_dict()
        pd.DataFrame([summary]).to_csv(out, index=False)
        return out

    def __repr__(self) -> str:  # pragma: no cover
        m = self.metadata
        return (
            f"CIE10Dataset("
            f"validos={m['validos_finales']}, "
            f"codigos={m['n_codigos_unicos']}, "
            f"top20={m['n_top20_codigos']} códigos / {m['records_top20']} registros, "
            f"otros={m['n_otros_codigos']} códigos / {m['records_otros']} registros"
            f")"
        )


# ---------------------------------------------------------------------------
# Contrato para la fase 2 (placeholder documentado)
# ---------------------------------------------------------------------------
# La fase 2 consumirá este módulo así:
#
#   ds = CIE10Dataset.from_pkl()
#
#   if ds.is_top20(codigo):
#       # top20 → ajustar distribución normal por código
#       records = ds.get_records_for_code(codigo)
#       mean_cirugia = records["tiempo_cirugia"].mean()
#       std_cirugia  = records["tiempo_cirugia"].std()
#       t_cirugia    = max(1.0, np.random.normal(mean_cirugia, std_cirugia))
#
#   else:
#       # otros → muestreo empírico directo (resample de un registro aleatorio)
#       records = ds.get_records_for_code(codigo)
#       if len(records) == 0:
#           # código nunca visto → fallback al grupo "otros" completo
#           records = ds.df_otros
#       row = records.sample(1).iloc[0]
#       t_cirugia = row["tiempo_cirugia"]
#
# Este contrato NO se implementa todavía para no romper la arquitectura actual.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Entrypoint de diagnóstico rápido (ejecutar directamente)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== PKL Loader — Diagnóstico ===\n")

    pkl_arg = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        ds = CIE10Dataset.from_pkl(pkl_arg)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    summary = ds.summary_dict()
    print("Resumen de preparación de datos:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print()
    print(repr(ds))

    print()
    print("Primeros 5 registros del dataset limpio:")
    print(ds.df_clean.head().to_string())

    print()
    print("Top 10 códigos CIE10 (por frecuencia):")
    freq = ds.df_clean["codigo_cie10"].value_counts().head(10)
    for code, count in freq.items():
        grupo = "TOP20" if ds.is_top20(code) else "otros"
        print(f"  [{grupo}] {code}: {count} registros")

    # Exportar resumen CSV de trazabilidad
    out_csv = Path(__file__).parent.parent / "datasets" / "data_preparation_summary.csv"
    ds.export_summary_csv(out_csv)
    print(f"\nResumen CSV exportado a: {out_csv}")
