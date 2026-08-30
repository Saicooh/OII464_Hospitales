"""
raw_trace_writer.py — Exportador del CSV de trazabilidad cruda de lotes simulados.

Propósito:
    Persistir TODOS los datos crudos usados para construir cada lote simulado.
    Cada fila del CSV representa un job (cirugía) dentro de una simulación,
    con su código CIE10, grupo, tiempos y estrategia de muestreo.

Formato del CSV:
    simulation_id, job_id, grupo, codigo_cie10, source_record_id,
    tiempo_cirugia, tiempo_anestesia, tiempo_transicion, tiempo_preparacion,
    tiempo_limpieza, setup_qx_anestesia, estrategia_muestreo

Estrategia de escritura — un archivo por simulación:
    joblib/loky corre las simulaciones en procesos separados (no threads).
    Un threading.Lock() NO protege escrituras entre procesos, lo que puede
    corromper un CSV compartido.

    Para evitar corrupción sin dependencias externas, cada simulación escribe
    su propio archivo:
        results/csv/raw_batch_trace_sim_<simulation_id>.csv

    Esto es naturalmente seguro porque cada proceso escribe en su propio
    archivo independiente. No hay lock necesario.

    Para obtener la traza consolidada de todos las simulaciones, usar:
        consolidate_traces(base_dir)  →  raw_batch_trace_all.csv

Compatibilidad:
    - `write_batch_trace()` y `reset_trace_file()` mantienen sus firmas
      originales; solo cambia la semántica de `output_path` (se usa como
      directorio base para construir el nombre por sim_id).
    - `get_default_trace_path()` devuelve el directorio base (results/csv/).

Autor: Fase 2 — integración datos reales.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Columnas canónicas del CSV (orden fijo y documentado)
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "simulation_id",
    "job_id",
    "grupo",
    "codigo_cie10",
    "source_record_id",
    "tiempo_cirugia",
    "tiempo_anestesia",
    "tiempo_transicion",
    "tiempo_preparacion",
    "tiempo_limpieza",
    "setup_qx_anestesia",
    "estrategia_muestreo",
    # Columnas de trazabilidad semántica por operación (modelo corregido):
    #   transition_to_or       = Tiempo Pabellón a Quirófano (transporte paciente al QX)
    #   setup_op1              = Tiempo Quirófano a Anestesia (preparación del QX)
    #   anesthesia_duration    = Tiempo Anestesia a Intervención (fase de anestesia)
    #   cleanup_op1            = 0.0 (no hay limpieza entre operaciones)
    #   setup_op2              = 0.0 (no hay setup entre anestesia y cirugía)
    #   cleanup_op2            = tiempo_limpieza (limpieza post-intervención)
    "transition_to_or",
    "setup_op1",
    "anesthesia_duration",
    "cleanup_op1",
    "setup_op2",
    "cleanup_op2",
    # Columna de trazabilidad de integración:
    # Indica si los tiempos dinámicos por operación fueron usados realmente
    # en el cálculo del schedule (True) o si el scheduler usó los estáticos
    # del config como fallback (False). Siempre True para datos PKL reales.
    "tiempos_dinamicos_en_simulacion",
]


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _trace_path_for_sim(base_dir: str | Path, simulation_id: str | int) -> Path:
    """
    Construye la ruta del CSV de trazabilidad para una simulación específica.

    Esquema: <base_dir>/raw_batch_trace_sim_<simulation_id>.csv

    Cada simulación tiene su propio archivo, lo que garantiza seguridad
    con joblib/loky sin necesidad de locks entre procesos.
    """
    return Path(base_dir) / f"raw_batch_trace_sim_{simulation_id}.csv"


# ---------------------------------------------------------------------------
# Función principal de escritura
# ---------------------------------------------------------------------------


def write_batch_trace(
    trace_rows: Sequence[dict],
    output_path: str | Path,
    simulation_id: str | int,
) -> None:
    """
    Escribe las filas de trazabilidad cruda de un lote al CSV de la simulación.

    Cada simulación escribe su propio archivo:
        <output_path>/raw_batch_trace_sim_<simulation_id>.csv

    Esto es inherentemente seguro con joblib/loky porque cada proceso escribe
    en su propio archivo (no hay escrituras concurrentes sobre el mismo archivo).

    Parámetros
    ----------
    trace_rows : Sequence[dict]
        Lista de dicts devuelta por `generate_day_surgeries_from_pkl()`.
        Cada dict representa un job del lote.
    output_path : str | Path
        Directorio base donde se escriben los CSV por simulación.
        (históricamente era la ruta del archivo único; ahora es el directorio
        padre — compatible con get_default_trace_path()).
    simulation_id : str | int
        Identificador de la simulación. Forma parte del nombre del archivo
        y se agrega como columna a cada fila.

    Comportamiento
    --------------
    - Crea el directorio si no existe.
    - Cada simulación escribe un archivo propio (overwrite completo, no append),
      ya que cada simulación es un lote único y no acumula sobre sí misma.
    - Seguro entre procesos: sin locks, sin riesgo de corrupción.
    """
    if not trace_rows:
        return

    # output_path puede ser un archivo .csv (legado) o un directorio.
    # Si es archivo .csv, usamos su directorio padre como base.
    out_path = Path(output_path)
    if out_path.suffix == ".csv":
        base_dir = out_path.parent
    else:
        base_dir = out_path

    base_dir.mkdir(parents=True, exist_ok=True)

    sim_csv = _trace_path_for_sim(base_dir, simulation_id)

    with open(sim_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=_CSV_COLUMNS,
            extrasaction="ignore",  # ignora campos extra en el dict
        )
        writer.writeheader()

        for row in trace_rows:
            enriched = {"simulation_id": simulation_id, **row}
            writer.writerow(enriched)


def reset_trace_file(output_path: str | Path) -> None:
    """
    Elimina todos los CSV de trazabilidad por simulación del directorio.

    Con la estrategia de un-CSV-por-simulación, "resetear" significa
    eliminar todos los archivos `raw_batch_trace_sim_*.csv` del directorio
    base. No hay un único archivo que eliminar.

    Parámetros
    ----------
    output_path : str | Path
        Ruta de referencia. Si es un archivo .csv, se usa su directorio padre.
        Si es directorio, se usa directamente.
    """
    out_path = Path(output_path)
    if out_path.suffix == ".csv":
        base_dir = out_path.parent
    else:
        base_dir = out_path

    if not base_dir.exists():
        return

    deleted = 0
    for f in base_dir.glob("raw_batch_trace_sim_*.csv"):
        f.unlink()
        deleted += 1

    # Si existe el legacy raw_batch_trace.csv (archivo único anterior), borrarlo también
    legacy = base_dir / "raw_batch_trace.csv"
    if legacy.exists():
        legacy.unlink()
        deleted += 1


def consolidate_traces(
    output_path: str | Path,
    *,
    out_filename: str = "raw_batch_trace_all.csv",
    delete_partials: bool = False,
) -> Path | None:
    """
    Consolida todos los CSV por simulación en un único CSV acumulado.

    Útil para análisis post-ejecución o para tener la traza completa de
    todas las simulaciones en un solo lugar.

    Parámetros
    ----------
    output_path : str | Path
        Directorio base (o ruta legado .csv — se usa su directorio padre).
    out_filename : str
        Nombre del archivo consolidado. Por defecto 'raw_batch_trace_all.csv'.
    delete_partials : bool
        Si True, elimina los archivos individuales tras consolidar.
        Por defecto False (se conservan para reproducibilidad).

    Retorna
    -------
    Path al archivo consolidado, o None si no había archivos que consolidar.
    """
    out_path = Path(output_path)
    if out_path.suffix == ".csv":
        base_dir = out_path.parent
    else:
        base_dir = out_path

    partial_files = sorted(base_dir.glob("raw_batch_trace_sim_*.csv"))

    if not partial_files:
        return None

    consolidated = base_dir / out_filename

    with open(consolidated, mode="w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for partial in partial_files:
            with open(partial, newline="", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    writer.writerow(row)

    if delete_partials:
        for partial in partial_files:
            partial.unlink()

    return consolidated


def get_default_trace_path(base_dir: str | Path = "results") -> Path:
    """
    Devuelve el directorio base donde se escriben los CSV de trazabilidad.

    Con la estrategia de un-CSV-por-simulación, esta función devuelve el
    directorio (results/csv/) en vez de una ruta de archivo único.

    Para construir la ruta de un sim específico: _trace_path_for_sim(base, sim_id)
    Para el consolidado: base_dir / 'raw_batch_trace_all.csv'

    Parámetros
    ----------
    base_dir : str | Path
        Directorio raíz de resultados. Por defecto 'results/'.

    Retorna
    -------
    Path
        Ruta al directorio csv/ dentro de base_dir.
    """
    return Path(base_dir) / "csv"
