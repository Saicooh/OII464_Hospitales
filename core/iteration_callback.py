"""
core/iteration_callback.py

Contratos y handler para snapshots por iteración interna de algoritmos
en analysis mode. No escribe disco; acumula snapshots serializables en memoria
para que el proceso padre decida qué persistir.

Este módulo NO depende de ningún algoritmo específico — es inyectado como
parámetro callable (`on_iteration`) en las funciones `run()` de cada algoritmo.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------


class ArtifactSaveMode:
    """Constantes para la política de persistencia de artefactos por iteración."""

    BEST_ONLY = "best_only"
    """Solo persiste si el fitness mejora el mejor global. Evita saturación de disco."""

    ALL = "all"
    """Persiste todas las iteraciones, sin filtro."""

    SAMPLED = "sampled"
    """Persiste un subconjunto muestreado (reservado para implementación futura)."""


# ---------------------------------------------------------------------------
# Snapshot data structure
# ---------------------------------------------------------------------------


@dataclass
class IterationSnapshot:
    """Snapshot serializable de una iteración interna del algoritmo.

    Contiene solo escalares — no objetos numpy ni referencias a soluciones
    completas — para ser seguro en IPC entre procesos.

    Attributes
    ----------
    algo_step:
        Paso algorítmico neutral (entero ≥ 1). Reemplaza 'generation'/'iteration'
        para ser agnóstico al algoritmo (GA, DPSO, SBOA, DMSHOA).
    best_fitness:
        Mejor fitness global acumulado hasta este paso (gbest o best_overall).
    best_makespan:
        Mejor makespan global acumulado hasta este paso.
    iteration_fitness:
        Mejor fitness del paso actual (puede ser peor que best_fitness).
    iteration_makespan:
        Mejor makespan del paso actual.
    """

    algo_step: int
    best_fitness: float
    best_makespan: float
    iteration_fitness: float
    iteration_makespan: float
    best_solution_snapshot: Optional[dict] = field(default=None)


# ---------------------------------------------------------------------------
# IterationCallback Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IterationCallback(Protocol):
    """Protocolo para callbacks de iteración de algoritmos.

    Cualquier callable con esta firma puede usarse como ``on_iteration``
    en los algoritmos. No requiere herencia — solo cumplir la firma.
    """

    def __call__(
        self,
        algo_step: int,
        best_fitness: float,
        best_makespan: float,
        iteration_fitness: float,
        iteration_makespan: float,
        best_solution_snapshot: Optional[dict] = None,
    ) -> None:
        """Invocado al final de cada paso algorítmico."""
        ...


# ---------------------------------------------------------------------------
# AnalysisIterationHandler
# ---------------------------------------------------------------------------


class AnalysisIterationHandler:
    """Acumula snapshots de iteraciones según la política configurada.

    Diseñado para ser inyectado como ``on_iteration`` callback en algoritmos
    de optimización. No realiza I/O — el proceso padre decide qué persistir.

    Parameters
    ----------
    policy:
        Política de acumulación. Una de ``ArtifactSaveMode.BEST_ONLY``,
        ``ArtifactSaveMode.ALL`` o ``ArtifactSaveMode.SAMPLED``.

    Attributes
    ----------
    snapshots:
        Lista de ``IterationSnapshot`` acumulados en la simulación actual.
    best_fitness:
        Mejor fitness visto hasta ahora. ``None`` antes del primer snapshot.
    """

    def __init__(self, policy: str = ArtifactSaveMode.BEST_ONLY) -> None:
        self._policy = policy
        self.snapshots: list[IterationSnapshot] = []
        self.best_fitness: Optional[float] = None

    def __call__(
        self,
        algo_step: int,
        best_fitness: float,
        best_makespan: float,
        iteration_fitness: float,
        iteration_makespan: float,
        best_solution_snapshot: Optional[dict] = None,
    ) -> None:
        """Evalúa la política y acumula el snapshot si corresponde."""
        if self._policy == ArtifactSaveMode.ALL:
            self._save(
                algo_step,
                best_fitness,
                best_makespan,
                iteration_fitness,
                iteration_makespan,
                best_solution_snapshot,
            )
        elif self._policy == ArtifactSaveMode.BEST_ONLY:
            if self.best_fitness is None or best_fitness < self.best_fitness:
                self._save(
                    algo_step,
                    best_fitness,
                    best_makespan,
                    iteration_fitness,
                    iteration_makespan,
                    best_solution_snapshot,
                )
        elif self._policy == ArtifactSaveMode.SAMPLED:
            # Reservado para implementación futura (Fase 3/4)
            self._save(
                algo_step,
                best_fitness,
                best_makespan,
                iteration_fitness,
                iteration_makespan,
                best_solution_snapshot,
            )

    def _save(
        self,
        algo_step: int,
        best_fitness: float,
        best_makespan: float,
        iteration_fitness: float,
        iteration_makespan: float,
        best_solution_snapshot: Optional[dict] = None,
    ) -> None:
        """Persiste el snapshot y actualiza best_fitness."""
        # Deep copy para evitar que el caller mute el objeto almacenado
        snapshot_copy = copy.deepcopy(best_solution_snapshot)
        self.snapshots.append(
            IterationSnapshot(
                algo_step=algo_step,
                best_fitness=best_fitness,
                best_makespan=best_makespan,
                iteration_fitness=iteration_fitness,
                iteration_makespan=iteration_makespan,
                best_solution_snapshot=snapshot_copy,
            )
        )
        if self.best_fitness is None or best_fitness < self.best_fitness:
            self.best_fitness = best_fitness

    def reset(self) -> None:
        """Limpia el estado acumulado para reutilización en la siguiente simulación."""
        self.snapshots = []
        self.best_fitness = None


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------


def serialize_solution(solution: Any) -> Optional[dict]:
    """Convierte una solución candidata a un dict JSON-safe.

    Garantiza:
    - Solo tipos nativos Python (sin numpy ni referencias compartidas).
    - Retorna None si la solución es None o no es un dict.

    Parameters
    ----------
    solution:
        Solución candidata del algoritmo.

    Returns
    -------
    dict | None: copia profunda JSON-safe de la solución, o None.
    """
    if solution is None or not isinstance(solution, dict):
        return None

    raw = copy.deepcopy(solution)

    # Normalizar room_assignment: convertir claves numpy/int a int nativo y
    # valores numpy/str a str nativo para garantizar JSON-serializability.
    if "job_sequence_base" in raw:
        raw["job_sequence_base"] = [int(j) for j in raw["job_sequence_base"]]

    if "room_assignment" in raw:
        normalized: dict = {}
        for job_id, ops in raw["room_assignment"].items():
            normalized_ops: dict = {}
            for op_num, room in ops.items():
                normalized_ops[int(op_num)] = str(room)
            normalized[int(job_id)] = normalized_ops
        raw["room_assignment"] = normalized

    return raw
