"""
Tests para core/iteration_callback.py — Lote 1 (actualizado para nuevo schema).

Cubre:
- IterationCallback como Protocol (duck-typing verificable)
- AnalysisIterationHandler: acumula snapshots según política best_only / all / sampled
- Contratos de serialización: snapshots son dicts planos sin objetos no serializables
- Nuevo schema: 4 métricas (best_fitness, best_makespan, iteration_fitness, iteration_makespan)
"""

import pytest

from core.iteration_callback import (
    AnalysisIterationHandler,
    ArtifactSaveMode,
    IterationSnapshot,
)


# Helper para construir snapshot con los 4 campos requeridos
def _snap(
    algo_step=1,
    best_fitness=100.0,
    best_makespan=90.0,
    iteration_fitness=110.0,
    iteration_makespan=105.0,
):
    return IterationSnapshot(
        algo_step=algo_step,
        best_fitness=best_fitness,
        best_makespan=best_makespan,
        iteration_fitness=iteration_fitness,
        iteration_makespan=iteration_makespan,
    )


# Helper para llamar al handler con 4 métricas
def _call(
    handler,
    algo_step,
    best_fitness,
    best_makespan=None,
    iteration_fitness=None,
    iteration_makespan=None,
):
    bm = best_makespan if best_makespan is not None else best_fitness
    itf = iteration_fitness if iteration_fitness is not None else best_fitness
    itm = iteration_makespan if iteration_makespan is not None else bm
    handler(
        algo_step=algo_step,
        best_fitness=best_fitness,
        best_makespan=bm,
        iteration_fitness=itf,
        iteration_makespan=itm,
    )


# ---------------------------------------------------------------------------
# ArtifactSaveMode — constantes de política
# ---------------------------------------------------------------------------


class TestArtifactSaveMode:
    def test_best_only_constant_exists(self):
        """ArtifactSaveMode.BEST_ONLY debe existir con valor 'best_only'."""
        assert ArtifactSaveMode.BEST_ONLY == "best_only"

    def test_all_constant_exists(self):
        """ArtifactSaveMode.ALL debe existir con valor 'all'."""
        assert ArtifactSaveMode.ALL == "all"

    def test_sampled_constant_exists(self):
        """ArtifactSaveMode.SAMPLED debe existir con valor 'sampled'."""
        assert ArtifactSaveMode.SAMPLED == "sampled"


# ---------------------------------------------------------------------------
# IterationSnapshot — estructura básica del snapshot (nuevo schema)
# ---------------------------------------------------------------------------


class TestIterationSnapshot:
    def test_snapshot_has_four_required_fields(self):
        """IterationSnapshot debe poder instanciarse con 4 métricas."""
        snap = _snap(
            algo_step=1,
            best_fitness=100.0,
            best_makespan=90.0,
            iteration_fitness=110.0,
            iteration_makespan=105.0,
        )
        assert snap.algo_step == 1
        assert abs(snap.best_fitness - 100.0) < 1e-9
        assert abs(snap.best_makespan - 90.0) < 1e-9
        assert abs(snap.iteration_fitness - 110.0) < 1e-9
        assert abs(snap.iteration_makespan - 105.0) < 1e-9

    def test_snapshot_algo_step_type(self):
        """algo_step debe ser int."""
        snap = _snap(algo_step=10)
        assert isinstance(snap.algo_step, int)

    def test_snapshot_best_fitness_type(self):
        """best_fitness debe ser float."""
        snap = _snap(best_fitness=99.9)
        assert isinstance(snap.best_fitness, float)

    def test_snapshot_four_metrics_independent(self):
        """Los 4 valores deben ser independientes."""
        snap = _snap(
            best_fitness=300.0,
            best_makespan=280.0,
            iteration_fitness=320.0,
            iteration_makespan=310.0,
        )
        assert snap.best_fitness != snap.iteration_fitness
        assert snap.best_makespan != snap.iteration_makespan


# ---------------------------------------------------------------------------
# AnalysisIterationHandler — política 'all'
# ---------------------------------------------------------------------------


class TestAnalysisIterationHandlerAll:
    """Con política 'all', TODOS los snapshots deben acumularse."""

    def _make_handler(self):
        return AnalysisIterationHandler(policy=ArtifactSaveMode.ALL)

    def test_no_snapshots_initially(self):
        """Antes de cualquier llamada, snapshots debe estar vacío."""
        handler = self._make_handler()
        assert handler.snapshots == []

    def test_single_call_accumulates_one_snapshot(self):
        """Una llamada debe producir exactamente 1 snapshot."""
        handler = self._make_handler()
        _call(handler, algo_step=1, best_fitness=500.0)
        assert len(handler.snapshots) == 1

    def test_all_policy_saves_every_call(self):
        """Con 'all', cada llamada debe acumularse aunque el fitness no mejore."""
        handler = self._make_handler()
        _call(handler, algo_step=1, best_fitness=500.0)
        _call(handler, algo_step=2, best_fitness=510.0)  # peor
        _call(handler, algo_step=3, best_fitness=490.0)  # mejor
        assert len(handler.snapshots) == 3

    def test_snapshot_values_stored_correctly(self):
        """El snapshot almacenado debe reflejar los valores recibidos."""
        handler = self._make_handler()
        handler(
            algo_step=7,
            best_fitness=123.4,
            best_makespan=118.0,
            iteration_fitness=130.0,
            iteration_makespan=125.0,
        )
        snap = handler.snapshots[0]
        assert snap.algo_step == 7
        assert abs(snap.best_fitness - 123.4) < 1e-9
        assert abs(snap.best_makespan - 118.0) < 1e-9
        assert abs(snap.iteration_fitness - 130.0) < 1e-9


# ---------------------------------------------------------------------------
# AnalysisIterationHandler — política 'best_only'
# ---------------------------------------------------------------------------


class TestAnalysisIterationHandlerBestOnly:
    """Con política 'best_only', solo se acumulan los snapshots que mejoran."""

    def _make_handler(self):
        return AnalysisIterationHandler(policy=ArtifactSaveMode.BEST_ONLY)

    def test_first_call_always_saved(self):
        """El primer snapshot siempre debe guardarse (no hay best previo)."""
        handler = self._make_handler()
        _call(handler, algo_step=1, best_fitness=500.0)
        assert len(handler.snapshots) == 1

    def test_improvement_is_saved(self):
        """Si el nuevo fitness es menor que el previo, debe guardarse."""
        handler = self._make_handler()
        _call(handler, algo_step=1, best_fitness=500.0)
        _call(handler, algo_step=2, best_fitness=490.0)  # mejora
        assert len(handler.snapshots) == 2

    def test_no_improvement_is_not_saved(self):
        """Si el fitness no mejora, NO debe acumularse."""
        handler = self._make_handler()
        _call(handler, algo_step=1, best_fitness=500.0)
        _call(handler, algo_step=2, best_fitness=510.0)  # peor
        assert len(handler.snapshots) == 1

    def test_equal_fitness_is_not_saved(self):
        """Fitness igual al best no es una mejora; no debe acumularse."""
        handler = self._make_handler()
        _call(handler, algo_step=1, best_fitness=500.0)
        _call(handler, algo_step=2, best_fitness=500.0)
        assert len(handler.snapshots) == 1

    def test_mixed_sequence_saves_only_improvements(self):
        """Secuencia mixta: solo pasos que mejoran el best global."""
        handler = self._make_handler()
        _call(handler, algo_step=1, best_fitness=600.0)
        _call(handler, algo_step=2, best_fitness=580.0)
        _call(handler, algo_step=3, best_fitness=590.0)  # NOT saved
        _call(handler, algo_step=4, best_fitness=560.0)
        assert len(handler.snapshots) == 3
        assert handler.snapshots[0].algo_step == 1
        assert handler.snapshots[1].algo_step == 2
        assert handler.snapshots[2].algo_step == 4

    def test_best_fitness_tracked_correctly(self):
        """handler.best_fitness debe reflejar el mínimo visto hasta el momento."""
        handler = self._make_handler()
        _call(handler, algo_step=1, best_fitness=600.0)
        _call(handler, algo_step=2, best_fitness=580.0)
        _call(handler, algo_step=3, best_fitness=620.0)
        assert abs(handler.best_fitness - 580.0) < 1e-9


# ---------------------------------------------------------------------------
# AnalysisIterationHandler — reset()
# ---------------------------------------------------------------------------


class TestAnalysisIterationHandlerReset:
    def test_reset_clears_snapshots(self):
        """Tras reset(), snapshots debe quedar vacío."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.ALL)
        _call(handler, algo_step=1, best_fitness=100.0)
        handler.reset()
        assert handler.snapshots == []

    def test_reset_clears_best_fitness(self):
        """Tras reset(), best_fitness debe volver a None (sin best previo)."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.BEST_ONLY)
        _call(handler, algo_step=1, best_fitness=100.0)
        handler.reset()
        # Después de reset, el primer call debe guardarse siempre
        _call(handler, algo_step=1, best_fitness=999.0)
        assert len(handler.snapshots) == 1

    def test_reset_allows_reuse_for_next_simulation(self):
        """reset() permite reutilizar el handler para la siguiente simulación."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.BEST_ONLY)
        _call(handler, algo_step=1, best_fitness=500.0)
        _call(handler, algo_step=2, best_fitness=490.0)
        assert len(handler.snapshots) == 2
        handler.reset()
        _call(handler, algo_step=1, best_fitness=600.0)
        assert len(handler.snapshots) == 1


# ---------------------------------------------------------------------------
# IterationCallback Protocol — duck-typing compliance
# ---------------------------------------------------------------------------


class TestIterationCallbackProtocol:
    def test_handler_is_callable(self):
        """AnalysisIterationHandler debe ser callable (implementa __call__)."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.ALL)
        assert callable(handler)

    def test_plain_function_satisfies_protocol(self):
        """Una función con la firma nueva debe ser usable como callback."""
        calls = []

        def my_callback(
            algo_step,
            best_fitness,
            best_makespan,
            iteration_fitness,
            iteration_makespan,
        ):
            calls.append(algo_step)

        my_callback(
            algo_step=1,
            best_fitness=100.0,
            best_makespan=90.0,
            iteration_fitness=110.0,
            iteration_makespan=105.0,
        )
        my_callback(
            algo_step=2,
            best_fitness=90.0,
            best_makespan=85.0,
            iteration_fitness=95.0,
            iteration_makespan=90.0,
        )
        assert calls == [1, 2]


# ---------------------------------------------------------------------------
# Phase 3 (RED → GREEN): IterationSnapshot y callback usan 'algo_step'
# ---------------------------------------------------------------------------


class TestIterationSnapshotAlgoStep:
    """IterationSnapshot debe usar 'algo_step' en lugar de 'generation'."""

    def test_snapshot_has_algo_step_field(self):
        """IterationSnapshot debe tener el atributo 'algo_step'."""
        snap = IterationSnapshot(
            algo_step=1,
            best_fitness=100.0,
            best_makespan=90.0,
            iteration_fitness=110.0,
            iteration_makespan=105.0,
        )
        assert snap.algo_step == 1

    def test_snapshot_has_no_generation_field(self):
        """IterationSnapshot NO debe tener el atributo 'generation' (legacy)."""
        snap = IterationSnapshot(
            algo_step=5,
            best_fitness=200.0,
            best_makespan=190.0,
            iteration_fitness=210.0,
            iteration_makespan=200.0,
        )
        assert not hasattr(snap, "generation"), (
            "Legacy 'generation' field must not exist on IterationSnapshot"
        )

    def test_snapshot_algo_step_type(self):
        """algo_step debe ser int."""
        snap = IterationSnapshot(
            algo_step=10,
            best_fitness=99.9,
            best_makespan=95.0,
            iteration_fitness=105.0,
            iteration_makespan=100.0,
        )
        assert isinstance(snap.algo_step, int)


class TestAnalysisIterationHandlerAlgoStep:
    """AnalysisIterationHandler debe aceptar 'algo_step' como parámetro."""

    def test_handler_call_accepts_algo_step(self):
        """Handler.__call__ debe aceptar 'algo_step' como kwarg (no 'generation')."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.ALL)
        handler(
            algo_step=1,
            best_fitness=500.0,
            best_makespan=480.0,
            iteration_fitness=510.0,
            iteration_makespan=490.0,
        )
        assert len(handler.snapshots) == 1

    def test_snapshot_stores_algo_step_not_generation(self):
        """El snapshot acumulado debe exponer algo_step, no generation."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.ALL)
        handler(
            algo_step=7,
            best_fitness=300.0,
            best_makespan=290.0,
            iteration_fitness=310.0,
            iteration_makespan=295.0,
        )
        snap = handler.snapshots[0]
        assert snap.algo_step == 7
        assert not hasattr(snap, "generation"), "Legacy 'generation' must not exist"

    def test_handler_rejects_generation_as_kwarg(self):
        """Handler NO debe aceptar 'generation' como kwarg (nombre legacy eliminado)."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.ALL)
        with pytest.raises(TypeError):
            handler(
                generation=1,
                best_fitness=500.0,
                best_makespan=480.0,
                iteration_fitness=510.0,
                iteration_makespan=490.0,
            )


class TestIterationCallbackProtocolAlgoStep:
    """El protocolo IterationCallback debe usar 'algo_step' en su firma."""

    def test_plain_function_with_algo_step_satisfies_protocol(self):
        """Una función con 'algo_step' debe ser invocable como callback."""
        calls = []

        def my_callback(
            algo_step,
            best_fitness,
            best_makespan,
            iteration_fitness,
            iteration_makespan,
        ):
            calls.append(algo_step)

        my_callback(
            algo_step=1,
            best_fitness=100.0,
            best_makespan=90.0,
            iteration_fitness=110.0,
            iteration_makespan=105.0,
        )
        assert calls == [1]


# ---------------------------------------------------------------------------
# Phase 4 (RED): best_solution_snapshot — campo opcional JSON-safe en snapshot
# ---------------------------------------------------------------------------


class TestIterationSnapshotBestSolutionSnapshot:
    """IterationSnapshot debe aceptar best_solution_snapshot opcional y JSON-safe."""

    def test_snapshot_accepts_none_snapshot(self):
        """best_solution_snapshot=None debe ser el default."""
        snap = IterationSnapshot(
            algo_step=1,
            best_fitness=100.0,
            best_makespan=90.0,
            iteration_fitness=110.0,
            iteration_makespan=105.0,
        )
        assert snap.best_solution_snapshot is None

    def test_snapshot_accepts_dict_snapshot(self):
        """best_solution_snapshot debe aceptar un dict Python nativo."""
        payload = {
            "job_sequence_base": [1, 2, 3],
            "room_assignment": {1: {1: "P1", 2: "P2"}},
        }
        snap = IterationSnapshot(
            algo_step=2,
            best_fitness=90.0,
            best_makespan=85.0,
            iteration_fitness=95.0,
            iteration_makespan=90.0,
            best_solution_snapshot=payload,
        )
        assert snap.best_solution_snapshot is not None
        assert snap.best_solution_snapshot["job_sequence_base"] == [1, 2, 3]

    def test_snapshot_json_serializable(self):
        """El snapshot debe ser JSON-serializable (sin numpy ni tipos no nativos)."""
        import json

        payload = {
            "job_sequence_base": [1, 2, 3],
            "room_assignment": {"1": {"1": "P1", "2": "P2"}},
        }
        snap = IterationSnapshot(
            algo_step=3,
            best_fitness=80.0,
            best_makespan=75.0,
            iteration_fitness=85.0,
            iteration_makespan=80.0,
            best_solution_snapshot=payload,
        )
        # Must not raise
        serialized = json.dumps(snap.best_solution_snapshot)
        restored = json.loads(serialized)
        assert restored["job_sequence_base"] == [1, 2, 3]


class TestAnalysisIterationHandlerSnapshot:
    """AnalysisIterationHandler debe propagar best_solution_snapshot al IterationSnapshot."""

    def test_handler_accepts_snapshot_kwarg(self):
        """Handler.__call__ debe aceptar best_solution_snapshot como kwarg."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.ALL)
        payload = {"job_sequence_base": [3, 1, 2], "room_assignment": {}}
        handler(
            algo_step=1,
            best_fitness=500.0,
            best_makespan=480.0,
            iteration_fitness=510.0,
            iteration_makespan=490.0,
            best_solution_snapshot=payload,
        )
        assert len(handler.snapshots) == 1
        assert handler.snapshots[0].best_solution_snapshot == payload

    def test_handler_stores_none_when_snapshot_not_provided(self):
        """Si no se pasa best_solution_snapshot, debe quedar None en el snapshot."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.ALL)
        handler(
            algo_step=1,
            best_fitness=500.0,
            best_makespan=480.0,
            iteration_fitness=510.0,
            iteration_makespan=490.0,
        )
        assert handler.snapshots[0].best_solution_snapshot is None

    def test_handler_snapshot_is_independent_copy(self):
        """El snapshot almacenado debe ser una copia independiente del dict original."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.ALL)
        payload = {"job_sequence_base": [1, 2, 3], "room_assignment": {}}
        handler(
            algo_step=1,
            best_fitness=500.0,
            best_makespan=480.0,
            iteration_fitness=510.0,
            iteration_makespan=490.0,
            best_solution_snapshot=payload,
        )
        # Mutating the original must not affect the stored snapshot
        payload["job_sequence_base"].append(99)
        assert 99 not in handler.snapshots[0].best_solution_snapshot["job_sequence_base"]
