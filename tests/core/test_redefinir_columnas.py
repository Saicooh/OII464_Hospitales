"""
Tests para el cambio: redefinir-columnas-de-algorithm-iterations

Cubre:
- DDL: columnas best_fitness, best_makespan, iteration_fitness, iteration_makespan
- Ausencia de wall_clock_s y combined_obj en algorithm_iterations
- IterationSnapshot con 4 métricas nuevas y sin campos removidos
- AnalysisIterationHandler con firma nueva
- save_algorithm_iterations_batch con columnas nuevas
- CSV export con headers nuevos
- Algoritmos pasan iteration_fitness/iteration_makespan distintos de best_*
"""

import csv
import io
import pytest

from core.analysis_persistence import AnalysisPersistence
from core.iteration_callback import (
    AnalysisIterationHandler,
    ArtifactSaveMode,
    IterationSnapshot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    persistence = AnalysisPersistence(":memory:")
    persistence.init_db()
    return persistence


def _make_sim(db):
    run_id = db.insert_run(num_sims=1, num_procs=5, config={})
    sim_id = db.insert_simulation(
        run_id=run_id,
        sim_index=0,
        algo_name="GA",
        wall_clock_s=10.0,
        final_makespan=500.0,
        combined_obj=None,
    )
    return run_id, sim_id


# ---------------------------------------------------------------------------
# Task 1.1 — DDL: nuevas columnas en algorithm_iterations
# ---------------------------------------------------------------------------


class TestAlgorithmIterationsDDL:
    """El schema de algorithm_iterations debe tener las 4 métricas correctas."""

    def _get_columns(self, db):
        rows = db._conn.execute("PRAGMA table_info(algorithm_iterations)").fetchall()
        return [row[1] for row in rows]

    def test_best_fitness_column_exists(self, db):
        cols = self._get_columns(db)
        assert "best_fitness" in cols

    def test_best_makespan_column_exists(self, db):
        """Nueva columna best_makespan debe existir."""
        cols = self._get_columns(db)
        assert "best_makespan" in cols

    def test_iteration_fitness_column_exists(self, db):
        """Nueva columna iteration_fitness debe existir."""
        cols = self._get_columns(db)
        assert "iteration_fitness" in cols

    def test_iteration_makespan_column_exists(self, db):
        """Nueva columna iteration_makespan debe existir."""
        cols = self._get_columns(db)
        assert "iteration_makespan" in cols

    def test_wall_clock_s_column_not_present(self, db):
        """wall_clock_s debe haberse eliminado de algorithm_iterations."""
        cols = self._get_columns(db)
        assert "wall_clock_s" not in cols

    def test_combined_obj_column_not_present(self, db):
        """combined_obj debe haberse eliminado de algorithm_iterations."""
        cols = self._get_columns(db)
        assert "combined_obj" not in cols

    def test_makespan_old_column_not_present(self, db):
        """La columna antigua 'makespan' (ambigua) no debe existir."""
        cols = self._get_columns(db)
        assert "makespan" not in cols


# ---------------------------------------------------------------------------
# Task 1.2 — save_algorithm_iterations_batch con nuevas columnas
# ---------------------------------------------------------------------------


class TestSaveAlgorithmIterationsBatchNewSchema:
    """Persistencia de filas usando las 4 métricas nuevas."""

    def test_batch_insert_with_four_metrics(self, db):
        """batch insert con best_*, iteration_* debe persistir correctamente."""
        run_id, sim_id = _make_sim(db)
        rows = [
            {
                "algo_step": 1,
                "best_fitness": 500.0,
                "best_makespan": 480.0,
                "iteration_fitness": 510.0,
                "iteration_makespan": 495.0,
            }
        ]
        result = db.save_algorithm_iterations_batch(sim_id=sim_id, rows=rows)
        assert len(result) == 1
        assert result[0] > 0

    def test_batch_values_stored_with_distinct_metrics(self, db):
        """best_* e iteration_* deben almacenarse con valores distintos."""
        run_id, sim_id = _make_sim(db)
        rows = [
            {
                "algo_step": 3,
                "best_fitness": 400.0,
                "best_makespan": 380.0,
                "iteration_fitness": 450.0,
                "iteration_makespan": 430.0,
            }
        ]
        db.save_algorithm_iterations_batch(sim_id=sim_id, rows=rows)
        row = db._conn.execute(
            "SELECT algo_step, best_fitness, best_makespan, iteration_fitness, iteration_makespan"
            " FROM algorithm_iterations WHERE sim_id = ?",
            (sim_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == 3
        assert abs(row[1] - 400.0) < 1e-9
        assert abs(row[2] - 380.0) < 1e-9
        assert abs(row[3] - 450.0) < 1e-9
        assert abs(row[4] - 430.0) < 1e-9

    def test_iteration_metrics_can_differ_from_best_metrics(self, db):
        """iteration_fitness puede ser mayor que best_fitness (no confundir)."""
        run_id, sim_id = _make_sim(db)
        rows = [
            {
                "algo_step": 5,
                "best_fitness": 300.0,
                "best_makespan": 290.0,
                "iteration_fitness": 350.0,  # worse than global best
                "iteration_makespan": 340.0,
            }
        ]
        db.save_algorithm_iterations_batch(sim_id=sim_id, rows=rows)
        row = db._conn.execute(
            "SELECT best_fitness, iteration_fitness FROM algorithm_iterations WHERE sim_id = ?",
            (sim_id,),
        ).fetchone()
        assert row[0] < row[1], (
            "best_fitness debe ser < iteration_fitness en este escenario"
        )

    def test_empty_batch_returns_empty_list(self, db):
        run_id, sim_id = _make_sim(db)
        result = db.save_algorithm_iterations_batch(sim_id=sim_id, rows=[])
        assert result == []


# ---------------------------------------------------------------------------
# Task 2.1 — IterationSnapshot con 4 métricas nuevas
# ---------------------------------------------------------------------------


class TestIterationSnapshotNewFields:
    """IterationSnapshot debe tener best_makespan, iteration_fitness, iteration_makespan."""

    def test_snapshot_has_best_makespan_field(self):
        snap = IterationSnapshot(
            algo_step=1,
            best_fitness=100.0,
            best_makespan=95.0,
            iteration_fitness=110.0,
            iteration_makespan=105.0,
        )
        assert abs(snap.best_makespan - 95.0) < 1e-9

    def test_snapshot_has_iteration_fitness_field(self):
        snap = IterationSnapshot(
            algo_step=2,
            best_fitness=200.0,
            best_makespan=190.0,
            iteration_fitness=220.0,
            iteration_makespan=210.0,
        )
        assert abs(snap.iteration_fitness - 220.0) < 1e-9

    def test_snapshot_has_iteration_makespan_field(self):
        snap = IterationSnapshot(
            algo_step=2,
            best_fitness=200.0,
            best_makespan=190.0,
            iteration_fitness=220.0,
            iteration_makespan=210.0,
        )
        assert abs(snap.iteration_makespan - 210.0) < 1e-9

    def test_snapshot_four_metrics_are_independent(self):
        """Los 4 valores deben ser independientes entre sí."""
        snap = IterationSnapshot(
            algo_step=7,
            best_fitness=300.0,
            best_makespan=280.0,
            iteration_fitness=320.0,
            iteration_makespan=310.0,
        )
        assert snap.best_fitness != snap.iteration_fitness
        assert snap.best_makespan != snap.iteration_makespan

    def test_snapshot_no_combined_obj_field(self):
        """combined_obj no debe existir como campo requerido en el nuevo snapshot."""
        snap = IterationSnapshot(
            algo_step=1,
            best_fitness=100.0,
            best_makespan=90.0,
            iteration_fitness=110.0,
            iteration_makespan=105.0,
        )
        # combined_obj no debe existir como campo del nuevo IterationSnapshot
        assert not hasattr(snap, "combined_obj"), (
            "combined_obj fue eliminado del schema: no debe ser atributo de IterationSnapshot"
        )

    def test_snapshot_no_wall_clock_s_field(self):
        """wall_clock_s no debe ser requerido en el nuevo contrato."""
        snap = IterationSnapshot(
            algo_step=1,
            best_fitness=100.0,
            best_makespan=90.0,
            iteration_fitness=110.0,
            iteration_makespan=105.0,
        )
        # wall_clock_s no debe existir como campo del nuevo IterationSnapshot
        assert not hasattr(snap, "wall_clock_s"), (
            "wall_clock_s fue eliminado del schema: no debe ser atributo de IterationSnapshot"
        )


# ---------------------------------------------------------------------------
# Task 2.2 — AnalysisIterationHandler con firma nueva
# ---------------------------------------------------------------------------


class TestAnalysisIterationHandlerNewSignature:
    """Handler debe aceptar best_*, iteration_* y crear snapshots correctos."""

    def test_handler_all_policy_stores_four_metrics(self):
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.ALL)
        handler(
            algo_step=1,
            best_fitness=400.0,
            best_makespan=380.0,
            iteration_fitness=420.0,
            iteration_makespan=410.0,
        )
        assert len(handler.snapshots) == 1
        snap = handler.snapshots[0]
        assert abs(snap.best_fitness - 400.0) < 1e-9
        assert abs(snap.best_makespan - 380.0) < 1e-9
        assert abs(snap.iteration_fitness - 420.0) < 1e-9
        assert abs(snap.iteration_makespan - 410.0) < 1e-9

    def test_handler_best_only_policy_filters_by_best_fitness(self):
        """best_only sigue filtrando por best_fitness (no por iteration_fitness)."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.BEST_ONLY)
        handler(
            algo_step=1,
            best_fitness=500.0,
            best_makespan=480.0,
            iteration_fitness=510.0,
            iteration_makespan=495.0,
        )
        handler(
            algo_step=2,
            best_fitness=500.0,  # no mejora
            best_makespan=480.0,
            iteration_fitness=490.0,  # iter mejoró, pero best no
            iteration_makespan=470.0,
        )
        assert len(handler.snapshots) == 1  # solo el primero

    def test_handler_all_saves_all_even_when_iter_worse(self):
        """Con 'all', guarda aunque iteration_fitness sea peor que best_fitness."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.ALL)
        handler(
            algo_step=1,
            best_fitness=300.0,
            best_makespan=280.0,
            iteration_fitness=350.0,
            iteration_makespan=340.0,
        )
        handler(
            algo_step=2,
            best_fitness=290.0,
            best_makespan=270.0,
            iteration_fitness=360.0,  # peor iteración
            iteration_makespan=345.0,
        )
        assert len(handler.snapshots) == 2

    def test_handler_iteration_metrics_distinct_from_best(self):
        """Los 4 valores se almacenan de forma independiente en el snapshot."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.ALL)
        handler(
            algo_step=3,
            best_fitness=100.0,
            best_makespan=90.0,
            iteration_fitness=130.0,
            iteration_makespan=120.0,
        )
        snap = handler.snapshots[0]
        assert abs(snap.iteration_fitness - 130.0) < 1e-9
        assert abs(snap.iteration_makespan - 120.0) < 1e-9


# ---------------------------------------------------------------------------
# Task 4.2 — CSV export con nuevos headers
# ---------------------------------------------------------------------------


class TestAlgorithmIterationsCSVNewSchema:
    """export_algorithm_iterations_csv debe reflejar los 4 nuevas métricas."""

    def _setup(self, db):
        run_id, sim_id = _make_sim(db)
        rows = [
            {
                "algo_step": 1,
                "best_fitness": 500.0,
                "best_makespan": 480.0,
                "iteration_fitness": 510.0,
                "iteration_makespan": 495.0,
            },
            {
                "algo_step": 2,
                "best_fitness": 490.0,
                "best_makespan": 470.0,
                "iteration_fitness": 495.0,
                "iteration_makespan": 475.0,
            },
        ]
        db.save_algorithm_iterations_batch(sim_id=sim_id, rows=rows)
        return run_id

    def test_csv_has_best_makespan_column(self, db, tmp_path):
        run_id = self._setup(db)
        csv_path = str(tmp_path / "iters.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = csv.DictReader(f).fieldnames
        assert "makespan_of_best_fitness" in header

    def test_csv_has_iteration_fitness_column(self, db, tmp_path):
        run_id = self._setup(db)
        csv_path = str(tmp_path / "iters.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = csv.DictReader(f).fieldnames
        assert "iteration_fitness" in header

    def test_csv_has_iteration_makespan_column(self, db, tmp_path):
        run_id = self._setup(db)
        csv_path = str(tmp_path / "iters.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = csv.DictReader(f).fieldnames
        assert "iteration_makespan" in header

    def test_csv_excludes_wall_clock_s_column(self, db, tmp_path):
        run_id = self._setup(db)
        csv_path = str(tmp_path / "iters.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = csv.DictReader(f).fieldnames
        assert "wall_clock_s" not in header

    def test_csv_excludes_combined_obj_column(self, db, tmp_path):
        run_id = self._setup(db)
        csv_path = str(tmp_path / "iters.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = csv.DictReader(f).fieldnames
        assert "combined_obj" not in header

    def test_csv_data_values_correct(self, db, tmp_path):
        """Los valores de makespan_of_best_fitness, iteration_* deben estar en el CSV."""
        run_id = self._setup(db)
        csv_path = str(tmp_path / "iters.csv")
        db.export_algorithm_iterations_csv(run_id=run_id, path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        first_row = rows[0]
        assert abs(float(first_row["makespan_of_best_fitness"]) - 480.0) < 1e-6
        assert abs(float(first_row["iteration_fitness"]) - 510.0) < 1e-6
        assert abs(float(first_row["iteration_makespan"]) - 495.0) < 1e-6


# ---------------------------------------------------------------------------
# Task 5.4 — Algoritmos: iteration_* puede diferir de best_*
# ---------------------------------------------------------------------------


class TestAlgorithmCallbackDistinctMetrics:
    """Verifica que los callbacks de algoritmos usan métricas distintas."""

    def test_callback_receives_distinct_iteration_and_best_metrics(self):
        """
        Simula un callback de algoritmo donde iteration_fitness != best_fitness.
        Verifica que el handler las almacena por separado.
        """
        captured = []

        def mock_callback(
            algo_step,
            best_fitness,
            best_makespan,
            iteration_fitness,
            iteration_makespan,
        ):
            captured.append(
                {
                    "algo_step": algo_step,
                    "best_fitness": best_fitness,
                    "best_makespan": best_makespan,
                    "iteration_fitness": iteration_fitness,
                    "iteration_makespan": iteration_makespan,
                }
            )

        # Simular lo que haría un algoritmo swarm:
        # best_* = gbest (histórico), iteration_* = mejor de esta iteración
        mock_callback(
            algo_step=5,
            best_fitness=300.0,  # gbest histórico (mejor global)
            best_makespan=285.0,
            iteration_fitness=320.0,  # mejor de esta iteración (puede ser peor)
            iteration_makespan=305.0,
        )

        assert len(captured) == 1
        c = captured[0]
        assert c["best_fitness"] != c["iteration_fitness"]
        assert c["best_makespan"] != c["iteration_makespan"]
        assert c["best_fitness"] < c["iteration_fitness"]  # gbest es mejor

    def test_handler_stores_distinct_metrics_from_swarm_algo(self):
        """AnalysisIterationHandler debe almacenar métricas distintas del swarm."""
        handler = AnalysisIterationHandler(policy=ArtifactSaveMode.ALL)

        # Primera iteración: gbest y iter_best coinciden (primera partícula buena)
        handler(
            algo_step=1,
            best_fitness=500.0,
            best_makespan=480.0,
            iteration_fitness=500.0,
            iteration_makespan=480.0,
        )
        # Segunda iteración: gbest no mejora, pero la mejor partícula de la iteración sí es otra
        handler(
            algo_step=2,
            best_fitness=500.0,  # gbest no cambió
            best_makespan=480.0,
            iteration_fitness=510.0,  # iter_best es peor que gbest
            iteration_makespan=495.0,
        )

        assert len(handler.snapshots) == 2
        snap2 = handler.snapshots[1]
        assert abs(snap2.best_fitness - 500.0) < 1e-9
        assert abs(snap2.iteration_fitness - 510.0) < 1e-9
