"""
Tests para core/file_manager.py — Lote 1.

Cubre:
- Modo normal: setup_elective_directories() sigue produciendo 'results/elective/'
- Nuevo modo analysis: setup_analysis_directories(timestamp, scenario) produce
  'results/<timestamp>/<scenario>/'
- Helpers de checkpoint: checkpoint_dir(checkpoint_id)
- Paths de artifacts bajo timestamp base
"""

import os
import pytest

from core.file_manager import FileManager


# ---------------------------------------------------------------------------
# Backward compatibility — modo normal
# ---------------------------------------------------------------------------


class TestNormalModeBackwardCompatibility:
    """El modo normal no debe verse afectado por los cambios del Lote 1."""

    def test_setup_elective_returns_csv_key(self, tmp_path):
        """setup_elective_directories() debe devolver un dict con key 'csv'."""
        fm = FileManager(base_dir=str(tmp_path))
        dirs = fm.setup_elective_directories()
        assert "csv" in dirs

    def test_setup_elective_returns_plots_key(self, tmp_path):
        """setup_elective_directories() debe devolver un dict con key 'plots'."""
        fm = FileManager(base_dir=str(tmp_path))
        dirs = fm.setup_elective_directories()
        assert "plots" in dirs

    def test_setup_elective_creates_csv_directory(self, tmp_path):
        """El directorio CSV debe existir tras setup_elective_directories()."""
        fm = FileManager(base_dir=str(tmp_path))
        dirs = fm.setup_elective_directories()
        assert os.path.isdir(dirs["csv"])

    def test_setup_elective_creates_plots_directory(self, tmp_path):
        """El directorio plots debe existir tras setup_elective_directories()."""
        fm = FileManager(base_dir=str(tmp_path))
        dirs = fm.setup_elective_directories()
        assert os.path.isdir(dirs["plots"])

    def test_setup_elective_path_contains_elective(self, tmp_path):
        """Los paths de elective deben incluir 'elective' en su ruta."""
        fm = FileManager(base_dir=str(tmp_path))
        dirs = fm.setup_elective_directories()
        assert "elective" in dirs["csv"]

# ---------------------------------------------------------------------------
# Analysis mode — setup_analysis_directories(timestamp, scenario)
# ---------------------------------------------------------------------------


class TestSetupAnalysisDirectories:
    """Nueva API para modo analysis con estructura results/<ts>/<scenario>/."""

    def test_returns_dict_with_csv_and_plots(self, tmp_path):
        """setup_analysis_directories debe devolver dict con 'csv' y 'plots'."""
        fm = FileManager(base_dir=str(tmp_path))
        dirs = fm.setup_analysis_directories(
            timestamp="20260416_120000", scenario="elective"
        )
        assert "csv" in dirs
        assert "plots" in dirs

    def test_csv_path_contains_timestamp(self, tmp_path):
        """El path CSV debe incluir el timestamp dado."""
        fm = FileManager(base_dir=str(tmp_path))
        dirs = fm.setup_analysis_directories(
            timestamp="20260416_120000", scenario="elective"
        )
        assert "20260416_120000" in dirs["csv"]

    def test_csv_path_contains_scenario(self, tmp_path):
        """El path CSV debe incluir el nombre del escenario."""
        fm = FileManager(base_dir=str(tmp_path))
        dirs = fm.setup_analysis_directories(
            timestamp="20260416_120000", scenario="elective"
        )
        assert "elective" in dirs["csv"]

    def test_directories_are_created_on_disk(self, tmp_path):
        """Los directorios deben existir en disco después de la llamada."""
        fm = FileManager(base_dir=str(tmp_path))
        dirs = fm.setup_analysis_directories(
            timestamp="20260416_120000", scenario="elective"
        )
        assert os.path.isdir(dirs["csv"])
        assert os.path.isdir(dirs["plots"])

    def test_different_timestamps_produce_different_paths(self, tmp_path):
        """Timestamps distintos deben producir rutas distintas."""
        fm = FileManager(base_dir=str(tmp_path))
        dirs1 = fm.setup_analysis_directories(
            timestamp="20260416_100000", scenario="elective"
        )
        dirs2 = fm.setup_analysis_directories(
            timestamp="20260416_110000", scenario="elective"
        )
        assert dirs1["csv"] != dirs2["csv"]

    def test_different_scenarios_produce_different_paths(self, tmp_path):
        """Escenarios distintos bajo el mismo timestamp producen rutas distintas."""
        fm = FileManager(base_dir=str(tmp_path))
        dirs1 = fm.setup_analysis_directories(
            timestamp="20260416_120000", scenario="elective"
        )
        dirs2 = fm.setup_analysis_directories(
            timestamp="20260416_120000", scenario="sweep_n20"
        )
        assert dirs1["csv"] != dirs2["csv"]

    def test_plot_subdirs_are_created(self, tmp_path):
        """Los subdirectorios de plots (boxplot, barplot, etc.) deben crearse."""
        fm = FileManager(base_dir=str(tmp_path))
        dirs = fm.setup_analysis_directories(
            timestamp="20260416_120000", scenario="elective"
        )
        plots_dir = dirs["plots"]
        for subdir in FileManager.PLOT_SUBDIRS:
            assert os.path.isdir(os.path.join(plots_dir, subdir)), (
                f"Subdir '{subdir}' not found under {plots_dir}"
            )


# ---------------------------------------------------------------------------
# Analysis mode — checkpoint_dir(base_dir, checkpoint_id)
# ---------------------------------------------------------------------------


class TestCheckpointDir:
    """Helper para obtener/crear el directorio de checkpoint."""

    def test_returns_path_with_checkpoint_id(self, tmp_path):
        """checkpoint_dir debe devolver una ruta con el checkpoint_id."""
        fm = FileManager(base_dir=str(tmp_path))
        base = str(tmp_path / "results" / "20260416_120000" / "elective")
        cp_dir = fm.get_checkpoint_dir(base_dir=base, checkpoint_id="cp_300")
        assert "cp_300" in cp_dir

    def test_checkpoint_dir_is_created(self, tmp_path):
        """El directorio de checkpoint debe crearse en disco."""
        fm = FileManager(base_dir=str(tmp_path))
        base = str(tmp_path / "results" / "20260416_120000" / "elective")
        cp_dir = fm.get_checkpoint_dir(base_dir=base, checkpoint_id="cp_300")
        assert os.path.isdir(cp_dir)

    def test_different_checkpoint_ids_produce_different_dirs(self, tmp_path):
        """Checkpoint IDs distintos producen directorios distintos."""
        fm = FileManager(base_dir=str(tmp_path))
        base = str(tmp_path / "results" / "20260416_120000" / "elective")
        cp1 = fm.get_checkpoint_dir(base_dir=base, checkpoint_id="cp_300")
        cp2 = fm.get_checkpoint_dir(base_dir=base, checkpoint_id="cp_600")
        assert cp1 != cp2
