"""
File and directory management for the simulation system.

Lote 1 changes:
- FileManager now accepts optional `base_dir` parameter (default: "results")
  for testability without touching real filesystem.
- New method `setup_analysis_directories(timestamp, scenario)` creates
  `<base_dir>/<timestamp>/<scenario>/` structure for analysis mode isolation.
- New helper `get_checkpoint_dir(base_dir, checkpoint_id)` returns and
  creates checkpoint-specific subdirectory.
- Elective normal-mode output remains backward compatible.
"""

import os
from utils.logger import logger


class FileManager:
    """
    Manages the creation and organization of output directories.

    Parameters
    ----------
    base_dir:
        Root directory for all outputs. Defaults to ``"results"`` for
        backward compatibility with normal-mode callers that do not pass it.
    """

    BASE_DIR = "results"

    PLOT_SUBDIRS = ["boxplot", "barplot", "histograms", "convergence", "gantt", "personnel"]
    CHECKPOINT_PLOT_SUBDIRS = ["gantt", "histograms", "personnel", "convergence"]

    def __init__(self, base_dir: str | None = None) -> None:
        self._base_dir = base_dir if base_dir is not None else self.BASE_DIR

    def setup_elective_directories(self) -> dict:
        """Creates directory structure for elective simulations (normal mode)."""
        return self._setup_directories("elective")

    def setup_analysis_directories(
        self, timestamp: str, scenario: str, plot_subdirs: list | None = None
    ) -> dict:
        """Creates directory structure for analysis mode runs.

        Produces::

            <base_dir>/<timestamp>/<scenario>/csv/
            <base_dir>/<timestamp>/<scenario>/plots/<subdir>/

        Parameters
        ----------
        timestamp:
            String timestamp (e.g. ``"20260416_120000"``) that uniquely
            identifies this analysis run. All outputs are isolated under it.
        scenario:
            Scenario label (e.g. ``"elective"`` or ``"sweep_n20"``).
        plot_subdirs:
            Optional list of plot subdirectories to create. When ``None``,
            uses the default ``PLOT_SUBDIRS``. Pass ``CHECKPOINT_PLOT_SUBDIRS``
            for checkpoint reports to avoid empty folders.

        Returns
        -------
        dict with ``"csv"`` and ``"plots"`` keys pointing to created directories.
        """
        experiment_dir = os.path.join(self._base_dir, timestamp, scenario)
        return self._build_output_dirs(experiment_dir, plot_subdirs=plot_subdirs)

    def get_checkpoint_dir(self, base_dir: str, checkpoint_id: str) -> str:
        """Returns (and creates) the checkpoint-specific subdirectory.

        Parameters
        ----------
        base_dir:
            Base directory for the current run/scenario.
        checkpoint_id:
            Unique identifier for this checkpoint (e.g. ``"cp_300"``).

        Returns
        -------
        Path string to the created checkpoint directory.
        """
        checkpoint_dir = os.path.join(base_dir, "checkpoints", checkpoint_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_directories(self, mode: str) -> dict:
        """Internal method to create normal-mode directory structure.

        Args:
            mode: output mode name

        Returns:
            dict with paths to csv and plots directories.
        """
        experiment_dir = os.path.join(self._base_dir, mode)
        return self._build_output_dirs(experiment_dir)

    def _build_output_dirs(
        self, experiment_dir: str, plot_subdirs: list | None = None
    ) -> dict:
        """Creates csv + plots subdirs under experiment_dir and returns paths."""
        output_dirs = {
            "csv": os.path.join(experiment_dir, "csv"),
            "plots": os.path.join(experiment_dir, "plots"),
        }

        for path in output_dirs.values():
            os.makedirs(path, exist_ok=True)

        subdirs = plot_subdirs if plot_subdirs is not None else self.PLOT_SUBDIRS
        for subdir in subdirs:
            os.makedirs(os.path.join(output_dirs["plots"], subdir), exist_ok=True)

        logger.info(f"Output directory: {experiment_dir}")
        logger.info(f"   - CSV files: {output_dirs['csv']}")
        logger.info(f"   - Plots: {output_dirs['plots']}")

        return output_dirs
