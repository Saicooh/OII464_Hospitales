"""
Main entry point for the Hospital Scheduling Simulation System.
This file acts as a simple orchestrator, delegating all responsibilities.
"""

import os
import sys

# Ensure project root is in sys.path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.simulation_runner import SimulationRunner
from config.config import (
    ANALYSIS_MODE_ENABLED,
    ANALYSIS_TEMPORAL_ENABLED,
    ANALYSIS_SWEEP_ENABLED,
)
from utils.logger import logger


def main():
    """
    Main orchestrator: decides which simulation mode to run.

    Priority order:
      1. analysis_mode.enabled=True  → evaluate temporal_enabled and sweep_enabled independently
         - temporal_enabled=True  → run_elective_analysis_mode() [SQLite + CIE-10]
         - sweep_enabled=True     → run_sweep_mode() [no SQLite, CSV only]
         - Both can run in order: temporal first, then sweep.
         - Neither: logs a warning and exits cleanly.
       2. (default)                   → run_elective_mode()

    Normal mode behavior is UNCHANGED when ANALYSIS_MODE_ENABLED=False.
    Sweep-only mode never initializes AnalysisPersistence/SQLite.
    """
    try:
        runner = SimulationRunner()

        if ANALYSIS_MODE_ENABLED:
            if ANALYSIS_TEMPORAL_ENABLED:
                db_path = runner.run_elective_analysis_mode()
                if db_path and os.path.exists(db_path):
                    logger.info("\nRunning offline analysis and convergence tools...")
                    output_directory = os.path.dirname(os.path.abspath(db_path))
                    
                    # 1. Run iterations schedule and strategy exporter
                    try:
                        from offline.analysis_exporter import AnalysisExporter
                        exporter = AnalysisExporter()
                        logger.info("Exporting iteration schedules and strategies to CSVs...")
                        exporter.export_iteration_csvs(db_path=db_path, output_dir=output_directory)
                    except Exception as ex_export:
                        logger.error(f"Error executing offline AnalysisExporter: {ex_export}")

                    # 2. Run convergence analysis (Monte Carlo stability analysis)
                    try:
                        from offline.convergence_analysis import load_simulation_data, process_and_plot_run
                        from joblib import Parallel, delayed
                        out_dir = os.path.join(output_directory, "convergence")
                        os.makedirs(out_dir, exist_ok=True)
                        logger.info("Generating convergence stability plots...")
                        data = load_simulation_data(db_path)
                        Parallel(n_jobs=-1)(
                            delayed(process_and_plot_run)(run_id, run_data, 5, out_dir)
                            for run_id, run_data in sorted(data.items())
                        )
                        logger.info(f"Convergence plots saved to: {out_dir}")
                    except Exception as ex_conv:
                        logger.error(f"Error executing offline convergence analysis: {ex_conv}")

                    # 3. Generate comparison boxplots and makespan histograms per run
                    try:
                        from offline.generate_analysis_plots import generate_all_analysis_plots
                        logger.info("Generating comparison boxplots and makespan histograms per run...")
                        generate_all_analysis_plots(
                            db_path, output_dir=output_directory
                        )
                        logger.info("Boxplots and histograms generated successfully.")
                    except Exception as ex_plots:
                        logger.error(f"Error generating boxplots and histograms: {ex_plots}")

                    # 4. Generate advanced operational and scalability plots per run
                    try:
                        from offline.generate_advanced_plots import generate_all_advanced_plots
                        logger.info("Generating advanced operational and scalability plots...")
                        generate_all_advanced_plots(
                            db_path, output_dir=output_directory
                        )
                        logger.info("Advanced operational and scalability plots generated successfully.")
                    except Exception as ex_adv_plots:
                        logger.error(f"Error generating advanced plots: {ex_adv_plots}")

            if ANALYSIS_SWEEP_ENABLED:
                logger.info("\nStarting sweep mode...")
                runner.run_sweep_mode()
            if not ANALYSIS_TEMPORAL_ENABLED and not ANALYSIS_SWEEP_ENABLED:
                logger.warning(
                    "analysis_mode.enabled=True but both temporal_enabled=False "
                    "and sweep_enabled=False — no analysis will run."
                )
        else:
            runner.run_elective_mode()

    except Exception as e:
        logger.error(f"\n\nFatal error during simulation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
