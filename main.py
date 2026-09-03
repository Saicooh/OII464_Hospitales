"""Entry point for the synthetic simulation with historical report compatibility."""

from __future__ import annotations

import sys
import os

from config.config import INSTANCE_PATH, N_JOBS, NUM_SIMULATIONS
from core.legacy_runner import LegacyExperimentRunner
from core.simulation_runner import SimulationRunner
from utils.logger import logger


def run_selected_instance(solver, **runtime_settings):
    """Keep the typed single-solver API available for integrations and tests."""
    runner = SimulationRunner.from_instance(
        INSTANCE_PATH, solver=solver, **runtime_settings
    )
    return runner.run_instance_mode()


def main() -> dict[str, str]:
    """Run the configured algorithm comparison and historical report contract."""
    runner = LegacyExperimentRunner(
        instance_path=INSTANCE_PATH,
        num_simulations=NUM_SIMULATIONS,
        n_jobs=N_JOBS,
        base_dir=os.environ.get("HOSPITAL_OUTPUT_ROOT", "results"),
    )
    return runner.run_elective_mode()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logger.error("Fatal error during synthetic simulation: %s", error)
        sys.exit(1)
