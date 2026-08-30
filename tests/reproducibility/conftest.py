"""Shared fixtures for the persisted-instance validation tests.

The unit layer uses synthetic file-backed SQLite databases built by
:func:`build_synthetic_db`. They are written through the production
``AnalysisPersistence`` schema so the reader is exercised against the real
table definitions.

The synthetic databases are file-backed rather than ``:memory:`` because the
reader opens its connection through a read-only URI
(``file:<path>?mode=ro``), which has no in-memory equivalent.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from core.analysis_persistence import AnalysisPersistence

DEFAULT_ALGOS: tuple[str, ...] = ("GA", "SBOA", "dPSO", "dMShOA")


def _reload_project_config(config_path: str | None) -> None:
    """Reload ``config.config`` from the profile selected by *config_path*."""
    if config_path is None:
        os.environ.pop("HOSPITAL_CONFIG_PATH", None)
    else:
        os.environ["HOSPITAL_CONFIG_PATH"] = config_path
    for module_name in [name for name in sys.modules if name.startswith("config.")]:
        del sys.modules[module_name]
    importlib.import_module("config.config")


@pytest.fixture(scope="package", autouse=True)
def canonical_config_profile():
    """Pin the default configuration profile for this package.

    ``tests/config/test_config.py`` points ``HOSPITAL_CONFIG_PATH`` at a
    temporary YAML declaring 4 rooms, 1 anesthetist and 1 surgeon, reloads
    ``config.config`` from it, and never restores the previous value. Every
    test collected afterwards therefore inherits that profile.

    The reader resolves resource counts from configuration by design, so this
    package pins the profile its assertions describe. The previous state is
    restored on teardown so no test outside this package observes a change.
    """
    previous = os.environ.get("HOSPITAL_CONFIG_PATH")
    _reload_project_config(None)
    try:
        yield
    finally:
        try:
            _reload_project_config(previous)
        except Exception:  # pragma: no cover - the leaked profile may be gone
            _reload_project_config(None)

# Job field names accepted by build_synthetic_db.
JOB_FIELDS = ("transition", "setup", "anesthesia", "surgery", "cleanup")


def build_synthetic_db(
    path: Path | str,
    jobs: Mapping[int, Mapping[str, float]],
    *,
    algos: Sequence[str] = DEFAULT_ALGOS,
    sims_per_algo: int = 3,
    best_makespan: float | None = None,
    setup_spread: Mapping[int, float] | None = None,
    proc_spread: Mapping[int, float] | None = None,
    cleanup_spread: Mapping[int, float] | None = None,
    transition_spread: Mapping[int, float] | None = None,
    algo_op1_offset: Mapping[str, float] | None = None,
) -> Path:
    """Create a file-backed campaign-shaped database and return its path.

    Parameters
    ----------
    jobs:
        Mapping ``job_id -> {transition, setup, anesthesia, surgery, cleanup}``
        in minutes. These are the ground-truth values the reader must recover.
    sims_per_algo:
        Simulations written per algorithm. The first simulation of every
        algorithm schedules every job first, so ``MIN(op1_finish)`` reaches the
        physical floor ``max(transition, setup) + anesthesia``; later
        simulations are pushed away from that floor.
    best_makespan:
        Value stored as the minimum ``final_makespan``. Defaults to the longest
        job chain, which keeps the lower bound at or below the best schedule.
    setup_spread, proc_spread, cleanup_spread, transition_spread:
        Per-job value added to the corresponding column on the last simulation
        only, used to inject controlled drift into an otherwise fixed instance.
    algo_op1_offset:
        Per-algorithm value added to every ``op1_finish``, used to break the
        per-algorithm agreement of the anesthesia estimator.
    """
    setup_spread = setup_spread or {}
    proc_spread = proc_spread or {}
    cleanup_spread = cleanup_spread or {}
    transition_spread = transition_spread or {}
    algo_op1_offset = algo_op1_offset or {}

    chains = [
        max(spec["transition"], spec["setup"])
        + spec["anesthesia"]
        + spec["surgery"]
        + spec["cleanup"]
        for spec in jobs.values()
    ]
    if best_makespan is None:
        best_makespan = max(chains) if chains else 0.0

    total_sims = len(algos) * sims_per_algo
    persistence = AnalysisPersistence(str(path))
    persistence.init_db()
    run_id = persistence.insert_run(total_sims, len(jobs), {"synthetic": True})

    sim_ordinal = 0
    for algo in algos:
        for sim_index in range(sims_per_algo):
            is_last = sim_index == sims_per_algo - 1
            # Only the first simulation of the run carries the best makespan,
            # so MIN(final_makespan) is exactly `best_makespan`.
            makespan = best_makespan if sim_ordinal == 0 else best_makespan + 25.0
            sim_id = persistence.insert_simulation(
                run_id=run_id,
                sim_index=sim_index,
                algo_name=algo,
                wall_clock_s=1.0,
                final_makespan=makespan,
                combined_obj=None,
            )

            breakdowns = []
            waits = []
            for job_id, spec in jobs.items():
                setup = spec["setup"] + (setup_spread.get(job_id, 0.0) if is_last else 0.0)
                surgery = spec["surgery"] + (
                    proc_spread.get(job_id, 0.0) if is_last else 0.0
                )
                cleanup = spec["cleanup"] + (
                    cleanup_spread.get(job_id, 0.0) if is_last else 0.0
                )
                transition = spec["transition"] + (
                    transition_spread.get(job_id, 0.0) if is_last else 0.0
                )
                breakdowns.append(
                    {
                        "job_id": job_id,
                        "codigo_cie10": "Z00",
                        "grupo": "synthetic",
                        "setup_min": setup,
                        "proc_time_min": surgery,
                        # Stale in the real campaign; the reader must not read it.
                        "transition_min": 0.0,
                        "cleanup_min": cleanup,
                        "setup_op1": setup,
                        "setup_op2": 0.0,
                        "cleanup_op1": 0.0,
                        "cleanup_op2": cleanup,
                    }
                )
                floor = max(spec["transition"], spec["setup"]) + spec["anesthesia"]
                # sim_index 0 attains the floor; later simulations queue behind it.
                op1_finish = floor + sim_index * 7.0 + algo_op1_offset.get(algo, 0.0)
                waits.append(
                    {
                        "job_id": job_id,
                        "op1_room": "Pabellon_1",
                        "op2_room": "Pabellon_2",
                        "op1_finish": op1_finish,
                        "op2_start": op1_finish,
                        "transition_used": transition,
                        "extra_wait_min": 0.0,
                    }
                )

            persistence.save_breakdowns_batch(sim_id, breakdowns)
            persistence.save_patient_wait_batch(sim_id, waits)
            sim_ordinal += 1

    persistence.close()
    return Path(path)


#: Two-job instance with hand-computed values used across the unit layer.
#: Job 1 chain = max(20, 5) + 30 + 100 + 10 = 160.0
#: Job 2 chain = max(4, 9) + 12 + 40 + 6 = 67.0
HAND_COMPUTED_JOBS: dict[int, dict[str, float]] = {
    1: {
        "transition": 20.0,
        "setup": 5.0,
        "anesthesia": 30.0,
        "surgery": 100.0,
        "cleanup": 10.0,
    },
    2: {
        "transition": 4.0,
        "setup": 9.0,
        "anesthesia": 12.0,
        "surgery": 40.0,
        "cleanup": 6.0,
    },
}


@pytest.fixture
def synthetic_db(tmp_path: Path) -> Path:
    """Two-job, three-simulation-per-algorithm database with known values."""
    return build_synthetic_db(tmp_path / "analysis.db", HAND_COMPUTED_JOBS)
