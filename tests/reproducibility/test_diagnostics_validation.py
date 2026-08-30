"""Focused tests for strict offline reproducibility diagnostics."""

from pathlib import Path

import pandas as pd
import pytest


def test_validate_pairing_rejects_mismatched_value_added_ratio(tmp_path):
    from reproducibility.diagnostics.validate_pairing import (
        PairingValidationError,
        validate_pairing,
    )

    rows = [
        {"sim_index": 0, "algo_name": algo, "value_added_ratio": 0.5, "final_makespan": 10}
        for algo in ("GA", "SBOA")
    ]
    rows[-1]["value_added_ratio"] = 0.6
    pd.DataFrame(rows).to_csv(tmp_path / "simulation_summary_run1.csv", index=False)

    with pytest.raises(PairingValidationError, match="value_added_ratio"):
        validate_pairing(tmp_path, runs={1: 1}, algos=("GA", "SBOA"))


def test_validate_pairing_accepts_complete_pair(tmp_path):
    from reproducibility.diagnostics.validate_pairing import validate_pairing

    rows = [
        {"sim_index": i, "algo_name": algo, "value_added_ratio": 0.5, "final_makespan": 10 + i}
        for i in range(2)
        for algo in ("GA", "SBOA")
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "simulation_summary_run1.csv", index=False)
    report = validate_pairing(tmp_path, runs={1: 1}, algos=("GA", "SBOA"))
    assert report[1]["paired_rows"] == 2


def test_validate_fixed_instance_rejects_drift(tmp_path):
    from reproducibility.diagnostics.check_fixed_instance import (
        FixedInstanceValidationError,
        validate_fixed_instance,
    )

    import sqlite3

    db = Path(tmp_path) / "analysis.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE runs (run_id INTEGER, num_procedures INTEGER, config_snapshot TEXT);
        CREATE TABLE simulations (sim_id INTEGER, run_id INTEGER);
        CREATE TABLE cie10_breakdown (sim_id INTEGER, job_id INTEGER, proc_time_min REAL);
        """
    )
    conn.execute("INSERT INTO runs VALUES (1, 1, '{\"fixed_pool_per_run\": true}')")
    conn.executemany("INSERT INTO simulations VALUES (?, 1)", [(1,), (2,)])
    conn.executemany("INSERT INTO cie10_breakdown VALUES (?, 1, ?)", [(1, 10.0), (2, 11.0)])
    conn.commit()
    conn.close()

    with pytest.raises(FixedInstanceValidationError, match="proc_time varies"):
        validate_fixed_instance(db)
