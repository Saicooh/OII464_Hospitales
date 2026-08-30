"""Validate the fixed-instance contract of a persisted campaign database.

The default CLI keeps the historical reporting behaviour. Pass ``--strict``
to make any contract violation fail with a named exception and non-zero exit.
"""
from pathlib import Path
import argparse
import json
import sqlite3
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import pandas as pd
from utils.results_locator import get_analysis_db_path


class FixedInstanceValidationError(RuntimeError):
    """The persisted campaign does not satisfy the fixed-instance contract."""


DRIFT_TOLERANCE_MIN = 1e-9


def _connect_read_only(db_path):
    path = Path(db_path).resolve()
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def validate_fixed_instance(db_path=None):
    """Return evidence for every run or raise on a persisted inconsistency."""
    db_path = Path(db_path or get_analysis_db_path()).resolve()
    conn = _connect_read_only(db_path)
    try:
        runs = conn.execute(
            "SELECT run_id, num_procedures, config_snapshot FROM runs ORDER BY run_id"
        ).fetchall()
        drift_rows = conn.execute(
            "SELECT s.run_id, b.job_id, COUNT(*), "
            "MIN(b.proc_time_min), MAX(b.proc_time_min) "
            "FROM cie10_breakdown b JOIN simulations s ON b.sim_id=s.sim_id "
            "GROUP BY s.run_id, b.job_id "
            "ORDER BY s.run_id, b.job_id"
        ).fetchall()
    finally:
        conn.close()

    flags = []
    for run_id, _, snapshot in runs:
        try:
            config = json.loads(snapshot)
        except (TypeError, json.JSONDecodeError):
            flags.append(f"run_id={run_id}: invalid config_snapshot JSON")
            continue
        if config.get("fixed_pool_per_run") is not True:
            flags.append(f"run_id={run_id}: fixed_pool_per_run is not true")

    drift_rows = [
        row for row in drift_rows if row[4] - row[3] > DRIFT_TOLERANCE_MIN
    ]
    if drift_rows:
        flags.extend(
            f"run_id={run_id}, job_id={job_id}: proc_time varies "
            f"({low!r}..{high!r}, spread={high - low:.3g})"
            for run_id, job_id, distinct, low, high in drift_rows
        )
    if flags:
        raise FixedInstanceValidationError(
            f"Fixed-instance validation failed for {db_path}: " + "; ".join(flags)
        )
    return {"database": str(db_path), "runs": len(runs), "drift_rows": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="Persisted analysis.db path")
    parser.add_argument(
        "--strict", action="store_true", help="Exit non-zero when validation fails"
    )
    args = parser.parse_args()
    try:
        report = validate_fixed_instance(args.db)
    except FixedInstanceValidationError as exc:
        print(f"ERROR: {exc}")
        if args.strict:
            raise
        return

    print(f"Validated {report['runs']} fixed runs from {report['database']}")
    conn = _connect_read_only(report["database"])
    try:
        print("\n=== runs.config_snapshot (per run) ===")
        for rid, np_, cfg in conn.execute(
            "SELECT run_id, num_procedures, config_snapshot FROM runs"
        ).fetchall():
            try:
                c = json.loads(cfg)
            except Exception:
                c = {}
            keys = ["std_factor", "use_real_data", "fixed_pool_per_run", "num_procedures", "sims_per_run"]
            print(f"  run {rid} N={np_}: " + ", ".join(f"{k}={c.get(k)}" for k in keys))
        q = """SELECT b.job_id, COUNT(DISTINCT b.proc_time_min) distinct_proc,
                      MIN(b.proc_time_min) mn, MAX(b.proc_time_min) mx
               FROM cie10_breakdown b JOIN simulations s ON b.sim_id=s.sim_id
               WHERE s.run_id=4 AND s.algo_name='GA'
               GROUP BY b.job_id ORDER BY b.job_id LIMIT 6"""
        print("\n=== Distinct proc_time per job, run4 GA ===")
        print(pd.read_sql(q, conn).to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
