"""Validate paired algorithm summaries without modifying any output."""
from pathlib import Path
import argparse
import sys
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils.results_locator import get_results_dir

ALGOS = ["GA", "SBOA", "dPSO", "dMShOA Old"]
RUN_SIZES = {1: 15, 2: 20, 3: 25, 4: 30}


class PairingValidationError(RuntimeError):
    """Persisted summaries do not contain a complete paired design."""


def validate_pairing(results_dir=None, runs=None, algos=None, tolerance=1e-9):
    """Return per-run pairing evidence or raise on any inconsistency."""
    results_dir = Path(results_dir or get_results_dir())
    runs = runs or RUN_SIZES
    algos = list(algos or ALGOS)
    report = {}
    errors = []

    for run, size in runs.items():
        path = results_dir / f"simulation_summary_run{run}.csv"
        if not path.exists():
            errors.append(f"run={run}: missing {path}")
            continue
        df = pd.read_csv(path)
        required = {"sim_index", "algo_name", "value_added_ratio", "final_makespan"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"run={run}: missing columns {sorted(missing)}")
            continue
        df = df[df["algo_name"].isin(algos)]
        counts = df.groupby("algo_name")["sim_index"].nunique()
        if set(counts.index) != set(algos) or counts.nunique() != 1:
            errors.append(f"run={run}: algorithm replication counts are {counts.to_dict()}")
            continue
        duplicates = df.duplicated(["sim_index", "algo_name"]).any()
        if duplicates:
            errors.append(f"run={run}: duplicate (sim_index, algo_name) rows")
            continue
        var = df.pivot(index="sim_index", columns="algo_name", values="value_added_ratio")[algos].dropna()
        mk = df.pivot(index="sim_index", columns="algo_name", values="final_makespan")[algos].dropna()
        ranges = var.max(axis=1) - var.min(axis=1)
        inconsistent = ranges[ranges >= tolerance]
        if not inconsistent.empty:
            errors.append(
                f"run={run}: {len(inconsistent)} sim_index rows differ in value_added_ratio; "
                f"max_range={ranges.max():.3g}"
            )
        report[run] = {
            "num_procedures": size,
            "paired_rows": int(len(var)),
            "identical_value_added_ratio": int((ranges < tolerance).sum()),
            "max_value_added_ratio_range": float(ranges.max()) if len(ranges) else None,
            "makespan_across_replications_std": float(mk.mean(axis=1).std()),
        }

    if errors:
        raise PairingValidationError("Pairing validation failed: " + "; ".join(errors))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on inconsistency")
    args = parser.parse_args()
    try:
        report = validate_pairing(args.results_dir)
    except PairingValidationError as exc:
        print(f"ERROR: {exc}")
        if args.strict:
            raise
        return
    for run, evidence in report.items():
        print(f"INSTANCE {run} (N={evidence['num_procedures']}):")
        print(f"  paired reps: {evidence['paired_rows']}")
        print(
            "  identical VAR across all algorithms: "
            f"{evidence['identical_value_added_ratio']}/{evidence['paired_rows']}"
        )
        print(
            "  max per-rep VAR range: "
            f"{evidence['max_value_added_ratio_range']:.2e}\n"
        )


if __name__ == "__main__":
    main()
