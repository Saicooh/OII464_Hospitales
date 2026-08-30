"""
generate_analysis_plots.py -- Generates boxplots and makespan histograms from analysis.db

Reads the simulations table in SQLite analysis.db, groups makespans by run and algorithm,
and generates standard comparison plots (makespan boxplots and frequency histograms).
"""

import os
import sys
import sqlite3
import argparse
from pathlib import Path

# Ensure project root is in PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import logger
from utils.plotting import plot_boxplot, plot_makespan_histogram


def _connect_read_only(db_path):
    path = Path(db_path).resolve()
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def load_run_makespans(db_path: str) -> dict:
    """
    Query all runs and simulations, returns a nested dict:
    {
        run_id: {
            "num_procedures": int,
            "results": {
                algo_name: {
                    "makespan": [makespan1, makespan2, ...]
                }
            }
        }
    }
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = _connect_read_only(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get runs
    cursor.execute("SELECT run_id, num_procedures FROM runs ORDER BY run_id")
    runs = cursor.fetchall()

    run_data = {}
    for r in runs:
        run_id = r["run_id"]
        run_data[run_id] = {
            "num_procedures": r["num_procedures"],
            "results": {}
        }

    # Get simulations
    cursor.execute(
        "SELECT run_id, algo_name, final_makespan FROM simulations "
        "ORDER BY run_id, algo_name, sim_index"
    )
    sims = cursor.fetchall()

    for s in sims:
        run_id = s["run_id"]
        algo = s["algo_name"]
        makespan = s["final_makespan"]

        if run_id not in run_data:
            continue

        results = run_data[run_id]["results"]
        if algo not in results:
            results[algo] = {"makespan": []}
        results[algo]["makespan"].append(makespan)

    conn.close()
    return run_data


def generate_all_analysis_plots(
    db_path: str, output_dir: str | os.PathLike[str] | None = None
) -> None:
    """
    Generates all boxplots and histograms for each run in the database.
    Saves them below the supplied analysis output directory. When no output
    directory is supplied, the directory containing the database is used.
    """
    analysis_dir = os.fspath(output_dir) if output_dir is not None else os.path.dirname(
        os.path.abspath(db_path)
    )

    logger.info(f"Loading makespan data from database: {db_path}")
    try:
        run_data = load_run_makespans(db_path)
    except Exception as e:
        logger.error(f"Failed to load data from {db_path}: {e}")
        return

    for run_id, data in sorted(run_data.items()):
        num_procs = data["num_procedures"]
        results = data["results"]
        
        logger.info(f"Generating plots for Run {run_id} (procedures = {num_procs})...")

        # Define run specific output directory
        run_plots_dir = os.path.join(analysis_dir, "plots", f"run{run_id}")
        os.makedirs(run_plots_dir, exist_ok=True)

        # 1. Boxplot
        boxplot_png, _ = plot_boxplot(results, run_plots_dir, show_title=False)
        if boxplot_png and os.path.exists(boxplot_png):
            logger.info(f"  Run {run_id} boxplot saved to {boxplot_png}")
        else:
            logger.warning(f"  Failed to generate boxplot for Run {run_id}")

        # 2. Histograms
        for algo_name, res in results.items():
            makespans = [mk for mk in res["makespan"] if mk != float("inf")]
            if len(makespans) >= 2:
                hist_png, _ = plot_makespan_histogram(makespans, algo_name, run_plots_dir)
                if hist_png and os.path.exists(hist_png):
                    logger.info(
                        f"  Run {run_id} histogram for {algo_name} saved to {hist_png}"
                    )
                else:
                    logger.warning(f"  Failed to generate histogram for Run {run_id}, Algo {algo_name}")


def _find_latest_analysis_db(base_dir: str = "results") -> str | None:
    """Finds the most recent analysis.db file in results directory."""
    from utils.results_locator import get_analysis_db_path
    p = get_analysis_db_path()
    if p.exists():
        return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison boxplots and makespan histograms from analysis.db"
    )
    parser.add_argument("db_path", nargs="?", default=None, help="Path to the analysis SQLite database")
    args = parser.parse_args()

    db_path = args.db_path
    if not db_path:
        db_path = _find_latest_analysis_db()
        if not db_path:
            logger.error("No database path specified and no analysis.db found in results/.")
            sys.exit(1)
        logger.info(f"No database path specified. Using the latest one found: {db_path}")

    generate_all_analysis_plots(db_path)
    logger.info("Plot generation complete.")


if __name__ == "__main__":
    main()
