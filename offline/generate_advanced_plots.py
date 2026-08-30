"""
generate_advanced_plots.py -- Generates advanced operational and scalability plots from analysis.db

Queries the v_simulation_summary view in SQLite analysis.db to extract:
  - workload_std_min (Workload Balance)
  - total_overtime_min (Room Overtime)
  - value_added_ratio (Value-Added ratio vs Waste)
  - final_makespan (Scalability)

Generates:
  1. Boxplots for Workload Balance, Room Overtime, and Value-Added Ratio per Run.
  2. Scalability Line Plots (Makespan, Overtime, and Value-Added Ratio vs. Number of Procedures).
"""

import os
import sys
import sqlite3
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless plotting
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import logger
from utils.plotting import ALGORITHM_DATA_KEYS, ALGORITHM_ORDER, ALGORITHM_COLORS, PlotConfig

BOXPLOT_COLORS = ["lightblue", "lightsalmon", "lightgreen", "mediumpurple"]


def _connect_read_only(db_path):
    path = Path(db_path).resolve()
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def load_advanced_data(db_path: str) -> tuple[dict, dict]:
    """
    Query v_simulation_summary. Returns:
    1. per_run_data: {
           run_id: {
               "num_procedures": int,
               "results": {
                   algo_name: {
                       "workload_std": [],
                       "overtime": [],
                       "value_added": [],
                       "makespan": []
                   }
               }
           }
       }
    2. sweep_data: {
           algo_name: {
               "procedures": [],
               "avg_makespan": [],
               "avg_overtime": [],
               "avg_value_added": []
           }
       }
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = _connect_read_only(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get runs and their procedures
    cursor.execute("SELECT run_id, num_procedures FROM runs ORDER BY run_id")
    runs = cursor.fetchall()

    per_run_data = {}
    for r in runs:
        run_id = r["run_id"]
        per_run_data[run_id] = {
            "num_procedures": r["num_procedures"],
            "results": {}
        }

    # Query v_simulation_summary
    cursor.execute(
        "SELECT run_id, num_procedures, algo_name, final_makespan, "
        "       total_overtime_min, workload_std_min, value_added_ratio "
        "FROM v_simulation_summary "
        "ORDER BY run_id, algo_name, sim_index"
    )
    rows = cursor.fetchall()

    for row in rows:
        run_id = row["run_id"]
        algo = row["algo_name"]
        
        if run_id not in per_run_data:
            continue

        results = per_run_data[run_id]["results"]
        if algo not in results:
            results[algo] = {
                "workload_std": [],
                "overtime": [],
                "value_added": [],
                "makespan": []
            }
        
        # Guard against inf makespans
        mk = row["final_makespan"]
        if mk != float("inf") and mk is not None:
            results[algo]["makespan"].append(mk)
            
        ov = row["total_overtime_min"]
        if ov is not None:
            results[algo]["overtime"].append(ov)
            
        ws = row["workload_std_min"]
        if ws is not None:
            results[algo]["workload_std"].append(ws)
            
        va = row["value_added_ratio"]
        if va is not None:
            results[algo]["value_added"].append(va)

    conn.close()

    # Build sweep data
    # Aggregate stats per (algo, num_procedures)
    algo_sweep = {}
    
    # Identify unique procedures in order
    unique_procs = sorted(list(set(r["num_procedures"] for r in runs)))
    
    for algo_name in ALGORITHM_ORDER:
        algo_sweep[algo_name] = {
            "procedures": [],
            "avg_makespan": [],
            "avg_overtime": [],
            "avg_value_added": []
        }
        
        data_name = ALGORITHM_DATA_KEYS[algo_name]
        for num_proc in unique_procs:
            # Find the run_id that corresponds to this num_proc
            target_run_id = None
            for r_id, r_info in per_run_data.items():
                if r_info["num_procedures"] == num_proc:
                    target_run_id = r_id
                    break
            
            if target_run_id is None or data_name not in per_run_data[target_run_id]["results"]:
                continue

            res = per_run_data[target_run_id]["results"][data_name]
            
            algo_sweep[algo_name]["procedures"].append(num_proc)
            
            mks = [m for m in res["makespan"] if m != float("inf")]
            algo_sweep[algo_name]["avg_makespan"].append(np.mean(mks) if mks else 0.0)
            
            ovs = res["overtime"]
            algo_sweep[algo_name]["avg_overtime"].append(np.mean(ovs) if ovs else 0.0)
            
            vas = res["value_added"]
            algo_sweep[algo_name]["avg_value_added"].append(np.mean(vas) if vas else 0.0)

    return per_run_data, algo_sweep


def plot_advanced_boxplot(results: dict, key: str, title: str | None, ylabel: str,
                          filename: str, run_id: int, output_dir: str) -> None:
    """Generates comparison boxplot for a specific key (workload_std, overtime, value_added)"""
    fig, ax = plt.subplots(figsize=PlotConfig.DEFAULT_FIGSIZE)
    
    data_to_plot = []
    labels = []
    
    for algo_name in ALGORITHM_ORDER:
        data_name = ALGORITHM_DATA_KEYS[algo_name]
        if data_name in results and results[data_name][key]:
            data_to_plot.append(results[data_name][key])
            labels.append(algo_name)
            
    if not data_to_plot:
        plt.close(fig)
        return

    try:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showmeans=True)
    except TypeError:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
    
    # Style colors
    for patch, label in zip(bp["boxes"], labels):
        color = ALGORITHM_COLORS.get(label, "mediumpurple")
        patch.set_facecolor(color)
        
    ax.set_ylabel(ylabel, fontsize=PlotConfig.DEFAULT_AXIS_LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Save
    run_sub_dir = os.path.join(output_dir, "advanced")
    os.makedirs(run_sub_dir, exist_ok=True)
    
    png_path = os.path.join(run_sub_dir, f"{filename}.png")
    fig.savefig(png_path, dpi=PlotConfig.DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    
def plot_scalability_line(sweep_data: dict, key: str, title: str | None, ylabel: str,
                          filename: str, output_dir: str) -> None:
    """Generates line plot showing average metric vs procedures (X)"""
    fig, ax = plt.subplots(figsize=PlotConfig.DEFAULT_FIGSIZE)
    
    for algo_name in ALGORITHM_ORDER:
        if algo_name not in sweep_data:
            continue
        data = sweep_data[algo_name]
        if not data["procedures"]:
            continue
            
        color = ALGORITHM_COLORS.get(algo_name, "steelblue")
        ax.plot(
            data["procedures"], 
            data[key], 
            marker="o", 
            linewidth=2, 
            color=color, 
            label=algo_name
        )
        
    ax.set_xlabel("Number of Procedures (X)", fontsize=PlotConfig.DEFAULT_AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=PlotConfig.DEFAULT_AXIS_LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(sweep_data[ALGORITHM_ORDER[0]]["procedures"])
    ax.legend(fontsize=PlotConfig.DEFAULT_LEGEND_SIZE)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Save
    png_path = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(png_path, dpi=PlotConfig.DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    
def generate_all_advanced_plots(
    db_path: str, output_dir: str | os.PathLike[str] | None = None
) -> None:
    """Loads data and generates all advanced boxplots and scalability sweeps."""
    analysis_dir = os.fspath(output_dir) if output_dir is not None else os.path.dirname(
        os.path.abspath(db_path)
    )
    
    logger.info(f"Loading advanced metric data from database: {db_path}")
    try:
        per_run_data, sweep_data = load_advanced_data(db_path)
    except Exception as e:
        logger.error(f"Failed to load advanced data: {e}")
        return
        
    # 1. Generate per-run boxplots
    for run_id, data in sorted(per_run_data.items()):
        num_procs = data["num_procedures"]
        results = data["results"]
        
        logger.info(f"Generating advanced boxplots for Run {run_id} (procedures = {num_procs})...")
        run_plots_dir = os.path.join(analysis_dir, "plots", f"run{run_id}")
        os.makedirs(run_plots_dir, exist_ok=True)
        
        # Workload imbalance (Standard Deviation of work minutes)
        plot_advanced_boxplot(
            results=results,
            key="workload_std",
            title=None,
            ylabel="Workload Std Dev (minutes)",
            filename="workload_imbalance",
            run_id=run_id,
            output_dir=run_plots_dir,
        )
        
        # Room Overtime
        plot_advanced_boxplot(
            results=results,
            key="overtime",
            title=None,
            ylabel="Total Overtime (minutes)",
            filename="room_overtime",
            run_id=run_id,
            output_dir=run_plots_dir,
        )
        
        # Value added ratio
        plot_advanced_boxplot(
            results=results,
            key="value_added",
            title=None,
            ylabel="Value-Added Ratio",
            filename="value_added_ratio",
            run_id=run_id,
            output_dir=run_plots_dir,
        )
        
    # 2. Generate scalability sweep line plots
    logger.info("Generating scalability sweep line plots...")
    sweep_plots_dir = os.path.join(analysis_dir, "plots", "advanced")
    os.makedirs(sweep_plots_dir, exist_ok=True)
    
    # Makespan scalability
    plot_scalability_line(
        sweep_data=sweep_data,
        key="avg_makespan",
        title=None,
        ylabel="Average Makespan (minutes)",
        filename="sweep_makespan",
        output_dir=sweep_plots_dir,
    )
    
    # Overtime scalability
    plot_scalability_line(
        sweep_data=sweep_data,
        key="avg_overtime",
        title=None,
        ylabel="Average Overtime (minutes)",
        filename="sweep_overtime",
        output_dir=sweep_plots_dir,
    )
    
    # Value Added Ratio scalability
    plot_scalability_line(
        sweep_data=sweep_data,
        key="avg_value_added",
        title=None,
        ylabel="Average Value-Added Ratio",
        filename="sweep_value_added",
        output_dir=sweep_plots_dir,
    )
    
    logger.info("Advanced plots generation complete.")


def _find_latest_analysis_db(base_dir: str = "results") -> str | None:
    """Finds the most recent analysis.db file in results directory."""
    from utils.results_locator import get_analysis_db_path
    p = get_analysis_db_path()
    if p.exists():
        return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate advanced operational and scalability plots from analysis.db"
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

    generate_all_advanced_plots(db_path)
    logger.info("Advanced plot generation complete.")


if __name__ == "__main__":
    main()
