"""
Monte Carlo Convergence Analysis — Post-processing offline.

Reads a completed analysis SQLite database and generates convergence plots
showing how key metrics stabilize as more simulations are accumulated.

Usage:
    python offline/convergence_analysis.py [results/20260502_203057/analysis.db]

The script generates, for each (run_id, algo_name):
  1. Cumulative Mean Makespan vs. # Simulations
  2. Cumulative Std Dev of Makespan vs. # Simulations
  3. Best-so-far (running minimum) Makespan vs. # Simulations
  4. A combined summary plot with all algorithms overlaid

Output is saved next to the database file in a 'convergence/' subdirectory.
"""

import os
import sys
import sqlite3
import argparse

# Asegurar que el directorio raíz del proyecto está en el PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for faster rendering
import matplotlib.pyplot as plt


def _connect_read_only(db_path):
    path = os.path.abspath(db_path)
    return sqlite3.connect(f"file:{path.replace(os.sep, '/')}?mode=ro", uri=True)


# ──────────────────────────────────────────────────────────────────────
# Data extraction
# ──────────────────────────────────────────────────────────────────────

def load_simulation_data(db_path: str) -> dict:
    """
    Returns a nested dict:
        {run_id: {algo_name: [(sim_index, final_makespan), ...]}}
    sorted by sim_index within each group.
    """
    conn = _connect_read_only(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT s.run_id, r.num_procedures, s.sim_index, s.algo_name, "
        "       s.final_makespan, s.wall_clock_elapsed_s "
        "FROM simulations s "
        "JOIN runs r ON s.run_id = r.run_id "
        "ORDER BY s.run_id, s.algo_name, s.sim_index"
    ).fetchall()
    conn.close()

    data = {}
    for r in rows:
        run_id = r["run_id"]
        algo = r["algo_name"]
        if run_id not in data:
            data[run_id] = {"num_procedures": r["num_procedures"], "algos": {}}
        if algo not in data[run_id]["algos"]:
            data[run_id]["algos"][algo] = []
        data[run_id]["algos"][algo].append({
            "sim_index": r["sim_index"],
            "makespan": r["final_makespan"],
            "wall_s": r["wall_clock_elapsed_s"],
        })

    return data


def compute_convergence(makespans: list, interval: int = 5) -> dict:
    """
    Given a list of makespans (ordered by sim_index), computes cumulative
    statistics at every `interval` simulations.

    Returns dict with numpy arrays:
        n_sims, cum_mean, cum_std, cum_best, cum_worst, cum_median
    """
    arr = np.array(makespans)
    n = len(arr)

    # Evaluation points: every `interval` sims, always include the last one
    points = list(range(interval, n + 1, interval))
    if points and points[-1] != n:
        points.append(n)
    if not points:
        points = [n]

    n_sims = []
    cum_mean = []
    cum_std = []
    cum_best = []
    cum_worst = []
    cum_median = []

    for k in points:
        subset = arr[:k]
        n_sims.append(k)
        cum_mean.append(np.mean(subset))
        cum_std.append(np.std(subset, ddof=1) if k > 1 else 0.0)
        cum_best.append(np.min(subset))
        cum_worst.append(np.max(subset))
        cum_median.append(np.median(subset))

    return {
        "n_sims": np.array(n_sims),
        "cum_mean": np.array(cum_mean),
        "cum_std": np.array(cum_std),
        "cum_best": np.array(cum_best),
        "cum_worst": np.array(cum_worst),
        "cum_median": np.array(cum_median),
    }


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

ALGO_COLORS = {
    "GA": "#3498db",
    "dPSO": "#e74c3c",
    "SBOA": "#2ecc71",
    "dMShOA": "#9b59b6",
}

ALGO_ORDER = ["GA", "dPSO", "SBOA", "dMShOA"]


def _sorted_algos(algos_dict: dict) -> list:
    """Returns algo names sorted by ALGO_ORDER, unknown algos at the end."""
    known = [a for a in ALGO_ORDER if a in algos_dict]
    unknown = sorted(set(algos_dict.keys()) - set(ALGO_ORDER))
    return known + unknown


def plot_single_algo_convergence(conv: dict, algo: str, num_procs: int,
                                  run_id: int, out_dir: str):
    """Generates a 3-panel convergence plot for one algorithm."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Monte Carlo Convergence — {algo} | X={num_procs} (Run {run_id})",
        fontsize=14, fontweight="bold"
    )
    color = ALGO_COLORS.get(algo, "#555555")
    ns = conv["n_sims"]

    # Panel 1: Cumulative Mean ± Std
    ax = axes[0]
    ax.plot(ns, conv["cum_mean"], color=color, linewidth=2, label="Mean")
    ax.fill_between(ns,
                     conv["cum_mean"] - conv["cum_std"],
                     conv["cum_mean"] + conv["cum_std"],
                     alpha=0.2, color=color, label="±1 Std Dev")
    ax.set_xlabel("# Simulations")
    ax.set_ylabel("Makespan (min)")
    ax.set_title("Cumulative Mean ± Std Dev")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Std Dev convergence
    ax = axes[1]
    ax.plot(ns, conv["cum_std"], color=color, linewidth=2)
    ax.set_xlabel("# Simulations")
    ax.set_ylabel("Std Dev (min)")
    ax.set_title("Standard Deviation Convergence")
    ax.grid(True, alpha=0.3)

    # Panel 3: Best-so-far
    ax = axes[2]
    ax.plot(ns, conv["cum_best"], color=color, linewidth=2, label="Best (min)")
    ax.plot(ns, conv["cum_worst"], color=color, linewidth=1,
            linestyle="--", alpha=0.5, label="Worst (max)")
    ax.set_xlabel("# Simulations")
    ax.set_ylabel("Makespan (min)")
    ax.set_title("Best-so-far & Worst-so-far")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"convergence_{algo}_run{run_id}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combined_convergence(all_conv: dict, num_procs: int,
                               run_id: int, out_dir: str):
    """Overlays all algorithms in a single 2x2 figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Monte Carlo Convergence — All Algorithms | X={num_procs} (Run {run_id})",
        fontsize=14, fontweight="bold"
    )

    sorted_algos = _sorted_algos(all_conv)

    # Panel (0,0): Cumulative Mean
    ax = axes[0, 0]
    for algo in sorted_algos:
        c = all_conv[algo]
        color = ALGO_COLORS.get(algo, "#555")
        ax.plot(c["n_sims"], c["cum_mean"], color=color, linewidth=2, label=algo)
    ax.set_xlabel("# Simulations")
    ax.set_ylabel("Makespan (min)")
    ax.set_title("Cumulative Mean Makespan")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (0,1): Cumulative Std Dev
    ax = axes[0, 1]
    for algo in sorted_algos:
        c = all_conv[algo]
        color = ALGO_COLORS.get(algo, "#555")
        ax.plot(c["n_sims"], c["cum_std"], color=color, linewidth=2, label=algo)
    ax.set_xlabel("# Simulations")
    ax.set_ylabel("Std Dev (min)")
    ax.set_title("Std Dev Convergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (1,0): Best-so-far & Worst-so-far
    ax = axes[1, 0]
    for algo in sorted_algos:
        c = all_conv[algo]
        color = ALGO_COLORS.get(algo, "#555")
        # Plot Best
        ax.plot(c["n_sims"], c["cum_best"], color=color, linewidth=2, label=algo)
        # Plot Worst (dashed, thinner, no separate legend entry to avoid clutter)
        ax.plot(c["n_sims"], c["cum_worst"], color=color, linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("# Simulations")
    ax.set_ylabel("Makespan (min)")
    ax.set_title("Best-so-far (solid) & Worst-so-far (dashed)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (1,1): Mean ± Std band (all algos)
    ax = axes[1, 1]
    for algo in sorted_algos:
        c = all_conv[algo]
        color = ALGO_COLORS.get(algo, "#555")
        ax.plot(c["n_sims"], c["cum_mean"], color=color, linewidth=2, label=algo)
        ax.fill_between(c["n_sims"],
                         c["cum_mean"] - c["cum_std"],
                         c["cum_mean"] + c["cum_std"],
                         alpha=0.1, color=color)
    ax.set_xlabel("# Simulations")
    ax.set_ylabel("Makespan (min)")
    ax.set_title("Mean ± 1σ Band (all algorithms)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"convergence_combined_run{run_id}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_convergence_by_time(algos_data: dict, num_procs: int,
                               run_id: int, out_dir: str):
    """
    Convergence by wall-clock time.
    Shows: at each simulation completion, the best makespan found so far.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        f"Best Makespan vs. Wall-Clock Time | X={num_procs} (Run {run_id})",
        fontsize=14, fontweight="bold"
    )

    sorted_algos = _sorted_algos(algos_data)

    for algo in sorted_algos:
        sims = algos_data[algo]
        # Sort by wall_clock
        sims_sorted = sorted(sims, key=lambda s: s["wall_s"])

        if not sims_sorted:
            continue

        times_min = [0.0]
        bests = [sims_sorted[0]["makespan"]]  # Initial value (extrapolated back to t=0 for visual continuity)
        
        current_best = float("inf")
        for s in sims_sorted:
            if s["makespan"] < current_best:
                current_best = s["makespan"]
            times_min.append(s["wall_s"] / 60.0)
            bests.append(current_best)

        color = ALGO_COLORS.get(algo, "#555")
        ax.step(times_min, bests, where="post", color=color, linewidth=2, label=algo)

    ax.set_xlabel("Wall-Clock Time (minutes)")
    ax.set_ylabel("Best Makespan Found (min)")
    ax.set_title("Best-so-far by Wall-Clock Time")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"convergence_by_time_run{run_id}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _find_latest_analysis_db(base_dir: str = "results") -> str | None:
    """Busca recursivamente todos los archivos 'analysis.db' y retorna el más reciente."""
    candidate_dbs = []
    if not os.path.exists(base_dir):
        return None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "analysis.db":
                path = os.path.join(root, file)
                candidate_dbs.append((path, os.path.getmtime(path)))
    if not candidate_dbs:
        return None
    # Ordenar por mtime descendente (el más nuevo primero)
    candidate_dbs.sort(key=lambda x: x[1], reverse=True)
    return candidate_dbs[0][0]


def process_and_plot_run(run_id, run_data, interval, out_dir):
    """Calcula la convergencia y genera los gráficos para una corrida en particular."""
    num_procs = run_data["num_procedures"]
    algos = run_data["algos"]
    total_sims = max(len(v) for v in algos.values()) if algos else 0

    print(f"Procesando Run {run_id} (X={num_procs}, {total_sims} sims/algo)...")

    # Ajustar intervalo dinámicamente si total_sims es muy pequeño
    effective_interval = interval
    if total_sims > 0:
        if total_sims < 10:
            effective_interval = 1
        elif total_sims < 50:
            effective_interval = min(interval, 5)

    all_conv = {}
    for algo in _sorted_algos(algos):
        sims = algos[algo]
        makespans = [s["makespan"] for s in sims]
        conv = compute_convergence(makespans, interval=effective_interval)
        all_conv[algo] = conv

    # Gráfico combinado
    plot_combined_convergence(all_conv, num_procs, run_id, out_dir)

    # Gráfico por tiempo de reloj de pared
    plot_convergence_by_time(algos, num_procs, run_id, out_dir)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Convergence Analysis for Hospital Simulation"
    )
    parser.add_argument("db_path", nargs="?", default=None, help="Path to the analysis SQLite database")
    parser.add_argument("--interval", type=int, default=5,
                        help="Evaluate convergence every N simulations (default: 5)")
    parser.add_argument("--time-interval", type=float, default=300.0,
                        help="Wall-clock time interval in seconds for time-based plot (default: 300 = 5min)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of concurrent jobs for joblib parallel processing (default: -1)")
    args = parser.parse_args()

    db_path = args.db_path
    if not db_path:
        db_path = _find_latest_analysis_db()
        if not db_path:
            print("ERROR: No se especificó db_path y no se encontró ningún archivo 'analysis.db' en 'results/'.")
            sys.exit(1)
        print(f"No se especificó db_path. Usando la base de datos más reciente: {db_path}")

    if not os.path.exists(db_path):
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)

    out_dir = os.path.join(os.path.dirname(db_path), "convergence")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Cargando datos desde: {db_path}")
    data = load_simulation_data(db_path)

    # Procesar corridas en paralelo usando joblib
    from joblib import Parallel, delayed
    runs_sorted = sorted(data.items())
    
    print(f"Procesando {len(runs_sorted)} corridas en paralelo (n_jobs={args.n_jobs})...")
    Parallel(n_jobs=args.n_jobs)(
        delayed(process_and_plot_run)(run_id, run_data, args.interval, out_dir)
        for run_id, run_data in runs_sorted
    )

    print(f"\nTodos los gráficos de convergencia guardados en: {out_dir}")


if __name__ == "__main__":
    main()
