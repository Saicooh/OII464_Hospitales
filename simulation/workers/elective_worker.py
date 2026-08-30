"""Worker for elective simulation mode."""

import numpy as np
import time
from simulation.scheduler import run_static_schedule
from config.config import USE_REAL_DATA, TRACE_CSV_PATH


def _build_day_data(job_ids, std_factor, sim_i):
    """
    Genera los datos del día de cirugías usando datos reales (PKL) o sintéticos,
    según la configuración USE_REAL_DATA.

    Retorna:
        (day_data, batch_trace)
        - day_data: dict {job_id: {1: float, 2: float}} — contrato del scheduler
        - batch_trace: list[dict] | None — trazabilidad cruda (None si sintético)
    """
    if USE_REAL_DATA:
        from data.real_batch_generator import generate_day_surgeries_from_pkl

        day_data, batch_trace = generate_day_surgeries_from_pkl(
            job_ids,
            seed=sim_i,
            batch_trace_extras={"simulation_id": sim_i},
        )
        return day_data, batch_trace
    else:
        from data.data_generator import generate_day_surgeries_data

        day_data = generate_day_surgeries_data(job_ids, std_factor=std_factor)
        return day_data, None


class ElectiveWorker:
    """
    Executes a single elective simulation.
    """

    def __init__(self, job_ids, algorithms, std_factor, wall_clock_start=None, data_seed_override=None):
        self.job_ids = job_ids
        self.algorithms = algorithms
        self.std_factor = std_factor
        # Analysis mode: when set, run() returns a 3-tuple including elapsed wall-clock time.
        # Must be injected as a constructor arg because loky workers are separate processes
        # and globals do NOT propagate reliably across fork/pickle boundaries.
        self.wall_clock_start = wall_clock_start
        # Fixed pool mode: when set, all sims in the same run share the same
        # generated patient dataset (data_seed_override), while algorithm
        # stochasticity (np.random.seed(sim_i)) remains independent.
        self.data_seed_override = data_seed_override

    def run(self, sim_i):
        """
        Runs one simulation iteration.

        Args:
            sim_i (int): Simulation index

        Returns:
            tuple: (sim_i, sim_results) in normal mode, or
                   (sim_i, sim_results, wall_clock_elapsed_s) in analysis mode
                   (when wall_clock_start was set in the constructor).
        """
        np.random.seed(sim_i)
        data_seed = self.data_seed_override if self.data_seed_override is not None else sim_i
        day_data, batch_trace = _build_day_data(self.job_ids, self.std_factor, data_seed)

        # Persist raw trace if real data is enabled.
        # In analysis mode (wall_clock_start set), skip trace writing:
        # data is deterministically reconstructible from the seed, and
        # writing 260+ CSVs to results/csv/ clutters outside the timestamped folder.
        if batch_trace is not None and self.wall_clock_start is None:
            from data.raw_trace_writer import write_batch_trace

            write_batch_trace(batch_trace, TRACE_CSV_PATH, simulation_id=sim_i)

        sim_results = {}

        # Analysis mode: use AnalysisIterationHandler to collect per-iteration snapshots
        analysis_mode = self.wall_clock_start is not None

        for spec in self.algorithms:
            algo_name = spec["name"]
            t0 = time.time()

            try:
                if analysis_mode:
                    from core.iteration_callback import AnalysisIterationHandler
                    from config.config import ANALYSIS_ARTIFACT_SAVE_MODE

                    handler = AnalysisIterationHandler(
                        policy=ANALYSIS_ARTIFACT_SAVE_MODE,
                    )

                    # Wrap the algorithm runner to inject the iteration callback.
                    _original_runner = spec["runner"]

                    def _wrapped_runner(
                        surgeries_data, job_ids, seed, _h=handler, _r=_original_runner
                    ):
                        return _r(surgeries_data, job_ids, seed, on_iteration=_h)

                    effective_runner = _wrapped_runner
                else:
                    effective_runner = spec["runner"]
                    handler = None

                result = run_static_schedule(
                    algorithm_runner=effective_runner,
                    surgeries_data=day_data,
                    job_ids=self.job_ids,
                    seed=sim_i,
                )

                if not isinstance(result, tuple) or len(result) != 4:
                    print(
                        f"WARNING: {algo_name} returned invalid format in sim {sim_i}"
                    )
                    sim_results[algo_name] = {
                        "makespan": float("inf"),
                        "solution": [],
                        "time": time.time() - t0,
                        "best_hist": [],
                        "avg_hist": [],
                        "job_label_map": None,
                    }
                    continue

                schedule_details, makespan, best_hist, avg_hist = result

                elapsed = time.time() - t0

                sim_results[algo_name] = {
                    "makespan": makespan if schedule_details else float("inf"),
                    "solution": schedule_details,
                    "time": elapsed,
                    "best_hist": best_hist if isinstance(best_hist, list) else [],
                    "avg_hist": avg_hist if isinstance(avg_hist, list) else [],
                    # job_label_map: {job_id -> codigo_cie10} for Gantt labeling.
                    # Built from batch_trace if real data is used; None for synthetic.
                    "job_label_map": (
                        {row["job_id"]: row["codigo_cie10"] for row in batch_trace}
                        if batch_trace is not None
                        else None
                    ),
                    # job_grupo_map: {job_id -> grupo} for CIE-10 breakdown telemetry.
                    "job_grupo_map": (
                        {row["job_id"]: row["grupo"] for row in batch_trace}
                        if batch_trace is not None
                        else None
                    ),
                }

                # Analysis mode: attach accumulated iteration snapshots
                if analysis_mode and handler is not None:
                    sim_results[algo_name]["iteration_snapshots"] = list(
                        handler.snapshots
                    )

            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR: {algo_name} failed in sim {sim_i}: {e}")
                import traceback

                traceback.print_exc()

                sim_results[algo_name] = {
                    "makespan": float("inf"),
                    "solution": [],
                    "time": elapsed,
                    "best_hist": [],
                    "avg_hist": [],
                    "job_label_map": None,
                    "batch_trace": batch_trace,
                }

        if self.wall_clock_start is not None:
            import time as _time

            wall_clock_elapsed_s = _time.time() - self.wall_clock_start
            return sim_i, sim_results, batch_trace, wall_clock_elapsed_s

        return sim_i, sim_results
