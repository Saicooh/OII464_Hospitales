"""Synthetic execution bridge that preserves the repository's old reports."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed

from config.config import (
    ALPHA_TEST,
    INSTANCE_PATH,
    N_JOBS,
    NUM_SIMULATIONS,
    get_algorithms,
)
from core.file_manager import FileManager
from core.report_generator import ReportGenerator
from data.instance_loader import load_instance
from data.instance_model import InstanceContext
from simulation.result_model import SolverOutput
from simulation.scheduler import calculate_schedule_fitness
from utils.logger import logger


class LegacyInstanceData(dict[int, dict[int, float]]):
    """Mapping expected by the historical algorithms, backed by one instance."""

    def __init__(self, context: InstanceContext) -> None:
        super().__init__(
            {
                job.job_id: {
                    operation.operation_id: operation.duration
                    for operation in job.operations
                }
                for job in context.jobs
            }
        )
        self.context = context


def _set_instance_rooms(runner: Any, rooms: tuple[str, ...]) -> None:
    """Make old algorithm initializers use the selected instance's rooms."""
    module = __import__(runner.__module__, fromlist=["PABELLONES"])
    if hasattr(module, "PABELLONES"):
        setattr(module, "PABELLONES", list(rooms))


def _invoke_solver(spec: dict[str, Any], data: LegacyInstanceData, seed: int) -> Any:
    runner = spec["runner"]
    _set_instance_rooms(runner, data.context.rooms)
    if (
        spec.get("interface") == "context"
        or runner.__module__ in {"algorithms.dmshoa", "algorithms.mh"}
    ):
        return runner(data.context, seed)
    return runner(data, list(data), seed)


def _normalize_solver_output(raw: Any) -> tuple[float, dict[str, Any], list[float], list[float]]:
    if isinstance(raw, SolverOutput):
        return (
            float(raw.combined_objective),
            raw.solution.to_dict(),
            list(raw.best_fitness_history),
            list(raw.average_fitness_history),
        )
    if not isinstance(raw, tuple) or len(raw) < 4:
        raise TypeError("solver must return SolverOutput or the historical four-tuple")
    solution = raw[1]
    if hasattr(solution, "to_dict"):
        solution = solution.to_dict()
    if not isinstance(solution, dict):
        raise TypeError("solver returned no schedule solution")
    return float(raw[0]), solution, list(raw[2]), list(raw[3])


def _run_one_simulation(
    context: InstanceContext, algorithm_specs: tuple[dict[str, Any], ...], simulation_index: int
) -> tuple[int, dict[str, dict[str, Any]]]:
    data = LegacyInstanceData(context)
    results: dict[str, dict[str, Any]] = {}
    for spec in algorithm_specs:
        name = spec["name"]
        started = time.perf_counter()
        try:
            raw = _invoke_solver(spec, data, simulation_index)
            objective, solution, best_history, average_history = _normalize_solver_output(raw)
            fitness_result = calculate_schedule_fitness(
                solution, data, return_details=True
            )
            if not isinstance(fitness_result, tuple):
                raise TypeError("compatibility scheduler did not return details")
            _, makespan, schedule = fitness_result
            elapsed = time.perf_counter() - started
            results[name] = {
                "makespan": makespan if schedule else float("inf"),
                "solution": schedule,
                "time": elapsed,
                "best_hist": best_history,
                "avg_hist": average_history,
                "objective": objective,
                # Historical synthetic plots identify jobs by their numeric ID.
                # The YAML label is metadata and must not become a display label
                # such as "Twelve-room case 25" in the legacy Gantt chart.
                "job_label_map": None,
            }
        except Exception as error:
            elapsed = time.perf_counter() - started
            logger.error("%s failed in simulation %s: %s", name, simulation_index, error)
            results[name] = {
                "makespan": float("inf"),
                "solution": [],
                "time": elapsed,
                "best_hist": [],
                "avg_hist": [],
                "objective": float("inf"),
                "job_label_map": None,
                "error": str(error),
            }
    return simulation_index, results


def _aggregate_results(
    results: list[tuple[int, dict[str, dict[str, Any]]]],
    algorithm_specs: tuple[dict[str, Any], ...],
) -> tuple[dict[str, dict[str, list[Any]]], dict[str, dict[str, Any]]]:
    all_results: dict[str, dict[str, list[Any]]] = {
        spec["name"]: {
            "makespan": [], "solution": [], "time": [],
            "best_hist": [], "avg_hist": [],
        }
        for spec in algorithm_specs
    }
    best_overall: dict[str, dict[str, Any]] = {
        spec["name"]: {
            "makespan": float("inf"), "schedule": None,
            "sim_num": -1, "job_label_map": None,
        }
        for spec in algorithm_specs
    }
    for simulation_index, simulation_results in sorted(results, key=lambda item: item[0]):
        for name, result in simulation_results.items():
            aggregate = all_results[name]
            aggregate["makespan"].append(result["makespan"])
            aggregate["solution"].append(result["solution"])
            aggregate["time"].append(result["time"])
            aggregate["best_hist"].append(result["best_hist"])
            aggregate["avg_hist"].append(result["avg_hist"])
            if result["makespan"] < best_overall[name]["makespan"]:
                best_overall[name] = {
                    "makespan": result["makespan"],
                    "schedule": result["solution"],
                    "sim_num": simulation_index,
                    "job_label_map": result.get("job_label_map"),
                }
    return all_results, best_overall


class LegacyExperimentRunner:
    """Run all configured algorithms and emit the historical report contract."""

    def __init__(
        self,
        instance_path: str | Path = INSTANCE_PATH,
        num_simulations: int = NUM_SIMULATIONS,
        n_jobs: int = N_JOBS,
        base_dir: str | Path = "results",
    ) -> None:
        self.context = load_instance(instance_path)
        self.num_simulations = int(num_simulations)
        self.n_jobs = int(n_jobs)
        self.base_dir = str(base_dir)
        self.algorithm_specs = tuple(get_algorithms())
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be positive")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")

    def run_elective_mode(self) -> dict[str, str]:
        output_dirs = FileManager(base_dir=self.base_dir).setup_elective_directories()
        # reporting._build_room_schedules uses this module-level compatibility
        # list, while the runner still passes the selected rooms to plotting.
        from utils import reporting

        reporting.ALL_ROOMS = list(self.context.rooms)
        logger.info(
            "Running synthetic instance %s with %d algorithms and %d simulations",
            self.context.instance_id,
            len(self.algorithm_specs),
            self.num_simulations,
        )
        started = time.perf_counter()
        raw_results = Parallel(n_jobs=self.n_jobs)(
            delayed(_run_one_simulation)(
                self.context, self.algorithm_specs, simulation_index
            )
            for simulation_index in range(self.num_simulations)
        )
        all_results, best_overall = _aggregate_results(
            raw_results, self.algorithm_specs
        )
        ReportGenerator().generate_elective_reports(
            all_results,
            best_overall,
            output_dirs,
            list(self.context.rooms),
            ALPHA_TEST,
        )
        logger.info(
            "Historical-compatible reports completed in %.2fs: %s",
            time.perf_counter() - started,
            output_dirs,
        )
        return output_dirs


__all__ = ["LegacyExperimentRunner", "LegacyInstanceData"]
