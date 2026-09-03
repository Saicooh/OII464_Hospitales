"""Worker for the instance-driven simulation path."""

from __future__ import annotations

import time
from typing import Any

from data.instance_model import InstanceContext
from simulation.result_model import SimulationResult, SolverOutput
from simulation.scheduler import schedule_instance_solution


class InstanceWorker:
    """Run one solver seed against one immutable instance context."""

    def __init__(self, context: InstanceContext, solver: Any) -> None:
        self.context = context
        self.solver = solver

    def run(self, simulation_index: int, solver_seed: int) -> SimulationResult:
        started = time.perf_counter()
        try:
            raw = self.solver(self.context, solver_seed)
            if isinstance(raw, SolverOutput):
                output = raw
            else:
                if not isinstance(raw, tuple) or len(raw) < 4:
                    raise TypeError(
                        "solver must return SolverOutput or "
                        "(objective, solution, best_history, average_history)"
                    )
                output = SolverOutput(
                    combined_objective=float(raw[0]),
                    solution=raw[1],
                    best_fitness_history=tuple(raw[2]),
                    average_fitness_history=tuple(raw[3]),
                    iterations=tuple(raw[4]) if len(raw) > 4 else (),
                )

            makespan, schedule = schedule_instance_solution(
                self.context, output.solution
            )
            elapsed = time.perf_counter() - started
            return SimulationResult.from_context(
                context=self.context,
                simulation_index=simulation_index,
                solver_seed=solver_seed,
                status="completed",
                combined_objective=float(output.combined_objective),
                makespan=makespan,
                schedule=schedule,
                algorithm_seconds=elapsed,
                wall_clock_seconds=time.perf_counter() - started,
                best_fitness_history=output.best_fitness_history,
                average_fitness_history=output.average_fitness_history,
                iterations=output.iterations,
            )
        except Exception as error:
            elapsed = time.perf_counter() - started
            return SimulationResult.from_context(
                context=self.context,
                simulation_index=simulation_index,
                solver_seed=solver_seed,
                status="failed",
                error=str(error),
                algorithm_seconds=elapsed,
                wall_clock_seconds=elapsed,
            )


__all__ = ["InstanceWorker"]
