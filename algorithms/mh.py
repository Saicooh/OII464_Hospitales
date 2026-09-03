"""Plantilla ejecutable para la metaheurística de los alumnos.

La plantilla usa búsqueda aleatoria con reinicios para que el repositorio
pueda ejecutarse antes de que cada grupo implemente su propia metaheurística.
El trabajo del grupo consiste en reemplazar la lógica de ``run`` y conservar
el contrato de entrada y salida indicado en el README.
"""

from __future__ import annotations

import random
from statistics import fmean
from typing import Any, Callable

import numpy as np

from data.instance_model import InstanceContext
from simulation.result_model import ScheduleSolution, SolverOutput
from simulation.scheduler import schedule_instance_solution


IterationCallback = Callable[..., None]


def _random_solution(
    context: InstanceContext, rng: random.Random
) -> dict[str, Any]:
    """Create a valid-looking starter candidate from the selected instance.

    When the operations of a job share a room, both operations start in that
    room. This gives students a simple, reproducible baseline and avoids
    making the template depend on global room names or personnel counts.
    """
    job_ids = [job.job_id for job in context.jobs]
    room_assignment: dict[int, dict[int, str]] = {}

    for job in context.jobs:
        eligible_rooms = [set(operation.eligible_rooms) for operation in job.operations]
        common_rooms = set.intersection(*eligible_rooms) if eligible_rooms else set()
        if common_rooms:
            room = rng.choice(sorted(common_rooms))
            room_assignment[job.job_id] = {
                operation.operation_id: room for operation in job.operations
            }
        else:
            room_assignment[job.job_id] = {
                operation.operation_id: rng.choice(operation.eligible_rooms)
                for operation in job.operations
            }

    return {
        "job_sequence_base": rng.sample(job_ids, len(job_ids)),
        "room_assignment": room_assignment,
    }


def _fitness(context: InstanceContext, solution: dict[str, Any]) -> float:
    """Return the makespan or infinity when the candidate is infeasible."""
    try:
        makespan, _ = schedule_instance_solution(context, solution)
    except (IndexError, KeyError, TypeError, ValueError):
        return float("inf")
    return float(makespan)


def run(
    context: InstanceContext,
    seed: int,
    on_iteration: IterationCallback | None = None,
) -> SolverOutput:
    """Run the student slot using the common typed solver contract.

    Keep this signature when replacing the starter algorithm. The scheduler
    resolves personnel when no explicit personnel assignment is provided.
    """
    from config.config import MH_MAX_ITERATIONS, MH_POP_SIZE, VERBOSE_MODE
    from core.iteration_callback import serialize_solution

    if not isinstance(context, InstanceContext):
        raise TypeError("context must be an InstanceContext")

    rng = random.Random(int(seed))
    population_size = max(1, int(MH_POP_SIZE))
    iterations = max(1, int(MH_MAX_ITERATIONS))
    best_solution: dict[str, Any] | None = None
    best_value = float("inf")
    best_history: list[float] = []
    average_history: list[float] = []

    # TODO: replace this random-restart loop with the group's metaheuristic.
    for step in range(iterations):
        iteration_values: list[float] = []
        for _ in range(population_size):
            candidate = _random_solution(context, rng)
            value = _fitness(context, candidate)
            iteration_values.append(value)
            if value < best_value:
                best_value = value
                best_solution = candidate

        finite_values = [value for value in iteration_values if np.isfinite(value)]
        best_history.append(best_value)
        average_history.append(fmean(finite_values) if finite_values else float("inf"))

        if on_iteration is not None and best_solution is not None:
            on_iteration(
                algo_step=step + 1,
                best_fitness=best_value,
                best_makespan=best_value,
                iteration_fitness=average_history[-1],
                iteration_makespan=best_value,
                best_solution_snapshot=serialize_solution(best_solution),
            )

        if VERBOSE_MODE:
            print(f"  -> Iter {step + 1}/{iterations}, Best Fitness: {best_value:.2f}")

    if best_solution is None:
        raise RuntimeError("MH did not produce a feasible schedule solution")

    return SolverOutput(
        combined_objective=best_value,
        solution=ScheduleSolution.from_dict(best_solution),
        best_fitness_history=tuple(best_history),
        average_fitness_history=tuple(average_history),
    )


__all__ = ["run"]
