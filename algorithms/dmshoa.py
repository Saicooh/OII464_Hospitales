"""Seeded dMShOA solver for the validated synthetic-instance contract."""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np

from data.instance_model import InstanceContext


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, -500, 500)
    return 1 / (1 + np.exp(-clipped))


def _instance_solution(context: InstanceContext, rng: random.Random) -> dict[str, Any]:
    job_ids = [job.job_id for job in context.jobs]
    return {
        "job_sequence_base": rng.sample(job_ids, len(job_ids)),
        "room_assignment": {
            job.job_id: {
                operation.operation_id: rng.choice(operation.eligible_rooms)
                for operation in job.operations
            }
            for job in context.jobs
        },
    }


def _instance_fitness(context: InstanceContext, solution: dict[str, Any]) -> float:
    from simulation.scheduler import schedule_instance_solution

    try:
        makespan, _ = schedule_instance_solution(context, solution)
    except (TypeError, ValueError, KeyError):
        return float("inf")
    return float(makespan)


def run(context: InstanceContext, seed: int, on_iteration=None):
    """Optimize one immutable instance using the supplied solver seed."""
    from config.config import (
        MAX_ITERATIONS_MSHOA,
        MSHOA_K,
        MSHOA_LOWER_BOUND,
        MSHOA_POP_SIZE,
        MSHOA_UPPER_BOUND,
        VERBOSE_MODE,
    )
    from simulation.result_model import ScheduleSolution, SolverOutput

    if not isinstance(context, InstanceContext):
        raise TypeError("context must be an InstanceContext")
    py_rng = random.Random(int(seed))
    np_rng = np.random.default_rng(int(seed))
    job_ids = [job.job_id for job in context.jobs]
    dim_sequence = len(job_ids)
    dim_total = dim_sequence + 2 * dim_sequence
    population_size = max(1, int(MSHOA_POP_SIZE))
    iterations = max(1, int(MAX_ITERATIONS_MSHOA))

    positions = np_rng.uniform(
        MSHOA_LOWER_BOUND, MSHOA_UPPER_BOUND, (population_size, dim_total)
    )
    solutions = [_instance_solution(context, py_rng) for _ in range(population_size)]
    fitness = np.array([_instance_fitness(context, item) for item in solutions])

    best_index = int(np.argmin(fitness))
    best_value = float(fitness[best_index])
    best_position = positions[best_index].copy()
    best_solution = copy.deepcopy(solutions[best_index])
    pti = np_rng.integers(1, 4, population_size)
    best_history: list[float] = []
    average_history: list[float] = []

    for step in range(iterations):
        for index in range(population_size):
            current = positions[index]
            if pti[index] == 1 and population_size > 1:
                peers = [peer for peer in range(population_size) if peer != index]
                peer_position = positions[py_rng.choice(peers)]
                candidate_position = best_position - (current - best_position)
                candidate_position += np_rng.uniform(-1.0, 1.0) * (peer_position - current)
            elif pti[index] == 2:
                theta = np_rng.uniform(np.pi, 2 * np.pi)
                candidate_position = best_position * np.cos(theta)
            else:
                scale = np_rng.uniform(0.0, MSHOA_K)
                direction = 1.0 if py_rng.random() > 0.5 else -1.0
                candidate_position = best_position + best_position * scale * direction
            candidate_position = np.clip(
                candidate_position, MSHOA_LOWER_BOUND, MSHOA_UPPER_BOUND
            )

            candidate = copy.deepcopy(solutions[index])
            probabilities = np.asarray(_sigmoid(candidate_position - current))
            for position_index in range(dim_sequence):
                if py_rng.random() < probabilities[position_index]:
                    swap_index = py_rng.randrange(dim_sequence)
                    sequence = candidate["job_sequence_base"]
                    sequence[position_index], sequence[swap_index] = (
                        sequence[swap_index],
                        sequence[position_index],
                    )
            room_index = dim_sequence
            for job in context.jobs:
                for operation in job.operations:
                    if py_rng.random() < probabilities[room_index]:
                        candidate["room_assignment"][job.job_id][operation.operation_id] = (
                            py_rng.choice(operation.eligible_rooms)
                        )
                    room_index += 1

            candidate_fitness = _instance_fitness(context, candidate)
            if candidate_fitness < fitness[index]:
                positions[index] = candidate_position
                solutions[index] = candidate
                fitness[index] = candidate_fitness

        current_best = int(np.argmin(fitness))
        if fitness[current_best] < best_value:
            best_value = float(fitness[current_best])
            best_position = positions[current_best].copy()
            best_solution = copy.deepcopy(solutions[current_best])
        pti = np_rng.integers(1, 4, population_size)

        finite = fitness[np.isfinite(fitness)]
        best_history.append(best_value)
        average_history.append(float(np.mean(finite)) if len(finite) else float("inf"))
        if on_iteration is not None:
            from core.iteration_callback import serialize_solution
            from simulation.scheduler import schedule_instance_solution

            best_makespan, _ = schedule_instance_solution(context, best_solution)
            on_iteration(
                algo_step=step + 1,
                best_fitness=best_value,
                best_makespan=best_makespan,
                iteration_fitness=average_history[-1],
                iteration_makespan=best_makespan,
                best_solution_snapshot=serialize_solution(best_solution),
            )
        if VERBOSE_MODE:
            print(f"  -> Iter {step + 1}/{iterations}, Best Fitness: {best_value:.2f}")

    return SolverOutput(
        combined_objective=best_value,
        solution=ScheduleSolution.from_dict(best_solution),
        best_fitness_history=tuple(best_history),
        average_fitness_history=tuple(average_history),
    )
