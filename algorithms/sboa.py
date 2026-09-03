# /algorithms/sboa.py
"""
Module for the implementation of the Secretary Bird Optimization Algorithm (SBOA)
for the surgery scheduling problem.
"""

import random
import math
import numpy as np

# Import constants and the fitness function from our centralized modules
from config.config import (
    SBOA_POP_SIZE,
    SBOA_MAX_ITER,
    SBOA_LOWER_BOUND,
    SBOA_UPPER_BOUND,
    PABELLONES,
    VERBOSE_MODE,
)
from simulation.scheduler import calculate_schedule_fitness

# --- SBOA-Specific Helper Functions ---


def levy_flight(dim, beta=1.5):
    """
    Generates a Levy flight step vector.
    """
    s = 0.01
    num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    step = s * (u / (np.abs(v) ** (1 / beta)))
    return step


def _balance_room_assignment(room_assignment, job_ids):
    """
    Applies a balancing heuristic to room assignments to avoid
    excessive concentration in a few rooms.
    """
    # Check usage count per room
    room_usage = {room: 0 for room in PABELLONES}

    for job in job_ids:
        for op in [1, 2]:
            room = room_assignment[job][op]
            room_usage[room] += 1

    # For each operation type, check if there's imbalance
    room_lists = [PABELLONES, PABELLONES]
    for op_idx, room_list in enumerate(room_lists):
        op_num = op_idx + 1

        # Calculate usage for this operation type
        op_usage = {room: room_usage[room] for room in room_list}

        # If one room has >50% of jobs and others have 0, redistribute
        total_usage = sum(op_usage.values())
        if total_usage > 0:
            max_usage_room = max(op_usage, key=op_usage.get)
            max_usage = op_usage[max_usage_room]

            # If concentration > 70% and there are empty rooms
            if max_usage > total_usage * 0.7 and any(u == 0 for u in op_usage.values()):
                # Redistribute some jobs to empty rooms
                jobs_to_redistribute = []
                for job in job_ids:
                    if room_assignment[job][op_num] == max_usage_room:
                        jobs_to_redistribute.append(job)

                # Shuffle and reassign some to empty rooms
                empty_rooms = [r for r in room_list if op_usage[r] == 0]
                random.shuffle(jobs_to_redistribute)

                for i, job in enumerate(jobs_to_redistribute[: len(empty_rooms)]):
                    room_assignment[job][op_num] = empty_rooms[i]

    return room_assignment


def discretize_bird(position_vector, job_ids, apply_balancing=True):
    """
    Decodes a continuous SBOA position vector into a discrete solution
    representation (sequence and room assignment).

    Args:
        position_vector: Continuous position vector
        job_ids: List of job IDs
        apply_balancing: If True, applies balancing heuristic to room assignments
    """
    num_jobs = len(job_ids)

    # Part 1: Decode the job sequence using Rank-Order-Value
    seq_part = position_vector[:num_jobs]
    job_sequence_base = [job_ids[i] for i in np.argsort(seq_part)]

    # Part 2: Decode the room assignment
    room_part = position_vector[num_jobs:]
    room_assignment = {}
    room_lists = [PABELLONES, PABELLONES]
    idx = 0
    for job in job_ids:
        room_assignment[job] = {}
        for op in [1, 2]:
            room_list = room_lists[op - 1]
            val = room_part[idx]
            norm_val = (val - SBOA_LOWER_BOUND) / (SBOA_UPPER_BOUND - SBOA_LOWER_BOUND)
            room_index = min(int(norm_val * len(room_list)), len(room_list) - 1)
            room_assignment[job][op] = room_list[room_index]
            idx += 1

    # Apply balancing heuristic to avoid concentration
    if apply_balancing:
        room_assignment = _balance_room_assignment(room_assignment, job_ids)

    return {"job_sequence_base": job_sequence_base, "room_assignment": room_assignment}


# --- Main Algorithm Execution Function ---


def run(surgeries_data, job_ids=None, seed=None, on_iteration=None):
    """Executes the full SBOA cycle.

    Parameters
    ----------
    on_iteration : callable, optional
        Callback invoked at the end of each iteration with signature:
        ``on_iteration(algo_step: int, best_fitness: float, combined_obj=None)``.
        Used in analysis mode. Ignored when None.
    """
    from data.instance_model import InstanceContext

    if isinstance(surgeries_data, InstanceContext):
        from core.legacy_runner import LegacyInstanceData

        globals()["PABELLONES"] = list(surgeries_data.rooms)
        context_seed = int(job_ids) if seed is None else int(seed)
        return run(
            LegacyInstanceData(surgeries_data),
            [job.job_id for job in surgeries_data.jobs],
            context_seed,
            on_iteration=on_iteration,
        )
    if job_ids is None or seed is None:
        raise TypeError("SBOA requires surgeries_data, job_ids, and seed")

    random.seed(seed)
    np.random.seed(seed)

    num_jobs = len(job_ids)
    dim_total = num_jobs + (num_jobs * 2)

    # Initialization
    positions = np.random.uniform(
        SBOA_LOWER_BOUND, SBOA_UPPER_BOUND, (SBOA_POP_SIZE, dim_total)
    )
    fitness = np.full(SBOA_POP_SIZE, float("inf"))

    for i in range(SBOA_POP_SIZE):
        discrete_sol = discretize_bird(positions[i], job_ids)
        fitness[i] = calculate_schedule_fitness(discrete_sol, surgeries_data)

    best_idx = np.argmin(fitness)
    gbest_value = fitness[best_idx]
    gbest_position = positions[best_idx, :].copy()

    # NEW: Calculate initial makespan
    gbest_solution = discretize_bird(gbest_position, job_ids)
    _, gbest_makespan, _ = calculate_schedule_fitness(
        gbest_solution, surgeries_data, return_details=True
    )

    best_fitness_history = []
    avg_fitness_history = []

    # Calculate the interval to print 4 times during the process
    print_interval = max(1, SBOA_MAX_ITER // 4)

    for t in range(SBOA_MAX_ITER):
        new_evals = []  # Collect (fitness, position) for all newly evaluated candidates
        for i in range(SBOA_POP_SIZE):
            current_pos = positions[i, :].copy()
            current_fit = fitness[i]

            # --- Exploration Phase (Hunting) ---
            if t < SBOA_MAX_ITER / 3:
                r_indices = np.random.choice(
                    np.delete(np.arange(SBOA_POP_SIZE), i), 2, replace=False
                )
                r1, r2 = positions[r_indices[0], :], positions[r_indices[1], :]
                new_pos_p1 = current_pos + (r1 - r2) * np.random.rand(dim_total)
            elif t < 2 * SBOA_MAX_ITER / 3:
                term = (
                    np.exp((t / SBOA_MAX_ITER) ** 4)
                    * (np.random.randn(dim_total) - 0.5)
                    * (gbest_position - current_pos)
                )
                new_pos_p1 = gbest_position + term
            else:
                perturbation = (1 - t / SBOA_MAX_ITER) ** (2 * t / SBOA_MAX_ITER)
                new_pos_p1 = gbest_position + perturbation * current_pos * (
                    0.05 * levy_flight(dim_total)
                )

            new_pos_p1 = np.clip(new_pos_p1, SBOA_LOWER_BOUND, SBOA_UPPER_BOUND)
            new_fit_p1 = calculate_schedule_fitness(
                discretize_bird(new_pos_p1, job_ids), surgeries_data
            )
            new_evals.append((new_fit_p1, new_pos_p1))

            if new_fit_p1 < current_fit:
                current_fit = new_fit_p1
                current_pos = new_pos_p1

            # --- Exploitation Phase (Escape) ---
            if np.random.rand() < 0.5:
                perturbation_factor = (1 - t / SBOA_MAX_ITER) ** 2
                term = (
                    (2 * np.random.randn(dim_total) - 1)
                    * perturbation_factor
                    * current_pos
                )
                new_pos_p2 = gbest_position + term
            else:
                rand_idx = np.random.randint(0, SBOA_POP_SIZE)
                x_random = positions[rand_idx, :]
                K = round(1 + np.random.rand())
                new_pos_p2 = gbest_position + np.random.randn(dim_total) * (
                    x_random - K * current_pos
                )

            new_pos_p2 = np.clip(new_pos_p2, SBOA_LOWER_BOUND, SBOA_UPPER_BOUND)
            new_fit_p2 = calculate_schedule_fitness(
                discretize_bird(new_pos_p2, job_ids), surgeries_data
            )
            new_evals.append((new_fit_p2, new_pos_p2))

            if new_fit_p2 < current_fit:
                current_fit = new_fit_p2
                current_pos = new_pos_p2

            fitness[i] = current_fit
            positions[i, :] = current_pos

        # Update global best and save history
        # Compute iteration_* from new evaluations ONLY (isolated from survivors)
        valid_new_evals = [(f, p) for f, p in new_evals if f != float("inf")]
        if valid_new_evals:
            best_new_fit, best_new_pos = min(valid_new_evals, key=lambda x: x[0])
            iter_best_sol = discretize_bird(best_new_pos, job_ids)
            _, iter_makespan, _ = calculate_schedule_fitness(
                iter_best_sol, surgeries_data, return_details=True
            )
        else:
            iter_makespan = float("inf")

        # Update gbest from population (after greedy replacement)
        pop_best_idx = np.argmin(fitness)
        pop_best_value = fitness[pop_best_idx]
        if pop_best_value < gbest_value:
            gbest_value = pop_best_value
            gbest_position = positions[pop_best_idx, :].copy()
            gbest_solution_tmp = discretize_bird(gbest_position, job_ids)
            _, gbest_makespan, _ = calculate_schedule_fitness(
                gbest_solution_tmp, surgeries_data, return_details=True
            )

        valid_fitnesses = [f for f in fitness if f != float("inf")]
        avg_fitness_iter = np.mean(valid_fitnesses) if valid_fitnesses else float("inf")

        best_so_far = best_fitness_history[-1] if best_fitness_history else float("inf")
        best_fitness_history.append(min(gbest_value, best_so_far))
        avg_fitness_history.append(avg_fitness_iter)

        # Print at iteration 1 and then at intervals
        if VERBOSE_MODE:
            if t == 0 or (t + 1) % print_interval == 0 or t == SBOA_MAX_ITER - 1:
                print(
                    f"  -> Iter {t + 1}/{SBOA_MAX_ITER}, Best Fitness: {gbest_value:.2f} || Makespan (of Best Fitness): {gbest_makespan:.2f}"
                )

        # Emit snapshot for analysis mode instrumentation
        # best_* = gbest (historical); iteration_* = this iteration's best particle
        if on_iteration is not None:
            from core.iteration_callback import serialize_solution

            _gbest_sol = discretize_bird(gbest_position, job_ids)
            on_iteration(
                algo_step=t + 1,
                best_fitness=gbest_value,
                best_makespan=gbest_makespan,
                iteration_fitness=avg_fitness_iter,
                iteration_makespan=iter_makespan,
                best_solution_snapshot=serialize_solution(_gbest_sol),
            )

    best_solution_overall = discretize_bird(gbest_position, job_ids)

    return gbest_value, best_solution_overall, best_fitness_history, avg_fitness_history
