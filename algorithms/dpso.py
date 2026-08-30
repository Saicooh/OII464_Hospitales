# /algorithms/dpso.py
"""
Module for the implementation of the Discrete Particle Swarm Optimization
Algorithm (dPSO) for the surgery scheduling problem.
"""

import random
import copy
import numpy as np

# Import constants and the fitness function from our centralized modules
from config.config import (
    SWARM_SIZE_DPSO,
    MAX_ITERATIONS_DPSO,
    W_DPSO,
    C1_DPSO,
    C2_DPSO,
    VEL_HIGH_DPSO,
    VEL_LOW_DPSO,
    PABELLONES,
    VERBOSE_MODE,
)
from simulation.scheduler import calculate_schedule_fitness

# --- dPSO-Specific Helper Classes and Functions ---


def sigmoid(x):
    """Sigmoid function to map velocity to probability."""
    clipped_x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-clipped_x))


class DiscreteParticle:
    """Represents a particle in the dPSO discrete space."""

    def __init__(self, job_ids):
        self.job_ids = job_ids
        self.num_jobs = len(job_ids)

        # Discrete position (problem solution)
        self.position_sequence = random.sample(self.job_ids, self.num_jobs)

        # GUARANTEED BALANCED: Ensure every room gets at least one job
        num_jobs = len(self.job_ids)

        # Create cyclic assignment lists
        # Create cyclic assignment lists
        pab_cycle1 = (PABELLONES * ((num_jobs // len(PABELLONES)) + 1))[:num_jobs]
        pab_cycle2 = (PABELLONES * ((num_jobs // len(PABELLONES)) + 1))[:num_jobs]

        # Shuffle to add randomness while maintaining balance
        random.shuffle(pab_cycle1)
        random.shuffle(pab_cycle2)

        self.position_rooms = {}
        for idx, job in enumerate(self.job_ids):
            self.position_rooms[job] = {1: pab_cycle1[idx], 2: pab_cycle2[idx]}

        # Continuous velocity (used to guide the search)
        self.velocity_seq = np.random.uniform(
            VEL_LOW_DPSO, VEL_HIGH_DPSO, self.num_jobs
        )
        self.velocity_rooms_flat = np.random.uniform(
            VEL_LOW_DPSO, VEL_HIGH_DPSO, self.num_jobs * 2
        )

        # Best personal position found (pbest)
        self.pbest_sequence = copy.deepcopy(self.position_sequence)
        self.pbest_rooms = copy.deepcopy(self.position_rooms)
        self.pbest_value = float("inf")

        # Current value (fitness) of the current position
        self.current_value = float("inf")

    def get_discrete_position_representation(self):
        """Returns the current position in the standard format for simulation."""
        return {
            "job_sequence_base": self.position_sequence,
            "room_assignment": self.position_rooms,
        }


# --- Main Algorithm Execution Function ---


def run(surgeries_data, job_ids, seed, on_iteration=None):
    """Executes the full dPSO cycle.

    Parameters
    ----------
    on_iteration : callable, optional
        Callback invoked at the end of each iteration with signature:
        ``on_iteration(algo_step: int, best_fitness: float, combined_obj=None)``.
        Used in analysis mode. Ignored when None.
    """
    random.seed(seed)
    np.random.seed(seed)

    job_id_to_index = {job_id: i for i, job_id in enumerate(job_ids)}
    num_jobs = len(job_ids)

    # Initialize swarm of particles
    swarm = [DiscreteParticle(job_ids) for _ in range(SWARM_SIZE_DPSO)]

    # Initialize global best solution (gbest)
    gbest_sequence = None
    gbest_rooms = None
    gbest_value = float("inf")
    gbest_makespan = float("inf")  # NEW: Track actual makespan

    best_fitness_history = []
    avg_fitness_history = []

    print_interval = max(1, MAX_ITERATIONS_DPSO // 4)
    for iteration in range(MAX_ITERATIONS_DPSO):
        current_fitnesses = []
        iter_best_value = float("inf")
        iter_best_sol = (
            None  # Track solution of iteration best for makespan computation
        )
        # Evaluate each particle and update pbest/gbest
        for particle in swarm:
            solution_representation = particle.get_discrete_position_representation()
            fitness = calculate_schedule_fitness(
                solution_representation, surgeries_data
            )

            particle.current_value = fitness
            current_fitnesses.append(fitness)

            if fitness < particle.pbest_value:
                particle.pbest_value = fitness
                particle.pbest_sequence = copy.deepcopy(particle.position_sequence)
                particle.pbest_rooms = copy.deepcopy(particle.position_rooms)

            # Capture iteration best BEFORE updating gbest
            if fitness < iter_best_value:
                iter_best_value = fitness
                iter_best_sol = (
                    solution_representation  # keep reference, no copy needed
                )

            if fitness < gbest_value:
                gbest_value = fitness
                gbest_sequence = copy.deepcopy(particle.position_sequence)
                gbest_rooms = copy.deepcopy(particle.position_rooms)

                # NEW: Get actual makespan for gbest
                _, gbest_makespan, _ = calculate_schedule_fitness(
                    {
                        "job_sequence_base": gbest_sequence,
                        "room_assignment": gbest_rooms,
                    },
                    surgeries_data,
                    return_details=True,
                )

        # Compute real makespan for best particle of this iteration
        if iter_best_sol is not None and iter_best_value != float("inf"):
            _, iter_makespan, _ = calculate_schedule_fitness(
                iter_best_sol, surgeries_data, return_details=True
            )
        else:
            iter_makespan = float("inf")

        # --- History Tracking ---
        valid_fitnesses = [f for f in current_fitnesses if f != float("inf")]
        best_fitness_iter = (
            gbest_value
            if gbest_value != float("inf")
            else (min(valid_fitnesses) if valid_fitnesses else float("inf"))
        )
        avg_fitness_iter = np.mean(valid_fitnesses) if valid_fitnesses else float("inf")

        best_so_far = best_fitness_history[-1] if best_fitness_history else float("inf")
        best_fitness_history.append(min(best_fitness_iter, best_so_far))
        avg_fitness_history.append(avg_fitness_iter)

        # Only print when VERBOSE_MODE=True
        if VERBOSE_MODE:
            if (
                iteration == 0
                or (iteration + 1) % print_interval == 0
                or iteration == MAX_ITERATIONS_DPSO - 1
            ):
                print(
                    f"  -> Iter {iteration + 1}/{MAX_ITERATIONS_DPSO}, Best Fitness: {gbest_value:.2f} || Makespan (of Best Fitness): {gbest_makespan:.2f}"
                )

        if gbest_sequence is None:
            # Still emit iteration snapshot even when no valid solution yet
            if on_iteration is not None:
                on_iteration(
                    algo_step=iteration + 1,
                    best_fitness=gbest_value,
                    best_makespan=gbest_value,
                    iteration_fitness=avg_fitness_iter,
                    iteration_makespan=iter_makespan,
                    best_solution_snapshot=None,
                )
            continue

        # Emit snapshot for analysis mode instrumentation
        if on_iteration is not None:
            from core.iteration_callback import serialize_solution

            _gbest_sol = {
                "job_sequence_base": list(gbest_sequence),
                "room_assignment": {
                    job_id: {1: gbest_rooms[job_id][1], 2: gbest_rooms[job_id][2]}
                    for job_id in job_ids
                    if job_id in gbest_rooms
                },
            }
            on_iteration(
                algo_step=iteration + 1,
                best_fitness=gbest_value,
                best_makespan=gbest_makespan,
                iteration_fitness=avg_fitness_iter,
                iteration_makespan=iter_makespan,
                best_solution_snapshot=serialize_solution(_gbest_sol),
            )

        # --- Update velocity and position of each particle ---
        for particle in swarm:
            r1_seq, r2_seq = np.random.rand(num_jobs), np.random.rand(num_jobs)
            r1_room, r2_room = (
                np.random.rand(num_jobs * 2),
                np.random.rand(num_jobs * 2),
            )

            # --- Sequence Velocity ---
            def get_rank_vector(sequence, job_ids_map):
                rank_vec = np.zeros(len(job_ids_map))
                job_to_rank = {job_id: rank for rank, job_id in enumerate(sequence)}
                for job_id, index in job_ids_map.items():
                    rank_vec[index] = job_to_rank.get(job_id, len(job_ids_map))
                return rank_vec

            current_rank = get_rank_vector(particle.position_sequence, job_id_to_index)
            pbest_rank = get_rank_vector(particle.pbest_sequence, job_id_to_index)
            gbest_rank = get_rank_vector(gbest_sequence, job_id_to_index)

            particle.velocity_seq = (
                W_DPSO * particle.velocity_seq
                + C1_DPSO * r1_seq * (pbest_rank - current_rank)
                + C2_DPSO * r2_seq * (gbest_rank - current_rank)
            )

            # --- Room Assignment Velocity ---
            def get_room_vector(room_dict, job_ids_list):
                vec = np.zeros(len(job_ids_list) * 2)
                room_lists = [PABELLONES, PABELLONES]
                room_indices = {
                    room: idx / (len(lst) - 1) if len(lst) > 1 else 0.5
                    for op_idx, lst in enumerate(room_lists)
                    for idx, room in enumerate(lst)
                }
                current_idx = 0
                for job_id in job_ids_list:
                    job_rooms = room_dict.get(job_id, {})
                    for op in range(1, 3):
                        room_name = job_rooms.get(op)
                        vec[current_idx] = room_indices.get(room_name, 0.5)
                        current_idx += 1
                return vec

            current_rooms_vec = get_room_vector(particle.position_rooms, job_ids)
            pbest_rooms_vec = get_room_vector(particle.pbest_rooms, job_ids)
            gbest_rooms_vec = get_room_vector(gbest_rooms, job_ids)

            particle.velocity_rooms_flat = (
                W_DPSO * particle.velocity_rooms_flat
                + C1_DPSO * r1_room * (pbest_rooms_vec - current_rooms_vec)
                + C2_DPSO * r2_room * (gbest_rooms_vec - current_rooms_vec)
            )

            # Limit velocity
            particle.velocity_seq = np.clip(
                particle.velocity_seq, VEL_LOW_DPSO, VEL_HIGH_DPSO
            )
            particle.velocity_rooms_flat = np.clip(
                particle.velocity_rooms_flat, VEL_LOW_DPSO, VEL_HIGH_DPSO
            )

            # --- Discrete Position Update ---
            # 1. Sequence
            prob_swap = sigmoid(particle.velocity_seq)
            indices = list(range(num_jobs))
            random.shuffle(indices)
            for k in indices:
                if random.random() < prob_swap[k]:
                    if num_jobs >= 2:
                        l = random.choice([idx for idx in range(num_jobs) if idx != k])
                        particle.position_sequence[k], particle.position_sequence[l] = (
                            particle.position_sequence[l],
                            particle.position_sequence[k],
                        )

            # 2. Room Assignment Update
            prob_room_change = sigmoid(particle.velocity_rooms_flat)
            room_idx = 0
            for job_id in job_ids:
                if job_id not in particle.position_rooms:
                    continue
                for op in range(1, 3):
                    if op not in particle.position_rooms[job_id]:
                        continue
                    if random.random() < prob_room_change[room_idx]:
                        # First try to copy from pbest/gbest
                        copy_from_pbest = (
                            job_id in particle.pbest_rooms
                            and op in particle.pbest_rooms[job_id]
                        )
                        copy_from_gbest = (
                            gbest_rooms is not None
                            and job_id in gbest_rooms
                            and op in gbest_rooms[job_id]
                        )

                        use_pbest_prob = (
                            C1_DPSO / (C1_DPSO + C2_DPSO)
                            if (C1_DPSO + C2_DPSO) > 0
                            else 0.5
                        )
                        source_rooms = None

                        if random.random() < use_pbest_prob:
                            if copy_from_pbest:
                                source_rooms = particle.pbest_rooms
                            elif copy_from_gbest:
                                source_rooms = gbest_rooms
                        else:
                            if copy_from_gbest:
                                source_rooms = gbest_rooms
                            elif copy_from_pbest:
                                source_rooms = particle.pbest_rooms

                        if source_rooms:
                            new_room = source_rooms[job_id][op]
                        else:
                            # Random selection from available rooms
                            room_lists = [PABELLONES, PABELLONES]
                            new_room = random.choice(room_lists[op - 1])

                        particle.position_rooms[job_id][op] = new_room

                    room_idx += 1
                    if room_idx >= len(prob_room_change):
                        break
                if room_idx >= len(prob_room_change):
                    break

    # Build a dictionary with the best solution found
    best_solution_dict = None
    if gbest_sequence is not None and gbest_rooms is not None:
        best_solution_dict = {
            "job_sequence_base": gbest_sequence,
            "room_assignment": gbest_rooms,
        }

    return gbest_value, best_solution_dict, best_fitness_history, avg_fitness_history
