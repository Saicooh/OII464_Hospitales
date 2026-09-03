# /algorithms/ga.py
"""
Module for the implementation of the Genetic Algorithm (GA) for the
surgery scheduling problem.
"""

import random
import copy
import numpy as np

# Import constants and the fitness function from our centralized modules
from config.config import (
    POPULATION_SIZE_GA,
    MAX_GENERATIONS,
    CROSSOVER_PROBABILITY,
    MUTATION_PROBABILITY,
    ELITISM_COUNT,
    PABELLONES,
    VERBOSE_MODE,
)
from simulation.scheduler import calculate_schedule_fitness

# --- GA-Specific Helper Functions ---


def create_individual(job_ids):
    """Creates a random individual for the GA with GUARANTEED balanced room distribution."""
    individual = {}
    # 1. Random base sequence of jobs
    base_sequence = random.sample(job_ids, len(job_ids))

    # 2. Layer 1 must have TWO IDENTICAL COPIES of the base sequence
    individual["job_sequence_base"] = base_sequence * 2

    # 3. GUARANTEED BALANCED: Ensure every room gets at least one job
    num_jobs = len(job_ids)

    # Create cyclic assignment lists
    pab_cycle1 = (PABELLONES * ((num_jobs // len(PABELLONES)) + 1))[:num_jobs]
    pab_cycle2 = (PABELLONES * ((num_jobs // len(PABELLONES)) + 1))[:num_jobs]

    # Shuffle to add randomness while maintaining balance
    random.shuffle(pab_cycle1)
    random.shuffle(pab_cycle2)

    individual["room_assignment"] = {}
    for idx, job in enumerate(job_ids):
        individual["room_assignment"][job] = {1: pab_cycle1[idx], 2: pab_cycle2[idx]}

    return individual


def selection(population, fitnesses, job_ids):
    """Inverted roulette wheel selection based on fitness (minimization)."""
    valid_indices = [i for i, f in enumerate(fitnesses) if f != float("inf")]

    # If there are no valid individuals, a new random population is generated.
    if not valid_indices:
        return [create_individual(job_ids) for _ in range(len(population))]

    valid_pop = [population[i] for i in valid_indices]
    valid_fit = [fitnesses[i] for i in valid_indices]

    # Invert fitness so that the best (lowest) have a higher probability
    max_fit = max(valid_fit) + 1
    inverted_fitness = [(max_fit - f) for f in valid_fit]
    total_inverted_fitness = sum(inverted_fitness)

    if total_inverted_fitness == 0:
        return random.choices(valid_pop, k=len(population))

    probabilities = [f / total_inverted_fitness for f in inverted_fitness]

    # Choose with replacement from the valid population according to the calculated probabilities.
    chosen_indices = np.random.choice(
        len(valid_pop), size=len(population), replace=True, p=probabilities
    )
    return [valid_pop[i] for i in chosen_indices]


def crossover(parent1, parent2):
    """Performs crossover on sequence (Order Crossover - OX1) and room assignment."""
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    # Sequence crossover (OX1)
    if random.random() < CROSSOVER_PROBABILITY:
        seq1 = parent1["job_sequence_base"]
        seq2 = parent2["job_sequence_base"]
        n = len(seq1)
        if n >= 2:
            p1, p2 = sorted(random.sample(range(n), 2))

            sub1 = seq1[p1 : p2 + 1]
            remaining1 = [item for item in seq2 if item not in sub1]
            child1["job_sequence_base"] = (
                remaining1[-(n - (p2 + 1)) :] + sub1 + remaining1[: -(n - (p2 + 1))]
            )

            sub2 = seq2[p1 : p2 + 1]
            remaining2 = [item for item in seq1 if item not in sub2]
            child2["job_sequence_base"] = (
                remaining2[-(n - (p2 + 1)) :] + sub2 + remaining2[: -(n - (p2 + 1))]
            )

    # Room assignment crossover (single point)
    if random.random() < CROSSOVER_PROBABILITY:
        jobs_list = list(parent1["room_assignment"].keys())
        if len(jobs_list) > 1:
            cut_point = random.randint(1, len(jobs_list) - 1)
            jobs_to_swap = jobs_list[cut_point:]
            for job in jobs_to_swap:
                if (
                    job in parent1["room_assignment"]
                    and job in parent2["room_assignment"]
                ):
                    child1["room_assignment"][job], child2["room_assignment"][job] = (
                        parent2["room_assignment"][job],
                        parent1["room_assignment"][job],
                    )
    return child1, child2


def mutate(individual):
    """Performs mutation on sequence (swap) and room assignment."""
    ind = copy.deepcopy(individual)

    # Sequence mutation (swapping two positions)
    if random.random() < MUTATION_PROBABILITY:
        seq = ind["job_sequence_base"]
        if len(seq) >= 2:
            i1, i2 = random.sample(range(len(seq)), 2)
            seq[i1], seq[i2] = seq[i2], seq[i1]

    # Room assignment mutation (change to a random room)
    mutation_prob_per_room = MUTATION_PROBABILITY / 2
    for job in ind["room_assignment"]:
        for op in [1, 2]:
            if random.random() < mutation_prob_per_room:
                ind["room_assignment"][job][op] = random.choice(PABELLONES)
    return ind


# --- Main Algorithm Execution Function ---


def run(surgeries_data, job_ids=None, seed=None, on_iteration=None):
    """Executes the full Genetic Algorithm cycle.

    Parameters
    ----------
    on_iteration : callable, optional
        Callback invoked at the end of each generation with signature:
        ``on_iteration(algo_step: int, best_fitness: float, combined_obj=None)``.
        Used in analysis mode to collect per-iteration snapshots. Has no effect
        on the algorithm logic; ignored when None.
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
        raise TypeError("GA requires surgeries_data, job_ids, and seed")

    random.seed(seed)
    np.random.seed(seed)

    # Initialize population
    population = [create_individual(job_ids) for _ in range(POPULATION_SIZE_GA)]

    best_objective_overall = float("inf")
    best_solution_overall = None
    best_makespan_overall = float("inf")
    best_fitness_history = []
    avg_fitness_history = []

    print_interval = max(1, MAX_GENERATIONS // 4)

    for generation in range(MAX_GENERATIONS):
        fitnesses = [
            calculate_schedule_fitness(ind, surgeries_data) for ind in population
        ]

        valid_fitnesses = [f for f in fitnesses if f != float("inf")]
        best_fitness_gen = min(valid_fitnesses) if valid_fitnesses else float("inf")
        avg_fitness_gen = np.mean(valid_fitnesses) if valid_fitnesses else float("inf")

        best_so_far = best_fitness_history[-1] if best_fitness_history else float("inf")
        best_fitness_history.append(min(best_fitness_gen, best_so_far))
        avg_fitness_history.append(avg_fitness_gen)

        # Compute real makespan for the best individual of this generation
        if best_fitness_gen != float("inf"):
            best_gen_index = fitnesses.index(best_fitness_gen)
            _, iter_makespan_gen, _ = calculate_schedule_fitness(
                population[best_gen_index], surgeries_data, return_details=True
            )
        else:
            iter_makespan_gen = float("inf")

        if best_fitness_gen < best_objective_overall:
            best_objective_overall = best_fitness_gen
            best_solution_overall = copy.deepcopy(population[best_gen_index])
            best_makespan_overall = iter_makespan_gen

        if VERBOSE_MODE:
            if (
                generation == 0
                or (generation + 1) % print_interval == 0
                or generation == MAX_GENERATIONS - 1
            ):
                print(
                    f"  -> Gen {generation + 1}/{MAX_GENERATIONS}, Best Fitness: {best_objective_overall:.2f} || Makespan (of Best Fitness): {best_makespan_overall:.2f}"
                )

        # --- Create New Generation ---
        sorted_pop_indices = np.argsort(fitnesses)
        elite = [
            copy.deepcopy(population[i])
            for i in sorted_pop_indices[:ELITISM_COUNT]
            if fitnesses[i] != float("inf")
        ]

        selected_population = selection(population, fitnesses, job_ids)

        offspring_list = []
        while len(elite) + len(offspring_list) < POPULATION_SIZE_GA:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(parent1, parent2)
            offspring_list.append(mutate(child1))
            if len(elite) + len(offspring_list) < POPULATION_SIZE_GA:
                offspring_list.append(mutate(child2))

        # Evaluate offspring ONLY → used for iteration_* metrics (isolated from elite)
        fitnesses_offspring = [
            calculate_schedule_fitness(child, surgeries_data)
            for child in offspring_list
        ]
        valid_offspring_fitnesses = [
            f for f in fitnesses_offspring if f != float("inf")
        ]
        if valid_offspring_fitnesses:
            best_offspring_fit = min(valid_offspring_fitnesses)
            best_offspring_idx = fitnesses_offspring.index(best_offspring_fit)
            _, offspring_makespan, _ = calculate_schedule_fitness(
                offspring_list[best_offspring_idx], surgeries_data, return_details=True
            )
        else:
            best_offspring_fit = float("inf")
            offspring_makespan = float("inf")

        if best_offspring_fit < best_objective_overall:
            best_objective_overall = best_offspring_fit
            best_solution_overall = copy.deepcopy(offspring_list[best_offspring_idx])
            best_makespan_overall = offspring_makespan

        population = elite + offspring_list

        # Emit snapshot for analysis mode instrumentation
        # best_* = historical global best; iteration_* = offspring-only best (isolated)
        if on_iteration is not None:
            from core.iteration_callback import serialize_solution

            on_iteration(
                algo_step=generation + 1,
                best_fitness=best_objective_overall,
                best_makespan=best_makespan_overall,
                iteration_fitness=avg_fitness_gen,
                iteration_makespan=iter_makespan_gen if iter_makespan_gen != float("inf") else offspring_makespan,
                best_solution_snapshot=serialize_solution(best_solution_overall),
            )

    return (
        best_objective_overall,
        best_solution_overall,
        best_fitness_history,
        avg_fitness_history,
    )
