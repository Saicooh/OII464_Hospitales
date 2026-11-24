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
from algorithms._eligibility import cyclic_room_assignment, eligible_room_table
from simulation.scheduler import calculate_schedule_fitness

# --- GA-Specific Helper Functions ---


def create_individual(job_ids, eligible=None):
    """Creates a random individual for the GA with a balanced room distribution.

    The sequence is a single permutation of the jobs. The historical GA stored
    two copies of it, but the evaluator only reads the last occurrence of each
    job, so half of the chromosome was inert and order crossover produced
    sequences of variable length.
    """
    individual = {}
    individual["job_sequence_base"] = random.sample(job_ids, len(job_ids))
    table = eligible if eligible is not None else {}
    individual["room_assignment"] = cyclic_room_assignment(job_ids, PABELLONES, table)
    return individual


def selection(population, fitnesses, job_ids, eligible=None):
    """Inverted roulette wheel selection based on fitness (minimization)."""
    valid_indices = [i for i, f in enumerate(fitnesses) if f != float("inf")]

    # If there are no valid individuals, a new random population is generated.
    if not valid_indices:
        return [create_individual(job_ids, eligible) for _ in range(len(population))]

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


def order_crossover(donor, other, p1, p2):
    """Standard OX1: keep donor[p1..p2] in place and fill the remaining
    positions, starting after p2 and wrapping around, with the jobs of
    ``other`` in the order they appear after p2."""
    n = len(donor)
    child = [None] * n
    child[p1 : p2 + 1] = donor[p1 : p2 + 1]
    kept = set(donor[p1 : p2 + 1])
    fill = [other[(p2 + 1 + k) % n] for k in range(n)]
    fill = [job for job in fill if job not in kept]
    positions = [(p2 + 1 + k) % n for k in range(n - (p2 - p1 + 1))]
    for position, job in zip(positions, fill):
        child[position] = job
    return child


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
            child1["job_sequence_base"] = order_crossover(seq1, seq2, p1, p2)
            child2["job_sequence_base"] = order_crossover(seq2, seq1, p1, p2)

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


def mutate(individual, eligible=None):
    """Performs mutation on sequence (swap) and room assignment."""
    ind = copy.deepcopy(individual)

    # Sequence mutation (swapping two positions)
    if random.random() < MUTATION_PROBABILITY:
        seq = ind["job_sequence_base"]
        if len(seq) >= 2:
            i1, i2 = random.sample(range(len(seq)), 2)
            seq[i1], seq[i2] = seq[i2], seq[i1]

    # Room assignment mutation (change to a random eligible room)
    mutation_prob_per_room = MUTATION_PROBABILITY / 2
    for job in ind["room_assignment"]:
        for op in [1, 2]:
            if random.random() < mutation_prob_per_room:
                options = (eligible or {}).get((job, op), PABELLONES)
                ind["room_assignment"][job][op] = random.choice(options)
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

    eligible = eligible_room_table(surgeries_data, PABELLONES)

    import config.config as config

    # Every algorithm stops after the same number of evaluator calls when
    # ``evaluation_budget`` is set; otherwise after MAX_GENERATIONS.
    budget = config.EVALUATION_BUDGET
    evaluations = 0

    def evaluate(individual):
        nonlocal evaluations
        evaluations += 1
        return calculate_schedule_fitness(individual, surgeries_data)

    def makespan_of(individual):
        _, makespan, _ = calculate_schedule_fitness(
            individual, surgeries_data, return_details=True
        )
        return makespan

    # Initialize population. Each individual is evaluated once and its fitness
    # is kept: the historical loop re-evaluated the whole population every
    # generation, which spent evaluations without new information.
    size = POPULATION_SIZE_GA if budget is None else max(2, min(POPULATION_SIZE_GA, budget))
    population = [create_individual(job_ids, eligible) for _ in range(size)]
    fitnesses = [evaluate(ind) for ind in population]

    best_objective_overall = float("inf")
    best_solution_overall = None
    best_makespan_overall = float("inf")
    best_fitness_history = []
    avg_fitness_history = []

    def track_best(candidates, values):
        nonlocal best_objective_overall, best_solution_overall, best_makespan_overall
        finite = [(value, index) for index, value in enumerate(values) if value != float("inf")]
        if not finite:
            return float("inf"), float("inf")
        value, index = min(finite)
        makespan = makespan_of(candidates[index])
        if value < best_objective_overall:
            best_objective_overall = value
            best_solution_overall = copy.deepcopy(candidates[index])
            best_makespan_overall = makespan
        return value, makespan

    track_best(population, fitnesses)
    generation = -1
    while (budget is None and generation + 1 < MAX_GENERATIONS) or (
        budget is not None and evaluations < budget
    ):
        generation += 1
        valid_fitnesses = [f for f in fitnesses if f != float("inf")]
        avg_fitness_gen = np.mean(valid_fitnesses) if valid_fitnesses else float("inf")
        best_fitness_history.append(best_objective_overall)
        avg_fitness_history.append(avg_fitness_gen)

        if VERBOSE_MODE:
            print(
                f"  -> Gen {generation + 1}, Best Fitness: {best_objective_overall:.2f} "
                f"|| Makespan (of Best Fitness): {best_makespan_overall:.2f}"
            )

        # --- Create New Generation ---
        sorted_pop_indices = np.argsort(fitnesses)
        elite_indices = [
            i for i in sorted_pop_indices[:ELITISM_COUNT] if fitnesses[i] != float("inf")
        ]
        elite = [copy.deepcopy(population[i]) for i in elite_indices]
        elite_fitnesses = [fitnesses[i] for i in elite_indices]

        selected_population = selection(population, fitnesses, job_ids, eligible)

        n_offspring = len(population) - len(elite)
        if budget is not None:
            n_offspring = min(n_offspring, budget - evaluations)
        offspring_list = []
        while len(offspring_list) < n_offspring:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(parent1, parent2)
            offspring_list.append(mutate(child1, eligible))
            if len(offspring_list) < n_offspring:
                offspring_list.append(mutate(child2, eligible))

        fitnesses_offspring = [evaluate(child) for child in offspring_list]
        _, offspring_makespan = track_best(offspring_list, fitnesses_offspring)

        population = elite + offspring_list
        fitnesses = elite_fitnesses + fitnesses_offspring
        if len(population) < 2:
            break

        # Emit snapshot for analysis mode instrumentation
        # best_* = historical global best; iteration_* = offspring-only best (isolated)
        if on_iteration is not None:
            from core.iteration_callback import serialize_solution

            on_iteration(
                algo_step=generation + 1,
                best_fitness=best_objective_overall,
                best_makespan=best_makespan_overall,
                iteration_fitness=avg_fitness_gen,
                iteration_makespan=offspring_makespan,
                best_solution_snapshot=serialize_solution(best_solution_overall),
            )

    return (
        best_objective_overall,
        best_solution_overall,
        best_fitness_history,
        avg_fitness_history,
    )
