# /algorithms/dmshoa_old.py
"""
dMShOA Old — variante legacy anterior al rewrite paper-faithful.

Implementación original del dMShOA para el problema de scheduling quirúrgico.
Utiliza discretización vía sigmoid + swaps probabilísticos, y distribución
balanceada garantizada de salas en la inicialización.

Esta variante se preserva como "dmshoa_old" para referencia y benchmarking
comparativo contra la variante adaptada (dmshoa_adaptado).

NOTA: No usar como implementación principal. Ver algorithms/dmshoa_adaptado.py
para la versión paper-faithful activa.
"""
import random
import copy
import numpy as np

# Import constants and the fitness function from our centralized modules
from config.config import (
    MSHOA_POP_SIZE, MAX_ITERATIONS_MSHOA, MSHOA_LOWER_BOUND, MSHOA_UPPER_BOUND,
    MSHOA_K, PABELLONES,
    VERBOSE_MODE
)
from simulation.scheduler import calculate_schedule_fitness

# --- dMShOA-Specific Helper Functions ---

def _sigmoid(x):
    """Sigmoid function to map a value to a probability."""
    clipped_x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-clipped_x))

def _create_random_solution(job_ids):
    """Creates a random discrete solution with GUARANTEED balanced room distribution."""
    solution = {}
    solution['job_sequence_base'] = random.sample(job_ids, len(job_ids))
    
    num_jobs = len(job_ids)
    
    # Create cyclic assignment lists
    pab_cycle1 = (PABELLONES * ((num_jobs // len(PABELLONES)) + 1))[:num_jobs]
    pab_cycle2 = (PABELLONES * ((num_jobs // len(PABELLONES)) + 1))[:num_jobs]
    
    # Shuffle to add randomness while maintaining balance
    random.shuffle(pab_cycle1)
    random.shuffle(pab_cycle2)
    
    solution['room_assignment'] = {}
    for idx, job in enumerate(job_ids):
        solution['room_assignment'][job] = {
            1: pab_cycle1[idx],
            2: pab_cycle2[idx]
        }
    
    return solution

# --- Main Algorithm Execution Function ---

def run(surgeries_data, job_ids, seed, on_iteration=None):
    """Executes the full dMShOA Old (legacy) cycle.

    Parameters
    ----------
    surgeries_data : dict
    job_ids : list of int
    seed : int
    on_iteration : callable, optional
        Invoked at end of each iteration (for compatibility with new runner interface).

    Returns
    -------
    tuple
        (best_fitness, best_solution, best_fitness_history, avg_fitness_history)
    """
    random.seed(seed)
    np.random.seed(seed)

    num_jobs = len(job_ids)
    dim_sequence = num_jobs
    dim_rooms = num_jobs * 2
    dim_total = dim_sequence + dim_rooms

    # 1. Initialization
    population_pos = np.random.uniform(MSHOA_LOWER_BOUND, MSHOA_UPPER_BOUND, (MSHOA_POP_SIZE, dim_total))
    population_sol = [_create_random_solution(job_ids) for _ in range(MSHOA_POP_SIZE)]
    
    fitness = [calculate_schedule_fitness(sol, surgeries_data) for sol in population_sol]

    best_idx = np.argmin(fitness)
    gbest_value = fitness[best_idx]
    gbest_position = copy.deepcopy(population_pos[best_idx])
    gbest_solution = copy.deepcopy(population_sol[best_idx])
    
    _, gbest_makespan, _ = calculate_schedule_fitness(
        gbest_solution, surgeries_data, return_details=True
    )
    
    pti = np.random.randint(1, 4, MSHOA_POP_SIZE)
    best_fitness_history = []
    avg_fitness_history = []

    print_interval = max(1, MAX_ITERATIONS_MSHOA // 4)
    for t in range(MAX_ITERATIONS_MSHOA):
        for i in range(MSHOA_POP_SIZE):
            current_pos = population_pos[i]
            current_sol = population_sol[i]

            # A. CALCULATION OF THE NEW CONTINUOUS POSITION (Original MShOA Logic)
            new_pos = np.zeros_like(current_pos)
            if pti[i] == 1: # Foraging
                D = np.random.uniform(-1, 1)
                r_idx = np.random.choice(np.delete(np.arange(MSHOA_POP_SIZE), i))
                x_r = population_pos[r_idx]
                v = current_pos - gbest_position
                R_t = x_r - current_pos
                new_pos = gbest_position - v + D * R_t
            elif pti[i] == 2: # Attack
                theta = np.random.uniform(np.pi, 2 * np.pi)
                new_pos = gbest_position * np.cos(theta)
            else: # Shelter/Defense
                k_rand = np.random.uniform(0, MSHOA_K)
                direction = 1 if random.random() > 0.5 else -1
                new_pos = gbest_position + (gbest_position * k_rand * direction)
            
            new_pos = np.clip(new_pos, MSHOA_LOWER_BOUND, MSHOA_UPPER_BOUND)

            # B. DISCRETIZATION (Conversion from continuous movement to discrete change)
            velocity = new_pos - current_pos
            probabilities = _sigmoid(velocity)
            new_sol = copy.deepcopy(current_sol)

            # 1. Probabilistically update sequence (swap)
            prob_seq = probabilities[:dim_sequence]
            for j in range(dim_sequence):
                if random.random() < prob_seq[j]:
                    swap_with_idx = random.randint(0, dim_sequence - 1)
                    new_sol['job_sequence_base'][j], new_sol['job_sequence_base'][swap_with_idx] = \
                        new_sol['job_sequence_base'][swap_with_idx], new_sol['job_sequence_base'][j]

            # 2. Probabilistically update rooms (reassignment)
            prob_rooms = probabilities[dim_sequence:]
            room_idx = 0
            for job_id in job_ids:
                for op in [1, 2]:
                    if random.random() < prob_rooms[room_idx]:
                        room_lists = [PABELLONES, PABELLONES]
                        new_sol['room_assignment'][job_id][op] = random.choice(room_lists[op - 1])
                    room_idx += 1
            
            # C. EVALUATION AND UPDATE
            new_fitness = calculate_schedule_fitness(new_sol, surgeries_data)
            
            if new_fitness < fitness[i]:
                population_pos[i] = new_pos
                population_sol[i] = new_sol
                fitness[i] = new_fitness

        # Update gbest and PTI
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < gbest_value:
            gbest_value = fitness[current_best_idx]
            gbest_position = copy.deepcopy(population_pos[current_best_idx])
            gbest_solution = copy.deepcopy(population_sol[current_best_idx])
            
            _, gbest_makespan, _ = calculate_schedule_fitness(
                gbest_solution, surgeries_data, return_details=True
            )

        pti = np.random.randint(1, 4, MSHOA_POP_SIZE)

        # Save history
        valid_fitnesses = [f for f in fitness if f != float('inf')]
        avg_fitness_iter = np.mean(valid_fitnesses) if valid_fitnesses else float('inf')
        
        best_so_far = best_fitness_history[-1] if best_fitness_history else float('inf')
        best_fitness_history.append(min(gbest_value, best_so_far))
        avg_fitness_history.append(avg_fitness_iter)
        
        if VERBOSE_MODE:
            if t == 0 or (t + 1) % print_interval == 0 or t == MAX_ITERATIONS_MSHOA - 1:
                print(f"  -> Iter {t + 1}/{MAX_ITERATIONS_MSHOA}, Best Fitness: {gbest_value:.2f} || Makespan (of Best Fitness): {gbest_makespan:.2f}")

        # --- Callback (interface compatibility con variante adaptada) ---
        if on_iteration is not None:
            from core.iteration_callback import serialize_solution

            # iteration_fitness: best fitness found THIS iteration (not global best)
            iter_best_value = min(valid_fitnesses) if valid_fitnesses else float('inf')
            _, iter_makespan, _ = calculate_schedule_fitness(
                population_sol[current_best_idx], surgeries_data, return_details=True
            ) if iter_best_value != float('inf') else (None, float('inf'), None)

            on_iteration(
                algo_step=t + 1,
                best_fitness=gbest_value,
                best_makespan=gbest_makespan,
                iteration_fitness=avg_fitness_iter,
                iteration_makespan=iter_makespan,
                best_solution_snapshot=serialize_solution(gbest_solution),
            )

    return gbest_value, gbest_solution, best_fitness_history, avg_fitness_history
