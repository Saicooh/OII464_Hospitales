# /algorithms/dmshoa_adaptado.py
"""
dMShOA Adaptado — implementación paper-faithful PRE-GREEDY.

Variante "adaptada" del dMShOA: fiel al paper (MShOA, Algorithm 1 & 2),
sin el decoder greedy experimental que fue rechazado.

Key decisions:
- Continuous population matrix X[pop, dim_total], full replacement each iteration.
- SPV (Smallest Position Value / argsort) for deterministic sequence decoding.
- Bound-normalized binning for room assignment (no balancing heuristics).
- PTI updated per-agent from (X_old, X_new) via LPA cosine-angle classification.
- Out-of-bounds coordinates repaired by uniform resampling, not clipping.
- Strategy equations match paper exactly: foraging Eq.(12), attack Eq.(14),
  shelter/defense Eq.(15) — no extra MATLAB cosine branch.

Nota: este módulo es la fuente de verdad de la variante adaptada.
      algorithms/dmshoa.py re-exporta este módulo para backward compatibility.
"""

import copy
import math
import numpy as np

from config.config import (
    MSHOA_POP_SIZE,
    MAX_ITERATIONS_MSHOA,
    MSHOA_LOWER_BOUND,
    MSHOA_UPPER_BOUND,
    MSHOA_K,
    PABELLONES,
    VERBOSE_MODE,
)
from simulation.scheduler import calculate_schedule_fitness


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _initialize_pti(pop_size: int, rng: np.random.Generator) -> np.ndarray:
    """Initialise the PTI vector using paper Eq.(2): PTI_i = round(1 + 2*rand).

    Each value is independently drawn from U(0, 1) and mapped to {1, 2, 3}.

    Parameters
    ----------
    pop_size : int
    rng : np.random.Generator

    Returns
    -------
    np.ndarray of int, shape (pop_size,)
    """
    pti = np.array(
        [int(round(1.0 + 2.0 * rng.random())) for _ in range(pop_size)],
        dtype=int,
    )
    return pti


def _compute_lpa(old_pos: np.ndarray, new_pos: np.ndarray) -> float:
    """Compute Left Polarisation Angle (LPA) between old and new positions.

    Defined as the angle between the two vectors via cosine similarity,
    clamped to [0, π].  This is the paper's Eq.(3) interpretation for a
    vector-valued agent position.

    Returns
    -------
    float
        Angle in radians, ∈ [0, π].
    """
    norm_old = np.linalg.norm(old_pos)
    norm_new = np.linalg.norm(new_pos)
    if norm_old < 1e-300 or norm_new < 1e-300:
        return 0.0
    cos_sim = np.dot(old_pos, new_pos) / (norm_old * norm_new)
    # Clamp to [-1, 1] to guard against numerical drift
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
    return math.acos(cos_sim)


def _sample_attack_theta(rng: np.random.Generator) -> float:
    """Sample attack angle θ uniformly from [π, 2π] for Eq.(14).

    Kept separate from _sample_rpa_deg to avoid sharing RNG state
    between the PTI update and the attack strategy.
    """
    return rng.uniform(math.pi, 2.0 * math.pi)


def _pti_distances(angle_deg: float) -> tuple:
    """Compute the three polarization-channel distances for a given angle in degrees.

    MATLAB authority (getPolarization_MSHOA.m):
      fi=10, a=35, b=55, c=125, d=145.

    For angle <= 90 (else branch in MATLAB):
        LinH = angle
        LinV = 90 - angle
        PolC = |angle - 45|, set to 0 if angle ∈ [35, 55]

    For angle > 90 (if branch in MATLAB):
        LinH = 180 - angle
        LinV = angle - 90
        PolC = |angle - 135|, set to 0 if angle ∈ [125, 145]

    Parameters
    ----------
    angle_deg : float
        Angle in degrees, ∈ [0, 180].

    Returns
    -------
    tuple[float, float, float]
        (LinH, LinV, PolC) — all non-negative.
    """
    a, b = 35.0, 55.0
    c, d = 125.0, 145.0

    if angle_deg > 90.0:
        lin_h = 180.0 - angle_deg
        lin_v = angle_deg - 90.0
        pol_c = abs(angle_deg - 135.0)
        if c <= angle_deg <= d:
            pol_c = 0.0
    else:
        lin_h = angle_deg
        lin_v = 90.0 - angle_deg
        pol_c = abs(angle_deg - 45.0)
        if a <= angle_deg <= b:
            pol_c = 0.0

    return lin_h, lin_v, pol_c


def _pti_from_distances(lin_h: float, lin_v: float, pol_c: float) -> int:
    """Return the PTI label (1, 2, or 3) by selecting the argmin channel.

    MATLAB: ``[dif, Idx] = min([LinH, LinV, PolC])``.
    Result is 1-indexed: 1=LinH, 2=LinV, 3=PolC.

    Parameters
    ----------
    lin_h, lin_v, pol_c : float
        Channel distances from :func:`_pti_distances`.

    Returns
    -------
    int
        PTI label ∈ {1, 2, 3}.
    """
    if lin_h <= lin_v and lin_h <= pol_c:
        return 1
    if lin_v <= pol_c:
        return 2
    return 3


def _sample_rpa_deg(rng: np.random.Generator) -> int:
    """Sample right-eye angle as a random integer in [1, 90].

    MATLAB: ``randi([1, 90])``.

    Parameters
    ----------
    rng : np.random.Generator

    Returns
    -------
    int
        Integer in [1, 90].
    """
    return int(rng.integers(1, 91))


def _update_pti_vector(
    X_old: np.ndarray, X_new: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Update PTI vector from the old and new population matrices.

    Implements the MATLAB ``getPolarization`` dual-eye selection per agent:

    For each agent i:
      1. left_angle_deg  = degrees(acos(dot(norm(X_old[i]), norm(X_new[i]))))
      2. right_angle_deg = _sample_rpa_deg(rng)   [1, 90] integer
      3. dif_1, Idx_1    = min(_pti_distances(left_angle_deg))
      4. dif_2, Idx_2    = min(_pti_distances(right_angle_deg))
      5. PTI[i]          = Idx_1 if dif_1 <= dif_2 else Idx_2

    Parameters
    ----------
    X_old : np.ndarray, shape (pop_size, dim)
    X_new : np.ndarray, shape (pop_size, dim)
    rng : np.random.Generator

    Returns
    -------
    np.ndarray of int, shape (pop_size,)
    """
    pop_size = X_old.shape[0]

    # MATLAB: k1=randi(N); k2=k1+randi(N-1);
    #         Positions=circshift(Positions,k1,1); x=circshift(x,k2,1);
    # np.roll with axis=0 is equivalent to circshift(...,k,1) in MATLAB.
    # Guard: MATLAB randi(0) is undefined; for pop_size==1 skip shift entirely.
    if pop_size > 1:
        k1 = int(rng.integers(1, pop_size + 1))
        r = int(rng.integers(1, pop_size))
        k2 = k1 + r
    else:
        k1, k2 = 0, 0
    X_old_shifted = np.roll(X_old, k1, axis=0)
    X_new_shifted = np.roll(X_new, k2, axis=0)

    pti = np.empty(pop_size, dtype=int)
    for i in range(pop_size):
        # Left eye: cosine angle between SHIFTED old and new position vectors (degrees)
        left_angle_deg = math.degrees(_compute_lpa(X_old_shifted[i], X_new_shifted[i]))

        # Right eye: integer in [1, 90] per MATLAB randi([1, 90])
        right_angle_deg = float(_sample_rpa_deg(rng))

        # Distance triplets for each eye
        lin_h1, lin_v1, pol_c1 = _pti_distances(left_angle_deg)
        lin_h2, lin_v2, pol_c2 = _pti_distances(right_angle_deg)

        dif_1 = min(lin_h1, lin_v1, pol_c1)
        dif_2 = min(lin_h2, lin_v2, pol_c2)

        if dif_1 <= dif_2:
            pti[i] = _pti_from_distances(lin_h1, lin_v1, pol_c1)
        else:
            pti[i] = _pti_from_distances(lin_h2, lin_v2, pol_c2)

    return pti


def _decode_position(position: np.ndarray, job_ids: list) -> dict:
    """Decode a continuous position vector into a discrete schedule solution.

    Layout: position[0:n] encodes job sequence; position[n:3n] encodes rooms.

    Sequence decoding (SPV / rank-order):
        The job at rank k (0-indexed argsort) in position[0:n] is placed k-th
        in the sequence.

    Room decoding (bound-normalized binning):
        Each of the 2n room dimensions is normalised to [0, 1] relative to
        [MSHOA_LOWER_BOUND, MSHOA_UPPER_BOUND], then mapped to a room index via
        floor(value * num_rooms), clipped to [0, num_rooms-1].

    Parameters
    ----------
    position : np.ndarray, shape (3n,)
    job_ids : list of int

    Returns
    -------
    dict with keys 'job_sequence_base' and 'room_assignment'.
    """
    # Import at call time to respect test module reloads (avoids stale closure)
    from config.config import (  # noqa: PLC0415
        PABELLONES as _PABELLONES,
        MSHOA_LOWER_BOUND as _LB,
        MSHOA_UPPER_BOUND as _UB,
    )

    n = len(job_ids)
    seq_part = position[:n]
    room_part = position[n:]  # length 2n

    # --- Sequence via SPV (argsort) ---
    rank_order = np.argsort(seq_part)  # rank_order[k] = index of k-th smallest
    job_ids_arr = np.array(job_ids)
    job_sequence_base = list(job_ids_arr[rank_order])

    # --- Room assignment via binning ---
    num_rooms = len(_PABELLONES)
    lb = _LB
    ub = _UB
    span = ub - lb if ub != lb else 1.0

    room_assignment = {}
    for idx, job_id in enumerate(job_ids):
        rooms = {}
        for op in [1, 2]:
            dim_idx = idx * 2 + (op - 1)
            val = float(np.clip(room_part[dim_idx], lb, ub))
            normalised = (val - lb) / span  # [0, 1]
            room_idx = int(math.floor(normalised * num_rooms))
            room_idx = min(room_idx, num_rooms - 1)
            rooms[op] = _PABELLONES[room_idx]
        room_assignment[job_id] = rooms

    return {"job_sequence_base": job_sequence_base, "room_assignment": room_assignment}


def _repair_bounds(position: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Repair out-of-bounds coordinates by uniform resampling in [lb, ub].

    Elements within bounds are unchanged.  Out-of-bounds elements receive a
    fresh uniform sample — matching both the paper flow and the MATLAB
    implementation (random re-entry, not clipping).

    Parameters
    ----------
    position : np.ndarray (modified in-place and returned)
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
    """
    lb = MSHOA_LOWER_BOUND
    ub = MSHOA_UPPER_BOUND
    oob = (position < lb) | (position > ub)
    n_oob = int(np.sum(oob))
    if n_oob:
        position[oob] = rng.uniform(lb, ub, n_oob)
    return position


def _apply_strategy(
    position: np.ndarray,
    gbest_position: np.ndarray,
    population: np.ndarray,
    pti_i: int,
    rng: np.random.Generator,
    agent_idx: int,
) -> np.ndarray:
    """Apply one MShOA strategy to produce a candidate position.

    PTI 1 — Foraging (Eq. 12):
        new = gbest - v + D * R_t
        where v = current - gbest, R_t = x_r - current, D ∈ U(-1, 1).

    PTI 2 — Attack (Eq. 14):
        new = gbest * cos(θ), θ ∈ U[π, 2π].

    PTI 3 — Shelter/Defense (Eq. 15):
        new = gbest + gbest * k_rand * direction, k_rand ∈ U(0, K), direction ∈ {+1, -1}.

    Parameters
    ----------
    position : np.ndarray
        Current agent position.
    gbest_position : np.ndarray
        Global-best position.
    population : np.ndarray, shape (pop_size, dim)
        Current full population (for random peer selection in foraging).
    pti_i : int
        PTI class of this agent (1, 2, or 3).
    rng : np.random.Generator
    agent_idx : int
        Index of this agent in the population (to exclude from peer selection).

    Returns
    -------
    np.ndarray
        Candidate position (bounds NOT yet repaired).
    """
    pop_size = population.shape[0]

    if pti_i == 1:  # Foraging — Eq.(12)
        D = rng.uniform(-1.0, 1.0)
        peers = [j for j in range(pop_size) if j != agent_idx]
        r_idx = int(rng.choice(peers))
        x_r = population[r_idx]
        v = position - gbest_position
        R_t = x_r - position
        new_pos = gbest_position - v + D * R_t

    elif pti_i == 2:  # Attack — Eq.(14)
        theta = _sample_attack_theta(rng)
        new_pos = gbest_position * math.cos(theta)

    else:  # Shelter/Defense — Eq.(15)
        k_rand = rng.uniform(0.0, MSHOA_K)
        direction = 1.0 if rng.random() > 0.5 else -1.0
        new_pos = gbest_position + gbest_position * k_rand * direction

    return new_pos


# ---------------------------------------------------------------------------
# Main algorithm entry point
# ---------------------------------------------------------------------------


def run(surgeries_data, job_ids, seed, on_iteration=None):
    """Execute the full paper-faithful dMShOA Adaptado cycle.

    Parameters
    ----------
    surgeries_data : dict
        Surgery processing times keyed by job_id.
    job_ids : list of int
        Job identifiers for this scheduling instance.
    seed : int
        RNG seed for full reproducibility.
    on_iteration : callable, optional
        Invoked at the end of each iteration with signature::

            on_iteration(
                algo_step: int,
                best_fitness: float,
                best_makespan: float,
                iteration_fitness: float,
                iteration_makespan: float,
                best_solution_snapshot: dict | None,
            )

    Returns
    -------
    tuple
        (best_fitness, best_solution, best_fitness_history, avg_fitness_history)
    """
    rng = np.random.default_rng(seed)

    num_jobs = len(job_ids)
    dim_total = num_jobs + 2 * num_jobs  # sequence + rooms (n + 2n)

    lb = MSHOA_LOWER_BOUND
    ub = MSHOA_UPPER_BOUND

    # --- Initialisation ---
    X = rng.uniform(lb, ub, (MSHOA_POP_SIZE, dim_total))
    population_sol = [_decode_position(X[i], job_ids) for i in range(MSHOA_POP_SIZE)]
    fitness = np.array(
        [calculate_schedule_fitness(sol, surgeries_data) for sol in population_sol],
        dtype=float,
    )

    best_idx = int(np.argmin(fitness))
    gbest_value = float(fitness[best_idx])
    gbest_position = X[best_idx].copy()
    gbest_solution = copy.deepcopy(population_sol[best_idx])
    _, gbest_makespan, _ = calculate_schedule_fitness(
        gbest_solution, surgeries_data, return_details=True
    )

    # Initialise PTI from Eq.(2): PTI_i = round(1 + 2*rand)
    pti = _initialize_pti(MSHOA_POP_SIZE, rng)

    best_fitness_history: list[float] = []
    avg_fitness_history: list[float] = []

    print_interval = max(1, MAX_ITERATIONS_MSHOA // 4)

    for t in range(MAX_ITERATIONS_MSHOA):
        X_old = X.copy()

        # --- Generate candidate population ---
        X_candidate = np.empty_like(X)
        for i in range(MSHOA_POP_SIZE):
            cand = _apply_strategy(X[i], gbest_position, X, pti[i], rng, i)
            X_candidate[i] = _repair_bounds(cand, rng)

        # --- Decode and evaluate all candidates ---
        candidate_sols = [
            _decode_position(X_candidate[i], job_ids) for i in range(MSHOA_POP_SIZE)
        ]
        candidate_fitness = np.array(
            [
                calculate_schedule_fitness(sol, surgeries_data)
                for sol in candidate_sols
            ],
            dtype=float,
        )

        # --- Iteration metrics (new evaluations only, isolated from gbest) ---
        valid_mask = np.isfinite(candidate_fitness)
        if np.any(valid_mask):
            iter_best_idx = int(np.argmin(candidate_fitness))
            iter_best_value = float(candidate_fitness[iter_best_idx])
            _, iter_makespan, _ = calculate_schedule_fitness(
                candidate_sols[iter_best_idx], surgeries_data, return_details=True
            )
        else:
            iter_best_value = float("inf")
            iter_makespan = float("inf")

        # --- Full population replacement (Algorithm 2, paper) ---
        X = X_candidate
        fitness = candidate_fitness
        population_sol = candidate_sols

        # --- Update PTI from (X_old, X_new) ---
        pti = _update_pti_vector(X_old, X, rng)

        # --- Update global best ---
        pop_best_idx = int(np.argmin(fitness))
        pop_best_value = float(fitness[pop_best_idx])
        if pop_best_value < gbest_value:
            gbest_value = pop_best_value
            gbest_position = X[pop_best_idx].copy()
            gbest_solution = copy.deepcopy(population_sol[pop_best_idx])
            _, gbest_makespan, _ = calculate_schedule_fitness(
                gbest_solution, surgeries_data, return_details=True
            )

        # --- History ---
        best_so_far = best_fitness_history[-1] if best_fitness_history else float("inf")
        best_fitness_history.append(min(gbest_value, best_so_far))

        valid_fitnesses = fitness[np.isfinite(fitness)]
        avg_fitness_iter = float(np.mean(valid_fitnesses)) if len(valid_fitnesses) else float("inf")
        avg_fitness_history.append(avg_fitness_iter)

        if VERBOSE_MODE:
            if t == 0 or (t + 1) % print_interval == 0 or t == MAX_ITERATIONS_MSHOA - 1:
                print(
                    f"  -> Iter {t + 1}/{MAX_ITERATIONS_MSHOA}, "
                    f"Best Fitness: {gbest_value:.2f} || Makespan (of Best Fitness): {gbest_makespan:.2f}"
                )

        # --- Callback ---
        if on_iteration is not None:
            from core.iteration_callback import serialize_solution

            on_iteration(
                algo_step=t + 1,
                best_fitness=gbest_value,
                best_makespan=gbest_makespan,
                iteration_fitness=avg_fitness_iter,
                iteration_makespan=iter_makespan,
                best_solution_snapshot=serialize_solution(gbest_solution),
            )

    return gbest_value, gbest_solution, best_fitness_history, avg_fitness_history
