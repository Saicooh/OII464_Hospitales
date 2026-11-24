"""dPSO con control aprendido de la perturbación (entrada ``MH`` del curso).

El componente metaheurístico es la dPSO del repositorio (misma actualización
de velocidades y posiciones). El componente de aprendizaje controla una
mutación adicional en dos niveles:

* nivel de intensidad: al inicio de cada bloque de ``block`` iteraciones se
  elige la probabilidad ``p`` de mutar cada partícula;
* nivel de movimiento: cada vez que una partícula se muta se elige el
  movimiento (intercambio, inserción o reasignación de sala).

Ambos niveles usan estimaciones de valor con paso constante y selección
épsilon-voraz. ``METHODS`` define las variantes de comparación y ablación.
Todas las variantes consumen el mismo presupuesto de llamadas al evaluador.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from data.instance_model import InstanceContext
from simulation.result_model import ScheduleSolution, SolverOutput
from simulation.scheduler import calculate_schedule_fitness, evaluate_solution, objective_from  # noqa: F401


IterationCallback = Callable[..., None]

MOVES = ("intercambio", "insercion", "sala")

# variante -> (control de intensidad, control de movimiento)
METHODS: dict[str, tuple[str, str]] = {
    "base": ("ninguno", "uniforme"),
    "fija": ("fija", "uniforme"),
    "aleatoria": ("aleatoria", "uniforme"),
    "reglas": ("reglas", "uniforme"),
    "aprende_p": ("aprendida", "uniforme"),
    "aprende_mov": ("fija", "aprendido"),
    "dos_niveles": ("aprendida", "aprendido"),
}


@dataclass(frozen=True)
class HybridParams:
    """Parámetros comunes a todas las variantes."""

    swarm_size: int = 30
    budget: int = 10_000
    block: int = 5
    probabilities: tuple[float, ...] = (0.05, 0.15, 0.30)
    fixed_p: float = 0.15
    epsilon: float = 0.1
    eta: float = 0.2
    eta_move: float = 0.05
    improvement_tolerance: float = 1e-6
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    vel_low: float = -4.0
    vel_high: float = 4.0


@dataclass
class HybridResult:
    """Resultado de una corrida, con los registros del controlador."""

    best_value: float
    best_solution: dict[str, Any] | None
    makespan: float
    total_wait: float
    max_wait: float
    evaluations: int
    elapsed: float
    best_history: list[float] = field(default_factory=list)
    average_history: list[float] = field(default_factory=list)
    eval_history: list[int] = field(default_factory=list)
    block_log: list[dict[str, Any]] = field(default_factory=list)
    move_counts: dict[str, int] = field(default_factory=dict)
    move_values: dict[str, float] = field(default_factory=dict)
    mutations: int = 0
    wait_violations: int = 0


class _Data(dict):
    """Mapa histórico que espera el evaluador de compatibilidad."""

    def __init__(self, context: InstanceContext) -> None:
        super().__init__(
            {
                job.job_id: {op.operation_id: op.duration for op in job.operations}
                for job in context.jobs
            }
        )
        self.context = context


def _waits(context: InstanceContext, solution: dict[str, Any]) -> tuple[float, float, int]:
    """Espera total, espera máxima y trabajos sobre su límite, según el evaluador."""
    decoded = evaluate_solution(context, solution)
    if decoded is None:
        return math.inf, math.inf, -1
    return decoded.total_wait, decoded.max_wait, decoded.wait_violations


class _ValueLearner:
    """Estimaciones de valor con paso constante y selección épsilon-voraz."""

    def __init__(self, n: int, epsilon: float, eta: float, rng: random.Random):
        self.q = [0.0] * n
        self.n = n
        self.epsilon = epsilon
        self.eta = eta
        self.rng = rng

    def greedy(self) -> int:
        top = max(self.q)
        return self.rng.choice([i for i, value in enumerate(self.q) if value == top])

    def select(self) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n)
        return self.greedy()

    def update(self, arm: int, reward: float) -> None:
        self.q[arm] += self.eta * (reward - self.q[arm])


class _Particle:
    __slots__ = (
        "sequence", "rooms", "vel_seq", "vel_rooms",
        "pbest_sequence", "pbest_rooms", "pbest_value", "value",
    )


def _relative_improvement(before: float, after: float) -> float:
    if not (math.isfinite(before) and math.isfinite(after)):
        return 0.0
    return min(1.0, max(0.0, (before - after) / max(1.0, abs(before))))


def run_hybrid(
    context: InstanceContext,
    seed: int,
    method: str = "dos_niveles",
    params: HybridParams | None = None,
    on_iteration: IterationCallback | None = None,
) -> HybridResult:
    """Ejecuta una variante con presupuesto fijo de llamadas al evaluador."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    prm = params or HybridParams()
    intensity_mode, move_mode = METHODS[method]
    started = time.perf_counter()

    # Flujos aleatorios separados: población inicial, dPSO, mutación y control.
    seed = int(seed)
    rng_init = random.Random(seed * 4 + 0)
    np_init = np.random.default_rng([seed, 0])
    rng_pso = random.Random(seed * 4 + 1)
    np_pso = np.random.default_rng([seed, 1])
    rng_mut = random.Random(seed * 4 + 2)
    rng_ctrl = random.Random(seed * 4 + 3)

    data = _Data(context)
    job_ids = [job.job_id for job in context.jobs]
    num_jobs = len(job_ids)
    job_index = {job_id: index for index, job_id in enumerate(job_ids)}
    rooms = list(context.rooms)
    room_scale = {
        room: (index / (len(rooms) - 1) if len(rooms) > 1 else 0.5)
        for index, room in enumerate(rooms)
    }
    eligible = {
        (job.job_id, op.operation_id): list(op.eligible_rooms)
        for job in context.jobs
        for op in job.operations
    }
    movable_ops = [key for key, options in eligible.items() if len(options) > 1]

    evaluations = 0

    def evaluate(particle: _Particle) -> tuple[float, float, list[dict[str, Any]]]:
        nonlocal evaluations
        evaluations += 1
        return calculate_schedule_fitness(
            {"job_sequence_base": particle.sequence, "room_assignment": particle.rooms},
            data,
            return_details=True,
        )

    # --- Población inicial (igual para todas las variantes con la misma semilla)
    swarm: list[_Particle] = []
    for _ in range(max(1, prm.swarm_size)):
        particle = _Particle()
        particle.sequence = rng_init.sample(job_ids, num_jobs)
        cycles = []
        for _op in (1, 2):
            cycle = (rooms * (num_jobs // len(rooms) + 1))[:num_jobs]
            rng_init.shuffle(cycle)
            cycles.append(cycle)
        particle.rooms = {
            job_id: {
                1: cycles[0][i] if cycles[0][i] in eligible[(job_id, 1)]
                else rng_init.choice(eligible[(job_id, 1)]),
                2: cycles[1][i] if cycles[1][i] in eligible[(job_id, 2)]
                else rng_init.choice(eligible[(job_id, 2)]),
            }
            for i, job_id in enumerate(job_ids)
        }
        particle.vel_seq = np_init.uniform(prm.vel_low, prm.vel_high, num_jobs)
        particle.vel_rooms = np_init.uniform(prm.vel_low, prm.vel_high, num_jobs * 2)
        particle.pbest_sequence = list(particle.sequence)
        particle.pbest_rooms = {j: dict(r) for j, r in particle.rooms.items()}
        particle.pbest_value = math.inf
        particle.value = math.inf
        swarm.append(particle)

    gbest_value = math.inf
    gbest_sequence: list[int] | None = None
    gbest_rooms: dict[int, dict[int, str]] | None = None
    gbest_details: list[dict[str, Any]] = []
    gbest_makespan = math.inf

    def register(particle: _Particle, fitness: float, makespan: float, details) -> None:
        nonlocal gbest_value, gbest_sequence, gbest_rooms, gbest_details, gbest_makespan
        particle.value = fitness
        if fitness < particle.pbest_value:
            particle.pbest_value = fitness
            particle.pbest_sequence = list(particle.sequence)
            particle.pbest_rooms = {j: dict(r) for j, r in particle.rooms.items()}
        if fitness < gbest_value:
            gbest_value = fitness
            gbest_sequence = list(particle.sequence)
            gbest_rooms = {j: dict(r) for j, r in particle.rooms.items()}
            gbest_details = details
            gbest_makespan = makespan

    def rank_vector(sequence: list[int]) -> np.ndarray:
        vector = np.empty(num_jobs)
        for rank, job_id in enumerate(sequence):
            vector[job_index[job_id]] = rank
        return vector

    def room_vector(assignment: dict[int, dict[int, str]]) -> np.ndarray:
        vector = np.empty(num_jobs * 2)
        for index, job_id in enumerate(job_ids):
            vector[2 * index] = room_scale[assignment[job_id][1]]
            vector[2 * index + 1] = room_scale[assignment[job_id][2]]
        return vector

    def dpso_move(particle: _Particle, gbest_rank: np.ndarray, gbest_room: np.ndarray) -> None:
        """Actualización de velocidad y posición de la dPSO del repositorio."""
        r1_seq, r2_seq = np_pso.random(num_jobs), np_pso.random(num_jobs)
        r1_room, r2_room = np_pso.random(num_jobs * 2), np_pso.random(num_jobs * 2)
        current_rank = rank_vector(particle.sequence)
        particle.vel_seq = np.clip(
            prm.w * particle.vel_seq
            + prm.c1 * r1_seq * (rank_vector(particle.pbest_sequence) - current_rank)
            + prm.c2 * r2_seq * (gbest_rank - current_rank),
            prm.vel_low, prm.vel_high,
        )
        current_room = room_vector(particle.rooms)
        particle.vel_rooms = np.clip(
            prm.w * particle.vel_rooms
            + prm.c1 * r1_room * (room_vector(particle.pbest_rooms) - current_room)
            + prm.c2 * r2_room * (gbest_room - current_room),
            prm.vel_low, prm.vel_high,
        )
        prob_swap = 1.0 / (1.0 + np.exp(-particle.vel_seq))
        if num_jobs >= 2:
            order = list(range(num_jobs))
            rng_pso.shuffle(order)
            sequence = particle.sequence
            for k in order:
                # la velocidad está indexada por trabajo, no por posición
                if rng_pso.random() < prob_swap[job_index[sequence[k]]]:
                    other = rng_pso.randrange(num_jobs - 1)
                    if other >= k:
                        other += 1
                    sequence[k], sequence[other] = sequence[other], sequence[k]
        prob_room = 1.0 / (1.0 + np.exp(-particle.vel_rooms))
        use_pbest = prm.c1 / (prm.c1 + prm.c2) if (prm.c1 + prm.c2) > 0 else 0.5
        for index, job_id in enumerate(job_ids):
            for op in (1, 2):
                if rng_pso.random() < prob_room[2 * index + op - 1]:
                    source = particle.pbest_rooms if rng_pso.random() < use_pbest else gbest_rooms
                    particle.rooms[job_id][op] = source[job_id][op]

    # --- Mutación adicional
    move_learner = _ValueLearner(len(MOVES), prm.epsilon, prm.eta_move, rng_ctrl)
    move_counts = {name: 0 for name in MOVES}

    def available(move: int) -> bool:
        if MOVES[move] == "sala":
            return bool(movable_ops)
        return num_jobs >= 2

    def choose_move() -> int | None:
        options = [m for m in range(len(MOVES)) if available(m)]
        if not options:
            return None
        if move_mode == "aprendido":
            move = move_learner.select()
            if available(move):
                return move
        return rng_mut.choice(options)

    def mutate(particle: _Particle, move: int) -> None:
        name = MOVES[move]
        if name == "intercambio":
            i, j = rng_mut.sample(range(num_jobs), 2)
            seq = particle.sequence
            seq[i], seq[j] = seq[j], seq[i]
            particle.vel_seq[job_index[seq[i]]] = 0.0
            particle.vel_seq[job_index[seq[j]]] = 0.0
        elif name == "insercion":
            i, j = rng_mut.sample(range(num_jobs), 2)
            job_id = particle.sequence.pop(i)
            particle.sequence.insert(j, job_id)
            particle.vel_seq[job_index[job_id]] = 0.0
        else:
            job_id, op = rng_mut.choice(movable_ops)
            current = particle.rooms[job_id][op]
            particle.rooms[job_id][op] = rng_mut.choice(
                [room for room in eligible[(job_id, op)] if room != current]
            )
            particle.vel_rooms[2 * job_index[job_id] + op - 1] = 0.0

    # --- Control de intensidad
    probabilities = list(prm.probabilities)
    intensity_learner = _ValueLearner(len(probabilities), prm.epsilon, prm.eta, rng_ctrl)
    warmup = list(range(len(probabilities)))
    rng_ctrl.shuffle(warmup)
    rule_level = min(range(len(probabilities)), key=lambda i: abs(probabilities[i] - 0.15))
    last_block_improved: bool | None = None

    def choose_intensity() -> tuple[float, int | None]:
        nonlocal rule_level
        if intensity_mode == "ninguno":
            return 0.0, None
        if intensity_mode == "fija":
            return prm.fixed_p, None
        if intensity_mode == "reglas":
            if last_block_improved is True:
                rule_level = max(0, rule_level - 1)
            elif last_block_improved is False:
                rule_level = min(len(probabilities) - 1, rule_level + 1)
            return probabilities[rule_level], rule_level
        if warmup:
            arm = warmup.pop(0)
        elif intensity_mode == "aleatoria":
            arm = rng_ctrl.randrange(len(probabilities))
        else:
            arm = intensity_learner.select()
        return probabilities[arm], arm

    best_history: list[float] = []
    average_history: list[float] = []
    eval_history: list[int] = []
    block_log: list[dict[str, Any]] = []
    mutations = 0

    def close_iteration(values: list[float], step: int) -> None:
        finite = [value for value in values if math.isfinite(value)]
        best_history.append(gbest_value)
        average_history.append(sum(finite) / len(finite) if finite else math.inf)
        eval_history.append(evaluations)
        if on_iteration is not None and gbest_sequence is not None:
            from core.iteration_callback import serialize_solution

            on_iteration(
                algo_step=step,
                best_fitness=gbest_value,
                best_makespan=gbest_makespan,
                iteration_fitness=average_history[-1],
                iteration_makespan=gbest_makespan,
                best_solution_snapshot=serialize_solution(
                    {"job_sequence_base": gbest_sequence, "room_assignment": gbest_rooms}
                ),
            )

    # --- Evaluación inicial
    budget = max(1, int(prm.budget))
    values = []
    for particle in swarm:
        if evaluations >= budget:
            break
        fitness, makespan, details = evaluate(particle)
        register(particle, fitness, makespan, details)
        values.append(fitness)
    close_iteration(values, 1)

    # --- Bloques de iteraciones
    step = 1
    while evaluations < budget:
        p, arm = choose_intensity()
        if gbest_sequence is None and intensity_mode != "ninguno":
            p, arm = prm.fixed_p, None  # sin solución factible todavía
        block_start_value = gbest_value
        block_start_evals = evaluations
        for _ in range(max(1, prm.block)):
            if evaluations >= budget:
                break
            step += 1
            values = []
            if gbest_sequence is not None:
                gbest_rank = rank_vector(gbest_sequence)
                gbest_room = room_vector(gbest_rooms)
            for particle in swarm:
                if evaluations >= budget:
                    break
                previous = particle.value
                if gbest_sequence is not None:
                    dpso_move(particle, gbest_rank, gbest_room)
                move = None
                if p > 0.0 and rng_mut.random() < p:
                    move = choose_move()
                    if move is not None:
                        mutate(particle, move)
                        mutations += 1
                        move_counts[MOVES[move]] += 1
                fitness, makespan, details = evaluate(particle)
                register(particle, fitness, makespan, details)
                values.append(fitness)
                if move is not None and move_mode == "aprendido":
                    move_learner.update(move, 1.0 if fitness < previous else 0.0)
            close_iteration(values, step)
        complete = evaluations - block_start_evals == prm.block * len(swarm)
        reward = _relative_improvement(block_start_value, gbest_value)
        if complete and math.isfinite(block_start_value):
            last_block_improved = reward > prm.improvement_tolerance
            if arm is not None and intensity_mode == "aprendida":
                intensity_learner.update(arm, reward)
        block_log.append(
            {
                "evaluations": evaluations,
                "p": p,
                "reward": reward,
                "complete": complete,
                "best": gbest_value,
                "q": list(intensity_learner.q),
                "q_move": list(move_learner.q),
            }
        )

    solution = (
        {"job_sequence_base": gbest_sequence, "room_assignment": gbest_rooms}
        if gbest_sequence is not None
        else None
    )
    total_wait, max_wait, wait_violations = (
        _waits(context, solution) if solution is not None else (math.inf, math.inf, -1)
    )
    return HybridResult(
        best_value=gbest_value,
        best_solution=solution,
        makespan=gbest_makespan,
        total_wait=total_wait,
        max_wait=max_wait,
        wait_violations=wait_violations,
        evaluations=evaluations,
        elapsed=time.perf_counter() - started,
        best_history=best_history,
        average_history=average_history,
        eval_history=eval_history,
        block_log=block_log,
        move_counts=move_counts,
        move_values=dict(zip(MOVES, move_learner.q)),
        mutations=mutations,
    )


def run(
    context: InstanceContext,
    seed: int,
    on_iteration: IterationCallback | None = None,
) -> SolverOutput:
    """Entrada ``MH`` del repositorio: variante completa de dos niveles.

    El presupuesto es ``evaluation_budget`` llamadas al evaluador, el mismo de
    los demás algoritmos. Si no está definido, es ``population_size *
    max_iterations``.
    """
    import config.config as config

    if not isinstance(context, InstanceContext):
        raise TypeError("context must be an InstanceContext")

    settings = dict(getattr(config, "MH_CONFIG", {}) or {})
    swarm_size = max(1, int(config.MH_POP_SIZE))
    iterations = max(1, int(config.MH_MAX_ITERATIONS))
    budget = getattr(config, "EVALUATION_BUDGET", None)
    params = HybridParams(
        swarm_size=swarm_size,
        budget=int(budget) if budget is not None else swarm_size * iterations,
        block=int(settings.get("block", HybridParams.block)),
        fixed_p=float(settings.get("fixed_p", HybridParams.fixed_p)),
        epsilon=float(settings.get("epsilon", HybridParams.epsilon)),
        eta=float(settings.get("eta", HybridParams.eta)),
        eta_move=float(settings.get("eta_move", HybridParams.eta_move)),
    )
    result = run_hybrid(
        context,
        seed,
        method=str(settings.get("method", "dos_niveles")),
        params=params,
        on_iteration=on_iteration,
    )
    if result.best_solution is None:
        raise RuntimeError("MH did not produce a feasible schedule solution")
    return SolverOutput(
        combined_objective=result.best_value,
        solution=ScheduleSolution.from_dict(result.best_solution),
        best_fitness_history=tuple(result.best_history),
        average_fitness_history=tuple(result.average_history),
    )


__all__ = ["HybridParams", "HybridResult", "METHODS", "MOVES", "run", "run_hybrid"]
