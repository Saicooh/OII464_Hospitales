"""Execution orchestration for one validated synthetic instance."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from joblib import Parallel, delayed

from config.config import N_JOBS, NUM_SIMULATIONS
from data.instance_loader import load_instance
from data.instance_model import InstanceContext
from simulation.result_model import SimulationResult
from simulation.workers.elective_worker import InstanceWorker


Solver = Callable[[InstanceContext, int], Any]


class SimulationRunner:
    """Run one immutable instance with deterministic solver-seed variation."""

    def __init__(
        self,
        context: InstanceContext,
        solver: Solver,
        num_simulations: int | None = None,
        n_jobs: int | None = None,
        solver_seeds: Sequence[int] | None = None,
    ) -> None:
        if not isinstance(context, InstanceContext):
            raise TypeError("context must be an InstanceContext")
        if not callable(solver):
            raise TypeError("solver must be callable")
        self.context = context
        self.solver = solver
        self.num_simulations = NUM_SIMULATIONS if num_simulations is None else int(num_simulations)
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be positive")
        self.n_jobs = N_JOBS if n_jobs is None else int(n_jobs)
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        self.solver_seeds = tuple(
            range(self.num_simulations) if solver_seeds is None else solver_seeds
        )
        if len(self.solver_seeds) != self.num_simulations:
            raise ValueError("solver_seeds must match num_simulations")
        if len(set(self.solver_seeds)) != len(self.solver_seeds):
            raise ValueError("solver seeds must be distinct")

    @classmethod
    def from_instance(
        cls,
        path: str,
        solver: Solver,
        **runtime_settings: Any,
    ) -> "SimulationRunner":
        """Validate the selected YAML before creating a runtime or pool."""
        context = load_instance(path)
        return cls(context=context, solver=solver, **runtime_settings)

    def run_instance_mode(self) -> tuple[SimulationResult, ...]:
        """Execute all configured seeds and return typed results in index order."""
        worker = InstanceWorker(self.context, self.solver)
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(worker.run)(simulation_index, solver_seed)
            for simulation_index, solver_seed in enumerate(self.solver_seeds)
        )
        return tuple(results)


__all__ = ["SimulationRunner"]
