"""Load the solver set used by the legacy-compatible experiment reports."""

from __future__ import annotations

from typing import Any

from algorithms.dmshoa import run as run_dmshoa
from algorithms.dpso import run as run_dpso
from algorithms.ga import run as run_ga
from algorithms.mh import run as run_mh
from algorithms.sboa import run as run_sboa


def load_algorithms(
    ga_enabled: bool,
    dpso_enabled: bool,
    sboa_enabled: bool,
    mshoa_enabled: bool,
    max_generations: int,
    max_iterations_dpso: int,
    sboa_max_iter: int,
    max_iterations_mshoa: int,
    all_rooms: list[str],
    mh_enabled: bool = False,
    mh_max_iterations: int = 100,
) -> list[dict[str, Any]]:
    """Return enabled algorithms in the historical comparison order."""
    algorithms: list[dict[str, Any]] = []
    if ga_enabled:
        algorithms.append({
            "name": "GA", "runner": run_ga, "iterations": max_generations,
            "all_rooms": list(all_rooms),
        })
    if dpso_enabled:
        algorithms.append({
            "name": "dPSO", "runner": run_dpso, "iterations": max_iterations_dpso,
            "all_rooms": list(all_rooms),
        })
    if sboa_enabled:
        algorithms.append({
            "name": "SBOA", "runner": run_sboa, "iterations": sboa_max_iter,
            "all_rooms": list(all_rooms),
        })
    if mshoa_enabled:
        algorithms.append({
            "name": "dMShOA", "runner": run_dmshoa, "iterations": max_iterations_mshoa,
            "all_rooms": list(all_rooms), "interface": "context",
        })
    if mh_enabled:
        algorithms.append({
            "name": "MH", "runner": run_mh, "iterations": mh_max_iterations,
            "all_rooms": list(all_rooms), "interface": "context",
        })
    if not algorithms:
        raise ValueError("No algorithms are enabled in config.yaml")
    return algorithms


__all__ = ["load_algorithms"]
