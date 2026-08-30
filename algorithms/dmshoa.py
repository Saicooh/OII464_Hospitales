# /algorithms/dmshoa.py
"""
dMShOA — wrapper de backward compatibility.

Este módulo re-exporta todo desde algorithms.dmshoa_adaptado, que es la
implementación paper-faithful PRE-GREEDY (la variante activa y correcta).

El decoder greedy experimental fue RECHAZADO y NO está activo aquí.

Para las dos variantes usables:
  - algorithms.dmshoa_adaptado  →  paper-faithful, SPV + full-replacement
  - algorithms.dmshoa_old       →  legacy, sigmoid + swaps probabilísticos

Este archivo existe SÓLO para no romper imports existentes que usen
``from algorithms.dmshoa import run`` o ``from algorithms import dmshoa``.
"""

# Re-export completo desde la variante adaptada (fuente de verdad)
from algorithms.dmshoa_adaptado import (  # noqa: F401
    run,
    _initialize_pti,
    _compute_lpa,
    _sample_attack_theta,
    _pti_distances,
    _pti_from_distances,
    _sample_rpa_deg,
    _update_pti_vector,
    _decode_position,
    _repair_bounds,
    _apply_strategy,
)
