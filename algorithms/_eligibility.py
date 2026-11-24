"""Room-eligibility helpers shared by the historical algorithms.

The historical GA, dPSO, and SBOA drew rooms from the global room list. That is
only correct when every room is eligible for every operation. These helpers
read the eligibility declared by the selected instance, so the algorithms never
propose an ineligible room. When every room is eligible, they return the same
lists in the same order as before and consume the random stream identically.
"""

from __future__ import annotations

import random
from typing import Any, Callable


def eligible_room_table(surgeries_data: Any, rooms: list[str]) -> dict[tuple[int, int], list[str]]:
    """Map (job, operation) to its eligible rooms, falling back to ``rooms``."""
    context = getattr(surgeries_data, "context", None)
    if context is None:
        return {
            (int(job_id), op): list(rooms)
            for job_id in surgeries_data
            for op in (1, 2)
        }
    return {
        (job.job_id, operation.operation_id): list(operation.eligible_rooms)
        for job in context.jobs
        for operation in job.operations
    }


def cyclic_room_assignment(
    job_ids: list[int],
    rooms: list[str],
    table: dict[tuple[int, int], list[str]],
    shuffle: Callable[[list[str]], None] = random.shuffle,
    choice: Callable[[list[str]], str] = random.choice,
) -> dict[int, dict[int, str]]:
    """Spread operations over rooms in shuffled cycles, respecting eligibility."""
    num_jobs = len(job_ids)
    cycles = []
    for _ in (1, 2):
        cycle = (list(rooms) * ((num_jobs // len(rooms)) + 1))[:num_jobs]
        shuffle(cycle)
        cycles.append(cycle)
    assignment: dict[int, dict[int, str]] = {}
    for index, job in enumerate(job_ids):
        assignment[job] = {}
        for op in (1, 2):
            room = cycles[op - 1][index]
            options = table.get((job, op), list(rooms))
            assignment[job][op] = room if room in options else choice(options)
    return assignment
