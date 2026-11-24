"""Decode and evaluate a candidate schedule for one instance.

There is a single scheduling model, used by every algorithm, by the reports and
by the typed runtime:

* A candidate is a priority sequence of jobs plus a room per operation (and,
  optionally, a person per operation).
* A list scheduler dispatches, at each step, the operation that can start
  first. Ties go to surgeries and then follow the priority sequence.
* Each operation occupies its room and person from the start of preparation
  (``max(transition, setup)``) through processing and cleanup.
* Blocking: an anesthetized patient stays in the anesthesia room with the
  anesthetist until the surgery starts. The anesthesia room is cleaned when the
  patient leaves it; if the surgery uses the same room, the patient never
  leaves and the room passes from anesthesia to surgery after the anesthesia
  cleanup. Dispatches that would close a cycle of patients waiting for each
  other's rooms are postponed, so the decoder cannot deadlock.
* The wait of a job is the time between the end of anesthesia processing and
  the start of surgery processing. Waits above the instance limit are not
  infeasible; they are penalized in the objective.

The objective is

    f = Cmax + alpha * sum(S) + beta * sum(W) + gamma * max(W) + delta * Psi
        + omega * sum(max(0, W - W_limit))

with the weights of ``config/config.yaml``.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

from data.instance_model import InstanceContext


class IneligibleAssignmentError(ValueError):
    """Raised when a candidate names a resource outside operation eligibility."""


class _Infeasible(Exception):
    """Internal signal: the candidate cannot be decoded into a schedule."""


@dataclass
class _Decoded:
    makespan: float
    details: list[dict[str, Any]]
    total_start: float
    total_wait: float
    max_wait: float
    excess_wait: float
    wait_violations: int
    imbalance: float = 0.0
    waits: dict[int, float] = field(default_factory=dict)


def _as_dict(solution: Any) -> dict[str, Any] | None:
    if hasattr(solution, "to_dict"):
        solution = solution.to_dict()
    elif hasattr(solution, "job_sequence_base"):
        solution = {
            "job_sequence_base": solution.job_sequence_base,
            "room_assignment": solution.room_assignment,
            "personnel_assignment": getattr(solution, "personnel_assignment", None),
        }
    return solution if isinstance(solution, dict) else None


def _decode(context: InstanceContext, solution: Any, strict: bool) -> _Decoded:
    """Dispatch one candidate. Raises ``_Infeasible`` or, if ``strict``,
    ``IneligibleAssignmentError`` / ``ValueError`` with a message."""
    solution = _as_dict(solution)
    if solution is None:
        raise _Infeasible("solution is not a mapping")

    job_ids = [job.job_id for job in context.jobs]
    raw_sequence = list(solution.get("job_sequence_base", ()))
    if set(raw_sequence) != set(job_ids):
        if strict:
            raise ValueError("job_sequence_base must contain every instance job")
        raise _Infeasible("incomplete sequence")
    # Historical candidates may repeat jobs; the first occurrence sets priority.
    sequence = list(dict.fromkeys(raw_sequence))
    rooms_by_job = solution.get("room_assignment", {}) or {}
    people_by_job = solution.get("personnel_assignment", {}) or {}

    operations = {}
    chosen_room: dict[tuple[int, int], str] = {}
    chosen_person: dict[tuple[int, int], str | None] = {}
    for job in context.jobs:
        if len(job.operations) != 2:
            raise _Infeasible("every job must have anesthesia and surgery")
        assigned_rooms = rooms_by_job.get(job.job_id, {}) or {}
        assigned_people = people_by_job.get(job.job_id, {}) or {}
        for operation in job.operations:
            key = (job.job_id, operation.operation_id)
            operations[key] = operation
            room = assigned_rooms.get(operation.operation_id)
            if room not in operation.eligible_rooms:
                if strict:
                    raise IneligibleAssignmentError(
                        f"job {job.job_id} operation {operation.operation_id}: "
                        f"ineligible room {room!r}"
                    )
                raise _Infeasible("ineligible room")
            chosen_room[key] = room
            person = assigned_people.get(operation.operation_id)
            if person is not None and person not in operation.eligible_personnel:
                if strict:
                    raise IneligibleAssignmentError(
                        f"job {job.job_id} operation {operation.operation_id}: "
                        f"ineligible personnel {person!r}"
                    )
                raise _Infeasible("ineligible personnel")
            if not operation.eligible_personnel:
                raise _Infeasible("operation without eligible personnel")
            chosen_person[key] = person

    inf = math.inf
    room_release = {room: 0.0 for room in context.rooms}
    room_holder: dict[str, int | None] = {room: None for room in context.rooms}
    person_release = {
        person: 0.0 for _, people in context.personnel_by_operation for person in people
    }
    for operation in operations.values():
        for person in operation.eligible_personnel:
            person_release.setdefault(person, 0.0)

    priority = {job_id: index for index, job_id in enumerate(sequence)}
    counter = {job_id: index for index, job_id in enumerate(job_ids)}
    anesthesia: dict[int, dict[str, Any]] = {}
    details: list[dict[str, Any]] = []
    waits: dict[int, float] = {}

    queue: list[tuple[float, int, int, int, int]] = [
        (0.0, priority[job_id], counter[job_id], job_id, 1) for job_id in job_ids
    ]
    heapq.heapify(queue)
    # Operations that wait for a resource held by a waiting patient, or whose
    # dispatch would close a cycle. Only a surgery dispatch can change that,
    # so they return to the queue after each surgery dispatch.
    parked: list[tuple[float, int, int, int, int]] = []

    def pick_person(key: tuple[int, int]) -> tuple[str, float]:
        fixed = chosen_person[key]
        if fixed is not None:
            return fixed, person_release[fixed]
        person = min(operations[key].eligible_personnel, key=lambda p: (person_release[p], p))
        return person, person_release[person]

    def closes_cycle(anesthesia_room: str, surgery_room: str) -> bool:
        """Would a patient waiting in ``anesthesia_room`` for ``surgery_room``
        close a cycle of patients waiting for each other's rooms?"""
        room, seen = surgery_room, set()
        while room_holder.get(room) is not None and room not in seen:
            seen.add(room)
            room = anesthesia[room_holder[room]]["wanted"]
            if room == anesthesia_room:
                return True
        return False

    while queue or parked:
        if not queue:
            raise _Infeasible("deadlock: no pending operation can start")
        key, prio, count, job_id, op_id = heapq.heappop(queue)
        operation = operations[(job_id, op_id)]
        room = chosen_room[(job_id, op_id)]
        person, person_ready = pick_person((job_id, op_id))

        if op_id == 1:
            ready = max(room_release[room], person_ready)
        else:
            first = anesthesia[job_id]
            room_ready = first["finish"] if room == first["room"] else room_release[room]
            ready = max(room_ready, person_ready, first["processing_end"])
        if ready == inf:
            parked.append((key, prio, count, job_id, op_id))
            continue
        if ready > key + 1e-9:
            # The stored ready time is stale: other operations took the
            # resources meanwhile. Reinsert with the current ready time.
            heapq.heappush(queue, (ready, prio, count, job_id, op_id))
            continue

        surgery_room = chosen_room[(job_id, 2)]
        if op_id == 1 and surgery_room != room and closes_cycle(room, surgery_room):
            # Starting now could deadlock: wait for the next surgery dispatch.
            parked.append((key, prio, count, job_id, op_id))
            continue

        setup_start = ready
        start = setup_start + max(operation.transition, operation.setup)
        processing_end = start + operation.duration
        finish = processing_end + operation.cleanup
        detail = {
            "Job": job_id,
            "Operation": op_id,
            "Resource": room,
            "Personnel": person,
            "Start": setup_start,
            "ProcessingEnd": processing_end,
            "Finish": finish,
            "SetupUsed": operation.setup,
            "TransitionUsed": operation.transition,
            "CleanupUsed": operation.cleanup,
        }
        # The keys and their order are the historical CSV contract. Waits are
        # available through ``evaluate_solution(...).waits``.
        details.append(detail)

        if op_id == 1:
            # Blocking: the room and the anesthetist stay with the patient
            # until the surgery starts.
            room_release[room] = inf
            room_holder[room] = job_id if surgery_room != room else None
            person_release[person] = inf
            anesthesia[job_id] = {
                "room": room, "person": person, "processing_end": processing_end,
                "finish": finish, "cleanup": operation.cleanup, "wanted": surgery_room,
                "detail": detail,
            }
            heapq.heappush(
                queue, (processing_end, -1, counter[job_id], job_id, 2)
            )
        else:
            first = anesthesia[job_id]
            wait = max(0.0, start - first["processing_end"])
            waits[job_id] = wait
            if room != first["room"]:
                # The patient leaves the anesthesia room now; it is cleaned.
                room_release[first["room"]] = setup_start + first["cleanup"]
                room_holder[first["room"]] = None
                first["detail"]["Finish"] = setup_start + first["cleanup"]
            else:
                first["detail"]["Finish"] = setup_start
            person_release[first["person"]] = max(first["finish"], setup_start)
            room_release[room] = finish
            person_release[person] = finish
            # Resources held by this patient are free again: parked operations
            # return to the queue (their ready time is at least now).
            for item in parked:
                heapq.heappush(queue, (setup_start,) + item[1:])
            parked.clear()

    makespan = max((d["Finish"] for d in details if d["Operation"] == 2), default=0.0)
    limits = {job.job_id: job.operations[1].max_wait for job in context.jobs}
    excess = {job_id: max(0.0, wait - limits[job_id]) for job_id, wait in waits.items()}

    usage = {room: 0 for room in context.rooms}
    for item in details:
        usage[item["Resource"]] += 1
    counts = list(usage.values())
    total_usage = sum(counts)
    if total_usage and counts:
        average = total_usage / len(counts)
        variance = sum((value - average) ** 2 for value in counts) / len(counts)
        unused = sum(value == 0 for value in counts)
        imbalance = variance**0.5 / average + unused * (unused + 1)
    else:
        imbalance = 0.0

    return _Decoded(
        makespan=makespan,
        details=details,
        total_start=sum(item["Start"] for item in details),
        total_wait=sum(waits.values()),
        max_wait=max(waits.values(), default=0.0),
        excess_wait=sum(excess.values()),
        wait_violations=sum(value > 1e-9 for value in excess.values()),
        imbalance=imbalance,
        waits=waits,
    )


def schedule_instance_solution(
    context: InstanceContext, solution: Any
) -> tuple[float, tuple]:
    """Decode one candidate into typed schedule entries.

    Uses the same model as ``calculate_schedule_fitness``. Ineligible rooms or
    personnel raise ``IneligibleAssignmentError``; an incomplete sequence or an
    undecodable candidate raises ``ValueError``.
    """
    from simulation.result_model import ScheduleEntry

    try:
        decoded = _decode(context, solution, strict=True)
    except _Infeasible as error:
        raise ValueError(str(error)) from error
    schedule = tuple(
        ScheduleEntry(
            item["Job"], item["Operation"], item["Resource"], item["Personnel"],
            item["Start"], item["ProcessingEnd"], item["Finish"],
            item["SetupUsed"], item["TransitionUsed"], item["CleanupUsed"],
        )
        for item in sorted(decoded.details, key=lambda d: (d["Job"], d["Operation"]))
    )
    return decoded.makespan, schedule


def _legacy_context(surgeries_data: Any) -> InstanceContext:
    """Build a context for callers that still pass the historical data map."""
    from data.instance_model import Job, Operation
    from config.config import (
        ALL_ROOMS,
        CLEANUP_TIMES,
        MAX_WAIT_TIMES,
        PERSONNEL_BY_OPERATION,
        SETUP_TIMES,
    )

    job_ids = sorted(int(job_id) for job_id in surgeries_data)
    jobs = []
    for job_id in job_ids:
        raw = surgeries_data[job_id]
        operations = []
        for operation_id in (1, 2):
            setup_by_op = raw.get("setup_by_op", {})
            transition_by_op = raw.get("transition_by_op", {})
            cleanup_by_op = raw.get("cleanup_by_op", {})
            operations.append(
                Operation(
                    operation_id=operation_id,
                    duration=float(raw[operation_id]),
                    setup=float(setup_by_op.get(operation_id, SETUP_TIMES.get(operation_id, 0.0))),
                    transition=float(transition_by_op.get(operation_id, 0.0)),
                    cleanup=float(cleanup_by_op.get(operation_id, CLEANUP_TIMES.get(operation_id, 0.0))),
                    max_wait=float(raw.get("max_wait_by_op", {}).get(operation_id, MAX_WAIT_TIMES.get(operation_id, 10_000.0))),
                    eligible_rooms=tuple(ALL_ROOMS),
                    eligible_personnel=tuple(PERSONNEL_BY_OPERATION.get(operation_id, ())),
                )
            )
        jobs.append(Job(job_id=job_id, label="", operations=tuple(operations)))
    return InstanceContext(
        schema_version=1,
        instance_id="legacy-map",
        family="legacy-compatibility",
        classification="fully synthetic instance",
        generation_seed=0,
        rooms=tuple(ALL_ROOMS),
        personnel_by_operation=tuple(
            (operation_id, tuple(people))
            for operation_id, people in sorted(PERSONNEL_BY_OPERATION.items())
        ),
        jobs=tuple(jobs),
        digest="0" * 64,
    )


def evaluate_solution(context: InstanceContext, solution: Any) -> _Decoded | None:
    """Decode a candidate and return its components, or ``None`` if infeasible."""
    try:
        return _decode(context, solution, strict=False)
    except _Infeasible:
        return None


def objective_from(decoded: _Decoded) -> float:
    """Composite objective of a decoded schedule."""
    from config.config import ALPHA, BETA, DELTA, GAMMA, OMEGA

    return (
        decoded.makespan
        + ALPHA * decoded.total_start
        + BETA * decoded.total_wait
        + GAMMA * decoded.max_wait
        + DELTA * decoded.imbalance
        + OMEGA * decoded.excess_wait
    )


def calculate_schedule_fitness(
    solution: Any, surgeries_data: Any, return_details: bool = False
) -> float | tuple[float, float, list[dict[str, Any]]]:
    """Composite objective of a candidate (``inf`` if it cannot be decoded).

    With ``return_details`` it returns ``(objective, makespan, details)``.
    """
    context = getattr(surgeries_data, "context", None)
    if context is None:
        context = _legacy_context(surgeries_data)
    decoded = evaluate_solution(context, solution)
    if decoded is None:
        return (math.inf, math.inf, []) if return_details else math.inf
    objective = objective_from(decoded)
    return (objective, decoded.makespan, decoded.details) if return_details else objective


__all__ = [
    "IneligibleAssignmentError",
    "calculate_schedule_fitness",
    "evaluate_solution",
    "objective_from",
    "schedule_instance_solution",
]
