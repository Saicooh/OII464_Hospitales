"""Decode and validate a candidate schedule for one instance."""

from typing import Any

from data.instance_model import InstanceContext


class IneligibleAssignmentError(ValueError):
    """Raised when a candidate names a resource outside operation eligibility."""


def schedule_instance_solution(
    context: InstanceContext, solution: Any
) -> tuple[float, tuple]:
    """Decode one candidate using only the selected instance's immutable data."""
    from simulation.result_model import ScheduleEntry

    job_ids = [job.job_id for job in context.jobs]
    if hasattr(solution, "job_sequence_base"):
        sequence = list(solution.job_sequence_base)
        room_assignments = solution.room_assignment
        personnel_assignments = solution.personnel_assignment or {}
    else:
        sequence = list(solution.get("job_sequence_base", ()))
        room_assignments = solution.get("room_assignment", {})
        personnel_assignments = solution.get("personnel_assignment", {})
    # The historical GA encoded the base sequence twice.  Keep accepting that
    # representation at the compatibility boundary while retaining a strict
    # one-job-per-instance schedule internally.
    if len(sequence) < len(job_ids) or set(sequence) != set(job_ids):
        raise ValueError("job_sequence_base must contain every instance job exactly once")
    sequence = list(dict.fromkeys(sequence))

    room_release = {room: 0.0 for room in context.rooms}
    personnel_release = {
        person: 0.0
        for _, people in context.personnel_by_operation
        for person in people
    }
    jobs = {job.job_id: job for job in context.jobs}
    schedule: list[ScheduleEntry] = []

    for job_id in sequence:
        job = jobs[job_id]
        assignments = room_assignments.get(job_id, {})
        people = personnel_assignments.get(job_id, {})
        resolved: list[tuple[str, str]] = []
        for operation in job.operations:
            room = assignments.get(operation.operation_id)
            if room not in operation.eligible_rooms:
                raise IneligibleAssignmentError(
                    f"job {job_id} operation {operation.operation_id}: "
                    f"ineligible room {room!r}"
                )
            person = people.get(operation.operation_id)
            if person is None:
                person = min(
                    operation.eligible_personnel,
                    key=lambda item: (personnel_release[item], item),
                )
            if person not in operation.eligible_personnel:
                raise IneligibleAssignmentError(
                    f"job {job_id} operation {operation.operation_id}: "
                    f"ineligible personnel {person!r}"
                )
            resolved.append((room, person))

        anesthesia, surgery = job.operations
        room_1, person_1 = resolved[0]
        setup_start = max(room_release[room_1], personnel_release[person_1])
        anesthesia_start = setup_start + max(anesthesia.transition, anesthesia.setup)
        anesthesia_end = anesthesia_start + anesthesia.duration
        anesthesia_finish = anesthesia_end + anesthesia.cleanup

        room_2, person_2 = resolved[1]
        surgery_setup_start = max(
            anesthesia_finish, room_release[room_2], personnel_release[person_2]
        )
        surgery_start = surgery_setup_start + max(surgery.transition, surgery.setup)
        wait = surgery_start - anesthesia_finish
        if wait > surgery.max_wait:
            raise ValueError(
                f"job {job_id} operation 2 wait {wait} exceeds {surgery.max_wait}"
            )
        surgery_end = surgery_start + surgery.duration
        finish = surgery_end + surgery.cleanup

        schedule.extend(
            (
                ScheduleEntry(
                    job_id,
                    1,
                    room_1,
                    person_1,
                    setup_start,
                    anesthesia_end,
                    anesthesia_finish,
                    anesthesia.setup,
                    anesthesia.transition,
                    anesthesia.cleanup,
                ),
                ScheduleEntry(
                    job_id,
                    2,
                    room_2,
                    person_2,
                    surgery_setup_start,
                    surgery_end,
                    finish,
                    surgery.setup,
                    surgery.transition,
                    surgery.cleanup,
                ),
            )
        )
        room_release[room_1] = finish
        room_release[room_2] = finish
        personnel_release[person_1] = surgery_start
        personnel_release[person_2] = finish

    makespan = max((entry.finish for entry in schedule), default=0.0)
    return makespan, tuple(schedule)


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


def _fitness_for_context(
    context: InstanceContext, solution: Any, return_details: bool = False
) -> float | tuple[float, float, list[dict[str, Any]]]:
    """Historical priority-queue scheduler parameterized by one instance."""
    import heapq
    from config.config import ALPHA, BETA, DELTA, GAMMA, MAX_WAIT_TIMES

    if hasattr(solution, "to_dict"):
        solution = solution.to_dict()
    if not isinstance(solution, dict):
        return (float("inf"), float("inf"), []) if return_details else float("inf")

    job_ids = [job.job_id for job in context.jobs]
    sequence = list(solution.get("job_sequence_base", ()))
    assignments = solution.get("room_assignment", {})
    if len(sequence) < len(job_ids) or set(sequence) != set(job_ids):
        return (float("inf"), float("inf"), []) if return_details else float("inf")
    if set(assignments) != set(job_ids):
        return (float("inf"), float("inf"), []) if return_details else float("inf")

    operations = {
        (job.job_id, operation.operation_id): operation
        for job in context.jobs
        for operation in job.operations
    }
    personnel_release = {
        person: 0.0
        for _, people in context.personnel_by_operation
        for person in people
    }
    room_release = {room: 0.0 for room in context.rooms}
    job_priority = {job_id: index for index, job_id in enumerate(sequence)}
    job_counter = {job_id: index for index, job_id in enumerate(job_ids)}
    job_processing_end = {job_id: {0: 0.0} for job_id in job_ids}
    job_used_resources: dict[int, dict[int, tuple[str | None, str | None]]] = {
        job_id: {0: (None, None)} for job_id in job_ids
    }
    next_operation = {job_id: 1 for job_id in job_ids}
    queue: list[tuple[float, int, int, int, int]] = []
    details: list[dict[str, Any]] = []

    for job_id in job_ids:
        operation = operations[(job_id, 1)]
        room = assignments[job_id].get(1)
        if room not in operation.eligible_rooms:
            return (float("inf"), float("inf"), []) if return_details else float("inf")
        earliest_person = min(
            (personnel_release[person] for person in operation.eligible_personnel),
            default=float("inf"),
        )
        heapq.heappush(
            queue,
            (
                max(room_release[room], earliest_person),
                job_priority[job_id],
                job_counter[job_id],
                job_id,
                1,
            ),
        )

    total_inter_operation_wait = 0.0
    max_inter_operation_wait = 0.0
    operations_done = 0
    total_operations = 2 * len(job_ids)

    while operations_done < total_operations:
        if not queue:
            return (float("inf"), float("inf"), []) if return_details else float("inf")
        _, _, _, job_id, operation_id = heapq.heappop(queue)
        operation = operations[(job_id, operation_id)]
        room = assignments[job_id].get(operation_id)
        if room not in operation.eligible_rooms:
            return (float("inf"), float("inf"), []) if return_details else float("inf")
        personnel = min(
            operation.eligible_personnel,
            key=lambda person: (personnel_release[person], person),
        )
        previous_end = job_processing_end[job_id][operation_id - 1]
        setup_start = max(room_release[room], personnel_release[personnel])
        if operation_id > 1:
            setup_start = max(setup_start, previous_end)
            actual_start = setup_start + operation.transition + operation.setup
        else:
            actual_start = setup_start + max(operation.transition, operation.setup)
        wait = max(0.0, actual_start - previous_end) if operation_id > 1 else 0.0
        # The historical runner was configured with a legacy feasibility
        # budget (500 minutes in the previous repository).  Synthetic YAML
        # instances can contain tighter clinical wait metadata, but applying
        # it directly here makes the old dispatch order reject every
        # candidate on large instances.  Keep the larger configured legacy
        # budget for this compatibility evaluator; the typed scheduler above
        # still enforces the instance's declared max_wait.
        historical_wait_limit = max(
            float(operation.max_wait),
            float(MAX_WAIT_TIMES.get(operation_id, operation.max_wait)),
        )
        if wait > historical_wait_limit:
            return (float("inf"), float("inf"), []) if return_details else float("inf")
        total_inter_operation_wait += wait
        max_inter_operation_wait = max(max_inter_operation_wait, wait)
        processing_end = actual_start + operation.duration
        finish = processing_end + operation.cleanup

        job_processing_end[job_id][operation_id] = processing_end
        job_used_resources[job_id][operation_id] = (room, personnel)
        details.append(
            {
                "Job": job_id,
                "Operation": operation_id,
                "Resource": room,
                "Personnel": personnel,
                "Start": setup_start,
                "ProcessingEnd": processing_end,
                "Finish": finish,
                "SetupUsed": operation.setup,
                "TransitionUsed": operation.transition,
                "CleanupUsed": operation.cleanup,
            }
        )

        room_release[room] = finish
        personnel_release[personnel] = finish
        if operation_id > 1:
            previous_room, previous_personnel = job_used_resources[job_id][operation_id - 1]
            if previous_room is not None:
                room_release[previous_room] = max(room_release[previous_room], setup_start)
            if previous_personnel is not None:
                personnel_release[previous_personnel] = max(
                    personnel_release[previous_personnel], setup_start
                )

        operations_done += 1
        next_operation[job_id] += 1
        following = next_operation[job_id]
        if following <= 2:
            next_data = operations[(job_id, following)]
            next_room = assignments[job_id].get(following)
            if next_room not in next_data.eligible_rooms:
                return (float("inf"), float("inf"), []) if return_details else float("inf")
            next_personnel_release = min(
                (personnel_release[person] for person in next_data.eligible_personnel),
                default=float("inf"),
            )
            priority = -1 if following > 1 else job_priority[job_id]
            heapq.heappush(
                queue,
                (
                    max(job_processing_end[job_id][operation_id], room_release[next_room], next_personnel_release),
                    priority,
                    job_counter[job_id],
                    job_id,
                    following,
                ),
            )

    final_makespan = max(
        (job_processing_end[job_id][2] + operations[(job_id, 2)].cleanup for job_id in job_ids),
        default=0.0,
    )
    room_usage = {room: 0 for room in context.rooms}
    for job_id in job_ids:
        for operation_id in (1, 2):
            room = job_used_resources[job_id][operation_id][0]
            if room is not None:
                room_usage[room] += 1
    usages = list(room_usage.values())
    total_usage = sum(usages)
    if total_usage and usages:
        average_usage = total_usage / len(usages)
        variance = sum((value - average_usage) ** 2 for value in usages) / len(usages)
        unused = sum(value == 0 for value in usages)
        imbalance = variance**0.5 / average_usage + unused * (unused + 1)
    else:
        imbalance = 0.0
    total_start = sum(
        item["Start"]
        for item in details
    )
    objective = (
        final_makespan
        + ALPHA * total_start
        + BETA * total_inter_operation_wait
        + GAMMA * max_inter_operation_wait
        + DELTA * imbalance
    )
    return (objective, final_makespan, details) if return_details else objective


def calculate_schedule_fitness(
    solution: Any, surgeries_data: Any, return_details: bool = False
) -> float | tuple[float, float, list[dict[str, Any]]]:
    """Evaluate legacy algorithm candidates using selected synthetic data."""
    context = getattr(surgeries_data, "context", None)
    if context is None:
        context = _legacy_context(surgeries_data)
    return _fitness_for_context(context, solution, return_details=return_details)


__all__ = [
    "IneligibleAssignmentError",
    "calculate_schedule_fitness",
    "schedule_instance_solution",
]
