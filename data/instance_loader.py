import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from data.instance_model import InstanceContext, Job, Operation, canonical_digest
_KEYS = {"schema_version", "instance_id", "family", "classification", "provenance", "generation", "dimensions", "resources", "jobs", "bounds", "validation", "digest"}
_PROFILE_KEYS = {"duration_distributions", "dependence", "assignment", "room_eligibility", "congestion_bands", "replica_policy"}
@dataclass(frozen=True, slots=True)
class ValidationIssue:
    stage: str
    code: str
    field: str
    message: str
class InstanceValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{item.field}: {item.message}" for item in issues))
def _issue(issues, stage, code, field, message):
    issues.append(ValidationIssue(stage, code, field, message))
def validate_document(document: Any) -> InstanceContext:
    issues: list[ValidationIssue] = []
    if not isinstance(document, dict):
        raise InstanceValidationError([ValidationIssue("schema", "document", "$", "expected a mapping")])
    missing = _KEYS - document.keys()
    if missing:
        _issue(issues, "schema", "missing_required", "$", f"missing {sorted(missing)}")
    if document.get("schema_version") != 1:
        _issue(issues, "schema", "unsupported_version", "schema_version", "expected version 1")
    structures = ("generation", "dimensions", "resources", "bounds", "validation")
    for field in structures:
        if field in document and not isinstance(document[field], dict):
            _issue(issues, "schema", "structure", field, "expected a mapping")
    if "jobs" in document and not isinstance(document["jobs"], list):
        _issue(issues, "schema", "structure", "jobs", "expected a list")
    if issues:
        raise InstanceValidationError(issues)

    resources, dimensions, jobs = document["resources"], document["dimensions"], document["jobs"]
    rooms, personnel = resources.get("rooms", []), resources.get("personnel", {})
    if not rooms or len(set(rooms)) != len(rooms) or any(not str(item).strip() for item in rooms):
        _issue(issues, "dimensions", "resource_ids", "resources.rooms", "room IDs must be unique and non-empty")
    people = [item for group in personnel.values() for item in group]
    if set(personnel) != {"1", "2"} or not people or len(set(people)) != len(people) or any(not str(item).strip() for item in people):
        _issue(issues, "dimensions", "personnel_ids", "resources.personnel", "qualified personnel IDs must be grouped, unique, and non-empty")
    if dimensions.get("jobs") != len(jobs):
        _issue(issues, "dimensions", "job_count", "dimensions.jobs", "must match jobs")
    if dimensions.get("operations_per_job") != 2:
        _issue(issues, "dimensions", "operation_count", "dimensions.operations_per_job", "expected 2")
    if [job.get("id") for job in jobs] != list(range(1, len(jobs) + 1)):
        _issue(issues, "dimensions", "job_ids", "jobs", "job IDs must be contiguous")

    for index, job in enumerate(jobs):
        operations = job.get("operations", [])
        if [op.get("id") for op in operations] != [1, 2]:
            _issue(issues, "dimensions", "operation_ids", f"jobs[{index}].operations", "expected operations 1 and 2")
            continue
        for op in operations:
            base = f"jobs[{index}].operations[{op['id'] - 1}]"
            values = (op.get("duration"), op.get("setup"), op.get("transition"), op.get("cleanup"), op.get("max_wait"))
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
                _issue(issues, "dimensions", "finite_timing", base, "timings must be finite")
            elif values[0] <= 0:
                _issue(issues, "dimensions", "positive_duration", f"{base}.duration", "must be positive")
            elif any(value < 0 for value in values[1:]):
                _issue(issues, "dimensions", "nonnegative_timing", base, "timings must be non-negative")

            eligible_rooms = op.get("eligible_rooms", [])
            eligible_people = op.get("eligible_personnel", [])
            group = personnel.get(str(op["id"]), [])
            if not eligible_rooms or not set(eligible_rooms) <= set(rooms):
                _issue(issues, "references", "room_reference", f"{base}.eligible_rooms", "must reference declared rooms")
            elif set(eligible_rooms) != set(rooms):
                _issue(issues, "references", "all_rooms_policy", f"{base}.eligible_rooms", "must list all declared rooms")
            if not eligible_people or not set(eligible_people) <= set(group):
                _issue(issues, "references", "personnel_reference", f"{base}.eligible_personnel", "must reference qualified personnel")

    profile = document["generation"].get("profile", {})
    if not _PROFILE_KEYS <= profile.keys():
        _issue(issues, "evidence", "generation_profile", "generation.profile", "generation evidence is incomplete")
    if not {"method", "version", "seed", "profile"} <= document["generation"].keys():
        _issue(issues, "evidence", "generation_evidence", "generation", "method, version, seed, and profile are required")
    if not {"method", "version", "outcome"} <= document["validation"].keys() or document["validation"].get("outcome") != "passed":
        _issue(issues, "evidence", "validation_evidence", "validation", "validation evidence must pass")
    if document["classification"] == "replica":
        _issue(issues, "evidence", "replica_forbidden", "classification", "authorized record-level evidence is absent")
    elif document["classification"] not in {"fully synthetic instance", "calibrated synthetic instance"}:
        _issue(issues, "evidence", "classification", "classification", "classification is unsupported")
    if document["bounds"].get("status") == "verified":
        _issue(issues, "evidence", "verified_bounds", "bounds.status", "verification evidence is incomplete")
    elif document["bounds"].get("status") not in {"pending", "heuristic"}:
        _issue(issues, "evidence", "bounds_status", "bounds.status", "expected pending or heuristic")
    try:
        digest = canonical_digest(document)
    except (TypeError, ValueError):
        digest = ""
        _issue(issues, "evidence", "canonical_content", "digest", "content is not canonical finite data")
    if document.get("digest") != digest:
        _issue(issues, "evidence", "digest_mismatch", "digest", "does not match canonical content")
    if issues:
        raise InstanceValidationError(issues)

    def operation(raw):
        return Operation(raw["id"], *map(float, (raw["duration"], raw["setup"], raw["transition"], raw["cleanup"], raw["max_wait"])), tuple(raw["eligible_rooms"]), tuple(raw["eligible_personnel"]))

    return InstanceContext(1, document["instance_id"], document["family"], document["classification"], int(document["generation"]["seed"]), tuple(rooms), tuple((op, tuple(personnel[str(op)])) for op in (1, 2)), tuple(Job(job["id"], job.get("label", ""), tuple(operation(op) for op in job["operations"])) for job in jobs), digest)
def load_instance(path: str | Path) -> InstanceContext:
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise InstanceValidationError([ValidationIssue("parse", "file", "$", str(error))]) from error
    except yaml.YAMLError as error:
        raise InstanceValidationError([ValidationIssue("parse", "yaml", "$", str(error))]) from error
    return validate_document(document)
def load_catalog(root: str | Path) -> tuple[InstanceContext, ...]:
    root = Path(root)
    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    entries = metadata.get("instances", []) if isinstance(metadata, dict) else []
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1 or metadata.get("instance_count") != len(entries):
        raise InstanceValidationError([ValidationIssue("evidence", "metadata", "metadata.yaml", "catalog metadata is inconsistent")])
    return tuple(load_instance(root / entry["file"]) for entry in entries)
