import hashlib
import json
from dataclasses import dataclass
from typing import Any
def canonical_digest(document: dict[str, Any]) -> str:
    """Hash canonical parsed content, excluding its top-level digest."""
    content = {key: value for key, value in document.items() if key != "digest"}
    encoded = json.dumps(
        content, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
@dataclass(frozen=True, slots=True)
class Operation:
    operation_id: int
    duration: float
    setup: float
    transition: float
    cleanup: float
    max_wait: float
    eligible_rooms: tuple[str, ...]
    eligible_personnel: tuple[str, ...]
@dataclass(frozen=True, slots=True)
class Job:
    job_id: int
    label: str
    operations: tuple[Operation, ...]
@dataclass(frozen=True, slots=True)
class InstanceContext:
    schema_version: int
    instance_id: str
    family: str
    classification: str
    generation_seed: int
    rooms: tuple[str, ...]
    personnel_by_operation: tuple[tuple[int, tuple[str, ...]], ...]
    jobs: tuple[Job, ...]
    digest: str
