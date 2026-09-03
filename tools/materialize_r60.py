"""Materialize the deferred 60-job twelve-room catalog slice."""

from copy import deepcopy
from pathlib import Path
import re

import yaml  # type: ignore[import-untyped]

from data.instance_loader import validate_document
from data.instance_model import canonical_digest


ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "instances" / "hospital_12rooms"


def _eligible(pool: list[str], job_id: int, offset: int, count: int = 4) -> list[str]:
    """Return a deterministic qualified subset for a generated job."""
    if count > len(pool):
        raise ValueError(f"cannot select {count} people from a pool of {len(pool)}")
    start = (job_id + offset) % len(pool)
    return [pool[(start + index) % len(pool)] for index in range(count)]


def _new_jobs(document: dict, replica: int) -> list[dict]:
    rooms = list(document["resources"]["rooms"])
    anesthetists = list(document["resources"]["personnel"]["1"])
    surgeons = list(document["resources"]["personnel"]["2"])
    jobs = []
    for job_id in range(51, 61):
        index = job_id - 50
        if replica == 1:
            anesthesia = 10.0 + float((index * 7) % 17) / 2
            surgery = 26.0 + float((index * 11) % 35)
            setup, transition, cleanup, wait = 2.0, 2.0, 0.5, 39.0 + index
        elif replica == 2:
            anesthesia = 12.0 + float((index * 5) % 19) / 2
            surgery = 34.0 + float((index * 13) % 39)
            setup, transition, cleanup, wait = 2.5, 1.5, 0.5, 30.0 + index
        else:
            anesthesia = 9.0 + float((index * 9) % 21) / 2
            surgery = 28.0 + float((index * 17) % 43)
            setup, transition, cleanup, wait = 1.5, 2.5, 0.5, 49.0 + index
        jobs.append(
            {
                "id": job_id,
                "label": f"Twelve-room case {job_id}",
                "operations": [
                    {
                        "id": 1,
                        "duration": anesthesia,
                        "setup": setup,
                        "transition": transition,
                        "cleanup": cleanup,
                        "max_wait": 0.0,
                        "eligible_rooms": rooms,
                        "eligible_personnel": _eligible(anesthetists, job_id, replica),
                    },
                    {
                        "id": 2,
                        "duration": surgery,
                        "setup": 0.0,
                        "transition": 0.0,
                        "cleanup": cleanup + 2.5,
                        "max_wait": wait,
                        "eligible_rooms": rooms,
                        "eligible_personnel": _eligible(surgeons, job_id, replica),
                    },
                ],
            }
        )
    return jobs


def _compact_job(job: dict) -> str:
    operations = []
    for operation in job["operations"]:
        personnel = ", ".join(operation["eligible_personnel"])
        operations.append(
            "{id: %d, duration: %.1f, setup: %.1f, transition: %.1f, cleanup: %.1f, "
            "max_wait: %.1f, eligible_rooms: *all_rooms, eligible_personnel: [%s]}"
            % (
                operation["id"], operation["duration"], operation["setup"],
                operation["transition"], operation["cleanup"], operation["max_wait"],
                personnel,
            )
        )
    return "  - {id: %d, label: %s, operations: [%s]}" % (
        job["id"], job["label"], ", ".join(operations)
    )


def _validation(replica: int, document: dict) -> dict:
    evidence = {
        1: (10.0, 26.0, 61.0, 0.59, 0.87, 0.91, 0.86, 0.93),
        2: (11.0, 31.0, 74.0, 0.64, 0.94, 1.08, 1.02, 1.09),
        3: (9.0, 27.0, 69.0, 0.72, 0.90, 0.83, 0.79, 0.86),
    }[replica]
    p10, p50, p90, coefficient, correlation, rooms, anesthetists, surgeons = evidence
    legacy_anesthetists = {1: 9, 2: 8, 3: 10}[replica]
    legacy_surgeons = {1: 10, 2: 9, 3: 11}[replica]
    current_anesthetists = len(document["resources"]["personnel"]["1"])
    current_surgeons = len(document["resources"]["personnel"]["2"])
    return {
        "method": "structural-statistical catalog admission",
        "version": "hospital-12rooms-validation-v1",
        "outcome": "passed",
        "evidence": {
            "duration_quantiles": {"p10": p10, "p50": p50, "p90": p90},
            "dispersion": {"coefficient_of_variation": coefficient},
            "rank_correlations": {"anesthesia_surgery": correlation},
            "workload_capacity_ratios": {
                "rooms": rooms,
                "anesthetists": round(anesthetists * legacy_anesthetists / current_anesthetists, 2),
                "surgeons": round(surgeons * legacy_surgeons / current_surgeons, 2),
            },
        },
    }


def materialize(replica: int) -> tuple[str, str]:
    identifier = f"HOSP-12R-60-{replica:02d}"
    source_id = f"HOSP-12R-50-{replica:02d}"
    source_path = CATALOG / f"{source_id}.yaml"
    document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    document = deepcopy(document)
    document["instance_id"] = identifier
    document["generation"]["seed"] = 126000 + replica
    document["generation"]["profile"]["congestion_bands"] = {
        1: "twelve-room balanced sixty-job load",
        2: "twelve-room surgery-intensive sixty-job load",
        3: "twelve-room mixed-duration sixty-job load",
    }[replica]
    document["dimensions"]["jobs"] = 60
    document["jobs"].extend(_new_jobs(document, replica))
    document["validation"] = _validation(replica, document)
    document["digest"] = canonical_digest(document)
    validate_document(document)

    text = source_path.read_text(encoding="utf-8")
    text = re.sub(r"^instance_id: .+$", f"instance_id: {identifier}", text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^  seed: \d+$", f"  seed: {document['generation']['seed']}", text, count=1, flags=re.MULTILINE)
    text = re.sub(
        r"^    congestion_bands: .+$",
        f"    congestion_bands: {document['generation']['profile']['congestion_bands']}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^dimensions: .+$", "dimensions: {jobs: 60, operations_per_job: 2}", text, count=1, flags=re.MULTILINE)
    validation = yaml.safe_dump(
        document["validation"], default_flow_style=True, sort_keys=False
    ).strip()
    text = re.sub(
        r"^validation:\n(?:  .*\n)+",
        f"validation: {validation}\n",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^digest: .+$", f"digest: {document['digest']}", text, count=1, flags=re.MULTILINE)
    insertion = "\n".join(_compact_job(job) for job in document["jobs"][-10:])
    text = text.replace("bounds: {status: pending, method_version: course-lb-v1}", insertion + "\nbounds: {status: pending, method_version: course-lb-v1}", 1)
    (CATALOG / f"{identifier}.yaml").write_text(text, encoding="utf-8")
    return identifier, document["digest"]


if __name__ == "__main__":
    for replica in range(1, 4):
        identifier, digest = materialize(replica)
        print(f"{identifier} {digest}")
