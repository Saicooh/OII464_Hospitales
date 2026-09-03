from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml  # type: ignore[import-untyped]

from data.instance_loader import validate_document
from data.instance_model import canonical_digest
def required_replica_count(metrics: Iterable[dict], current: int = 3) -> int:
    """Add one replica pair when any pilot CI half-width exceeds 10%."""
    expand = any(
        abs(item["ci_half_width"]) > 0.1 * abs(item["mean"])
        for item in metrics
        if item["mean"] != 0
    )
    return min(10, current + 2) if expand else current
def build_catalog(output_dir: str | Path, documents: Iterable[dict]) -> Path:
    """Validate and materialize supplied instance documents under output_dir."""
    root = Path(output_dir)
    prepared = []
    for source in documents:
        document = deepcopy(source)
        document["digest"] = canonical_digest(document)
        validate_document(document)
        prepared.append(document)
    root.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "instance_count": len(prepared),
        "instances": [],
    }
    for document in prepared:
        filename = f"{document['instance_id']}.yaml"
        (root / filename).write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        metadata["instances"].append({
            "instance_id": document["instance_id"], "file": filename,
            **{key: document[key] for key in ("provenance", "generation", "dimensions", "resources", "classification", "bounds", "validation", "digest")},
        })
    (root / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return root
