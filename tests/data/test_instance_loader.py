import pickle
from dataclasses import FrozenInstanceError

import pytest
import yaml
from data.instance_loader import InstanceValidationError, load_instance
from data.instance_model import canonical_digest
from tests.data.test_instance_catalog import instance_payload
@pytest.mark.parametrize(
    ("document", "code"),
    [("jobs: [", "yaml"), (yaml.safe_dump({"schema_version": 99}), "unsupported_version")],
)
def test_loader_rejects_malformed_or_unsupported_yaml(tmp_path, document, code):
    path = tmp_path / "invalid.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(InstanceValidationError) as caught:
        load_instance(path)
    assert code in {issue.code for issue in caught.value.issues}
def test_loader_reports_ordered_dimension_reference_and_evidence_errors(tmp_path):
    payload = instance_payload()
    payload["dimensions"]["jobs"] = 2
    payload["jobs"][0]["operations"][0]["duration"] = 0.0
    payload["jobs"][0]["operations"][0]["eligible_rooms"] = ["R9"]
    payload["jobs"][0]["operations"][0]["eligible_personnel"] = ["X1"]
    payload["resources"]["personnel"]["1"] = ["A1", "A1"]
    payload["classification"] = "replica"
    payload["bounds"]["status"] = "verified"
    payload["generation"].pop("version")
    payload["validation"]["outcome"] = "failed"
    payload["digest"] = canonical_digest(payload)
    path = tmp_path / "ordered-errors.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(InstanceValidationError) as caught:
        load_instance(path)
    stages = [issue.stage for issue in caught.value.issues]
    codes = {issue.code for issue in caught.value.issues}
    assert stages == sorted(stages, key=("schema", "dimensions", "references", "evidence").index)
    assert {"job_count", "positive_duration", "personnel_ids", "room_reference", "personnel_reference", "generation_evidence", "validation_evidence", "replica_forbidden", "verified_bounds"} <= codes
def test_loaded_context_is_frozen_and_pickle_safe(tmp_path):
    path = tmp_path / "valid.yaml"
    path.write_text(yaml.safe_dump(instance_payload()), encoding="utf-8")

    context = load_instance(path)
    restored = pickle.loads(pickle.dumps(context))
    assert restored == context
    assert restored.rooms == ("R1", "R2")
    assert restored.jobs[0].operations[1].eligible_personnel == ("S1",)
    with pytest.raises(FrozenInstanceError):
        context.instance_id = "changed"
