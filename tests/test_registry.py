"""Run registry tests."""

import pytest

from ptk.config.defaults import scaffold_config
from ptk.config.schema import TrainingMethod
from ptk.registry.runs import RunRegistry, RunStatus
from ptk.registry.store import LocalStore


def test_create_and_list_runs(tmp_path):
    store = LocalStore(tmp_path / "registry")
    registry = RunRegistry(store=store)
    config = scaffold_config(TrainingMethod.SFT, run_name="registry-test")
    record = registry.create_run(config)
    assert record.run_id
    assert record.status == RunStatus.PENDING

    fetched = registry.get_run(record.run_id)
    assert fetched is not None
    assert fetched.run_name == "registry-test"

    registry.update_run(record.run_id, status=RunStatus.COMPLETED, global_step=10)
    updated = registry.get_run(record.run_id)
    assert updated.status == RunStatus.COMPLETED
    assert updated.global_step == 10

    runs = registry.list_runs()
    assert len(runs) == 1


def test_resume_drift_detection(tmp_path):
    from ptk.exceptions import ResumeDriftError

    store = LocalStore(tmp_path / "registry")
    registry = RunRegistry(store=store)
    config = scaffold_config(TrainingMethod.SFT, run_name="drift-test")
    record = registry.create_run(config)

    modified = scaffold_config(TrainingMethod.LORA, run_name="drift-test")
    with pytest.raises(ResumeDriftError):
        registry.validate_resume(record.run_id, modified)
