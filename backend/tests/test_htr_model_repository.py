import pytest

from app.htr.domain.entities import ModelRef, ModelVersionStatus
from app.htr.domain.errors import HTRError
from app.htr.infrastructure.model_repository import (
    SqlAlchemyModelRepository,
    SqlAlchemyTrainingRunRepository,
)
from app.htr.infrastructure.storage import HTRStorage

DEFAULT = ModelRef(id="default", path="/models/default.mlmodel")


@pytest.fixture()
def repo(db_session, tmp_path):
    return SqlAlchemyModelRepository(db_session, HTRStorage(tmp_path), DEFAULT)


def test_default_model_used_when_no_custom_model(repo, author):
    assert repo.get_default_model() == DEFAULT
    assert repo.get_active_model(author.id) is None
    assert repo.get_active_model_ref(author.id) == DEFAULT


def test_versions_do_not_overwrite_each_other(repo, author):
    v1 = repo.create_version(author.id, "default", {"epochs": 1}, "hash1")
    v2 = repo.create_version(author.id, "default", {"epochs": 1}, "hash2")
    assert (v1.version, v2.version) == (1, 2)
    assert v1.id != v2.id
    assert v1.file_path != v2.file_path
    assert [v.version for v in repo.list_versions(author.id)] == [1, 2]


def test_at_most_one_active_version_per_author(repo, author):
    v1 = repo.create_version(author.id, "default", {}, "h1")
    v2 = repo.create_version(author.id, "default", {}, "h2")
    repo.mark_ready(v1.id, v1.file_path, {"cer": 0.1})
    repo.mark_ready(v2.id, v2.file_path, {"cer": 0.05})

    repo.activate_model(author.id, v1.id)
    repo.activate_model(author.id, v2.id)

    versions = {v.version: v.status for v in repo.list_versions(author.id)}
    assert versions == {1: ModelVersionStatus.READY, 2: ModelVersionStatus.ACTIVE}
    active = repo.get_active_model(author.id)
    assert active.id == v2.id
    assert repo.get_active_model_ref(author.id).path == v2.file_path


def test_only_ready_versions_can_be_activated(repo, author):
    v1 = repo.create_version(author.id, "default", {}, "h1")  # still TRAINING
    with pytest.raises(HTRError):
        repo.activate_model(author.id, v1.id)
    repo.mark_failed(v1.id, "boom")
    with pytest.raises(HTRError):
        repo.activate_model(author.id, v1.id)


def test_failed_version_does_not_disturb_active(repo, author):
    v1 = repo.create_version(author.id, "default", {}, "h1")
    repo.mark_ready(v1.id, v1.file_path, {})
    repo.activate_model(author.id, v1.id)

    v2 = repo.create_version(author.id, "default", {}, "h2")
    repo.mark_failed(v2.id, "cuda OOM")

    assert repo.get_active_model(author.id).id == v1.id


def test_metadata_round_trip(repo, author):
    config = {"epochs": 10, "learning_rate": 0.0001}
    v = repo.create_version(author.id, "default", config, "deadbeef",
                            environment={"packages": {"kraken": "5.0"}})
    repo.mark_ready(v.id, v.file_path, {"validation": {"cer": 0.02}})
    stored = repo.list_versions(author.id)[0]
    assert stored.training_config == config
    assert stored.dataset_hash == "deadbeef"
    assert stored.metrics == {"validation": {"cer": 0.02}}
    assert stored.base_model_id == "default"
