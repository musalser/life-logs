from app.htr.application.training_service import (
    NO_HOLDOUT_NOTE,
    HandwritingTrainingService,
)
from app.htr.domain.entities import (
    ModelVersionStatus,
    TrainingConfig,
    TrainingDataset,
    TrainingOutcome,
    TrainingSample,
)
from app.htr.domain.errors import DatasetBuildError
from tests.htr_fakes import (
    FakeDatasetBuilder,
    FakeModelRepository,
    FakeTrainingRunRepository,
    RecordingTrainer,
)

AUTHOR_ID = 7


def make_dataset(n_samples=3, author_id=AUTHOR_ID, dataset_hash="abc123"):
    samples = [
        TrainingSample(page_id=i, line_id=i * 10, image_path=f"/crops/{i}.png",
                       transcription=f"строка {i}")
        for i in range(1, n_samples + 1)
    ]
    return TrainingDataset(author_id=author_id, samples=samples, dataset_hash=dataset_hash)


def make_service(dataset=None, error=None, trainer=None, model_repo=None, run_repo=None,
                 min_samples=1):
    return HandwritingTrainingService(
        dataset_builder=FakeDatasetBuilder(dataset=dataset, error=error),
        model_repository=model_repo or FakeModelRepository(),
        training_run_repository=run_repo or FakeTrainingRunRepository(),
        trainer=trainer or RecordingTrainer(),
        config=TrainingConfig(),
        min_training_samples=min_samples,
    )


def test_successful_training_creates_and_activates_new_version():
    trainer = RecordingTrainer()
    model_repo = FakeModelRepository()
    run_repo = FakeTrainingRunRepository()
    service = make_service(dataset=make_dataset(), trainer=trainer,
                           model_repo=model_repo, run_repo=run_repo)

    result = service.train_author(AUTHOR_ID)

    assert result.outcome == TrainingOutcome.SUCCESS
    active = model_repo.get_active_model(AUTHOR_ID)
    assert active is not None and active.version == 1
    assert run_repo.runs[result.training_run_id]["status"] == "SUCCEEDED"
    assert result.dataset_hash == "abc123"


def test_base_model_is_always_default_never_previous_custom():
    trainer = RecordingTrainer()
    model_repo = FakeModelRepository()
    service = make_service(dataset=make_dataset(), trainer=trainer, model_repo=model_repo)

    service.train_author(AUTHOR_ID)  # creates active custom v1
    assert model_repo.get_active_model(AUTHOR_ID) is not None
    service.train_author(AUTHOR_ID)  # must still start from default
    service.train_author(AUTHOR_ID)

    assert [call.base_model.id for call in trainer.calls] == ["default"] * 3


def test_each_success_creates_new_version_and_old_ones_are_kept():
    model_repo = FakeModelRepository()
    service = make_service(dataset=make_dataset(), model_repo=model_repo)

    service.train_author(AUTHOR_ID)
    service.train_author(AUTHOR_ID)

    versions = model_repo.list_versions(AUTHOR_ID)
    assert [v.version for v in versions] == [1, 2]
    assert versions[0].status == ModelVersionStatus.READY  # demoted, not deleted
    assert versions[1].status == ModelVersionStatus.ACTIVE


def test_failed_training_keeps_previous_active_model():
    model_repo = FakeModelRepository()
    service_ok = make_service(dataset=make_dataset(), model_repo=model_repo)
    service_ok.train_author(AUTHOR_ID)
    v1 = model_repo.get_active_model(AUTHOR_ID)

    run_repo = FakeTrainingRunRepository()
    service_fail = make_service(
        dataset=make_dataset(), trainer=RecordingTrainer(fail=True),
        model_repo=model_repo, run_repo=run_repo,
    )
    result = service_fail.train_author(AUTHOR_ID)

    assert result.outcome == TrainingOutcome.FAILED
    versions = model_repo.list_versions(AUTHOR_ID)
    assert versions[1].status == ModelVersionStatus.FAILED
    active = model_repo.get_active_model(AUTHOR_ID)
    assert active is not None and active.id == v1.id  # v1 still active
    assert run_repo.runs[result.training_run_id]["status"] == "FAILED"


def test_insufficient_data():
    run_repo = FakeTrainingRunRepository()
    model_repo = FakeModelRepository()
    service = make_service(dataset=make_dataset(n_samples=1), min_samples=5,
                           model_repo=model_repo, run_repo=run_repo)

    result = service.train_author(AUTHOR_ID)

    assert result.outcome == TrainingOutcome.INSUFFICIENT_DATA
    assert result.message
    assert model_repo.list_versions(AUTHOR_ID) == []
    assert run_repo.runs[result.training_run_id]["status"] == "INSUFFICIENT_DATA"


def test_dataset_build_error_creates_no_model_version():
    model_repo = FakeModelRepository()
    service = make_service(error=DatasetBuildError("bad line"), model_repo=model_repo)

    result = service.train_author(AUTHOR_ID)

    assert result.outcome == TrainingOutcome.FAILED
    assert model_repo.list_versions(AUTHOR_ID) == []


def test_all_confirmed_samples_are_passed_to_trainer():
    trainer = RecordingTrainer()
    dataset = make_dataset(n_samples=4)
    service = make_service(dataset=dataset, trainer=trainer)

    service.train_author(AUTHOR_ID)

    assert trainer.calls[0].dataset is dataset
    assert len(trainer.calls[0].dataset.samples) == 4


def test_metrics_flag_missing_holdout():
    service = make_service(dataset=make_dataset())
    result = service.train_author(AUTHOR_ID)
    assert result.metrics["holdout_used"] is False
    assert result.metrics["note"] == NO_HOLDOUT_NOTE
    assert result.metrics["validation"] == {"cer": 0.02, "wer": 0.05}
