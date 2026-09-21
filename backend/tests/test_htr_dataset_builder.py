import pytest

from app.htr.application.dataset_builder import TrainingDatasetBuilder
from app.htr.domain.entities import PageStatus
from app.htr.domain.errors import DatasetBuildError
from tests.htr_fakes import FakeLineCropper, FakePageRepository, make_line, make_page


def build(pages, cropper=None):
    return TrainingDatasetBuilder(
        page_repository=FakePageRepository(pages),
        line_cropper=cropper or FakeLineCropper(),
        crop_path_provider=lambda page_id, line_id: f"/crops/{page_id}/{line_id}.png",
    )


def test_only_confirmed_pages_of_author_are_used():
    pages = [
        make_page(1, author_id=1, status=PageStatus.CONFIRMED),
        make_page(2, author_id=1, status=PageStatus.EDITING),
        make_page(3, author_id=1, status=PageStatus.RECOGNIZED),
        make_page(4, author_id=2, status=PageStatus.CONFIRMED),  # other author
        make_page(5, author_id=1, status=PageStatus.CONFIRMED),
    ]
    dataset = build(pages).build_for_author(1)
    assert {s.page_id for s in dataset.samples} == {1, 5}
    assert dataset.author_id == 1


def test_all_confirmed_pages_included_each_time():
    pages = [make_page(i, status=PageStatus.CONFIRMED) for i in range(1, 4)]
    dataset = build(pages).build_for_author(1)
    assert {s.page_id for s in dataset.samples} == {1, 2, 3}


def test_corrected_text_is_ground_truth_and_never_autofixed():
    raw = "  Уж очен дед был похож на еврея,,  "
    page = make_page(
        1, lines=[make_line(10, predicted="Уж очень дед", corrected=raw)]
    )
    dataset = build([page]).build_for_author(1)
    # user text is taken verbatim: no spellfix, no punctuation/space normalization
    assert dataset.samples[0].transcription == raw


def test_prediction_used_when_no_correction():
    page = make_page(1, lines=[make_line(10, predicted="Уж очень дед", corrected=None)])
    dataset = build([page]).build_for_author(1)
    assert dataset.samples[0].transcription == "Уж очень дед"


def test_explicitly_empty_line_is_valid_but_excluded():
    page = make_page(
        1,
        lines=[
            make_line(10, predicted="текст"),
            make_line(11, order=1, predicted="мусор", corrected=""),
        ],
    )
    dataset = build([page]).build_for_author(1)
    assert [s.line_id for s in dataset.samples] == [10]


def test_implicitly_empty_line_fails_the_build():
    page = make_page(1, lines=[make_line(10, predicted="", corrected=None)])
    with pytest.raises(DatasetBuildError):
        build([page]).build_for_author(1)


def test_non_confirmed_page_from_repository_is_rejected():
    class LeakyRepo(FakePageRepository):
        def get_confirmed_pages(self, author_id):
            return self.pages  # buggy repo returns everything

    builder = TrainingDatasetBuilder(
        page_repository=LeakyRepo([make_page(1, status=PageStatus.EDITING)]),
        line_cropper=FakeLineCropper(),
        crop_path_provider=lambda p, l: f"/crops/{p}/{l}.png",
    )
    with pytest.raises(DatasetBuildError):
        builder.build_for_author(1)


def test_corrupt_image_page_is_excluded():
    pages = [
        make_page(1, file_path="/pages/ok.png"),
        make_page(2, file_path="/pages/broken.png"),
    ]
    cropper = FakeLineCropper(corrupt_paths={"/pages/broken.png"})
    dataset = build(pages, cropper).build_for_author(1)
    assert {s.page_id for s in dataset.samples} == {1}


def test_dataset_hash_is_stable_for_identical_content():
    pages = [make_page(1), make_page(2)]
    h1 = build(pages).build_for_author(1).dataset_hash
    h2 = build(list(reversed(pages))).build_for_author(1).dataset_hash
    assert h1 == h2


def test_dataset_hash_changes_with_content():
    base = build([make_page(1)]).build_for_author(1).dataset_hash
    edited_page = make_page(1, lines=[make_line(100, corrected="другой текст")])
    edited = build([edited_page]).build_for_author(1).dataset_hash
    grown = build([make_page(1), make_page(2)]).build_for_author(1).dataset_hash
    assert base != edited
    assert base != grown
