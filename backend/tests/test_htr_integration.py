"""Integration test of the full workflow:

page upload -> recognition result -> word/line editing -> confirmation ->
dataset build -> training -> new model version -> activation.

Uses real SQLAlchemy repositories (sqlite), the real dataset builder with a
Pillow line cropper and a fake trainer backend.
"""
import io

import pytest

from app.htr.application.correction_context import CorrectionContextBuilder
from app.htr.application.lexicon import LexiconAnnotator
from app.htr.application.dataset_builder import TrainingDatasetBuilder
from app.htr.application.page_service import HandwritingPageService
from app.htr.application.training_service import HandwritingTrainingService
from app.htr.domain.entities import (
    BoundingBox,
    ModelRef,
    ModelVersionStatus,
    PageStatus,
    RecognitionResult,
    RecognizedLine,
    RecognizedWord,
    TrainingConfig,
    TrainingOutcome,
)
from app.htr.domain.errors import NotFoundError, PageStateError
from app.htr.infrastructure.model_repository import (
    SqlAlchemyModelRepository,
    SqlAlchemyTrainingRunRepository,
)
from app.htr.infrastructure.page_repository import SqlAlchemyPageRepository
from app.htr.infrastructure.storage import HTRStorage, PilLineCropper
from tests.htr_fakes import FakeCorrector, FakeLexiconProvider, RecordingTrainer

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

DEFAULT = ModelRef(id="default", path="/models/default.mlmodel")


def png_bytes(width=200, height=120) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), color=255).save(buf, format="PNG")
    return buf.getvalue()


def recognition_result() -> RecognitionResult:
    line1 = BoundingBox(0, 0, 200, 40)
    line2 = BoundingBox(0, 50, 200, 90)
    word2 = BoundingBox(35, 0, 90, 40)
    word7 = BoundingBox(45, 50, 120, 90)
    return RecognitionResult(
        page_width=200,
        page_height=120,
        lines=[
            RecognizedLine(
                id="l1",
                bbox=line1,
                text="Уж очен дед был похож",
                # skewed outlines, like the neural baseline segmenter produces
                polygon=[(line1.x1, line1.y1), (line1.x2, 8), (line1.x2, line1.y2), (line1.x1, 32)],
                words=[
                    RecognizedWord("w1", BoundingBox(0, 0, 30, 40), "Уж", 0.98),
                    RecognizedWord("w2", word2, "очен", 0.55,
                                   polygon=[(35, 0), (90, 6), (90, 40), (35, 34)]),
                    RecognizedWord("w3", BoundingBox(95, 0, 130, 40), "дед", 0.95),
                    RecognizedWord("w4", BoundingBox(135, 0, 165, 40), "был", 0.92),
                    RecognizedWord("w5", BoundingBox(170, 0, 200, 40), "похож", 0.80),
                ],
            ),
            RecognizedLine(
                id="l2",
                bbox=line2,
                text="на еврея",
                words=[
                    RecognizedWord("w6", BoundingBox(0, 50, 40, 90), "на", 0.97),
                    RecognizedWord("w7", word7, "еврея", 0.75,
                                   polygon=[(45, 50), (120, 54), (120, 90), (45, 86)]),
                ],
            ),
        ],
    )


@pytest.fixture()
def env(db_session, tmp_path, user, author):
    storage = HTRStorage(tmp_path / "htr")
    page_repo = SqlAlchemyPageRepository(db_session)
    model_repo = SqlAlchemyModelRepository(db_session, storage, DEFAULT)
    page_service = HandwritingPageService(
        page_repository=page_repo,
        model_repository=model_repo,
        image_store=storage,
    )
    trainer = RecordingTrainer(write_artifact=True)
    training_service = HandwritingTrainingService(
        dataset_builder=TrainingDatasetBuilder(
            page_repository=page_repo,
            line_cropper=PilLineCropper(),
            crop_path_provider=storage.line_crop_path,
        ),
        model_repository=model_repo,
        training_run_repository=SqlAlchemyTrainingRunRepository(db_session),
        trainer=trainer,
        config=TrainingConfig(device="cpu"),
        # the test pages are tiny; the production default threshold is 50 lines
        min_training_lines=1,
    )
    return page_service, training_service, model_repo, trainer, user, author


def run_page_cycle(page_service, user, author, edit=True):
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    assert page.status == PageStatus.UPLOADED
    page = page_service.apply_recognition(page.id, recognition_result())
    assert page.status == PageStatus.RECOGNIZED
    if edit:
        # word-level edit: fix "очен" -> "очень"
        word = page.lines[0].words[1]
        page = page_service.update_word(page.id, page.lines[0].id, word.id, "очень")
        assert page.status == PageStatus.EDITING
        # line-level edit: replace whole second line
        page = page_service.update_line(page.id, page.lines[1].id, "на еврея,")
    return page_service.confirm_page(page.id)


def test_full_training_workflow(env):
    page_service, training_service, model_repo, trainer, user, author = env

    page = run_page_cycle(page_service, user, author)
    assert page.status == PageStatus.CONFIRMED
    assert page.confirmed_at is not None
    # prediction quality was measured against the user's ground truth
    assert page.prediction_cer is not None and page.prediction_cer > 0
    assert page.prediction_wer is not None

    # word edit rebuilt the canonical line transcription
    assert page.lines[0].corrected_text == "Уж очень дед был похож"
    assert page.lines[0].words_stale is False
    # line edit marked word alignment stale
    assert page.lines[1].corrected_text == "на еврея,"
    assert page.lines[1].words_stale is True

    result = training_service.train_author(author.id)
    assert result.outcome == TrainingOutcome.SUCCESS

    versions = model_repo.list_versions(author.id)
    assert len(versions) == 1
    v1 = versions[0]
    assert v1.status == ModelVersionStatus.ACTIVE
    assert v1.version == 1
    assert v1.base_model_id == "default"
    assert v1.dataset_hash == result.dataset_hash

    # trainer received all confirmed lines with the corrected transcriptions
    dataset = trainer.calls[0].dataset
    assert sorted(s.transcription for s in dataset.samples) == sorted(
        ["Уж очень дед был похож", "на еврея,"]
    )
    # line crops were physically produced
    import os
    assert all(os.path.isfile(s.image_path) for s in dataset.samples)


def test_second_page_retrains_from_default_with_full_corpus(env):
    page_service, training_service, model_repo, trainer, user, author = env

    run_page_cycle(page_service, user, author)
    training_service.train_author(author.id)
    run_page_cycle(page_service, user, author, edit=False)
    result = training_service.train_author(author.id)

    assert result.outcome == TrainingOutcome.SUCCESS
    versions = model_repo.list_versions(author.id)
    assert [v.version for v in versions] == [1, 2]
    assert versions[0].status == ModelVersionStatus.READY  # kept, not deleted
    assert versions[1].status == ModelVersionStatus.ACTIVE

    # second run: base is still default, dataset covers BOTH confirmed pages
    second_call = trainer.calls[1]
    assert second_call.base_model.id == "default"
    assert len({s.page_id for s in second_call.dataset.samples}) == 2
    assert len(second_call.dataset.samples) == 4

    # next recognition would use the new active model
    active_ref = model_repo.get_active_model_ref(author.id)
    assert active_ref.path == versions[1].file_path


def test_recognition_records_active_model_version(env):
    page_service, training_service, model_repo, trainer, user, author = env

    run_page_cycle(page_service, user, author)
    training_service.train_author(author.id)
    active = model_repo.get_active_model(author.id)

    page = page_service.upload_page(user.id, author.id, "p2.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())
    assert page.recognition_model_version_id == active.id


def test_deleted_confirmed_page_leaves_the_training_corpus(env):
    page_service, training_service, _, _, user, author = env

    first = run_page_cycle(page_service, user, author)
    second = run_page_cycle(page_service, user, author, edit=False)

    dataset = training_service.dataset_builder.build_for_author(author.id)
    assert {s.page_id for s in dataset.samples} == {first.id, second.id}

    page_service.delete_page(second.id)

    dataset = training_service.dataset_builder.build_for_author(author.id)
    assert {s.page_id for s in dataset.samples} == {first.id}
    assert page_service.page_repository.get_page(second.id) is None


def test_recognition_polygons_round_trip_through_the_database(env):
    page_service, _, _, _, user, author = env

    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    result = recognition_result()
    page = page_service.apply_recognition(page.id, result)

    line = page.lines[0]
    assert line.polygon == [(0, 0), (200, 8), (200, 40), (0, 32)]
    # the skewed word keeps its curved outline, not just the envelope bbox
    assert line.words[1].polygon == [(35, 0), (90, 6), (90, 40), (35, 34)]

    # re-read from the database rather than trusting the returned object
    fresh = page_service.page_repository.get_page(page.id)
    assert fresh.lines[0].polygon == line.polygon
    assert fresh.lines[0].words[1].polygon == line.words[1].polygon


def test_force_recognition_reopens_a_confirmed_page(env):
    page_service, _, _, _, user, author = env

    confirmed = run_page_cycle(page_service, user, author)
    assert confirmed.status == PageStatus.CONFIRMED
    assert confirmed.confirmed_at is not None
    assert confirmed.prediction_cer is not None

    # a confirmed page is ground truth: re-segmentation needs an explicit force
    with pytest.raises(PageStateError):
        page_service.apply_recognition(confirmed.id, recognition_result())

    refreshed = page_service.apply_recognition(confirmed.id, recognition_result(), force=True)

    assert refreshed.status == PageStatus.RECOGNIZED
    assert refreshed.confirmed_at is None
    assert refreshed.prediction_cer is None and refreshed.prediction_wer is None
    # the new prediction replaced the confirmed transcript
    assert [line.corrected_text for line in refreshed.lines] == [None, None]


# ---------------------------------------------------------------------------
# Model proposals: kept apart from the transcription
# ---------------------------------------------------------------------------


def test_model_proposal_never_touches_the_transcription(env):
    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())

    corrector = FakeCorrector({"Уж очен дед был похож": "Уж очень дед был похож"})
    page_service.corrector = corrector
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )

    suggested = page_service.suggest_corrections(page.id)

    first = suggested.lines[0]
    assert first.suggested_text == "Уж очень дед был похож"
    assert first.suggested_by == corrector.model
    assert [change.after for change in first.suggestion_changes] == ["очень"]
    # the recognition is exactly as kraken produced it
    assert first.corrected_text is None
    assert first.corrected_by is None
    assert first.predicted_text == "Уж очен дед был похож"
    assert first.effective_text == "Уж очен дед был похож"
    # nothing was edited, so the page is still only recognized
    assert suggested.status == PageStatus.RECOGNIZED
    # neighbouring lines are offered as context, the target itself is not
    assert corrector.calls[0][1] == ["на еврея"]


def test_line_without_changes_gets_no_proposal(env):
    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())

    page_service.corrector = FakeCorrector()  # every line comes back unchanged
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )

    suggested = page_service.suggest_corrections(page.id)

    assert all(line.suggested_text is None for line in suggested.lines)
    assert all(not line.has_suggestion for line in suggested.lines)


def test_accepting_a_proposal_moves_it_into_the_transcription(env):
    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())
    page_service.corrector = FakeCorrector({"Уж очен дед был похож": "Уж очень дед был похож"})
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )
    suggested = page_service.suggest_corrections(page.id)
    line_id = suggested.lines[0].id

    accepted = page_service.accept_suggestion(page.id, line_id)

    first = accepted.lines[0]
    assert first.corrected_text == "Уж очень дед был похож"
    assert first.corrected_by == "user"          # the human decided
    assert first.suggested_text is None          # the proposal has been used
    assert first.predicted_text == "Уж очен дед был похож"
    assert accepted.status == PageStatus.EDITING
    # accepting rewrites the line, so the word boxes may no longer match
    assert first.words_stale is True
    # the other line had no proposal and stays recognised
    assert accepted.lines[1].corrected_text is None


def test_dismissing_a_proposal_keeps_the_transcription(env):
    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())
    page_service.corrector = FakeCorrector({"на еврея": "на еврея,"})
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )
    suggested = page_service.suggest_corrections(page.id)
    line_id = suggested.lines[1].id
    assert suggested.lines[1].suggested_text == "на еврея,"

    dismissed = page_service.dismiss_suggestion(page.id, line_id)

    assert dismissed.lines[1].suggested_text is None
    assert dismissed.lines[1].corrected_text is None
    assert dismissed.lines[1].effective_text == "на еврея"
    assert dismissed.status == PageStatus.RECOGNIZED


def test_editing_a_line_drops_its_pending_proposal(env):
    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())
    page_service.corrector = FakeCorrector({"Уж очен дед был похож": "Уж очень дед был похож"})
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )
    suggested = page_service.suggest_corrections(page.id)
    line_id = suggested.lines[0].id

    edited = page_service.update_line(page.id, line_id, "Уж очень дед был")

    assert edited.lines[0].corrected_text == "Уж очень дед был"
    assert edited.lines[0].suggested_text is None


def test_accepting_one_word_keeps_the_rest_of_the_proposal(env):
    """The user may like some proposed words but not others."""
    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())
    page_service.corrector = FakeCorrector(
        {"Уж очен дед был похож": "Уж очень дед был похож всегда"}
    )
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )
    suggested = page_service.suggest_corrections(page.id)
    line = suggested.lines[0]
    assert [(c.before, c.after) for c in line.suggestion_changes] == [
        ("очен", "очень"),
        ("", "всегда"),
    ]

    # take the spelling fix, leave the word the model wanted to add
    page = page_service.accept_suggestion_change(page.id, line.id, 0)

    first = page.lines[0]
    assert first.corrected_text == "Уж очень дед был похож"
    assert first.corrected_by == "user"
    assert first.predicted_text == "Уж очен дед был похож"
    # the remaining proposal is still there, now with a single change
    assert first.suggested_text == "Уж очень дед был похож всегда"
    assert first.has_suggestion is True
    assert [(c.before, c.after) for c in first.suggestion_changes] == [("", "всегда")]

    # ... and taking it too finishes the line
    page = page_service.accept_suggestion_change(page.id, line.id, 0)
    assert page.lines[0].corrected_text == "Уж очень дед был похож всегда"
    assert page.lines[0].has_suggestion is False
    assert page.lines[0].suggested_text == "Уж очень дед был похож всегда"


def test_accepting_a_change_out_of_range_is_a_not_found(env):
    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())
    page_service.corrector = FakeCorrector({"на еврея": "на еврея,"})  # punctuation only
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )
    suggested = page_service.suggest_corrections(page.id)
    line_id = suggested.lines[1].id

    with pytest.raises(NotFoundError):
        page_service.accept_suggestion_change(page.id, line_id, 0)


def test_accepting_and_dismissing_proposals_is_logged(env):
    """Point 17: the correction loop needs feedback about what the user did."""
    from app.models import HTRSuggestionEvent

    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())
    page_service.corrector = FakeCorrector(
        {"Уж очен дед был похож": "Уж очень дед был похож", "на еврея": "на еврея,"}
    )
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )
    suggested = page_service.suggest_corrections(page.id)
    first_id = suggested.lines[0].id
    second_id = suggested.lines[1].id

    page_service.accept_suggestion_change(page.id, first_id, 0)
    page_service.dismiss_suggestion(page.id, second_id)

    events = (
        page_service.page_repository.db.query(HTRSuggestionEvent)
        .order_by(HTRSuggestionEvent.id)
        .all()
    )
    assert [event.action for event in events] == ["accepted_change", "dismissed"]
    accepted = events[0]
    assert accepted.change_before == "очен"
    assert accepted.change_after == "очень"
    assert accepted.original_text == "Уж очен дед был похож"
    assert accepted.suggested_text == "Уж очень дед был похож"
    assert accepted.suggested_by == "fake-llm"
    dismissed = events[1]
    assert dismissed.line_id == second_id
    assert dismissed.suggested_text == "на еврея,"


def test_word_alternatives_survive_the_round_trip_to_the_database(env):
    """Alternatives are written once at recognition and read back for the editor."""
    from app.htr.domain.entities import WordAlternative

    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    result = recognition_result()
    first_line = result.lines[0]
    first_line.words[0].alternatives = [
        WordAlternative(text="Ужъ", score=-1.5),
        WordAlternative(text="Уш", score=-2.5),
    ]

    page = page_service.apply_recognition(page.id, result)

    word = page.lines[0].words[0]
    assert [item.text for item in word.alternatives] == ["Ужъ", "Уш"]
    assert word.alternatives[0].score == -1.5
    # a word without alternatives simply has an empty list
    assert page.lines[0].words[1].alternatives == []


def test_word_alternatives_reach_the_api_response(env):
    from app.htr.domain.entities import WordAlternative

    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    result = recognition_result()
    result.lines[0].words[0].alternatives = [WordAlternative(text="Ужъ", score=-1.5)]

    page = page_service.apply_recognition(page.id, result)

    assert page.lines[0].words[0].alternatives[0].text == "Ужъ"


def test_recognition_hands_the_author_vocabulary_to_the_decoder(env):
    """Beam scoring must prefer the author's own words, not just the dictionary."""
    from tests.htr_fakes import FakeRecognizer

    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    recognizer = FakeRecognizer(result=recognition_result())
    page_service.recognizer = recognizer
    page_service.lexicon_annotator = LexiconAnnotator(
        FakeLexiconProvider(known={"уж", "очень"})
    )

    page_service.recognize_page(page.id)

    checker = recognizer.last_word_checker
    assert checker is not None
    assert checker.is_known("уж")


def test_bulk_accept_logs_every_line(env):
    from app.models import HTRSuggestionEvent

    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())
    # a real word replacement, so the change is verifiable by the dictionary
    page_service.corrector = FakeCorrector({"на еврея": "на евреев"})
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )
    page_service.lexicon_annotator = LexiconAnnotator(
        FakeLexiconProvider(known={"уж", "очень", "на", "еврея", "евреев"})
    )
    page_service.suggest_corrections(page.id)
    _, accepted = page_service.accept_verified_suggestions(page.id)
    assert accepted == 1

    events = page_service.page_repository.db.query(HTRSuggestionEvent).all()
    assert [event.action for event in events] == ["accepted_bulk"]


def test_bulk_accept_takes_only_verified_proposals(env):
    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())
    page_service.corrector = FakeCorrector(
        {
            "Уж очен дед был похож": "Уж очень дед был похож",   # verified
            "на еврея": "на еврея, Захарьевка",                  # invented place
        }
    )
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )
    page_service.lexicon_annotator = LexiconAnnotator(
        FakeLexiconProvider(known={"уж", "очень", "дед", "был", "похож", "на", "еврея"})
    )
    page_service.suggest_corrections(page.id)

    accepted_page, count = page_service.accept_verified_suggestions(page.id)

    assert count == 1
    # the verified fix became the user's text ...
    assert accepted_page.lines[0].corrected_text == "Уж очень дед был похож"
    # ... while the line with the invented word stays untouched and is still
    # offered for review
    assert accepted_page.lines[1].corrected_text is None
    assert accepted_page.lines[1].suggested_text == "на еврея, Захарьевка"
    assert accepted_page.lines[1].suggestion_verified is False


def test_every_proposal_is_unverified_without_a_dictionary(env):
    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())
    page_service.corrector = FakeCorrector({"Уж очен дед был похож": "Уж очень дед был похож"})
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )
    page_service.suggest_corrections(page.id)  # no lexicon annotator at all

    page = page_service.page_repository.get_page(page.id)
    page = page_service.get_page(page.id)

    assert page.lines[0].suggestion_changes[0].in_lexicon is None
    assert page.lines[0].suggestion_verified is False
    _, count = page_service.accept_verified_suggestions(page.id)
    assert count == 0


def test_suggestions_are_best_effort_when_the_model_is_unavailable(env):
    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())

    page_service.corrector = FakeCorrector(fail=True)
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )

    unchanged = page_service.suggest_corrections(page.id)

    assert [line.suggested_text for line in unchanged.lines] == [None, None]
    assert [line.corrected_text for line in unchanged.lines] == [None, None]
    assert unchanged.status == PageStatus.RECOGNIZED


def test_suggestions_stop_after_repeated_failures(env):
    """A hanging/unreachable LLM must not cost one timeout per line."""
    page_service, _, _, _, user, author = env
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())
    corrector = FakeCorrector(fail=True)
    page_service.corrector = corrector
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )

    page_service.suggest_corrections(page.id)

    assert len(corrector.calls) == 2  # gives up after two failures in a row


def test_suggestions_are_refused_for_a_confirmed_page(env):
    page_service, _, _, _, user, author = env
    confirmed = run_page_cycle(page_service, user, author)
    page_service.corrector = FakeCorrector({"Уж очень дед": "испорчено"})
    page_service.correction_context_builder = CorrectionContextBuilder(
        page_service.page_repository
    )

    with pytest.raises(PageStateError):
        page_service.suggest_corrections(confirmed.id)

    fresh = page_service.page_repository.get_page(confirmed.id)
    assert fresh.status == PageStatus.CONFIRMED
    assert [line.corrected_text for line in fresh.lines] == ["Уж очень дед был похож", "на еврея,"]
    assert [line.suggested_text for line in fresh.lines] == [None, None]
