"""Unit tests of the dictionary (OOV) check.

The feature decides which words the user is asked to look at, so the tests
focus on the two failure modes that matter: a word that *is* fine must not be
flagged (the author's own vocabulary, old spellings, numbers), and a word that
is genuinely unknown must be.
"""
from __future__ import annotations

import pytest

from app.htr.application.lexicon import LexiconAnnotator
from app.htr.domain.entities import PageStatus, PageSummary
from app.htr.domain.text import iter_words, normalize_word, word_variants
from app.htr.infrastructure.lexicon import (
    AuthorCorpusCache,
    BloomFilter,
    FileLexicon,
    LayeredLexiconChecker,
    SqlAlchemyAuthorCorpus,
    load_file_lexicon,
    words_of,
)
from app.htr.infrastructure.page_repository import SqlAlchemyPageRepository
from tests.htr_fakes import FakeLexiconProvider, make_line, make_page, make_word, recognition_result


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------


def test_iter_words_drops_punctuation_but_keeps_hyphens():
    assert iter_words("Уж очень дед, 1917г. кто-то «Шпалозавод»") == [
        "Уж", "очень", "дед", "1917г", "кто-то", "Шпалозавод"
    ]
    assert iter_words(None) == []
    assert iter_words("   ") == []


@pytest.mark.parametrize(
    ("old", "modern"),
    [
        ("Домъ", "дом"),
        ("Всѣ", "все"),
        ("Міръ", "мир"),
        ("Ѳедоръ", "федор"),
        ("Сѵнодъ", "синод"),
    ],
)
def test_normalize_word_modernizes_old_orthography(old, modern):
    assert normalize_word(old) == modern
    assert normalize_word(old) == normalize_word(modern)


def test_normalize_word_is_a_noop_for_modern_words():
    assert normalize_word("Корова") == "корова"
    assert normalize_word("Нового") == "нового"


def test_word_variants_add_modern_endings_as_extra_candidates():
    assert "нового" in word_variants("Новаго")
    assert "синие" in word_variants("Синія")
    assert "новые" in word_variants("новыя")
    assert "её" in word_variants("ея")
    # the original spelling always stays available
    assert "новаго" in word_variants("Новаго")
    assert "россия" in word_variants("Россия")


def test_words_of_normalizes_a_transcription():
    forms = words_of("Домъ стоитъ, 1917 г.")
    assert "дом" in forms
    assert "стоит" in forms


# ---------------------------------------------------------------------------
# Bloom filter artifact
# ---------------------------------------------------------------------------


def test_bloom_filter_has_no_false_negatives_across_a_save_load_cycle(tmp_path):
    words = [f"слово{index}" for index in range(500)] + ["мама", "мыла", "раму"]
    bloom = BloomFilter.build(capacity=len(words), false_positive_rate=0.01)
    for word in words:
        bloom.add(word)

    path = bloom.save(tmp_path / "lexicon.bloom")
    restored = BloomFilter.load(path)

    assert all(word in restored for word in words)
    assert restored.hash_count == bloom.hash_count
    assert restored.item_count == len(words)


def test_bloom_filter_false_positive_rate_is_small():
    words = [f"форма{index}" for index in range(4000)]
    bloom = BloomFilter.build(capacity=len(words), false_positive_rate=0.01)
    for word in words:
        bloom.add(word)

    probes = [f"выдумка{index}" for index in range(4000)]
    false_positives = sum(1 for probe in probes if probe in bloom)
    assert false_positives / len(probes) < 0.04


def test_bloom_filter_rejects_foreign_artifacts(tmp_path):
    path = tmp_path / "broken.bloom"
    path.write_bytes(b"not a lexicon")
    with pytest.raises(ValueError):
        BloomFilter.load(path)


def test_load_file_lexicon_reports_a_missing_artifact(tmp_path):
    lexicon = load_file_lexicon(tmp_path / "absent.bloom")
    assert lexicon.is_available is False
    assert "что-то" not in lexicon


def test_load_file_lexicon_reads_the_artifact(tmp_path):
    bloom = BloomFilter.build(capacity=3)
    bloom.add("корова")
    path = bloom.save(tmp_path / "lexicon.bloom")

    lexicon = load_file_lexicon(path)
    assert lexicon.is_available is True
    assert "корова" in lexicon
    assert isinstance(lexicon, FileLexicon)


# ---------------------------------------------------------------------------
# the checker itself
# ---------------------------------------------------------------------------


def test_checker_accepts_numbers_abbreviations_and_hyphenated_words():
    checker = LayeredLexiconChecker(base=frozenset({"кто", "то", "мама"}))
    assert checker.is_known("1917")
    assert checker.is_known("1917г")
    assert checker.is_known("...")
    assert checker.is_known("кто-то")       # both parts are known
    assert checker.is_known("мама")


def test_checker_never_flags_a_one_letter_word():
    # word lists have no one-letter entries, but "и", "в", "с", "а", "я" are words
    checker = LayeredLexiconChecker(base=frozenset({"дом"}))
    for letter in ("и", "в", "с", "к", "у", "а", "я", "п", "И"):
        assert checker.is_known(letter)


def test_checker_flags_an_unknown_word_and_old_spellings_of_known_ones():
    checker = LayeredLexiconChecker(base=frozenset({"нового", "дом"}))
    assert checker.is_known("Новаго")
    assert checker.is_known("Домъ")
    assert not checker.is_known("Шпалозавод")
    assert not checker.is_known("карова")


def test_checker_uses_the_author_vocabulary_before_the_dictionary():
    checker = LayeredLexiconChecker(base=frozenset(), extra={"шпалозавод"})
    assert checker.is_known("Шпалозавод")
    assert not checker.is_known("Паровоз")


def test_checker_without_a_dictionary_is_unavailable_but_never_crashes():
    checker = LayeredLexiconChecker(base=None)
    assert checker.is_available is False
    assert checker.is_known("что угодно") is False
    # the author lexicon is consulted, but does not make the check available on
    # its own: without the general dictionary every other word would be flagged
    with_words = LayeredLexiconChecker(base=None, extra={"дом"})
    assert with_words.is_available is False
    assert with_words.is_known("Домъ")
    forced = LayeredLexiconChecker(base=None, extra={"дом"}, available=True)
    assert forced.is_available is True


# ---------------------------------------------------------------------------
# page annotation
# ---------------------------------------------------------------------------


def _page_with_words():
    words = [
        make_word(1, order=0, text="Уж"),
        make_word(2, order=1, text="очень"),
        make_word(3, order=2, text="дед"),
    ]
    line = make_line(10, order=0, predicted="Уж очень дед", words=words)
    return make_page(7, lines=[line])


def test_annotate_marks_words_and_counts_the_line():
    page = _page_with_words()
    annotator = LexiconAnnotator(FakeLexiconProvider(known={"уж", "очень"}))

    annotator.annotate(page)

    assert page.lexicon_available is True
    assert [word.in_lexicon for word in page.lines[0].words] == [True, True, False]
    assert page.lines[0].oov_count == 1
    assert page.lines[0].oov_words == ["дед"]
    assert page.oov_count == 1


def test_annotate_falls_back_to_the_word_text_when_the_line_was_rewritten():
    page = _page_with_words()
    page.lines[0].corrected_text = "Уж очень дед и баба"
    page.lines[0].words_stale = True
    annotator = LexiconAnnotator(FakeLexiconProvider(known={"уж", "очень", "и"}))

    annotator.annotate(page)

    # the boxes are stale, so each box is judged on its own (old) text
    assert [word.in_lexicon for word in page.lines[0].words] == [True, True, False]
    # while the badge and the chip list follow the transcription the user sees
    assert page.lines[0].oov_count == 2
    assert page.lines[0].oov_words == ["дед", "баба"]
    assert page.oov_count == 2


def test_annotate_leaves_everything_empty_without_a_dictionary():
    page = _page_with_words()
    annotator = LexiconAnnotator(FakeLexiconProvider(known={"уж"}, available=False))

    annotator.annotate(page)

    assert page.lexicon_available is False
    assert page.oov_count == 0
    assert page.lines[0].oov_words == []
    assert all(word.in_lexicon is None for word in page.lines[0].words)


def test_annotate_without_a_provider_is_a_noop():
    page = _page_with_words()
    LexiconAnnotator(None).annotate(page)
    assert page.lexicon_available is False
    assert page.oov_count == 0
    assert all(word.in_lexicon is None for word in page.lines[0].words)


def test_annotate_summaries_counts_oov_words_per_page():
    provider = FakeLexiconProvider(
        known={"уж", "очень"},
        texts={7: ["Уж очень дед", "на еврея"], 8: ["Уж очень дед"]},
    )
    summaries = [
        PageSummary(
            id=7, author_id=1, status=PageStatus.RECOGNIZED, file_path="/a.png",
            created_at=None, confirmed_at=None, line_count=2,
        ),
        PageSummary(
            id=8, author_id=1, status=PageStatus.RECOGNIZED, file_path="/b.png",
            created_at=None, confirmed_at=None, line_count=1,
        ),
    ]

    LexiconAnnotator(provider).annotate_summaries(summaries)

    assert [summary.oov_count for summary in summaries] == [3, 1]
    assert all(summary.lexicon_available for summary in summaries)


def test_annotate_summaries_reports_unavailable_lexicon_as_none():
    summaries = [
        PageSummary(
            id=7, author_id=1, status=PageStatus.RECOGNIZED, file_path="/a.png",
            created_at=None, confirmed_at=None, line_count=1,
        )
    ]
    LexiconAnnotator(FakeLexiconProvider(available=False)).annotate_summaries(summaries)
    assert summaries[0].oov_count is None
    assert summaries[0].lexicon_available is False


# ---------------------------------------------------------------------------
# the author corpus
# ---------------------------------------------------------------------------


@pytest.fixture()
def shared_cache():
    return AuthorCorpusCache()


@pytest.fixture()
def page_repo(db_session, shared_cache):
    return SqlAlchemyPageRepository(db_session, corpus_cache=shared_cache)


def test_corpus_counts_only_confirmed_pages_as_author_vocabulary(
    db_session, page_repo, shared_cache, user, author
):
    page = page_repo.create_page(user.id, author.id, "/p.png", 200, 120)
    page_repo.save_recognition(page.id, recognition_result(), None)
    corpus = SqlAlchemyAuthorCorpus(db_session, cache=shared_cache)

    # recognized but not confirmed: usable for the page totals, not as vocabulary
    assert corpus.page_texts(author.id)[page.id] == ["Уж очень дед", "на еврея"]
    assert corpus.known_words(author.id) == frozenset()

    page_repo.confirm_page(page.id, confirmed_at=None, prediction_cer=None, prediction_wer=None)

    known = corpus.known_words(author.id)
    assert {"уж", "очень", "дед", "на", "еврея"} <= known
    assert corpus.confirmed_texts(author.id)[page.id] == ["Уж очень дед", "на еврея"]


def test_corpus_cache_is_invalidated_by_a_confirmation_and_a_deletion(
    db_session, page_repo, shared_cache, user, author
):
    page = page_repo.create_page(user.id, author.id, "/p.png", 200, 120)
    page_repo.save_recognition(page.id, recognition_result(), None)
    corpus = SqlAlchemyAuthorCorpus(db_session, cache=shared_cache)

    # cached while the page is only recognized
    assert corpus.known_words(author.id) == frozenset()

    page_repo.confirm_page(page.id, confirmed_at=None, prediction_cer=None, prediction_wer=None)
    assert "еврея" in corpus.known_words(author.id)

    page_repo.delete_page(page.id)
    assert corpus.known_words(author.id) == frozenset()


def test_corpus_page_texts_follow_an_edit(db_session, page_repo, shared_cache, user, author):
    page = page_repo.create_page(user.id, author.id, "/p.png", 200, 120)
    page_repo.save_recognition(page.id, recognition_result(), None)
    corpus = SqlAlchemyAuthorCorpus(db_session, cache=shared_cache)
    assert corpus.page_texts(author.id)[page.id][0] == "Уж очень дед"

    line_id = page_repo.get_page(page.id).lines[0].id
    page_repo.apply_line_update(page.id, line_id, "Совсем другой текст")

    assert corpus.page_texts(author.id)[page.id][0] == "Совсем другой текст"


def test_corpus_includes_knowledge_vocabulary(db_session, shared_cache, user, author):
    from app.models import Entity

    db_session.add(
        Entity(
            user_id=user.id,
            entity_type="PLACE",
            canonical_name="Шпалозавод",
            normalized_name="шпалозавод",
        )
    )
    db_session.commit()

    corpus = SqlAlchemyAuthorCorpus(db_session, cache=shared_cache)
    assert "шпалозавод" in corpus.known_words(author.id)
