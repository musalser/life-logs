"""The character LM is rebuilt from the confirmed corpus at training time.

Kept apart from the CLI in ``scripts/build_htr_lm.py``: these tests pin the
artifact contract the recognizer depends on (``meta["running_text_chars"]``, the
target file name) and the corpus rule — only confirmed pages, and a page returned
to editing disappears.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.htr.infrastructure.lm.builder import (
    author_lines,
    build_author_lm,
    rebuild_author_char_lm,
    word_form_texts,
)


def confirmed_page(db_session, author, text):
    """A confirmed page carrying one line, as the corpus sees it."""
    from app.models import HTRLine, HTRPage

    page = HTRPage(
        user_id=author.user_id,
        author_id=author.id,
        status="CONFIRMED",
        file_path="page.png",
    )
    db_session.add(page)
    db_session.flush()
    db_session.add(
        HTRLine(
            page_id=page.id,
            order_index=0,
            x1=0,
            y1=0,
            x2=100,
            y2=20,
            predicted_text=text,
            corrected_text=text,
        )
    )
    db_session.commit()
    return page


@pytest.fixture()
def storage(tmp_path, monkeypatch):
    """A minimal storage tree with one prose file and no word lists."""
    from app.config import settings

    (tmp_path / "lm" / "texts").mkdir(parents=True)
    (tmp_path / "lm" / "texts" / "ru.txt").write_text(
        "дед был кузнецом и жил у реки\nбабушка пряла шерсть\n", encoding="utf-8"
    )
    monkeypatch.setattr(settings, "htr_storage_dir", str(tmp_path))
    return tmp_path


def test_author_lines_come_only_from_confirmed_pages(db_session, author, storage):
    draft = confirmed_page(db_session, author, "черновая строка")
    draft.status = "EDITING"
    db_session.commit()
    confirmed_page(db_session, author, "подтверждённая строка")

    assert author_lines(db_session, author.id) == ["подтверждённая строка"]


def test_built_model_keeps_the_artifact_contract(db_session, author, storage):
    confirmed_page(db_session, author, "дед был кузнецом")

    model = build_author_lm(db_session, author.id, order=4, author_weight=5)

    assert model.order == 4
    meta = model.meta
    assert meta["author_id"] == author.id
    assert meta["author_weight"] == 5
    assert meta["author_chars"] > 0
    # the recognizer refuses beam search when this is too small to be trusted
    assert meta["running_text_chars"] >= meta["author_chars"] > 0
    assert meta["total_chars"] > 0


def test_rebuild_writes_where_the_recognizer_looks(db_session, author, storage):
    confirmed_page(db_session, author, "дед был кузнецом")

    path = rebuild_author_char_lm(db_session, author.id, order=4)

    expected = storage / "lm" / f"author_{author.id}_char_lm.npz"
    assert path == expected
    assert expected.is_file()

    from app.htr.infrastructure.lm.char_ngram import CharNGram

    reloaded = CharNGram.load(expected)
    assert reloaded.meta["author_id"] == author.id


def test_rebuild_is_skipped_without_confirmed_text(db_session, author, storage):
    """With nothing confirmed yet the general model is the right one to keep."""
    assert rebuild_author_char_lm(db_session, author.id) is None
    assert not (storage / "lm" / f"author_{author.id}_char_lm.npz").exists()


def test_word_form_files_are_read_as_cyrillic_forms(storage):
    downloads = storage / "lexicon" / "downloads"
    downloads.mkdir(parents=True)
    (downloads / "russian_nouns.txt").write_text(
        "дед\n# комментарий\n\nкузнец\n", encoding="cp1251"
    )

    parts = word_form_texts(sorted(downloads.glob("*russian*.txt")))

    assert len(parts) == 1
    assert parts[0].split("\n") == ["дед", "кузнец"]


def test_author_id_in_meta_survives_a_round_trip(db_session, author, storage):
    confirmed_page(db_session, author, "старая строка")

    path = Path(rebuild_author_char_lm(db_session, author.id, order=3))
    assert path.name == f"author_{author.id}_char_lm.npz"
