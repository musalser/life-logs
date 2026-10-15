"""Тесты сервиса библиотеки: дерево авторов → папок → страниц, CRUD папок, перемещение."""
from __future__ import annotations

import pytest

from app.models import DiaryPage, Entity, Folder, HTRAuthor, HTRPage, User
from app.services.library_service import LibraryError, LibraryService


@pytest.fixture()
def self_entity(db_session, user):
    entity = Entity(
        user_id=user.id,
        entity_type="person",
        canonical_name="Я",
        normalized_name="я",
    )
    db_session.add(entity)
    db_session.commit()
    user.self_entity_id = entity.id
    db_session.commit()
    return entity


@pytest.fixture()
def author_entity(db_session, user):
    entity = Entity(
        user_id=user.id,
        entity_type="person",
        canonical_name="Баба Галя",
        normalized_name="баба галя",
    )
    db_session.add(entity)
    db_session.commit()
    author = HTRAuthor(user_id=user.id, name="Баба Галя", entity_id=entity.id)
    db_session.add(author)
    db_session.commit()
    return entity


def test_tree_has_self_then_authors_with_nested_folders(db_session, user, self_entity, author_entity):
    root = Folder(user_id=user.id, owner_entity_id=author_entity.id, name="1975")
    db_session.add(root)
    db_session.commit()
    child = Folder(user_id=user.id, owner_entity_id=author_entity.id, name="письма", parent_id=root.id)
    db_session.add(child)
    db_session.commit()
    page = HTRPage(
        user_id=user.id,
        author_id=db_session.query(HTRAuthor).filter(HTRAuthor.entity_id == author_entity.id).one().id,
        file_path="x.jpg",
        file_name="scan.jpg",
        order_index=0,
        status="CONFIRMED",
        folder_id=child.id,
    )
    db_session.add(page)
    db_session.commit()
    diary = DiaryPage(user_id=user.id, title="Запись", content="текст")
    db_session.add(diary)
    db_session.commit()

    tree = LibraryService(db_session).tree(user)

    assert [a.entity_id for a in tree.authors] == [self_entity.id, author_entity.id]
    assert tree.authors[0].is_self is True
    # дневник лежит под «Я»
    assert [p.ref_id for p in tree.authors[0].pages] == [diary.id]
    # у автора — вложенная папка со страницей
    author_node = tree.authors[1]
    assert author_node.folders[0].name == "1975"
    assert author_node.folders[0].children[0].name == "письма"
    assert author_node.folders[0].children[0].pages[0].ref_id == page.id
    assert author_node.folders[0].children[0].pages[0].type == "htr"


def test_create_and_nest_folder(db_session, user, author_entity):
    service = LibraryService(db_session)
    parent = service.create_folder(user, "1980", author_entity.id, None)
    child = service.create_folder(user, "внутри", author_entity.id, parent.id)

    assert child.parent_id == parent.id
    assert child.owner_entity_id == author_entity.id


def test_folder_owner_must_be_an_author(db_session, user, self_entity):
    stranger = Entity(user_id=user.id, entity_type="person", canonical_name="X", normalized_name="x")
    db_session.add(stranger)
    db_session.commit()
    with pytest.raises(LibraryError):
        LibraryService(db_session).create_folder(user, "папка", stranger.id, None)


def test_delete_folder_promotes_children(db_session, user, author_entity):
    service = LibraryService(db_session)
    parent = service.create_folder(user, "1980", author_entity.id, None)
    child = service.create_folder(user, "внутри", author_entity.id, parent.id)
    page = HTRPage(
        user_id=user.id,
        author_id=db_session.query(HTRAuthor).filter(HTRAuthor.entity_id == author_entity.id).one().id,
        file_path="x.jpg",
        file_name="scan.jpg",
        status="CONFIRMED",
        folder_id=child.id,
    )
    db_session.add(page)
    db_session.commit()

    service.delete_folder(user, parent.id)

    # дочерняя папка поднялась в корень автора, а страница осталась в ней
    assert db_session.query(Folder).filter(Folder.id == child.id).one().parent_id is None
    assert db_session.query(HTRPage).filter(HTRPage.id == page.id).one().folder_id == child.id


def test_move_page_and_owner_guard(db_session, user, self_entity, author_entity):
    diary = DiaryPage(user_id=user.id, title="Запись", content="текст")
    db_session.add(diary)
    db_session.commit()
    self_folder = LibraryService(db_session).create_folder(user, "моё", self_entity.id, None)
    author_folder = LibraryService(db_session).create_folder(user, "чужое", author_entity.id, None)

    service = LibraryService(db_session)
    service.move_page(user, "diary", diary.id, self_folder.id)
    assert db_session.query(DiaryPage).filter(DiaryPage.id == diary.id).one().folder_id == self_folder.id

    with pytest.raises(LibraryError):
        service.move_page(user, "diary", diary.id, author_folder.id)
