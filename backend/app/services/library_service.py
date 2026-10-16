"""Единая библиотека «Записи»: дерево авторов → папок → страниц.

Раньше дневник и рукописи жили на разных вкладках, а «папки» рукописей были
лишь выводом из клиентского пути загрузки (``source_path``). Теперь это одна
структура: в корне — объекты авторов («Я» и объекты, к которым привязаны
авторы рукописей), внутри — вложенные папки, которые настраивает пользователь,
и страницы двух типов (``diary`` и ``htr``).

Сервис только читает и двигает дерево; сами редакторы страниц остаются на
своих эндпоинтах (``/diary/pages``, ``/htr/pages``).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ..models import DiaryPage, Entity, Folder, HTRAuthor, HTRPage, User
from ..schemas import (
    FolderResponse,
    LibraryAuthorNode,
    LibraryFolderNode,
    LibraryPageNode,
    LibraryTreeResponse,
)

DIARY = "diary"
HTR = "htr"


class LibraryError(RuntimeError):
    """The tree cannot be changed the way it was asked."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def clean_folder_name(name: str | None) -> str:
    """A folder name as it will be stored: one space between words, capped."""
    return " ".join((name or "").split())[:255]


class LibraryService:
    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Author objects (the roots of the tree)
    # ------------------------------------------------------------------

    def _author_entities(self, user: User) -> list[Entity]:
        """Roots of the tree: «Я» first, then objects bound to manuscript authors."""
        self_id = user.self_entity_id
        bound = (
            self.db.query(Entity)
            .join(HTRAuthor, HTRAuthor.entity_id == Entity.id)
            .filter(HTRAuthor.user_id == user.id)
            .order_by(Entity.canonical_name, Entity.id)
            .all()
        )
        ordered: list[Entity] = []
        seen: set[int] = set()
        if self_id:
            self_entity = self.db.query(Entity).filter(Entity.id == self_id).first()
            if self_entity is not None:
                ordered.append(self_entity)
                seen.add(self_entity.id)
        for entity in bound:
            if entity.id not in seen:
                ordered.append(entity)
                seen.add(entity.id)
        return ordered

    def _is_author_entity(self, user: User, entity_id: int | None) -> bool:
        if user.self_entity_id == entity_id:
            return True
        return (
            self.db.query(HTRAuthor)
            .filter(HTRAuthor.user_id == user.id, HTRAuthor.entity_id == entity_id)
            .first()
            is not None
        )

    # ------------------------------------------------------------------
    # The tree
    # ------------------------------------------------------------------

    def tree(self, user: User) -> LibraryTreeResponse:
        authors: list[LibraryAuthorNode] = []
        for entity in self._author_entities(user):
            authors.append(self._author_node(user, entity))
        return LibraryTreeResponse(authors=authors)

    def _author_node(self, user: User, entity: Entity) -> LibraryAuthorNode:
        folders = (
            self.db.query(Folder)
            .filter(Folder.user_id == user.id, Folder.owner_entity_id == entity.id)
            .order_by(Folder.order_index, Folder.name, Folder.id)
            .all()
        )
        pages = self._pages_for(user, entity)
        pages_by_folder: dict[int | None, list[LibraryPageNode]] = {}
        for page in pages:
            pages_by_folder.setdefault(page.folder_id, []).append(page)

        root_folders, root_pages = self._build_folders(folders, pages_by_folder)
        htr_author = (
            self.db.query(HTRAuthor)
            .filter(HTRAuthor.user_id == user.id, HTRAuthor.entity_id == entity.id)
            .order_by(HTRAuthor.id)
            .first()
        )
        return LibraryAuthorNode(
            entity_id=entity.id,
            name=entity.canonical_name,
            is_self=entity.id == user.self_entity_id,
            author_id=htr_author.id if htr_author is not None else None,
            folders=root_folders,
            pages=root_pages,
        )

    def _pages_for(self, user: User, entity: Entity) -> list[LibraryPageNode]:
        pages: list[LibraryPageNode] = []
        if entity.id == user.self_entity_id:
            rows = (
                self.db.query(DiaryPage)
                .filter(DiaryPage.user_id == user.id)
                .order_by(DiaryPage.id.desc())
                .all()
            )
            for row in rows:
                pages.append(
                    LibraryPageNode(
                        type=DIARY,
                        ref_id=row.id,
                        title=row.title or f"Запись #{row.id}",
                        folder_id=row.folder_id,
                        order_index=0,
                        status=None,
                        created_at=row.created_at,
                    )
                )
        htr_rows = (
            self.db.query(HTRPage)
            .join(HTRAuthor, HTRAuthor.id == HTRPage.author_id)
            .filter(HTRAuthor.user_id == user.id, HTRAuthor.entity_id == entity.id)
            .order_by(HTRPage.order_index, HTRPage.id)
            .all()
        )
        for row in htr_rows:
            pages.append(
                LibraryPageNode(
                    type=HTR,
                    ref_id=row.id,
                    title=row.file_name or f"Стр. #{row.id}",
                    folder_id=row.folder_id,
                    order_index=row.order_index,
                    status=row.status,
                    created_at=row.created_at,
                )
            )
        return pages

    @staticmethod
    def _build_folders(
        folders: list[Folder], pages_by_folder: dict[int | None, list[LibraryPageNode]]
    ) -> tuple[list[LibraryFolderNode], list[LibraryPageNode]]:
        by_parent: dict[int | None, list[Folder]] = {}
        for folder in folders:
            by_parent.setdefault(folder.parent_id, []).append(folder)

        def build(parent_id: int | None) -> list[LibraryFolderNode]:
            result: list[LibraryFolderNode] = []
            for folder in by_parent.get(parent_id, []):
                result.append(
                    LibraryFolderNode(
                        id=folder.id,
                        name=folder.name,
                        children=build(folder.id),
                        pages=pages_by_folder.get(folder.id, []),
                    )
                )
            return result

        return build(None), pages_by_folder.get(None, [])

    # ------------------------------------------------------------------
    # Folder CRUD
    # ------------------------------------------------------------------

    def _get_folder(self, user: User, folder_id: int) -> Folder:
        folder = (
            self.db.query(Folder)
            .filter(Folder.id == folder_id, Folder.user_id == user.id)
            .first()
        )
        if folder is None:
            raise LibraryError("Папка не найдена")
        return folder

    def _validate_parent(
        self, user: User, owner_entity_id: int | None, parent_id: int | None
    ) -> None:
        if parent_id is None:
            return
        parent = self._get_folder(user, parent_id)
        if parent.owner_entity_id != owner_entity_id:
            raise LibraryError("Папка должна принадлежать тому же автору")

    def _validate_no_cycle(self, folder: Folder, new_parent_id: int | None) -> None:
        if new_parent_id is None:
            return
        node = new_parent_id
        while node is not None:
            if node == folder.id:
                raise LibraryError("Нельзя вложить папку саму в себя")
            parent = self.db.query(Folder).filter(Folder.id == node).first()
            node = parent.parent_id if parent is not None else None

    def create_folder(
        self, user: User, name: str, owner_entity_id: int | None, parent_id: int | None
    ) -> FolderResponse:
        name = clean_folder_name(name)
        if not name:
            raise LibraryError("Имя папки не может быть пустым")
        if not self._is_author_entity(user, owner_entity_id):
            raise LibraryError("Папку можно создать только для автора (объекта)")
        self._validate_parent(user, owner_entity_id, parent_id)
        folder = Folder(
            user_id=user.id,
            owner_entity_id=owner_entity_id,
            name=name,
            parent_id=parent_id,
        )
        self.db.add(folder)
        self.db.commit()
        self.db.refresh(folder)
        return self._folder_response(folder)

    def ensure_folder(
        self, user: User, name: str, owner_entity_id: int | None, parent_id: int | None
    ) -> tuple[FolderResponse, bool]:
        """The folder of this name, created when it does not exist yet.

        Returns the folder and whether this call created it. A bulk import (a
        PDF) names its folder after the file, so re-importing the same PDF lands
        in the same folder instead of piling up «Дневник», «Дневник (2)» … A
        folder that is already there is reused whatever it holds.
        """
        cleaned = clean_folder_name(name)
        if not cleaned:
            raise LibraryError("Имя папки не может быть пустым")
        if not self._is_author_entity(user, owner_entity_id):
            raise LibraryError("Папку можно создать только для автора (объекта)")
        self._validate_parent(user, owner_entity_id, parent_id)
        # Case is compared in Python, not in SQL: `lower()` does not know Cyrillic
        # in every backend this runs on (SQLite in the tests), and the author's
        # folders are a handful of rows, exactly like the page names above.
        siblings = (
            self.db.query(Folder)
            .filter(
                Folder.user_id == user.id,
                Folder.owner_entity_id == owner_entity_id,
                Folder.parent_id == parent_id,
            )
            .order_by(Folder.id)
            .all()
        )
        wanted = cleaned.casefold()
        for folder in siblings:
            if folder.name.casefold() == wanted:
                return self._folder_response(folder), False
        return self.create_folder(user, cleaned, owner_entity_id, parent_id), True

    def folder_for_upload(
        self, user: User, folder_id: int, owner_entity_id: int | None
    ) -> FolderResponse:
        """The folder an upload may go into, checked against the author's object."""
        folder = self._get_folder(user, folder_id)
        if folder.owner_entity_id != owner_entity_id:
            raise LibraryError("Страницу можно положить только в папку её автора")
        return self._folder_response(folder)

    def update_folder(
        self, user: User, folder_id: int, name: str | None, parent_id: int | None
    ) -> FolderResponse:
        folder = self._get_folder(user, folder_id)
        if name is not None:
            cleaned = clean_folder_name(name)
            if not cleaned:
                raise LibraryError("Имя папки не может быть пустым")
            folder.name = cleaned
        if parent_id is not None:
            self._validate_parent(user, folder.owner_entity_id, parent_id)
            self._validate_no_cycle(folder, parent_id)
            folder.parent_id = parent_id
        folder.updated_at = _now()
        self.db.commit()
        self.db.refresh(folder)
        return self._folder_response(folder)

    def delete_folder(self, user: User, folder_id: int) -> None:
        folder = self._get_folder(user, folder_id)
        parent_id = folder.parent_id
        # move subfolders and pages one level up, so deleting a folder never
        # deletes its contents
        for child in self.db.query(Folder).filter(Folder.parent_id == folder.id).all():
            child.parent_id = parent_id
        self.db.query(DiaryPage).filter(DiaryPage.folder_id == folder.id).update(
            {DiaryPage.folder_id: parent_id}, synchronize_session=False
        )
        self.db.query(HTRPage).filter(HTRPage.folder_id == folder.id).update(
            {HTRPage.folder_id: parent_id}, synchronize_session=False
        )
        self.db.delete(folder)
        self.db.commit()

    # ------------------------------------------------------------------
    # Moving a page between folders
    # ------------------------------------------------------------------

    def move_page(
        self, user: User, source_type: str, ref_id: int, folder_id: int | None
    ) -> None:
        if source_type == DIARY:
            page = (
                self.db.query(DiaryPage)
                .filter(DiaryPage.id == ref_id, DiaryPage.user_id == user.id)
                .first()
            )
            if page is None:
                raise LibraryError("Запись дневника не найдена")
            if folder_id is not None:
                folder = self._get_folder(user, folder_id)
                if folder.owner_entity_id != user.self_entity_id:
                    raise LibraryError("Запись дневника можно положить только в папку «Я»")
            page.folder_id = folder_id
        elif source_type == HTR:
            page = (
                self.db.query(HTRPage)
                .filter(HTRPage.id == ref_id, HTRPage.user_id == user.id)
                .first()
            )
            if page is None:
                raise LibraryError("Страница рукописи не найдена")
            author = (
                self.db.query(HTRAuthor).filter(HTRAuthor.id == page.author_id).first()
            )
            owner = author.entity_id if author is not None else None
            if folder_id is not None:
                folder = self._get_folder(user, folder_id)
                if folder.owner_entity_id != owner:
                    raise LibraryError("Страницу можно положить только в папку её автора")
            page.folder_id = folder_id
        else:
            raise LibraryError(f"Неизвестный тип страницы: {source_type!r}")
        self.db.commit()

    # ------------------------------------------------------------------
    # Payload
    # ------------------------------------------------------------------

    @staticmethod
    def _folder_response(folder: Folder) -> FolderResponse:
        return FolderResponse(
            id=folder.id,
            name=folder.name,
            parent_id=folder.parent_id,
            owner_entity_id=folder.owner_entity_id,
            order_index=folder.order_index,
            created_at=folder.created_at,
        )
