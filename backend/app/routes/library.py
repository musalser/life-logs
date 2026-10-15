import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..deps import get_current_user, get_db
from ..models import User
from ..schemas import (
    FolderCreateRequest,
    FolderResponse,
    FolderUpdateRequest,
    LibraryTreeResponse,
    MovePageRequest,
)
from ..services.library_service import LibraryError, LibraryService

router = APIRouter(prefix="/library", tags=["Library"])
logger = logging.getLogger(__name__)


def _get_user(db: Session, username: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def _service(db: Session) -> LibraryService:
    return LibraryService(db)


def _map_error(exc: LibraryError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/tree", response_model=LibraryTreeResponse)
def tree(db: Session = Depends(get_db), username: str = Depends(get_current_user)):
    """Дерево «Записей»: авторы (объекты) → вложенные папки → страницы."""
    user = _get_user(db, username)
    return _service(db).tree(user)


@router.post("/folders", response_model=FolderResponse, status_code=status.HTTP_201_CREATED)
def create_folder(
    request: FolderCreateRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    user = _get_user(db, username)
    try:
        return _service(db).create_folder(user, request.name, request.owner_entity_id, request.parent_id)
    except LibraryError as exc:
        raise _map_error(exc)


@router.patch("/folders/{folder_id}", response_model=FolderResponse)
def update_folder(
    folder_id: int,
    request: FolderUpdateRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    user = _get_user(db, username)
    try:
        return _service(db).update_folder(user, folder_id, request.name, request.parent_id)
    except LibraryError as exc:
        raise _map_error(exc)


@router.delete("/folders/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_folder(
    folder_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    user = _get_user(db, username)
    try:
        _service(db).delete_folder(user, folder_id)
    except LibraryError as exc:
        raise _map_error(exc)
    return None


@router.patch("/pages/{source_type}/{ref_id}/folder", status_code=status.HTTP_204_NO_CONTENT)
def move_page(
    source_type: str,
    ref_id: int,
    request: MovePageRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Переместить страницу в папку (``folder_id`` = null — в корень автора)."""
    user = _get_user(db, username)
    try:
        _service(db).move_page(user, source_type, ref_id, request.folder_id)
    except LibraryError as exc:
        raise _map_error(exc)
    return None
