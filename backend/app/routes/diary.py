
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from ..deps import get_current_user, get_db, get_llama_service
from ..models import DiaryPage, User
from ..services.llama_client import LlamaService
from ..schemas import (
	DiaryPageCreateRequest,
	DiaryPageResponse,
	DiaryPageUpdateRequest,
)


router = APIRouter(prefix="/diary", tags=["Diary"])


def _get_user_by_username(db: Session, username: str) -> User:
	user = db.query(User).filter(User.username == username).first()
	if not user:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="User not found",
		)
	return user


@router.post("/pages", response_model=DiaryPageResponse, status_code=status.HTTP_201_CREATED)
def create_diary_page(
	request: DiaryPageCreateRequest,
	db: Session = Depends(get_db),
	username: str = Depends(get_current_user),
	llama_service: LlamaService | None = Depends(get_llama_service),
):
	user = _get_user_by_username(db, username)
	title = "Новая запись"
	if llama_service is not None:
		title = llama_service.generate_diary_title(request.content)

	diary_page = DiaryPage(
		user_id=user.id,
		title=title,
		content=request.content,
	)
	db.add(diary_page)
	db.commit()
	db.refresh(diary_page)
	return diary_page


@router.get("/pages", response_model=List[DiaryPageResponse])
def list_diary_pages(
	limit: int = 50,
	offset: int = 0,
	db: Session = Depends(get_db),
	username: str = Depends(get_current_user),
):
	user = _get_user_by_username(db, username)

	pages = (
		db.query(DiaryPage)
		.filter(DiaryPage.user_id == user.id)
		.order_by(DiaryPage.id.desc())
		.offset(offset)
		.limit(limit)
		.all()
	)
	return pages


@router.get("/pages/{page_id}", response_model=DiaryPageResponse)
def get_diary_page(
	page_id: int,
	db: Session = Depends(get_db),
	username: str = Depends(get_current_user),
):
	user = _get_user_by_username(db, username)
	diary_page = (
		db.query(DiaryPage)
		.filter(DiaryPage.id == page_id, DiaryPage.user_id == user.id)
		.first()
	)
	if not diary_page:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail="Diary page not found",
		)
	return diary_page


@router.put("/pages/{page_id}", response_model=DiaryPageResponse)
def update_diary_page(
	page_id: int,
	request: DiaryPageUpdateRequest,
	db: Session = Depends(get_db),
	username: str = Depends(get_current_user),
):
	user = _get_user_by_username(db, username)
	diary_page = (
		db.query(DiaryPage)
		.filter(DiaryPage.id == page_id, DiaryPage.user_id == user.id)
		.first()
	)
	if not diary_page:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail="Diary page not found",
		)

	diary_page.content = request.content
	db.commit()
	db.refresh(diary_page)
	return diary_page


@router.delete("/pages/{page_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_diary_page(
	page_id: int,
	db: Session = Depends(get_db),
	username: str = Depends(get_current_user),
):
	user = _get_user_by_username(db, username)
	diary_page = (
		db.query(DiaryPage)
		.filter(DiaryPage.id == page_id, DiaryPage.user_id == user.id)
		.first()
	)
	if not diary_page:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail="Diary page not found",
		)

	db.delete(diary_page)
	db.commit()
	return Response(status_code=status.HTTP_204_NO_CONTENT)

