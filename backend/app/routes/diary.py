
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from ..deps import (
	get_current_user,
	get_db,
	get_diary_service,
	get_knowledge_source_service,
)
from ..models import DiaryPage, User
from app.services.diary_service import DiaryService
from app.services.knowledge_source_service import KnowledgeSourceService
from ..schemas import (
	DiaryPageCreateRequest,
	DiaryPageResponse,
	DiaryPageUpdateRequest,
	ExtractionSummaryResponse,
)


router = APIRouter(prefix="/diary", tags=["Diary"])
logger = logging.getLogger(__name__)


def _get_user_by_username(db: Session, username: str) -> User:
	user = db.query(User).filter(User.username == username).first()
	if not user:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="User not found",
		)
	return user


@router.post("/pages", response_model=DiaryPageResponse, status_code=status.HTTP_201_CREATED)
async def create_diary_page(
	request: DiaryPageCreateRequest,
	db: Session = Depends(get_db),
	username: str = Depends(get_current_user),
	diary_service: DiaryService = Depends(get_diary_service),
	source_service: KnowledgeSourceService = Depends(get_knowledge_source_service),
):
	logger.info("Creating diary page for user %s", username)
	user = _get_user_by_username(db, username)
	title = "Новая запись"
	try:
		title = await diary_service.generate_diary_title(request.content)
	except Exception as e:
		logger.exception("Failed to generate diary title")

	diary_page = DiaryPage(
		user_id=user.id,
		title=title,
		content=request.content,
	)
	db.add(diary_page)
	db.commit()
	db.refresh(diary_page)

	# The page becomes a knowledge source; sync mode runs the pipeline here,
	# async mode hands it to Celery (7.5).
	try:
		source, created, changed = source_service.refresh_from_diary_page(diary_page)
		result = await source_service.request_extraction(source)
		logger.info(
			"Knowledge source for page_id=%s: created=%s changed=%s result=%s",
			diary_page.id, created, changed, result,
		)
	except Exception:
		logger.exception("Knowledge extraction failed for page_id=%s", diary_page.id)

	return diary_page


@router.post("/pages/{page_id}/extract", response_model=ExtractionSummaryResponse)
async def extract_page_knowledge(
	page_id: int,
	db: Session = Depends(get_db),
	username: str = Depends(get_current_user),
	source_service: KnowledgeSourceService = Depends(get_knowledge_source_service),
):
	logger.info("Manual knowledge extraction for page %s by user %s", page_id, username)
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

	source, _, _ = source_service.refresh_from_diary_page(diary_page)
	summary = await source_service.request_extraction(source)
	logger.info("Knowledge extraction for page_id=%s: %s", diary_page.id, summary)
	return {"page_id": diary_page.id, "summary": summary}


@router.get("/pages", response_model=List[DiaryPageResponse])
async def list_diary_pages(
	limit: int = 50,
	offset: int = 0,
	db: Session = Depends(get_db),
	username: str = Depends(get_current_user),
):
	logger.info("Listing diary pages for user %s with limit=%s offset=%s", username, limit, offset)
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
async def get_diary_page(
	page_id: int,
	db: Session = Depends(get_db),
	username: str = Depends(get_current_user),
):
	logger.info("Fetching diary page %s for user %s", page_id, username)
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
async def update_diary_page(
	page_id: int,
	request: DiaryPageUpdateRequest,
	db: Session = Depends(get_db),
	username: str = Depends(get_current_user),
	source_service: KnowledgeSourceService = Depends(get_knowledge_source_service),
):
	logger.info("Updating diary page %s for user %s", page_id, username)
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

	# An edited page means edited text: the source is refreshed, and a changed
	# hash marks the old knowledge stale (7.5, переизвлечение при изменении текста)
	try:
		source, _, changed = source_service.refresh_from_diary_page(diary_page)
		result = await source_service.request_extraction(source)
		logger.info(
			"Knowledge source for page_id=%s: changed=%s result=%s", diary_page.id, changed, result
		)
	except Exception:
		logger.exception("Knowledge extraction failed for page_id=%s", diary_page.id)
	return diary_page


@router.delete("/pages/{page_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_diary_page(
	page_id: int,
	db: Session = Depends(get_db),
	username: str = Depends(get_current_user),
	source_service: KnowledgeSourceService = Depends(get_knowledge_source_service),
):
	logger.info("Deleting diary page %s for user %s", page_id, username)
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

	# the source and everything read from it go with the page
	source = source_service.get("diary", diary_page.id, user.id)
	if source is not None:
		source_service.delete_source(source)
	db.delete(diary_page)
	db.commit()
	return Response(status_code=status.HTTP_204_NO_CONTENT)

