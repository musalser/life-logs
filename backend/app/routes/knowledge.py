import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session, joinedload

from ..config import settings
from ..deps import get_current_user, get_db, get_knowledge_source_service
from ..models import (
    DiaryPage,
    Entity,
    EntityMention,
    EntityRelation,
    Fact,
    FactConflict,
    Goal,
    HTRPage,
    KnowledgeSource,
    User,
)
from ..schemas import (
    AnalyticsRefreshRequest,
    AnalyticsResponse,
    EntityFactResponse,
    FactConflictResponse,
    FactResponse,
    GoalResponse,
    KnowledgeObjectCreateRequest,
    KnowledgeObjectResponse,
    KnowledgeObjectUpdateRequest,
    KnowledgeResponse,
    KnowledgeStatsResponse,
    ObjectMergeRequest,
    RelationGraphResponse,
    RelationWithPath,
)
from ..services.analytics_service import AnalyticsService
from ..services.entity_facts import facts_for_entity
from ..services.entity_merge import MergeError, merge_entities
from ..services.knowledge_graph_service import KnowledgeGraphService
from ..services.knowledge_source_service import (
    KnowledgeSourceService,
    SourceError,
    source_status_payload,
)


router = APIRouter(prefix="/knowledge", tags=["Knowledge"])
logger = logging.getLogger(__name__)

SOURCE_TYPES = ("diary", "htr")


def _get_user_by_username(db: Session, username: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


def _parse_aliases(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(value) for value in values] if isinstance(values, list) else []


def _object_payload(entity: Entity, *, self_entity_id: int | None, relation_count: int = 0) -> KnowledgeObjectResponse:
    return KnowledgeObjectResponse(
        id=entity.id,
        name=entity.canonical_name,
        entity_type=entity.entity_type,
        description=entity.description,
        aliases=_parse_aliases(entity.aliases_json),
        is_self=entity.id == self_entity_id,
        relation_count=relation_count,
        last_seen_at=entity.last_seen_at,
    )


def _subject_or_self(db: Session, user: User, subject_entity_id: int | None) -> Entity:
    """Объект-перспектива: выбранный или собственный объект аккаунта."""
    if subject_entity_id is None:
        service = KnowledgeSourceService(db)
        entity = service.ensure_self_entity(user.id)
        db.commit()
        return entity
    entity = (
        db.query(Entity)
        .filter(Entity.id == subject_entity_id, Entity.user_id == user.id)
        .first()
    )
    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge object not found",
        )
    return entity


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------


@router.get("/objects", response_model=list[KnowledgeObjectResponse])
async def list_objects(
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Объекты для селектора: «Я» первым, дальше персоны по последней встрече."""
    user = _get_user_by_username(db, username)
    self_entity = _subject_or_self(db, user, None)
    entities = (
        db.query(Entity)
        .filter(Entity.user_id == user.id, Entity.entity_type == "person")
        .order_by(Entity.last_seen_at.desc().nullslast(), Entity.id)
        .all()
    )
    counts: dict[int, int] = {}
    for edge in db.query(EntityRelation).filter(EntityRelation.user_id == user.id).all():
        counts[edge.to_entity_id] = counts.get(edge.to_entity_id, 0) + 1
    ordered = [entity for entity in entities if entity.id == self_entity.id] + [
        entity for entity in entities if entity.id != self_entity.id
    ]
    return [
        _object_payload(entity, self_entity_id=self_entity.id, relation_count=counts.get(entity.id, 0))
        for entity in ordered
    ]


@router.post("/objects", response_model=KnowledgeObjectResponse, status_code=status.HTTP_201_CREATED)
async def create_object(
    request: KnowledgeObjectCreateRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Новый объект-персона: к нему можно привязать автора рукописи."""
    user = _get_user_by_username(db, username)
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Name is required")
    normalized = " ".join(name.split()).lower()[:255]
    existing = (
        db.query(Entity)
        .filter_by(user_id=user.id, entity_type="person", normalized_name=normalized)
        .first()
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Объект с таким именем уже есть",
        )
    entity = Entity(
        user_id=user.id,
        entity_type="person",
        canonical_name=name[:255],
        normalized_name=normalized,
        description=(request.description or None),
    )
    entity.aliases_json = json.dumps([alias for alias in request.aliases if alias], ensure_ascii=False) or None
    db.add(entity)
    db.commit()
    db.refresh(entity)
    self_entity = _subject_or_self(db, user, None)
    return _object_payload(entity, self_entity_id=self_entity.id)


@router.patch("/objects/{entity_id}", response_model=KnowledgeObjectResponse)
async def update_object(
    entity_id: int,
    request: KnowledgeObjectUpdateRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Переименование объекта и его алиасы."""
    user = _get_user_by_username(db, username)
    entity = (
        db.query(Entity)
        .filter(Entity.id == entity_id, Entity.user_id == user.id)
        .first()
    )
    if entity is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge object not found")
    if request.name is not None:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Name cannot be empty")
        entity.canonical_name = name[:255]
        entity.normalized_name = " ".join(name.split()).lower()[:255]
    if request.description is not None:
        entity.description = request.description or None
    if request.aliases is not None:
        entity.aliases_json = (
            json.dumps([alias for alias in request.aliases if alias], ensure_ascii=False) or None
        )
    db.commit()
    db.refresh(entity)
    self_entity = _subject_or_self(db, user, None)
    return _object_payload(entity, self_entity_id=self_entity.id)


@router.get("/objects/{entity_id}/facts", response_model=list[EntityFactResponse])
async def object_facts(
    entity_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Факты, в которых участвует объект: субъект, дополнение или упоминание в цитате."""
    user = _get_user_by_username(db, username)
    entity = (
        db.query(Entity)
        .filter(Entity.id == entity_id, Entity.user_id == user.id)
        .first()
    )
    if entity is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge object not found")
    rows = facts_for_entity(db, user.id, entity)
    party_ids = {row.fact.subject_entity_id for row in rows} | {row.fact.object_entity_id for row in rows}
    party_ids.discard(None)
    names = (
        {e.id: e.canonical_name for e in db.query(Entity).filter(Entity.id.in_(party_ids)).all()}
        if party_ids
        else {}
    )
    return [
        EntityFactResponse(
            **FactResponse.model_validate(row.fact).model_dump(),
            match_reasons=row.reasons,
            subject_name=names.get(row.fact.subject_entity_id),
            object_name=names.get(row.fact.object_entity_id),
        )
        for row in rows
    ]


@router.post("/objects/merge", response_model=KnowledgeObjectResponse)
async def merge_objects(
    request: ObjectMergeRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Объединяет два объекта: всё знание ``source`` переносится в ``target``.

    Это ручной выход для случаев, когда извлечение создало объект, который не
    слился по эмбеддингам, но для пользователя это тот же человек (например,
    ещё один объект существующего автора рукописи).
    """
    user = _get_user_by_username(db, username)
    try:
        entity = merge_entities(db, user, request.source_id, request.target_id)
    except MergeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    self_entity = _subject_or_self(db, user, None)
    return _object_payload(entity, self_entity_id=self_entity.id)


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------


@router.get("", response_model=KnowledgeResponse)
async def get_knowledge(
    subject_entity_id: int | None = None,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Знания от перспективы выбранного объекта (по умолчанию — «Я»)."""
    logger.info("Fetching knowledge for user %s subject=%s", username, subject_entity_id)
    user = _get_user_by_username(db, username)
    subject = _subject_or_self(db, user, subject_entity_id)
    graph = KnowledgeGraphService(db, user.id)

    direct = {edge.to_entity_id: edge for edge in graph.edges(subject.id)}
    paths = graph.paths(subject.id)
    people = graph._names(set(direct) | set(paths) | {subject.id})

    relations: list[RelationWithPath] = []
    for entity_id, edge in direct.items():
        relations.append(
            RelationWithPath(
                id=edge.id,
                entity_id=entity_id,
                person_name=people.get(entity_id, "?"),
                relation_type=edge.relation_type,
                confidence=edge.confidence,
                evidence_text=edge.evidence_text,
                updated_at=edge.updated_at,
                direct=True,
                path=[],
                path_label=people.get(entity_id, "?"),
            )
        )
    for entity_id, chain in paths.items():
        last = chain[-1]
        relations.append(
            RelationWithPath(
                id=-entity_id,  # у пути нет собственного ребра
                entity_id=entity_id,
                person_name=people.get(entity_id, last.get("name", "?")),
                relation_type=last.get("relation_type", ""),
                confidence=None,
                evidence_text=None,
                updated_at=None,
                direct=False,
                path=chain,
                path_label=graph.path_label(chain),
            )
        )
    relations.sort(key=lambda item: (not item.direct, item.person_name))

    goals = (
        db.query(Goal)
        .options(joinedload(Goal.progress_notes))
        .filter(Goal.user_id == user.id, Goal.subject_entity_id == subject.id)
        .order_by(Goal.updated_at.desc())
        .all()
    )
    # Событие — факт с важностью: показывается в своём блоке по дате, а не
    # второй раз среди фактов. Привычки выводит аналитика (раздел «habits»).
    threshold = settings.knowledge_event_min_importance
    subject_facts = db.query(Fact).filter(Fact.user_id == user.id, Fact.subject_entity_id == subject.id)
    events = (
        subject_facts.filter(Fact.importance >= threshold)
        .order_by(Fact.time_iso.asc().nullslast(), Fact.id.desc())
        .limit(200)
        .all()
    )
    facts = (
        subject_facts.filter((Fact.importance.is_(None)) | (Fact.importance < threshold))
        .order_by(Fact.id.desc())
        .limit(200)
        .all()
    )
    entities = (
        db.query(Entity)
        .filter(Entity.user_id == user.id, Entity.entity_type == "person")
        .order_by(Entity.last_seen_at.desc().nullslast(), Entity.id)
        .all()
    )
    source_count = (
        db.query(KnowledgeSource)
        .filter(KnowledgeSource.user_id == user.id, KnowledgeSource.extraction_status == "done")
        .count()
    )

    return KnowledgeResponse(
        subject_entity_id=subject.id,
        subject_name=subject.canonical_name,
        goals=[GoalResponse.model_validate(goal) for goal in goals],
        relations=relations,
        events=[FactResponse.model_validate(event) for event in events],
        facts=[FactResponse.model_validate(fact) for fact in facts],
        entities=[
            _object_payload(entity, self_entity_id=subject.id)
            for entity in entities
            if entity.id != subject.id
        ],
        stats=KnowledgeStatsResponse(
            facts=len(facts),
            goals=len(goals),
            events=len(events),
            relations=len(direct),
            people=len({**direct, **paths}),
            sources=source_count,
        ),
    )


# ---------------------------------------------------------------------------
# Conflicts: второе значение функционального предиката
# ---------------------------------------------------------------------------


@router.get("/conflicts", response_model=list[FactConflictResponse])
async def list_conflicts(
    unresolved_only: bool = True,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Конфликты «на проверку».

    У человека одно рождение и одна смерть: если из нового текста приходит
    другое значение, факт не создаётся и не перезаписывается — он попадает
    сюда, чтобы пользователь решил, какое значение верно.
    """
    user = _get_user_by_username(db, username)
    query = db.query(FactConflict).filter(FactConflict.user_id == user.id)
    if unresolved_only:
        query = query.filter(FactConflict.resolved_at.is_(None))
    rows = query.order_by(FactConflict.id.desc()).limit(200).all()
    return [
        FactConflictResponse(
            id=row.id,
            subject_entity_id=row.subject_entity_id,
            subject_name=row.subject_entity.canonical_name if row.subject_entity else None,
            predicate_kind=row.predicate_kind,
            existing_fact_id=row.existing_fact_id,
            existing_value=row.existing_value,
            incoming_value=row.incoming_value,
            source_id=row.source_id,
            source_text=row.source_text,
            time_text=row.time_text,
            confidence=row.confidence,
            resolution=row.resolution,
            created_at=row.created_at,
        )
        for row in rows
    ]


@router.get("/graph", response_model=RelationGraphResponse)
async def relation_graph(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Граф связей целиком: корень и режим выбирает клиент.

    Отдаём весь архив сразу, а не окружение выбранного объекта: уровни считаются
    внутри компоненты родства с постоянным сдвигом, поэтому смена корня — это
    вычитание его уровня на клиенте, а не новый запрос. ``ETag`` — версия знаний:
    после перепрогона или слияния она меняется, и клиент забирает граф заново, а
    без изменений получает 304.
    """
    user = _get_user_by_username(db, username)
    payload = KnowledgeGraphService(db, user.id).relation_graph()
    etag = f'W/"{payload["version"]}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag})
    response.headers["ETag"] = etag
    return RelationGraphResponse(**payload)


# ---------------------------------------------------------------------------
# Sources (8.1)
# ---------------------------------------------------------------------------


def _source_type_or_404(source_type: str) -> str:
    if source_type not in SOURCE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown source type {source_type!r}; expected one of {', '.join(SOURCE_TYPES)}",
        )
    return source_type


@router.get("/source/{source_id}")
async def get_source(
    source_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Источник знания: тип, страница и название — чтобы открыть цитату.

    Карточка знания (факт, событие, цель) знает только ``source_id``; этот
    эндпоинт переводит его в конкретную страницу рукописи или запись дневника,
    куда ведёт клик по цитате.
    """
    user = _get_user_by_username(db, username)
    source = (
        db.query(KnowledgeSource)
        .filter(KnowledgeSource.id == source_id, KnowledgeSource.user_id == user.id)
        .first()
    )
    if source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source not found")
    return {
        "id": source.id,
        "source_type": source.source_type,
        "ref_id": source.ref_id,
        "title": source.title,
        "extraction_status": source.extraction_status,
        "author_entity_id": source.author_entity_id,
    }


@router.get("/sources/{source_type}/{ref_id}/status")
async def get_source_status(
    source_type: str,
    ref_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Extraction status of one source: the UI polls this after a confirmation."""
    user = _get_user_by_username(db, username)
    source = KnowledgeSourceService(db).get(_source_type_or_404(source_type), ref_id, user.id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge source not found for this page",
        )
    return source_status_payload(db, source)


@router.post("/sources/{source_type}/{ref_id}/extract")
async def extract_source(
    source_type: str,
    ref_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    source_service: KnowledgeSourceService = Depends(get_knowledge_source_service),
):
    """Re-runs the pipeline over one source, refreshing its text first.

    The text is re-read because the point of a manual re-run is usually an edit
    that the automatic path missed: a diary page may have changed while a
    manuscript page is only re-read while it is still confirmed.
    """
    user = _get_user_by_username(db, username)
    source_type = _source_type_or_404(source_type)
    source = source_service.get(source_type, ref_id, user.id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge source not found for this page",
        )

    try:
        if source_type == "diary":
            page = (
                db.query(DiaryPage)
                .filter(DiaryPage.id == ref_id, DiaryPage.user_id == user.id)
                .first()
            )
            if page is not None:
                source, _, _ = source_service.refresh_from_diary_page(page)
        else:
            page = (
                db.query(HTRPage)
                .filter(HTRPage.id == ref_id, HTRPage.user_id == user.id)
                .first()
            )
            if page is not None and page.status == "CONFIRMED":
                source, _, _ = source_service.refresh_from_htr_page(page)
    except SourceError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    return await source_service.request_extraction(source)


# ---------------------------------------------------------------------------
# Analytics (10)
# ---------------------------------------------------------------------------


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    subject_entity_id: int | None = None,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Кэш аналитики; ``stale=true`` означает «данные изменились, обновите»."""
    user = _get_user_by_username(db, username)
    subject = _subject_or_self(db, user, subject_entity_id)
    return AnalyticsService(db).get_cached(user.id, subject.id)


@router.post("/analytics/refresh", response_model=AnalyticsResponse)
async def refresh_analytics(
    request: AnalyticsRefreshRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Перегенерация аналитики моделью и сохранение кэша."""
    user = _get_user_by_username(db, username)
    subject = _subject_or_self(db, user, request.subject_entity_id)
    service = AnalyticsService(db)
    try:
        return await service.generate(user.id, subject.id)
    except Exception as exc:  # noqa: BLE001 - аналитика не должна ронять вкладку
        logger.exception("Analytics generation failed for user %s", user.id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Не удалось сгенерировать аналитику: {exc}",
        )
