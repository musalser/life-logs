from pydantic import BaseModel
from datetime import datetime


class ChatRequest(BaseModel):
    message: str
    tone: str = "coach"
    stream: bool = True
    #: искать ответ по архиву (RAG); тумблер в интерфейсе чата
    rag: bool = True
    #: перспектива поиска: чанки источников этого объекта (и всегда собственные)
    subject_entity_id: int | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    name: str | None = None


class UserResponse(BaseModel):
    id: int
    username: str
    name: str
    tone: str
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChatResponse(BaseModel):
    reply: str
    created_at: datetime


class DiaryPageCreateRequest(BaseModel):
    content: str


class DiaryPageUpdateRequest(BaseModel):
    content: str


class DiaryPageResponse(BaseModel):
    id: int
    user_id: int
    title: str | None = None
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class Entry(BaseModel):
    id: int
    user_id: int
    text: str
    reply: str
    created_at: datetime
    #: использованные чанки: [{"chunk_id", "text", "source_type", "ref_id", "title"}]
    context_json: str | None = None

    class Config:
        from_attributes = True


class GoalProgressResponse(BaseModel):
    id: int
    source_id: int
    progress_kind: str
    note: str | None = None
    source_text: str | None = None
    confidence: float | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class GoalResponse(BaseModel):
    id: int
    title: str
    description: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime | None = None
    progress_notes: list[GoalProgressResponse] = []

    class Config:
        from_attributes = True


class RelationResponse(BaseModel):
    id: int
    entity_id: int
    person_name: str
    relation_type: str
    confidence: float | None = None
    evidence_text: str | None = None
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class ExtractionSummaryResponse(BaseModel):
    page_id: int
    summary: dict

# ---------------------------------------------------------------------------
# Knowledge objects, graph paths and analytics (spec 8.1, 9, 10)
# ---------------------------------------------------------------------------


class KnowledgeObjectResponse(BaseModel):
    id: int
    name: str
    entity_type: str
    description: str | None = None
    aliases: list[str] = []
    is_self: bool = False
    relation_count: int = 0
    last_seen_at: datetime | None = None


class KnowledgeObjectCreateRequest(BaseModel):
    name: str
    aliases: list[str] = []
    description: str | None = None


class KnowledgeObjectUpdateRequest(BaseModel):
    name: str | None = None
    aliases: list[str] | None = None
    description: str | None = None


class RelationPathStep(BaseModel):
    entity_id: int
    name: str
    relation_type: str


class RelationWithPath(RelationResponse):
    """Связь от выбранного объекта: прямое ребро или цепочка до человека."""

    direct: bool = True
    #: цепочка рёбер от выбранного объекта до персоны: [{"entity_id", "name",
    #: "relation_type"}, ...]. Пустая у прямых связей.
    path: list[RelationPathStep] = []
    #: человеческая подпись цепочки, например «мама Бабы Гали»
    path_label: str | None = None


class FactResponse(BaseModel):
    id: int
    source_id: int
    subject_entity_id: int | None = None
    predicate: str
    object_entity_id: int | None = None
    object_text: str | None = None
    source_text: str
    start_offset: int | None = None
    end_offset: int | None = None
    time_text: str | None = None
    #: та же дата в виде ГГГГ[-ММ[-ДД]]
    time_iso: str | None = None
    #: событие — факт с важностью (см. settings.knowledge_event_min_importance)
    importance: float | None = None
    #: как текст говорит о регулярности: сырьё для привычек в аналитике
    recurrence: str | None = None
    confidence: float | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class EntityFactResponse(FactResponse):
    """Факт, найденный по объекту: с причинами совпадения и именами участников."""

    #: why the fact belongs to the object: subject | object | mention | text
    match_reasons: list[str] = []
    subject_name: str | None = None
    object_name: str | None = None


class FactConflictResponse(BaseModel):
    """Конфликт функционального предиката: второе, отличающееся значение."""

    id: int
    subject_entity_id: int | None = None
    subject_name: str | None = None
    predicate_kind: str
    existing_fact_id: int | None = None
    existing_value: str | None = None
    incoming_value: str | None = None
    source_id: int | None = None
    source_text: str | None = None
    time_text: str | None = None
    confidence: float | None = None
    resolution: str | None = None
    created_at: datetime


class PageKnowledgeLine(BaseModel):
    """Знание на странице рукописи: офсет переведён в строку и её координаты."""

    kind: str  # fact | mention | event
    id: int
    text: str
    quote: str | None = None
    line_id: int | None = None
    line_order: int | None = None
    confidence: float | None = None


class PageKnowledgeResponse(BaseModel):
    page_id: int
    source_id: int | None = None
    status: str | None = None
    items: list[PageKnowledgeLine] = []


class AnalyticsEvidence(BaseModel):
    kind: str
    id: int
    label: str | None = None


class AnalyticsItem(BaseModel):
    text: str
    confidence: float | None = None
    evidence: list[AnalyticsEvidence] = []


class AnalyticsPayload(BaseModel):
    insights: list[AnalyticsItem] = []
    #: привычки, выведенные из повторов в фактах (не из одной страницы)
    habits: list[AnalyticsItem] = []
    questions: list[AnalyticsItem] = []
    ideas: list[AnalyticsItem] = []
    conclusions: list[AnalyticsItem] = []


class AnalyticsResponse(BaseModel):
    subject_entity_id: int | None = None
    version: int = 0
    stale: bool = True
    generated_at: datetime | None = None
    model: str | None = None
    payload: AnalyticsPayload = AnalyticsPayload()


class AnalyticsRefreshRequest(BaseModel):
    subject_entity_id: int | None = None


class KnowledgeStatsResponse(BaseModel):
    facts: int = 0
    goals: int = 0
    events: int = 0
    relations: int = 0
    people: int = 0
    sources: int = 0


class KnowledgeResponse(BaseModel):
    #: от чьего лица показаны знания (объект-перспектива)
    subject_entity_id: int | None = None
    subject_name: str | None = None
    goals: list[GoalResponse] = []
    relations: list[RelationWithPath] = []
    #: события — факты с важностью, по дате
    events: list[FactResponse] = []
    facts: list[FactResponse] = []
    entities: list[KnowledgeObjectResponse] = []
    stats: KnowledgeStatsResponse = KnowledgeStatsResponse()


# ---------------------------------------------------------------------------
# Unified library («Записи»): authors -> folders -> pages
# ---------------------------------------------------------------------------


class LibraryPageNode(BaseModel):
    type: str  # 'diary' | 'htr'
    ref_id: int
    title: str | None = None
    folder_id: int | None = None
    order_index: int = 0
    status: str | None = None
    created_at: datetime | None = None


class LibraryFolderNode(BaseModel):
    id: int
    name: str
    children: list["LibraryFolderNode"] = []
    pages: list[LibraryPageNode] = []


class LibraryAuthorNode(BaseModel):
    entity_id: int
    name: str
    is_self: bool = False
    #: HTRAuthor id for uploading manuscript pages (None for a pure «Я» with no
    #: manuscript author bound to it)
    author_id: int | None = None
    folders: list[LibraryFolderNode] = []
    pages: list[LibraryPageNode] = []


class LibraryTreeResponse(BaseModel):
    authors: list[LibraryAuthorNode] = []


class FolderCreateRequest(BaseModel):
    name: str
    owner_entity_id: int
    parent_id: int | None = None


class FolderUpdateRequest(BaseModel):
    name: str | None = None
    parent_id: int | None = None


class FolderResponse(BaseModel):
    id: int
    name: str
    parent_id: int | None = None
    owner_entity_id: int | None = None
    order_index: int = 0
    created_at: datetime | None = None


class MovePageRequest(BaseModel):
    #: NULL = move to the author's root level
    folder_id: int | None = None


class ObjectMergeRequest(BaseModel):
    source_id: int
    target_id: int


# ---------------------------------------------------------------------------
# Граф связей (раздел 9): весь архив сразу, корень выбирает клиент
# ---------------------------------------------------------------------------


class RelationGraphNode(BaseModel):
    id: int
    name: str
    entity_type: str
    #: якорная сущность («Отец автора») — роль без имени, а не человек
    is_anchor: bool = False
    #: к кому привязан якорь
    anchor_of: int | None = None
    #: роль якоря из закрытого списка (father, neighbour, …)
    anchor_relation: str | None = None
    #: уровень поколений внутри компоненты. None — не определён (ритуальное,
    #: неоднозначное, вне родства); такие узлы не соединяют компоненты
    level: int | None = None
    #: id компоненты родства: у каждого дерева свой постоянный сдвиг уровней
    component: int | None = None


class RelationGraphEdge(BaseModel):
    id: int
    from_id: int
    to_id: int
    #: вид из закрытого списка — источник истины
    relation_kind: str | None = None
    #: слово из текста («папа», «тесть») — для подписи
    relation_type: str | None = None
    #: level(to) − level(from): знак считает бэкенд, не модель
    delta: int | None = None
    via_marriage: bool = False
    bond: str = "blood"
    canon_key: str
    confidence: float | None = None
    #: источники-подтверждения: ребро живёт, пока они есть
    sources: list[int] = []


class RelationGraphConflict(BaseModel):
    #: level_mismatch | functional_parent | anchor_named_pair | kind_disagreement
    kind: str
    child: int | None = None
    parents: list[int] = []
    #: переполненные родительские слоты: {"father": [76, 320]}
    duplicated_slots: dict[str, list[int]] = {}
    slot: str | None = None
    anchor: int | None = None
    named: int | None = None
    merge_candidates: list[int] = []
    edge_id: int | None = None
    from_id: int | None = None
    to_id: int | None = None
    relation_kind: str | None = None
    expected_delta: int | None = None
    actual_delta: int | None = None
    #: виды одного факта, названные источниками по-разному
    kinds: list[str] = []
    kept: str | None = None


class RelationGraphUnplaced(BaseModel):
    entity_id: int
    #: ritual | ambiguous | isolated
    reason: str


class RelationGraphDangling(BaseModel):
    #: ребро ссылается на узел вне графа — это битые данные, а не молчание
    edge_id: int | None = None
    from_id: int
    to_id: int
    missing: list[int] = []


class RelationGraphResponse(BaseModel):
    """Граф связей целиком: корень и режим выбирает клиент.

    ``version`` — версия знаний: по ней клиент кэширует, а сервер отдаёт 304.
    """

    version: int
    nodes: list[RelationGraphNode]
    edges: list[RelationGraphEdge]
    conflicts: list[RelationGraphConflict]
    unplaced: list[RelationGraphUnplaced]
    dangling: list[RelationGraphDangling] = []
