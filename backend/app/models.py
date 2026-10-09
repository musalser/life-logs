from sqlalchemy import (
    Boolean,
    Column,
    Index,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Float,
    UniqueConstraint,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from .db import Base

#: dimension of the embeddings written by app.services.embedding_service.
#: Kept as a literal (not settings.embedding_dim) so that the model metadata,
#: and therefore Alembic autogenerate, does not change with an environment
#: variable: switching the embedding model must be a migration, not a surprise.
EMBEDDING_DIM = 768


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    name = Column(String, default="User")
    tone = Column(String, default="coach")  # "coach", "friend", "critic"
    #: the person-object that represents this account ("Я" on the knowledge
    #: tab). Knowledge extracted from the diary is attributed to it, so it is
    #: created together with the account (see routes/auth.py).
    self_entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True, unique=True
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    entries = relationship("Entry", back_populates="user")
    diary_pages = relationship("DiaryPage", back_populates="user")
    #: two foreign keys now link users and entities (entities.user_id and
    #: users.self_entity_id), so the join must name its column
    entities = relationship("Entity", back_populates="user", foreign_keys="Entity.user_id")
    facts = relationship("Fact", back_populates="user")
    refresh_sessions = relationship("RefreshSession", back_populates="user")
    goals = relationship("Goal", back_populates="user")
    events = relationship("Event", back_populates="user")
    habits = relationship("Habit", back_populates="user")
    entity_relations = relationship("EntityRelation", back_populates="user")
    #: the cycle users.self_entity_id <-> entities.user_id needs one side to be
    #: written after the other, otherwise flush order is ambiguous
    self_entity = relationship("Entity", foreign_keys=[self_entity_id], post_update=True)


class RefreshSession(Base):
    __tablename__ = "refresh_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    username = Column(String, nullable=False, index=True)
    family_id = Column(String(64), nullable=False, index=True)
    token_jti = Column(String(64), nullable=False, unique=True, index=True)
    token_hash = Column(String(128), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    issued_at = Column(DateTime(timezone=True), server_default=func.now())
    rotated_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    reuse_detected_at = Column(DateTime(timezone=True), nullable=True)
    replaced_by_jti = Column(String(64), nullable=True, index=True)
    user_agent = Column(String(255), nullable=True)
    ip_address = Column(String(64), nullable=True)

    user = relationship("User", back_populates="refresh_sessions")


class Entry(Base):
    __tablename__ = "entries"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    text = Column(Text)
    reply = Column(Text)
    #: JSON list of the RAG chunks the answer was built from:
    #: [{"chunk_id": 1, "text": "...", "source_type": "htr", "ref_id": 12, "title": "..."}]
    context_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="entries")


class DiaryPage(Base):
    __tablename__ = "diary_pages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(120), nullable=True)
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="diary_pages")


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "entity_type",
            "normalized_name",
            name="uq_entity_user_type_name",
        ),
        Index(
            "ix_entities_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False, index=True)
    canonical_name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    #: JSON list of other spellings of the same object ("mama", "mom",
    #: "матушка"): a merge adds the variant instead of creating a new entity
    aliases_json = Column(Text, nullable=True)
    #: embedding of canonical_name + aliases, used to *retrieve* dedup
    #: candidates (the decision is the LLM resolver's, see the spec 7.2)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    first_seen_at = Column(DateTime(timezone=True), nullable=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="entities", foreign_keys=[user_id])
    mentions = relationship("EntityMention", back_populates="entity")
    from_relations = relationship(
        "EntityRelation",
        foreign_keys="EntityRelation.from_entity_id",
        back_populates="from_entity",
    )
    to_relations = relationship(
        "EntityRelation",
        foreign_keys="EntityRelation.to_entity_id",
        back_populates="to_entity",
    )
    subject_facts = relationship(
        "Fact",
        foreign_keys="Fact.subject_entity_id",
        back_populates="subject_entity",
    )
    object_facts = relationship(
        "Fact",
        foreign_keys="Fact.object_entity_id",
        back_populates="object_entity",
    )


class EntityMention(Base):
    __tablename__ = "entity_mentions"

    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    #: the text the mention was read from: a diary page or a manuscript page,
    #: see KnowledgeSource
    source_id = Column(
        Integer, ForeignKey("knowledge_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    mention_text = Column(String(255), nullable=False)
    sentence_text = Column(Text, nullable=True)
    #: offsets into KnowledgeSource.text (0-based, end exclusive)
    start_offset = Column(Integer, nullable=True)
    end_offset = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    sentiment_label = Column(String(32), nullable=True)
    emotion_label = Column(String(32), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    entity = relationship("Entity", back_populates="mentions")
    source = relationship("KnowledgeSource", back_populates="mentions")


class Fact(Base):
    __tablename__ = "facts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id = Column(
        Integer, ForeignKey("knowledge_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_entity_id = Column(Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True, index=True)
    predicate = Column(String(100), nullable=False, index=True)
    object_entity_id = Column(Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True, index=True)
    object_text = Column(Text, nullable=True)
    #: exact quote from the source, kept as a copy so a re-indexed or edited
    #: source still has a readable citation
    source_text = Column(Text, nullable=False)
    #: offsets of that quote in KnowledgeSource.text (0-based, end exclusive)
    start_offset = Column(Integer, nullable=True)
    end_offset = Column(Integer, nullable=True)
    time_text = Column(String(255), nullable=True)
    emotion_label = Column(String(32), nullable=True, index=True)
    sentiment_score = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="facts")
    source = relationship("KnowledgeSource", back_populates="facts")
    subject_entity = relationship(
        "Entity",
        foreign_keys=[subject_entity_id],
        back_populates="subject_facts",
    )
    object_entity = relationship(
        "Entity",
        foreign_keys=[object_entity_id],
        back_populates="object_facts",
    )


class Goal(Base):
    __tablename__ = "goals"
    __table_args__ = (
        Index(
            "ix_goals_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    #: whose goal it is: the author of the source text, not necessarily the
    #: account owner (a manuscript's author has their own goals)
    subject_entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    # active | completed | abandoned
    status = Column(String(32), nullable=False, default="active", index=True)
    created_from_source_id = Column(
        Integer, ForeignKey("knowledge_sources.id", ondelete="SET NULL"), nullable=True
    )
    #: embedding of the title, used to retrieve dedup candidates
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="goals")
    progress_notes = relationship(
        "GoalProgress", back_populates="goal", cascade="all, delete-orphan"
    )


class GoalProgress(Base):
    __tablename__ = "goal_progress"

    id = Column(Integer, primary_key=True, index=True)
    goal_id = Column(Integer, ForeignKey("goals.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id = Column(
        Integer, ForeignKey("knowledge_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # started | progress | completed | abandoned
    progress_kind = Column(String(32), nullable=False, default="progress")
    note = Column(Text, nullable=True)
    source_text = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    goal = relationship("Goal", back_populates="progress_notes")
    source = relationship("KnowledgeSource", back_populates="goal_progress")


class EntityRelation(Base):
    """A directed edge of the social graph: "from" is the perspective.

    A manuscript written by Баба Галя yields ``Баба Галя -> мама``, not
    ``me -> мама``: knowledge about the circle of contacts is always tied to
    the author of the text. The knowledge tab shows the edges of the selected
    object and the path to it, and never re-composes relationship names.
    """

    __tablename__ = "entity_relations"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "from_entity_id",
            "to_entity_id",
            "relation_type",
            name="uq_entity_relation_edge",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    from_entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    to_entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # sister, friend, colleague, mother, ...
    relation_type = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=True)
    evidence_text = Column(Text, nullable=True)
    #: where the edge was read (NULL for edges migrated from the old
    #: single-perspective table)
    source_id = Column(
        Integer, ForeignKey("knowledge_sources.id", ondelete="CASCADE"), nullable=True, index=True
    )
    start_offset = Column(Integer, nullable=True)
    end_offset = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="entity_relations")
    from_entity = relationship(
        "Entity", foreign_keys=[from_entity_id], back_populates="from_relations"
    )
    to_entity = relationship("Entity", foreign_keys=[to_entity_id], back_populates="to_relations")
    source = relationship("KnowledgeSource")


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        Index(
            "ix_events_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id = Column(
        Integer, ForeignKey("knowledge_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    time_text = Column(String(255), nullable=True)
    importance = Column(Float, nullable=True)
    source_text = Column(Text, nullable=True)
    start_offset = Column(Integer, nullable=True)
    end_offset = Column(Integer, nullable=True)
    #: embedding of title + time_text + description: the date must be part of
    #: the vector text, or "день рождения 2023" and "…2024" become one event
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="events")
    source = relationship("KnowledgeSource", back_populates="events")


class Habit(Base):
    __tablename__ = "habits"
    __table_args__ = (
        Index(
            "ix_habits_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    subject_entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title = Column(String(255), nullable=False)
    #: embedding of the title, used to retrieve dedup candidates
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="habits")
    logs = relationship("HabitLog", back_populates="habit", cascade="all, delete-orphan")


class HabitLog(Base):
    __tablename__ = "habit_logs"

    id = Column(Integer, primary_key=True, index=True)
    habit_id = Column(Integer, ForeignKey("habits.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id = Column(
        Integer, ForeignKey("knowledge_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    note = Column(Text, nullable=True)
    source_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    habit = relationship("Habit", back_populates="logs")
    source = relationship("KnowledgeSource", back_populates="habit_logs")


# ---------------------------------------------------------------------------
# Knowledge sources: one text to extract from (manuscript page or diary page)
# ---------------------------------------------------------------------------


class KnowledgeSource(Base):
    """The text a knowledge extraction run reads, whatever produced it.

    A confirmed manuscript page and a diary page are both *sources*: the
    pipeline, the offsets and the provenance are the same, and nothing is
    copied between the two (see the spec, section 6).
    """

    __tablename__ = "knowledge_sources"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "source_type", "ref_id", name="uq_knowledge_source_user_ref"
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    #: 'htr' (htr_pages.id) | 'diary' (diary_pages.id)
    source_type = Column(String(16), nullable=False, index=True)
    ref_id = Column(Integer, nullable=False, index=True)
    #: whose text this is: the object of the manuscript author, or the
    #: account's own object for the diary
    author_entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title = Column(String(255), nullable=True)
    text = Column(Text, nullable=False, default="")
    #: sha256 of text; a change means the knowledge derived from it is stale
    text_hash = Column(String(64), nullable=True, index=True)
    #: for 'htr': JSON [{"line_id": N, "start": 0, "end": 42}, ...] mapping
    #: global offsets in `text` back to the lines of the manuscript page;
    #: NULL for the diary, where offsets are simply indexes into content
    line_map = Column(Text, nullable=True)
    #: pending | running | done | failed
    extraction_status = Column(String(16), nullable=False, default="pending", index=True)
    last_extracted_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User")
    author_entity = relationship("Entity")
    mentions = relationship(
        "EntityMention", back_populates="source", cascade="all, delete-orphan"
    )
    facts = relationship("Fact", back_populates="source", cascade="all, delete-orphan")
    goal_progress = relationship(
        "GoalProgress", back_populates="source", cascade="all, delete-orphan"
    )
    events = relationship("Event", back_populates="source", cascade="all, delete-orphan")
    habit_logs = relationship(
        "HabitLog", back_populates="source", cascade="all, delete-orphan"
    )
    chunks = relationship("Chunk", back_populates="source", cascade="all, delete-orphan")
    runs = relationship(
        "ExtractionRun",
        back_populates="source",
        cascade="all, delete-orphan",
        order_by="ExtractionRun.id",
    )


class ExtractionRun(Base):
    """One execution of the extraction pipeline over one source."""

    __tablename__ = "extraction_runs"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(
        Integer, ForeignKey("knowledge_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: running | succeeded | failed
    status = Column(String(16), nullable=False, default="running", index=True)
    #: JSON counters per knowledge type (created / matched / errors)
    stats = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    #: which LLM produced the extraction
    model = Column(String(64), nullable=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    finished_at = Column(DateTime(timezone=True), nullable=True)

    source = relationship("KnowledgeSource", back_populates="runs")


class Chunk(Base):
    """A RAG passage of one source, with its embedding."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id = Column(
        Integer, ForeignKey("knowledge_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index = Column(Integer, nullable=False, default=0)
    text = Column(Text, nullable=False)
    start_offset = Column(Integer, nullable=True)
    end_offset = Column(Integer, nullable=True)
    #: embedded with the model's 'Document' prompt (queries use 'SearchQuery')
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    source = relationship("KnowledgeSource", back_populates="chunks")


class KnowledgeInsight(Base):
    """Cached analytics (insights / questions / ideas / conclusions).

    Regenerated when knowledge_state.version moves past the stored version.
    """

    __tablename__ = "knowledge_insights"
    __table_args__ = (
        UniqueConstraint("user_id", "subject_entity_id", name="uq_knowledge_insight_subject"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    #: the object the analytics was generated from
    subject_entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True, index=True
    )
    #: knowledge_state.version at generation time
    version = Column(Integer, nullable=False, default=0)
    #: JSON {"insights": [...], "questions": [...], "ideas": [...],
    #: "conclusions": [...]}, each item {text, confidence, evidence: [...]}
    payload = Column(Text, nullable=False, default="{}")
    model = Column(String(64), nullable=True)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")
    subject_entity = relationship("Entity")


class KnowledgeState(Base):
    """Per-user version of the knowledge base, bumped on every extraction."""

    __tablename__ = "knowledge_state"

    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    version = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User")


# ---------------------------------------------------------------------------
# HTR (handwritten text recognition & per-author model training)
# ---------------------------------------------------------------------------


class HTRAuthor(Base):
    __tablename__ = "htr_authors"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    #: the object this manuscript author is for the family: relations read from
    #: their pages are attributed to it, so extraction refuses to run without it
    entity_id = Column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    pages = relationship("HTRPage", back_populates="author")
    model_versions = relationship("HTRModelVersion", back_populates="author")
    entity = relationship("Entity")


class HTRPage(Base):
    __tablename__ = "htr_pages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    author_id = Column(Integer, ForeignKey("htr_authors.id", ondelete="CASCADE"), nullable=False, index=True)
    file_path = Column(String(1024), nullable=False)
    # the name of the uploaded file, shown in the sidebar instead of the page id;
    # source_path keeps the client-side path when it supplied one (folder upload),
    # so pages with equal names can be told apart by their full path
    file_name = Column(String(1024), nullable=True)
    source_path = Column(String(2048), nullable=True)
    # manual order inside the author's sidebar list (ascending)
    order_index = Column(Integer, nullable=False, default=0, server_default="0", index=True)
    # UPLOADED | RECOGNIZED | EDITING | CONFIRMED
    status = Column(String(32), nullable=False, default="UPLOADED", index=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    confirmed_at = Column(DateTime(timezone=True), nullable=True)
    recognition_model_version_id = Column(
        Integer, ForeignKey("htr_model_versions.id", ondelete="SET NULL"), nullable=True
    )
    # prediction quality vs. user ground truth, computed at confirmation
    prediction_cer = Column(Float, nullable=True)
    prediction_wer = Column(Float, nullable=True)

    author = relationship("HTRAuthor", back_populates="pages")
    lines = relationship(
        "HTRLine",
        back_populates="page",
        cascade="all, delete-orphan",
        order_by="HTRLine.order_index",
    )


class HTRLine(Base):
    __tablename__ = "htr_lines"

    id = Column(Integer, primary_key=True, index=True)
    page_id = Column(Integer, ForeignKey("htr_pages.id", ondelete="CASCADE"), nullable=False, index=True)
    order_index = Column(Integer, nullable=False, default=0)
    x1 = Column(Integer, nullable=False)
    y1 = Column(Integer, nullable=False)
    x2 = Column(Integer, nullable=False)
    y2 = Column(Integer, nullable=False)
    predicted_text = Column(Text, nullable=True)
    corrected_text = Column(Text, nullable=True)
    # who put corrected_text there: 'user' (accepted model proposals count as
    # user text, since the human made the decision)
    corrected_by = Column(String(16), nullable=True)
    # proposal of the language model: kept apart from corrected_text so the
    # kraken transcription stays exactly as recognized until the user accepts
    suggested_text = Column(Text, nullable=True)
    suggested_by = Column(String(32), nullable=True)
    # word bboxes no longer match the tokenization of corrected_text
    words_stale = Column(Boolean, nullable=False, default=False)
    # JSON [[x, y], ...] outline that follows the (curved) baseline
    polygon = Column(Text, nullable=True)

    page = relationship("HTRPage", back_populates="lines")
    words = relationship(
        "HTRWord",
        back_populates="line",
        cascade="all, delete-orphan",
        order_by="HTRWord.order_index",
    )


class HTRWord(Base):
    __tablename__ = "htr_words"

    id = Column(Integer, primary_key=True, index=True)
    line_id = Column(Integer, ForeignKey("htr_lines.id", ondelete="CASCADE"), nullable=False, index=True)
    order_index = Column(Integer, nullable=False, default=0)
    x1 = Column(Integer, nullable=False)
    y1 = Column(Integer, nullable=False)
    x2 = Column(Integer, nullable=False)
    y2 = Column(Integer, nullable=False)
    predicted_text = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    corrected_text = Column(Text, nullable=True)
    # JSON [[x, y], ...] outline of the word, following the line baseline
    polygon = Column(Text, nullable=True)
    # JSON [{"text": ..., "score": ...}, ...] — other readings the beam
    # considered for this word, best first (see domain.entities.WordAlternative)
    alternatives = Column(Text, nullable=True)

    line = relationship("HTRLine", back_populates="words")


class HTRModelVersion(Base):
    __tablename__ = "htr_model_versions"
    __table_args__ = (
        UniqueConstraint("author_id", "version", name="uq_htr_model_author_version"),
    )

    id = Column(Integer, primary_key=True, index=True)
    author_id = Column(Integer, ForeignKey("htr_authors.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    base_model_id = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)
    # TRAINING | READY | ACTIVE | FAILED
    status = Column(String(32), nullable=False, default="TRAINING", index=True)
    dataset_hash = Column(String(64), nullable=True)
    metrics = Column(Text, nullable=True)  # JSON
    training_config = Column(Text, nullable=True)  # JSON
    environment = Column(Text, nullable=True)  # JSON: package versions etc.
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    activated_at = Column(DateTime(timezone=True), nullable=True)

    author = relationship("HTRAuthor", back_populates="model_versions")


class HTRSuggestionEvent(Base):
    """What the user did with a model proposal.

    This is the feedback the correction loop was missing: without it there is
    no way to tell whether the language model helps, which of its changes were
    accepted, or whether the dictionary verdict predicted the user's decision.
    Every accept (whole line, single word, bulk) and every dismissal lands here.
    """

    __tablename__ = "htr_suggestion_events"

    id = Column(Integer, primary_key=True, index=True)
    page_id = Column(Integer, ForeignKey("htr_pages.id", ondelete="CASCADE"), nullable=False, index=True)
    line_id = Column(Integer, ForeignKey("htr_lines.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    # accepted_change | accepted_line | accepted_bulk | dismissed
    action = Column(String(24), nullable=False, index=True)
    # the proposal as a whole and the piece that was acted on
    suggested_text = Column(Text, nullable=True)
    original_text = Column(Text, nullable=True)
    change_before = Column(Text, nullable=True)
    change_after = Column(Text, nullable=True)
    # dictionary verdict of that change (True/False/None)
    in_lexicon = Column(Boolean, nullable=True)
    # model that made the proposal
    suggested_by = Column(String(32), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    page = relationship("HTRPage")
    line = relationship("HTRLine")


class HTRTrainingRun(Base):
    __tablename__ = "htr_training_runs"

    id = Column(Integer, primary_key=True, index=True)
    author_id = Column(Integer, ForeignKey("htr_authors.id", ondelete="CASCADE"), nullable=False, index=True)
    model_version_id = Column(
        Integer, ForeignKey("htr_model_versions.id", ondelete="SET NULL"), nullable=True
    )
    dataset_hash = Column(String(64), nullable=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    finished_at = Column(DateTime(timezone=True), nullable=True)
    # RUNNING | SUCCEEDED | FAILED | INSUFFICIENT_DATA
    status = Column(String(32), nullable=False, default="RUNNING", index=True)
    error = Column(Text, nullable=True)
    metrics = Column(Text, nullable=True)  # JSON


class HTRLexiconIgnore(Base):
    """A word the user took out of their own vocabulary.

    The vocabulary check treats the words of the author's confirmed pages as
    known, so their names and dialect words are not flagged. The flip side is
    that a typo confirmed once becomes "known" everywhere. This table is the
    escape hatch: an ignored word is subtracted from the author's vocabulary
    (and from the knowledge terms), so it is flagged again as unknown.
    """

    __tablename__ = "htr_lexicon_ignores"
    __table_args__ = (
        UniqueConstraint("author_id", "word", name="uq_htr_lexicon_ignore_author_word"),
    )

    id = Column(Integer, primary_key=True, index=True)
    author_id = Column(
        Integer, ForeignKey("htr_authors.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # normalized (casefolded) surface form, length covers any real word
    word = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())