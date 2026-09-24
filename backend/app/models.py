from sqlalchemy import (
    Boolean,
    Column,
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
from .db import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    name = Column(String, default="User")
    tone = Column(String, default="coach")  # "coach", "friend", "critic"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    entries = relationship("Entry", back_populates="user")
    diary_pages = relationship("DiaryPage", back_populates="user")
    entities = relationship("Entity", back_populates="user")
    facts = relationship("Fact", back_populates="user")
    refresh_sessions = relationship("RefreshSession", back_populates="user")
    goals = relationship("Goal", back_populates="user")
    events = relationship("Event", back_populates="user")
    habits = relationship("Habit", back_populates="user")
    entity_relations = relationship("EntityRelation", back_populates="user")


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
    entity_mentions = relationship("EntityMention", back_populates="diary_page")
    facts = relationship("Fact", back_populates="diary_page")


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "entity_type",
            "normalized_name",
            name="uq_entity_user_type_name",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False, index=True)
    canonical_name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    first_seen_at = Column(DateTime(timezone=True), nullable=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Alias stubs: optional synonym storage for one entity.
    # Example: ["mama", "mom", "mother"].
    # aliases_json = Column(Text, nullable=True)

    # Embedding stubs: vector search support for semantic retrieval.
    # Keep commented until pgvector or external vector store is added.
    # embedding_model = Column(String(100), nullable=True)
    # embedding_vector = Column(Text, nullable=True)

    user = relationship("User", back_populates="entities")
    mentions = relationship("EntityMention", back_populates="entity")
    relation = relationship("EntityRelation", back_populates="entity", uselist=False)
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
    diary_page_id = Column(Integer, ForeignKey("diary_pages.id", ondelete="CASCADE"), nullable=False, index=True)
    mention_text = Column(String(255), nullable=False)
    sentence_text = Column(Text, nullable=True)
    start_offset = Column(Integer, nullable=True)
    end_offset = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    sentiment_label = Column(String(32), nullable=True)
    emotion_label = Column(String(32), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    entity = relationship("Entity", back_populates="mentions")
    diary_page = relationship("DiaryPage", back_populates="entity_mentions")


class Fact(Base):
    __tablename__ = "facts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    diary_page_id = Column(Integer, ForeignKey("diary_pages.id", ondelete="CASCADE"), nullable=False, index=True)
    subject_entity_id = Column(Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True, index=True)
    predicate = Column(String(100), nullable=False, index=True)
    object_entity_id = Column(Integer, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True, index=True)
    object_text = Column(Text, nullable=True)
    source_text = Column(Text, nullable=False)
    time_text = Column(String(255), nullable=True)
    emotion_label = Column(String(32), nullable=True, index=True)
    sentiment_score = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Summary stubs: page-level or period-level summaries derived from facts.
    # Keep commented if summaries will be stored in a separate table later.
    # summary_short = Column(Text, nullable=True)
    # summary_long = Column(Text, nullable=True)

    # Clustering stubs: group semantically similar facts.
    # Store cluster id and centroid distance when clustering is added.
    # cluster_id = Column(Integer, nullable=True, index=True)
    # cluster_distance = Column(Float, nullable=True)

    # Embedding stubs: optional vector per fact for semantic search.
    # embedding_model = Column(String(100), nullable=True)
    # embedding_vector = Column(Text, nullable=True)

    user = relationship("User", back_populates="facts")
    diary_page = relationship("DiaryPage", back_populates="facts")
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

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    # active | completed | abandoned
    status = Column(String(32), nullable=False, default="active", index=True)
    created_from_page_id = Column(
        Integer, ForeignKey("diary_pages.id", ondelete="SET NULL"), nullable=True
    )
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
    diary_page_id = Column(
        Integer, ForeignKey("diary_pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # started | progress | completed | abandoned
    progress_kind = Column(String(32), nullable=False, default="progress")
    note = Column(Text, nullable=True)
    source_text = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    goal = relationship("Goal", back_populates="progress_notes")
    diary_page = relationship("DiaryPage")


class EntityRelation(Base):
    __tablename__ = "entity_relations"
    __table_args__ = (
        UniqueConstraint("user_id", "entity_id", name="uq_entity_relation_user_entity"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_id = Column(Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    # sister, friend, colleague, mother, ...
    relation_type = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=True)
    evidence_text = Column(Text, nullable=True)
    last_page_id = Column(Integer, ForeignKey("diary_pages.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="entity_relations")
    entity = relationship("Entity", back_populates="relation")


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    diary_page_id = Column(
        Integer, ForeignKey("diary_pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    time_text = Column(String(255), nullable=True)
    importance = Column(Float, nullable=True)
    source_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="events")
    diary_page = relationship("DiaryPage")


class Habit(Base):
    __tablename__ = "habits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="habits")
    logs = relationship("HabitLog", back_populates="habit", cascade="all, delete-orphan")


class HabitLog(Base):
    __tablename__ = "habit_logs"

    id = Column(Integer, primary_key=True, index=True)
    habit_id = Column(Integer, ForeignKey("habits.id", ondelete="CASCADE"), nullable=False, index=True)
    diary_page_id = Column(
        Integer, ForeignKey("diary_pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    note = Column(Text, nullable=True)
    source_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    habit = relationship("Habit", back_populates="logs")
    diary_page = relationship("DiaryPage")


# ---------------------------------------------------------------------------
# HTR (handwritten text recognition & per-author model training)
# ---------------------------------------------------------------------------


class HTRAuthor(Base):
    __tablename__ = "htr_authors"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    pages = relationship("HTRPage", back_populates="author")
    model_versions = relationship("HTRModelVersion", back_populates="author")


class HTRPage(Base):
    __tablename__ = "htr_pages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    author_id = Column(Integer, ForeignKey("htr_authors.id", ondelete="CASCADE"), nullable=False, index=True)
    file_path = Column(String(1024), nullable=False)
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