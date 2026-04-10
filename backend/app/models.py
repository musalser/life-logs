from sqlalchemy import (
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