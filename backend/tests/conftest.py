import sys
from pathlib import Path

# make `app` importable regardless of pytest invocation directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.models import HTRAuthor, User


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture()
def user(db_session):
    user = User(username="writer", password_hash="x", name="Writer")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture()
def author(db_session, user):
    author = HTRAuthor(user_id=user.id, name="Grandfather")
    db_session.add(author)
    db_session.commit()
    db_session.refresh(author)
    return author
