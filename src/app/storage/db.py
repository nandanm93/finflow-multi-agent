from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)


def get_session() -> Session:
    return SessionLocal()


def init_db() -> None:
    # Fallback for local dev runs when migrations were not executed yet.
    from app.storage import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
