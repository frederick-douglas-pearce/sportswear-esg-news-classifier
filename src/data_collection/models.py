"""SQLAlchemy models for ESG news database."""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    JSON,
    Column,
    DateTime,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Article(Base):
    """News article from NewsData.io API."""

    __tablename__ = "articles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    article_id = Column(String(255), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    description = Column(Text)
    full_content = Column(Text)
    url = Column(String(2048), nullable=False)
    image_url = Column(String(2048))
    published_at = Column(DateTime)
    source_name = Column(String(255))
    source_url = Column(String(2048))
    language = Column(String(10))
    country = Column(ARRAY(String(50)))
    category = Column(ARRAY(String(50)))
    keywords = Column(ARRAY(String(255)))
    brands_mentioned = Column(ARRAY(String(100)))
    raw_response = Column(JSON)
    scrape_status = Column(String(20), default="pending")
    scrape_error = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    scraped_at = Column(DateTime)
    embedding = Column(Vector(1536))

    def __repr__(self) -> str:
        return f"<Article(article_id={self.article_id!r}, title={self.title[:50]!r}...)>"


class CollectionRun(Base):
    """Log of each data collection run."""

    __tablename__ = "collection_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    started_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)
    api_calls_made = Column(Integer, default=0)
    articles_fetched = Column(Integer, default=0)
    articles_duplicates = Column(Integer, default=0)
    articles_scraped = Column(Integer, default=0)
    articles_scrape_failed = Column(Integer, default=0)
    status = Column(String(20), default="running")
    error_message = Column(Text)

    def __repr__(self) -> str:
        return f"<CollectionRun(id={self.id!r}, status={self.status!r})>"
