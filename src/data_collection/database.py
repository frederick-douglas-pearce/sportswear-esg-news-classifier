"""Database operations for ESG news storage."""

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from .api_client import ArticleData
from .config import settings
from .models import Article, Base, CollectionRun

logger = logging.getLogger(__name__)


class Database:
    """Database connection and operations manager."""

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or settings.database_url
        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def init_db(self) -> None:
        """Create all tables and enable pgvector extension."""
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        Base.metadata.create_all(self.engine)
        logger.info("Database initialized successfully")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_article(self, session: Session, article_data: ArticleData) -> tuple[Article | None, str]:
        """
        Insert or update an article.

        Returns:
            Tuple of (Article object or None, status string)
            Status is one of: "new", "duplicate_id", "duplicate_title"
        """
        # Check for existing article by ID
        existing = session.query(Article).filter_by(article_id=article_data.article_id).first()
        if existing:
            return existing, "duplicate_id"

        # Check for existing article by title (case-insensitive)
        if article_data.title:
            from sqlalchemy import func
            existing_title = (
                session.query(Article)
                .filter(func.lower(Article.title) == article_data.title.lower().strip())
                .first()
            )
            if existing_title:
                return None, "duplicate_title"

        article = Article(
            article_id=article_data.article_id,
            title=article_data.title,
            description=article_data.description,
            url=article_data.url,
            image_url=article_data.image_url,
            published_at=article_data.published_at,
            source_name=article_data.source_name,
            source_url=article_data.source_url,
            language=article_data.language,
            country=article_data.country,
            category=article_data.category,
            keywords=article_data.keywords,
            brands_mentioned=article_data.brands_mentioned,
            raw_response=article_data.raw_response,
            scrape_status="pending",
        )
        session.add(article)
        session.flush()
        return article, "new"

    def get_articles_pending_scrape(self, session: Session, limit: int = 100) -> list[Article]:
        """Get articles that need to be scraped."""
        return (
            session.query(Article)
            .filter(Article.scrape_status == "pending")
            .order_by(Article.created_at)
            .limit(limit)
            .all()
        )

    def update_article_content(
        self,
        session: Session,
        article_id: str,
        content: str | None,
        status: str,
        error: str | None = None,
    ) -> None:
        """Update article with scraped content."""
        article = session.query(Article).filter_by(article_id=article_id).first()
        if article:
            article.full_content = content
            article.scrape_status = status
            article.scrape_error = error
            article.scraped_at = datetime.now(timezone.utc)

    def create_collection_run(self, session: Session) -> CollectionRun:
        """Create a new collection run record."""
        run = CollectionRun(started_at=datetime.now(timezone.utc), status="running")
        session.add(run)
        session.flush()
        return run

    def complete_collection_run(
        self,
        session: Session,
        run: CollectionRun,
        api_calls: int,
        articles_fetched: int,
        articles_duplicates: int,
        articles_scraped: int,
        articles_scrape_failed: int,
        status: str = "success",
        error_message: str | None = None,
    ) -> None:
        """Update collection run with final statistics."""
        run.completed_at = datetime.now(timezone.utc)
        run.api_calls_made = api_calls
        run.articles_fetched = articles_fetched
        run.articles_duplicates = articles_duplicates
        run.articles_scraped = articles_scraped
        run.articles_scrape_failed = articles_scrape_failed
        run.status = status
        run.error_message = error_message

    def get_article_count(self, session: Session) -> int:
        """Get total number of articles in database."""
        return session.query(Article).count()

    def get_scraped_article_count(self, session: Session) -> int:
        """Get number of successfully scraped articles."""
        return session.query(Article).filter(Article.scrape_status == "success").count()


db = Database()
