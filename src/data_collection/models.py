"""SQLAlchemy models for ESG news database."""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Article(Base):
    """News article from NewsData.io API."""

    __tablename__ = "articles"
    __table_args__ = (
        # Case-insensitive unique constraint on title to prevent duplicate articles
        Index("ix_articles_title_lower", text("lower(title)"), unique=True),
    )

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

    # Labeling status tracking
    # Valid statuses: pending, chunked, embedded, labeled, skipped, false_positive, unlabelable, deduplicated
    labeling_status = Column(String(20), default="pending")
    labeling_error = Column(Text)
    labeled_at = Column(DateTime(timezone=True))
    skipped_at = Column(DateTime(timezone=True))  # Timestamp when article was skipped (for future relabeling)

    # Relationships
    chunks = relationship("ArticleChunk", back_populates="article", cascade="all, delete-orphan")
    brand_labels = relationship("BrandLabel", back_populates="article", cascade="all, delete-orphan")
    classifier_predictions = relationship(
        "ClassifierPrediction", back_populates="article", cascade="all, delete-orphan"
    )

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
    notes = Column(Text)  # For annotations, cleanup records, etc.

    def __repr__(self) -> str:
        return f"<CollectionRun(id={self.id!r}, status={self.status!r})>"


# =============================================================================
# Labeling Models
# =============================================================================


class ArticleChunk(Base):
    """Chunked article text for embeddings and evidence retrieval."""

    __tablename__ = "article_chunks"
    __table_args__ = (
        Index("ix_article_chunks_article_id", "article_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Order within article (0-indexed)
    chunk_text = Column(Text, nullable=False)
    char_start = Column(Integer, nullable=False)  # Start position in full_content
    char_end = Column(Integer, nullable=False)  # End position in full_content
    token_count = Column(Integer)  # Approximate token count
    embedding = Column(Vector(1536))  # text-embedding-3-small
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    article = relationship("Article", back_populates="chunks")
    evidence = relationship("LabelEvidence", back_populates="chunk")

    def __repr__(self) -> str:
        return f"<ArticleChunk(article_id={self.article_id!r}, index={self.chunk_index})>"


class BrandLabel(Base):
    """Per-brand ESG labels with sentiment for an article."""

    __tablename__ = "brand_labels"
    __table_args__ = (
        Index("ix_brand_labels_article_id", "article_id"),
        Index("ix_brand_labels_brand", "brand"),
        Index("ix_brand_labels_labeled_by", "labeled_by"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    brand = Column(String(100), nullable=False)

    # Top-level ESG categories (boolean: applies or not)
    environmental = Column(Boolean, default=False)
    social = Column(Boolean, default=False)
    governance = Column(Boolean, default=False)
    digital_transformation = Column(Boolean, default=False)

    # Sentiment per category (-1=negative, 0=neutral, 1=positive, NULL=not applicable)
    environmental_sentiment = Column(SmallInteger)
    social_sentiment = Column(SmallInteger)
    governance_sentiment = Column(SmallInteger)
    digital_sentiment = Column(SmallInteger)

    # Confidence and metadata
    confidence_score = Column(Float)
    reasoning = Column(Text)  # LLM's explanation for the classification

    # Labeling provenance
    labeled_by = Column(String(50), nullable=False)  # 'claude-sonnet', 'human', 'classifier-v1'
    model_version = Column(String(100))  # e.g., 'claude-3-5-sonnet-20241022'
    labeled_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Human review tracking
    human_reviewed = Column(Boolean, default=False)
    human_reviewer = Column(String(100))
    reviewed_at = Column(DateTime(timezone=True))

    # Relationships
    article = relationship("Article", back_populates="brand_labels")
    evidence = relationship("LabelEvidence", back_populates="brand_label", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<BrandLabel(article_id={self.article_id!r}, brand={self.brand!r})>"


class LabelEvidence(Base):
    """Supporting text excerpts that justify a label assignment."""

    __tablename__ = "label_evidence"
    __table_args__ = (
        Index("ix_label_evidence_brand_label_id", "brand_label_id"),
        Index("ix_label_evidence_category", "category"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_label_id = Column(UUID(as_uuid=True), ForeignKey("brand_labels.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("article_chunks.id", ondelete="SET NULL"))
    category = Column(String(30), nullable=False)  # environmental, social, governance, digital_transformation
    excerpt = Column(Text, nullable=False)  # Extracted quote from article
    relevance_score = Column(Float)  # How relevant this excerpt is (0.0 to 1.0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    brand_label = relationship("BrandLabel", back_populates="evidence")
    chunk = relationship("ArticleChunk", back_populates="evidence")

    def __repr__(self) -> str:
        return f"<LabelEvidence(brand_label_id={self.brand_label_id!r}, category={self.category!r})>"


class LabelingRun(Base):
    """Log of each labeling batch run."""

    __tablename__ = "labeling_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    started_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True))
    articles_processed = Column(Integer, default=0)
    brands_labeled = Column(Integer, default=0)
    chunks_created = Column(Integer, default=0)
    embeddings_generated = Column(Integer, default=0)
    llm_calls_made = Column(Integer, default=0)
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    estimated_cost_usd = Column(Float)
    status = Column(String(20), default="running")  # running, success, partial, failed
    error_message = Column(Text)
    config = Column(JSON)  # Store run configuration

    def __repr__(self) -> str:
        return f"<LabelingRun(id={self.id!r}, status={self.status!r})>"


# =============================================================================
# ML Classifier Predictions
# =============================================================================


class ClassifierPrediction(Base):
    """ML classifier predictions for audit trail and analysis.

    Stores predictions from FP (False Positive), EP (ESG Pre-filter), and
    ESG (Multi-label) classifiers. Used to track which articles were filtered
    by ML classifiers vs sent to LLM for labeling.
    """

    __tablename__ = "classifier_predictions"
    __table_args__ = (
        Index("ix_classifier_predictions_article_id", "article_id"),
        Index("ix_classifier_predictions_classifier_type", "classifier_type"),
        Index("ix_classifier_predictions_created_at", "created_at"),
        Index("ix_classifier_predictions_action_taken", "action_taken"),
        Index("ix_classifier_predictions_type_action", "classifier_type", "action_taken"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    article_id = Column(
        UUID(as_uuid=True),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Classifier identification
    classifier_type = Column(String(20), nullable=False)  # 'fp', 'ep', 'esg'
    model_version = Column(String(100))  # e.g., 'RF_tuned_v1'

    # Prediction result
    probability = Column(Float, nullable=False)  # Raw probability from model (0.0-1.0)
    prediction = Column(Boolean, nullable=False)  # Binary prediction result
    threshold_used = Column(Float, nullable=False)  # Threshold used for this prediction

    # FP-specific fields (nullable for other classifiers)
    risk_level = Column(String(20))  # 'low', 'medium', 'high' for FP classifier

    # EP/ESG-specific fields (nullable for FP)
    esg_categories = Column(JSON)  # For ESG multi-label predictions

    # Decision tracking
    action_taken = Column(String(50), nullable=False)  # 'skipped_llm', 'continued_to_llm', 'failed'
    skip_reason = Column(String(255))  # Reason for skipping LLM (if applicable)

    # Error handling
    error_message = Column(Text)  # If prediction failed

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    article = relationship("Article", back_populates="classifier_predictions")

    def __repr__(self) -> str:
        return f"<ClassifierPrediction(type={self.classifier_type!r}, action={self.action_taken!r})>"
