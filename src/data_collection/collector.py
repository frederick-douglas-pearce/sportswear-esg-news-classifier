"""Main collection orchestrator for ESG news data."""

import logging
import random
from dataclasses import dataclass, field
from typing import Literal

from .api_client import NewsDataClient
from .config import settings
from .database import Database, db
from .gdelt_client import GDELTClient
from .scraper import ArticleScraper

logger = logging.getLogger(__name__)

# Type alias for supported API sources
ApiSource = Literal["newsdata", "gdelt"]


def normalize_title(title: str | None) -> str:
    """Normalize title for deduplication comparison."""
    if not title:
        return ""
    return title.lower().strip()


@dataclass
class CollectionStats:
    """Statistics from a collection run."""

    api_calls: int = 0
    articles_fetched: int = 0
    articles_duplicates: int = 0
    articles_duplicate_title: int = 0  # Same title from different sources
    articles_no_brand: int = 0  # Filtered out - no tracked brand mentioned
    articles_scraped: int = 0
    articles_scrape_failed: int = 0
    errors: list[str] = field(default_factory=list)


class NewsCollector:
    """Orchestrates the news collection and scraping process."""

    def __init__(
        self,
        database: Database | None = None,
        api_client: NewsDataClient | GDELTClient | None = None,
        scraper: ArticleScraper | None = None,
        source: ApiSource = "newsdata",
    ):
        self.db = database or db
        self.source = source

        if api_client is not None:
            self.api_client = api_client
        elif source == "gdelt":
            self.api_client = GDELTClient()
        else:
            self.api_client = NewsDataClient()

        self.scraper = scraper or ArticleScraper()

    def collect_from_api(
        self,
        max_calls: int | None = None,
        dry_run: bool = False,
        brand_only: bool = True,
    ) -> CollectionStats:
        """
        Phase 1: Collect article metadata from API (NewsData.io or GDELT).

        Args:
            max_calls: Maximum API calls to make (default: from settings)
            dry_run: If True, don't save to database
            brand_only: If True, search only by brand names (no keywords)

        Returns:
            CollectionStats with results
        """
        max_calls = max_calls or settings.max_api_calls_per_day
        stats = CollectionStats()

        # In-memory deduplication to track articles seen this run
        seen_article_ids: set[str] = set()
        seen_titles: set[str] = set()

        queries = self.api_client.generate_search_queries(brand_only=brand_only)
        random.shuffle(queries)

        logger.info(f"Starting {self.source.upper()} collection with {len(queries)} queries, max {max_calls} calls")

        for query, category in queries:
            if self.api_client.api_calls_made >= max_calls:
                logger.info(f"Reached max API calls ({max_calls})")
                break

            logger.debug(f"Searching: {query}")

            # Call appropriate API based on source
            if self.source == "gdelt":
                articles, next_page = self.api_client.search_news(
                    query,
                    max_records=settings.gdelt_max_records,
                    timespan=settings.gdelt_timespan,
                )
            else:
                articles, next_page = self.api_client.search_news(query, category=category)

            stats.api_calls = self.api_client.api_calls_made

            if not articles:
                continue

            # Filter out articles already seen this run (by ID)
            new_articles = []
            for article in articles:
                if article.article_id in seen_article_ids:
                    stats.articles_duplicates += 1
                else:
                    seen_article_ids.add(article.article_id)
                    new_articles.append(article)

            # Filter out duplicate titles (same article from different sources)
            unique_articles = []
            for article in new_articles:
                normalized = normalize_title(article.title)
                if normalized and normalized in seen_titles:
                    stats.articles_duplicate_title += 1
                else:
                    if normalized:
                        seen_titles.add(normalized)
                    unique_articles.append(article)

            # Filter out articles that don't mention any tracked brand
            branded_articles = []
            for article in unique_articles:
                if article.brands_mentioned:
                    branded_articles.append(article)
                else:
                    stats.articles_no_brand += 1

            if dry_run:
                stats.articles_fetched += len(branded_articles)
                if branded_articles or unique_articles:
                    dup_id = len(articles) - len(new_articles)
                    dup_title = len(new_articles) - len(unique_articles)
                    no_brand = len(unique_articles) - len(branded_articles)
                    logger.info(
                        f"[DRY RUN] Would save {len(branded_articles)} articles "
                        f"({dup_id} dup ID, {dup_title} dup title, {no_brand} no brand)"
                    )
                continue

            with self.db.get_session() as session:
                for article_data in branded_articles:
                    try:
                        _, status = self.db.upsert_article(session, article_data)
                        if status == "new":
                            stats.articles_fetched += 1
                        elif status == "duplicate_id":
                            stats.articles_duplicates += 1
                        elif status == "duplicate_title":
                            stats.articles_duplicate_title += 1
                    except Exception as e:
                        error_msg = f"Failed to save article {article_data.article_id}: {e}"
                        logger.warning(error_msg)
                        stats.errors.append(error_msg)

        logger.info(
            f"API collection complete: {stats.api_calls} calls, "
            f"{stats.articles_fetched} new, {stats.articles_duplicates} dup ID, "
            f"{stats.articles_duplicate_title} dup title, {stats.articles_no_brand} no brand"
        )
        return stats

    def scrape_pending_articles(
        self,
        limit: int = 100,
        dry_run: bool = False,
    ) -> CollectionStats:
        """
        Phase 2: Scrape full content for pending articles.

        Args:
            limit: Maximum articles to scrape
            dry_run: If True, don't save to database

        Returns:
            CollectionStats with results
        """
        stats = CollectionStats()

        with self.db.get_session() as session:
            pending = self.db.get_articles_pending_scrape(session, limit=limit)
            logger.info(f"Found {len(pending)} articles pending scrape")

            for article in pending:
                logger.debug(f"Scraping: {article.url}")

                if dry_run:
                    logger.info(f"[DRY RUN] Would scrape: {article.url}")
                    stats.articles_scraped += 1
                    continue

                result = self.scraper.scrape_with_delay(article.url)

                if result.success:
                    self.db.update_article_content(
                        session,
                        article.article_id,
                        result.content,
                        "success",
                    )
                    stats.articles_scraped += 1
                else:
                    self.db.update_article_content(
                        session,
                        article.article_id,
                        None,
                        result.status,
                        result.error,
                    )
                    stats.articles_scrape_failed += 1

        logger.info(
            f"Scraping complete: {stats.articles_scraped} success, "
            f"{stats.articles_scrape_failed} failed"
        )
        return stats

    def collect_daily_news(
        self,
        max_calls: int | None = None,
        scrape_limit: int = 100,
        dry_run: bool = False,
        brand_only: bool = True,
    ) -> CollectionStats:
        """
        Run full daily collection: API fetch + scraping.

        Args:
            max_calls: Maximum API calls
            scrape_limit: Maximum articles to scrape
            dry_run: If True, don't save to database
            brand_only: If True, search only by brand names (no keywords)

        Note: The API source is determined by the `source` parameter passed to __init__.

        Returns:
            Combined CollectionStats
        """
        self.db.init_db()

        with self.db.get_session() as session:
            run = self.db.create_collection_run(session)
            run_id = run.id

        try:
            api_stats = self.collect_from_api(max_calls=max_calls, dry_run=dry_run, brand_only=brand_only)
            scrape_stats = self.scrape_pending_articles(limit=scrape_limit, dry_run=dry_run)

            combined = CollectionStats(
                api_calls=api_stats.api_calls,
                articles_fetched=api_stats.articles_fetched,
                articles_duplicates=api_stats.articles_duplicates,
                articles_duplicate_title=api_stats.articles_duplicate_title,
                articles_no_brand=api_stats.articles_no_brand,
                articles_scraped=scrape_stats.articles_scraped,
                articles_scrape_failed=scrape_stats.articles_scrape_failed,
                errors=api_stats.errors + scrape_stats.errors,
            )

            status = "success" if not combined.errors else "partial"

            with self.db.get_session() as session:
                run = session.query(type(run)).get(run_id)
                if run:
                    self.db.complete_collection_run(
                        session,
                        run,
                        combined.api_calls,
                        combined.articles_fetched,
                        combined.articles_duplicates,
                        combined.articles_scraped,
                        combined.articles_scrape_failed,
                        status=status,
                    )

            logger.info(f"Daily collection complete: {combined}")
            return combined

        except Exception as e:
            logger.error(f"Collection failed: {e}")
            with self.db.get_session() as session:
                run = session.query(type(run)).get(run_id)
                if run:
                    self.db.complete_collection_run(
                        session,
                        run,
                        self.api_client.api_calls_made,
                        0,
                        0,
                        0,
                        0,
                        status="failed",
                        error_message=str(e),
                    )
            raise
