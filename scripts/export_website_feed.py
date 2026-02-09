#!/usr/bin/env python3
"""Export ESG news data for website display.

Generates JSON and Atom feeds from labeled articles for display on
Jekyll/al-folio static website.

Usage:
    # Export JSON for Jekyll _data folder
    uv run python scripts/export_website_feed.py --format json -o ~/website/_data/esg_news.json

    # Export Atom feed for RSS subscribers
    uv run python scripts/export_website_feed.py --format atom -o ~/website/assets/feeds/esg_news.atom

    # Export both formats
    uv run python scripts/export_website_feed.py --format both \
        --json-output ~/website/_data/esg_news.json \
        --atom-output ~/website/assets/feeds/esg_news.atom

    # Limit number of articles
    uv run python scripts/export_website_feed.py --format json --limit 100

    # Preview without writing files
    uv run python scripts/export_website_feed.py --format json --dry-run
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import spacy

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import joinedload

from src.data_collection.database import db
from src.data_collection.models import Article, ArticleChunk, BrandLabel, LabelEvidence
from src.labeling.evidence_matcher import _get_confidence_label


# Default context size for evidence snippets
SNIPPET_CONTEXT_CHARS = 100
DEFAULT_SENTENCES_BEFORE = 1
DEFAULT_SENTENCES_AFTER = 1

# Default top-N evidence per category for export
DEFAULT_TOP_N_EVIDENCE_PER_CATEGORY = 3

# Lazy-loaded spaCy model
_nlp = None


def _get_nlp():
    """Get spaCy NLP model, loading lazily on first use."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_snippet_with_sentences(
    chunk_text: str,
    excerpt: str,
    sentences_before: int = DEFAULT_SENTENCES_BEFORE,
    sentences_after: int = DEFAULT_SENTENCES_AFTER,
) -> str:
    """Extract snippet with N sentences of context around the excerpt.

    Uses spaCy for sentence boundary detection to provide natural context
    around evidence excerpts.

    Args:
        chunk_text: Full text of the matched chunk
        excerpt: The evidence excerpt to find
        sentences_before: Number of sentences to include before the excerpt
        sentences_after: Number of sentences to include after the excerpt

    Returns:
        Snippet with sentence context around the excerpt, with ellipsis
        indicators if truncated. Falls back to character-based extraction
        if excerpt cannot be located in sentences.
    """
    if not chunk_text or not excerpt:
        return chunk_text or ""

    # Find excerpt position in text (case-insensitive)
    excerpt_lower = excerpt.lower()
    chunk_lower = chunk_text.lower()

    pos = chunk_lower.find(excerpt_lower)
    if pos == -1:
        # Try first few words as fallback
        first_words = " ".join(excerpt.split()[:5]).lower()
        if first_words:
            pos = chunk_lower.find(first_words)

    if pos == -1:
        # Cannot find excerpt - fall back to character-based extraction
        return extract_snippet_char_based(chunk_text, excerpt)

    excerpt_end = pos + len(excerpt)

    # Use spaCy to tokenize into sentences
    nlp = _get_nlp()
    doc = nlp(chunk_text)
    sentences = list(doc.sents)

    if not sentences:
        return extract_snippet_char_based(chunk_text, excerpt)

    # Find which sentence(s) contain the excerpt
    excerpt_sentence_indices = []
    for i, sent in enumerate(sentences):
        sent_start = sent.start_char
        sent_end = sent.end_char
        # Check if sentence overlaps with excerpt
        if sent_start <= excerpt_end and sent_end >= pos:
            excerpt_sentence_indices.append(i)

    if not excerpt_sentence_indices:
        # Excerpt spans sentence boundaries or not found
        return extract_snippet_char_based(chunk_text, excerpt)

    # Determine range of sentences to include
    first_excerpt_idx = min(excerpt_sentence_indices)
    last_excerpt_idx = max(excerpt_sentence_indices)

    start_idx = max(0, first_excerpt_idx - sentences_before)
    end_idx = min(len(sentences) - 1, last_excerpt_idx + sentences_after)

    # Build the snippet from sentences
    snippet_sentences = [sentences[i].text.strip() for i in range(start_idx, end_idx + 1)]
    snippet = " ".join(snippet_sentences)

    # Add ellipsis indicators
    prefix = "..." if start_idx > 0 else ""
    suffix = "..." if end_idx < len(sentences) - 1 else ""

    return f"{prefix}{snippet}{suffix}"


def extract_snippet_char_based(
    chunk_text: str, excerpt: str, context_chars: int = SNIPPET_CONTEXT_CHARS
) -> str:
    """Extract a condensed snippet around the matched excerpt using character windows.

    This is the fallback method when sentence-based extraction cannot locate
    the excerpt properly.

    Args:
        chunk_text: Full text of the matched chunk
        excerpt: The evidence excerpt to find
        context_chars: Characters of context around the excerpt

    Returns:
        Snippet with context around the excerpt, or truncated chunk if not found
    """
    if not chunk_text or not excerpt:
        return chunk_text or ""

    excerpt_lower = excerpt.lower()
    chunk_lower = chunk_text.lower()

    # Try to find exact position first
    pos = chunk_lower.find(excerpt_lower)
    if pos == -1:
        # Try to find the first few words of the excerpt
        first_words = " ".join(excerpt.split()[:5]).lower()
        if first_words:
            pos = chunk_lower.find(first_words)

    if pos == -1:
        # Can't find it - return truncated chunk
        if len(chunk_text) <= context_chars * 2:
            return chunk_text
        return chunk_text[: context_chars * 2] + "..."

    # Extract context around the match
    context_start = max(0, pos - context_chars)
    context_end = min(len(chunk_text), pos + len(excerpt) + context_chars)

    # Adjust to word boundaries
    if context_start > 0:
        # Find next space after context_start
        space_pos = chunk_text.find(" ", context_start)
        if space_pos != -1 and space_pos < pos:
            context_start = space_pos + 1

    if context_end < len(chunk_text):
        # Find last space before context_end
        space_pos = chunk_text.rfind(" ", pos + len(excerpt), context_end)
        if space_pos != -1:
            context_end = space_pos

    snippet = chunk_text[context_start:context_end].strip()

    # Add ellipsis indicators
    prefix = "..." if context_start > 0 else ""
    suffix = "..." if context_end < len(chunk_text) else ""

    return f"{prefix}{snippet}{suffix}"


def extract_snippet(
    chunk_text: str, excerpt: str, context_chars: int = SNIPPET_CONTEXT_CHARS
) -> str:
    """Extract a snippet around the matched excerpt with sentence context.

    Uses sentence-aware extraction by default, which provides 1 sentence
    before and 1 sentence after the excerpt for natural context.

    Args:
        chunk_text: Full text of the matched chunk
        excerpt: The evidence excerpt to find
        context_chars: Characters of context (used for fallback only)

    Returns:
        Snippet with context around the excerpt
    """
    return extract_snippet_with_sentences(chunk_text, excerpt)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_sentiment_label(sentiment: int | None) -> str | None:
    """Convert sentiment value to label."""
    if sentiment is None:
        return None
    return {-1: "negative", 0: "neutral", 1: "positive"}.get(sentiment)


def sanitize_text(text: str | None) -> str | None:
    """Remove or replace characters that cause JSON/YAML parsing issues.

    Removes emoji and other problematic Unicode characters that can cause
    issues with Jekyll's YAML parser.
    """
    if text is None:
        return None

    import re

    # Remove emoji and other symbol characters
    # This regex pattern matches most emoji and pictographic characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # variation selectors
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )

    return emoji_pattern.sub("", text)


# Scorecard configuration
SCORECARD_PERIOD_DAYS = 14
SCORECARD_TOP_N = 3
SCORECARD_BOTTOM_N = 3
SCORECARD_SIMILARITY_THRESHOLD = 0.72  # Articles with similarity >= this are considered duplicates

# Sentiment to points mapping: positive=+2, neutral=+1, negative=-1
SENTIMENT_POINTS = {1: 2, 0: 1, -1: -1}

# ESG categories for scorecard
SCORECARD_CATEGORIES = ['environmental', 'social', 'governance', 'digital_transformation']


def deduplicate_articles(
    articles: list[Article],
    similarity_threshold: float = SCORECARD_SIMILARITY_THRESHOLD,
) -> tuple[list[Article], int]:
    """Deduplicate articles based on content similarity using sentence embeddings.

    Uses the NoveltyScorer's sentence-transformer model (all-MiniLM-L6-v2) to compute
    embeddings and identify duplicate stories across different sources.

    Args:
        articles: List of articles to deduplicate
        similarity_threshold: Cosine similarity above which articles are considered duplicates

    Returns:
        Tuple of (deduplicated articles list, number of duplicates removed)
    """
    if len(articles) <= 1:
        return articles, 0

    from sklearn.metrics.pairwise import cosine_similarity

    from src.deployment.novelty import NoveltyScorer

    logger = logging.getLogger(__name__)

    # Get article texts (title + content for better matching)
    texts = []
    for article in articles:
        # Use title + first part of content for embedding
        content = article.full_content or article.description or ""
        # Limit content length to avoid very long texts dominating
        content_preview = content[:1000] if content else ""
        texts.append(f"{article.title} {content_preview}")

    # Compute embeddings using existing NoveltyScorer infrastructure
    logger.info(f"Computing embeddings for {len(articles)} articles...")
    scorer = NoveltyScorer()
    embeddings = scorer.embed_texts(texts, show_progress=False)

    # Compute pairwise similarities
    sim_matrix = cosine_similarity(embeddings)

    # Greedy clustering: keep first article, mark similar ones as duplicates
    keep_indices = []
    seen = set()

    for i in range(len(articles)):
        if i in seen:
            continue
        keep_indices.append(i)
        # Mark all similar articles as duplicates (seen)
        for j in range(i + 1, len(articles)):
            if j not in seen and sim_matrix[i, j] >= similarity_threshold:
                seen.add(j)

    deduplicated = [articles[i] for i in keep_indices]
    duplicates_removed = len(articles) - len(deduplicated)

    if duplicates_removed > 0:
        logger.info(
            f"Deduplication: {len(articles)} -> {len(deduplicated)} articles "
            f"({duplicates_removed} duplicates removed at threshold {similarity_threshold})"
        )

    return deduplicated, duplicates_removed


def calculate_brand_scores(
    articles: list[Article],
    period_days: int = SCORECARD_PERIOD_DAYS,
    dedupe: bool = True,
    similarity_threshold: float = SCORECARD_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Calculate ESG scores for each brand from recent articles.

    Scoring: positive=+2, neutral=+1, negative=-1

    Args:
        articles: List of Article objects with brand_labels loaded
        period_days: Number of days to look back for scoring
        dedupe: Whether to deduplicate similar articles before scoring
        similarity_threshold: Cosine similarity threshold for deduplication

    Returns:
        dict with:
        - period_days: int
        - period_start: str (ISO date)
        - period_end: str (ISO date)
        - articles_in_period: int (before deduplication)
        - articles_after_dedup: int (after deduplication, if enabled)
        - duplicates_removed: int
        - top_brands: list (top N with positive scores)
        - bottom_brands: list (bottom N, excluding top brands)
        - all_brand_scores: list (full ranking)
    """
    now = datetime.now(timezone.utc)
    period_start = now - timedelta(days=period_days)

    # Filter articles within period
    recent_articles = []
    for article in articles:
        pub_date = article.published_at
        if pub_date:
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
            if pub_date >= period_start:
                recent_articles.append(article)

    articles_in_period = len(recent_articles)
    duplicates_removed = 0

    # Deduplicate articles if enabled
    if dedupe and len(recent_articles) > 1:
        recent_articles, duplicates_removed = deduplicate_articles(
            recent_articles, similarity_threshold=similarity_threshold
        )

    # Initialize scores per brand
    # Structure: {brand: {category: score, 'total': score, 'article_count': count, 'article_ids': set}}
    brand_scores: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            'environmental': 0,
            'social': 0,
            'governance': 0,
            'digital_transformation': 0,
            'total': 0,
            'article_count': 0,
            'article_ids': set(),
        }
    )

    # Calculate scores from brand labels
    for article in recent_articles:
        for label in article.brand_labels:
            brand = label.brand
            scores = brand_scores[brand]

            # Track unique articles per brand
            article_id = str(article.id)
            if article_id not in scores['article_ids']:
                scores['article_ids'].add(article_id)
                scores['article_count'] += 1

            # Score each category
            category_sentiment_map = [
                ('environmental', label.environmental, label.environmental_sentiment),
                ('social', label.social, label.social_sentiment),
                ('governance', label.governance, label.governance_sentiment),
                ('digital_transformation', label.digital_transformation, label.digital_sentiment),
            ]

            for category, applies, sentiment in category_sentiment_map:
                if applies and sentiment is not None:
                    points = SENTIMENT_POINTS.get(sentiment, 0)
                    scores[category] += points
                    scores['total'] += points

    # Convert to list and sort by total score descending, then alphabetically
    all_scores = []
    for brand, scores in brand_scores.items():
        all_scores.append({
            'brand': brand,
            'total': scores['total'],
            'environmental': scores['environmental'],
            'social': scores['social'],
            'governance': scores['governance'],
            'digital_transformation': scores['digital_transformation'],
            'article_count': scores['article_count'],
        })

    # Sort by total score descending, then alphabetically by brand name
    all_scores.sort(key=lambda x: (-x['total'], x['brand']))

    # Top N brands with positive scores only, including ties at the cutoff
    positive_brands = [b for b in all_scores if b['total'] > 0]
    if len(positive_brands) <= SCORECARD_TOP_N:
        top_brands = positive_brands
    else:
        # Include all brands tied with the Nth position
        cutoff_score = positive_brands[SCORECARD_TOP_N - 1]['total']
        top_brands = [b for b in positive_brands if b['total'] >= cutoff_score]
    top_brand_names = {b['brand'] for b in top_brands}

    # Assign medals to top brands (ties get the same medal)
    medals = ['gold', 'silver', 'bronze']
    for i, brand in enumerate(top_brands):
        if i < len(medals):
            brand['medal'] = medals[i]
        else:
            # Check if tied with a medal position (compare scores)
            if len(top_brands) >= 3 and brand['total'] == top_brands[2]['total']:
                brand['medal'] = 'bronze'
            elif len(top_brands) >= 2 and brand['total'] == top_brands[1]['total']:
                brand['medal'] = 'silver'
            elif len(top_brands) >= 1 and brand['total'] == top_brands[0]['total']:
                brand['medal'] = 'gold'
            else:
                brand['medal'] = None

    # Last N brands: must have negative scores, excluding top brands, sorted worst first
    # Include ties at the cutoff position
    negative_brands = [b for b in all_scores if b['brand'] not in top_brand_names and b['total'] < 0]
    if len(negative_brands) <= SCORECARD_BOTTOM_N:
        bottom_brands = negative_brands[::-1]  # Reverse to show worst first
    else:
        # Sort by score ascending (worst first) to find the cutoff
        sorted_worst_first = sorted(negative_brands, key=lambda x: (x['total'], x['brand']))
        cutoff_score = sorted_worst_first[SCORECARD_BOTTOM_N - 1]['total']
        # Include all brands tied at or below the cutoff score
        bottom_brands = [b for b in sorted_worst_first if b['total'] <= cutoff_score]

    return {
        'period_days': period_days,
        'period_start': period_start.strftime('%Y-%m-%d'),
        'period_end': now.strftime('%Y-%m-%d'),
        'articles_in_period': articles_in_period,
        'articles_after_dedup': len(recent_articles),
        'duplicates_removed': duplicates_removed,
        'top_brands': top_brands,
        'bottom_brands': bottom_brands,
        'all_brand_scores': all_scores,
    }


def query_labeled_articles(session, limit: int | None = None) -> list[Article]:
    """Query labeled articles with all relationships loaded.

    Args:
        session: Database session
        limit: Maximum number of articles to return

    Returns:
        List of Article objects with brand_labels and evidence loaded
    """
    query = (
        session.query(Article)
        .filter(Article.labeling_status == "labeled")
        .options(
            joinedload(Article.brand_labels)
            .joinedload(BrandLabel.evidence)
            .joinedload(LabelEvidence.chunk)
        )
        .order_by(Article.published_at.desc())
    )

    if limit:
        query = query.limit(limit)

    return query.all()


def format_article_for_json(
    article: Article,
    top_n_evidence_per_category: int = DEFAULT_TOP_N_EVIDENCE_PER_CATEGORY,
) -> dict[str, Any]:
    """Format an article for JSON export.

    Args:
        article: Article object with loaded relationships
        top_n_evidence_per_category: Maximum evidence items per category (0 = no limit)

    Returns:
        Dictionary matching the JSON schema
    """
    # Collect all unique brands and categories for this article
    all_brands = set()
    all_categories = set()
    brand_details = []

    for label in article.brand_labels:
        all_brands.add(label.brand)

        # Build category info
        categories = {}
        for cat_name, cat_field, sent_field in [
            ("environmental", "environmental", "environmental_sentiment"),
            ("social", "social", "social_sentiment"),
            ("governance", "governance", "governance_sentiment"),
            ("digital_transformation", "digital_transformation", "digital_sentiment"),
        ]:
            applies = getattr(label, cat_field, False) or False
            sentiment = getattr(label, sent_field, None)

            if applies:
                all_categories.add(cat_name)

            categories[cat_name] = {
                "applies": applies,
                "sentiment": sentiment,
                "sentiment_label": get_sentiment_label(sentiment) if applies else None,
            }

        # Build evidence list, sorted by rerank_score (fallback to relevance_score)
        # Group evidence by category for top-N filtering
        evidence_by_category: dict[str, list[LabelEvidence]] = {}
        for ev in label.evidence:
            if ev.category not in evidence_by_category:
                evidence_by_category[ev.category] = []
            evidence_by_category[ev.category].append(ev)

        # Sort each category by rerank_score (desc), then relevance_score (desc)
        for cat_evidence in evidence_by_category.values():
            cat_evidence.sort(
                key=lambda e: (
                    e.rerank_score if e.rerank_score is not None else -1,
                    e.relevance_score if e.relevance_score is not None else -1,
                ),
                reverse=True,
            )

        # Build evidence list with top-N per category
        evidence_list = []
        for cat, cat_evidence in evidence_by_category.items():
            # Apply top-N limit if configured
            if top_n_evidence_per_category > 0:
                cat_evidence = cat_evidence[:top_n_evidence_per_category]

            for ev in cat_evidence:
                # Use rerank_score as primary score if available, else relevance_score
                primary_score = ev.rerank_score if ev.rerank_score is not None else ev.relevance_score
                evidence_item = {
                    "category": ev.category,
                    "excerpt": sanitize_text(ev.excerpt),
                    "relevance_score": ev.relevance_score,
                    "rerank_score": ev.rerank_score,
                    "match_method": ev.match_method,
                    "confidence": _get_confidence_label(primary_score or 0.0),
                }
                # Include condensed context snippet instead of full chunk text
                if ev.chunk and ev.chunk.chunk_text and ev.excerpt:
                    snippet = extract_snippet(ev.chunk.chunk_text, ev.excerpt)
                    evidence_item["context_snippet"] = sanitize_text(snippet)
                evidence_list.append(evidence_item)

        brand_details.append({
            "brand": label.brand,
            "categories": categories,
            "evidence": evidence_list,
            "confidence": label.confidence_score,
            "reasoning": label.reasoning,
        })

    # Format published date
    published_at = article.published_at
    if published_at and published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    return {
        "id": str(article.id),
        "title": article.title,
        "url": article.url,
        "source_name": article.source_name,
        "published_at": published_at.isoformat() if published_at else None,
        "published_date": published_at.strftime("%Y-%m-%d") if published_at else None,
        "brands": sorted(all_brands),
        "categories": sorted(all_categories),
        "brand_details": brand_details,
    }


def export_to_json(
    articles: list[Article],
    top_n_evidence_per_category: int = DEFAULT_TOP_N_EVIDENCE_PER_CATEGORY,
    include_scorecard: bool = True,
    scorecard_period_days: int = SCORECARD_PERIOD_DAYS,
    scorecard_dedupe: bool = True,
    scorecard_similarity_threshold: float = SCORECARD_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Export articles to JSON format.

    Args:
        articles: List of Article objects
        top_n_evidence_per_category: Maximum evidence items per category (0 = no limit)
        include_scorecard: Whether to include brand scorecard section
        scorecard_period_days: Number of days for scorecard calculation
        scorecard_dedupe: Whether to deduplicate articles for scorecard
        scorecard_similarity_threshold: Cosine similarity threshold for deduplication

    Returns:
        Dictionary with full JSON structure
    """
    # Collect all unique brands and categories across all articles
    all_brands = set()
    all_categories = set()

    formatted_articles = []
    for article in articles:
        formatted = format_article_for_json(article, top_n_evidence_per_category)
        formatted_articles.append(formatted)
        all_brands.update(formatted["brands"])
        all_categories.update(formatted["categories"])

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_articles": len(formatted_articles),
        "brands": sorted(all_brands),
        "categories": sorted(all_categories),
    }

    # Add scorecard section if requested
    if include_scorecard:
        result["scorecard"] = calculate_brand_scores(
            articles,
            period_days=scorecard_period_days,
            dedupe=scorecard_dedupe,
            similarity_threshold=scorecard_similarity_threshold,
        )

    result["articles"] = formatted_articles

    return result


def export_to_atom(articles: list[Article], base_url: str) -> str:
    """Export articles to Atom feed format.

    Args:
        articles: List of Article objects
        base_url: Base URL for the website

    Returns:
        Atom feed as XML string
    """
    from feedgen.feed import FeedGenerator

    fg = FeedGenerator()
    fg.id(f"{base_url}/esg-news/")
    fg.title("ESG Sportswear News")
    fg.subtitle("Environmental, Social, and Governance news for sportswear brands")
    fg.author({"name": "ESG News Classifier", "uri": base_url})
    fg.link(href=f"{base_url}/esg-news/", rel="alternate")
    fg.link(href=f"{base_url}/assets/feeds/esg_news.atom", rel="self")
    fg.language("en")
    fg.updated(datetime.now(timezone.utc))

    for article in articles:
        fe = fg.add_entry()
        fe.id(str(article.id))
        fe.title(article.title)
        fe.link(href=article.url)

        # Build summary from brands and categories
        brands = set()
        categories = set()
        for label in article.brand_labels:
            brands.add(label.brand)
            if label.environmental:
                categories.add("Environmental")
            if label.social:
                categories.add("Social")
            if label.governance:
                categories.add("Governance")
            if label.digital_transformation:
                categories.add("Digital Transformation")

        summary_parts = []
        if brands:
            summary_parts.append(f"Brands: {', '.join(sorted(brands))}")
        if categories:
            summary_parts.append(f"Categories: {', '.join(sorted(categories))}")
        if article.source_name:
            summary_parts.append(f"Source: {article.source_name}")

        fe.summary(" | ".join(summary_parts))

        # Set dates
        published_at = article.published_at
        if published_at:
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
            fe.published(published_at)
            fe.updated(published_at)

        # Add categories as tags
        for brand in brands:
            fe.category(term=brand, label=brand)
        for cat in categories:
            fe.category(term=cat.lower().replace(" ", "_"), label=cat)

    return fg.atom_str(pretty=True).decode("utf-8")


def write_json(data: dict, filepath: Path) -> None:
    """Write JSON data to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def write_atom(content: str, filepath: Path) -> None:
    """Write Atom feed to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def print_summary(articles: list[Article], json_data: dict | None = None) -> None:
    """Print export summary."""
    print(f"\n{'=' * 60}")
    print("ESG NEWS EXPORT SUMMARY")
    print("=" * 60)
    print(f"\nTotal articles: {len(articles)}")

    if json_data:
        print(f"Unique brands: {len(json_data['brands'])}")
        print(f"Categories: {', '.join(json_data['categories'])}")

        # Brand distribution
        from collections import Counter
        brand_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()

        for article in json_data["articles"]:
            for brand in article["brands"]:
                brand_counts[brand] += 1
            for cat in article["categories"]:
                category_counts[cat] += 1

        print("\nTop 10 brands:")
        for brand, count in brand_counts.most_common(10):
            print(f"  {brand}: {count} articles")

        print("\nCategory distribution:")
        for cat, count in category_counts.most_common():
            print(f"  {cat}: {count} articles")

        # Scorecard summary
        if "scorecard" in json_data:
            scorecard = json_data["scorecard"]
            print(f"\n{'-' * 40}")
            print("SUSTAINABILITY SCORECARD")
            print(f"Period: {scorecard['period_start']} to {scorecard['period_end']} ({scorecard['period_days']} days)")

            # Show deduplication stats
            articles_in_period = scorecard.get('articles_in_period', 0)
            articles_after_dedup = scorecard.get('articles_after_dedup', articles_in_period)
            duplicates_removed = scorecard.get('duplicates_removed', 0)
            if duplicates_removed > 0:
                print(f"Articles: {articles_in_period} total -> {articles_after_dedup} unique ({duplicates_removed} duplicates removed)")
            else:
                print(f"Articles: {articles_after_dedup} in period")

            if scorecard["top_brands"]:
                print("\nTop Performers:")
                medal_emoji = {"gold": "🥇", "silver": "🥈", "bronze": "🥉"}
                for brand in scorecard["top_brands"]:
                    medal = medal_emoji.get(brand.get("medal", ""), "")
                    print(f"  {medal} {brand['brand']}: +{brand['total']} (E:{brand['environmental']:+d} S:{brand['social']:+d} G:{brand['governance']:+d} D:{brand['digital_transformation']:+d})")
            else:
                print("\nTop Performers: None (no brands with positive scores)")

            if scorecard["bottom_brands"]:
                print("\nBack of the Pack:")
                for brand in scorecard["bottom_brands"]:
                    print(f"  {brand['brand']}: {brand['total']:+d} (E:{brand['environmental']:+d} S:{brand['social']:+d} G:{brand['governance']:+d} D:{brand['digital_transformation']:+d})")
            else:
                print("\nBack of the Pack: None (no brands with negative scores)")

    print("\n" + "=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export ESG news for website display",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["json", "atom", "both"],
        help="Output format: json, atom, or both",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (for single format exports)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        help="JSON output file path (for --format both)",
    )
    parser.add_argument(
        "--atom-output",
        type=str,
        help="Atom output file path (for --format both)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of articles to export",
    )
    parser.add_argument(
        "--top-n-evidence",
        type=int,
        default=DEFAULT_TOP_N_EVIDENCE_PER_CATEGORY,
        help=f"Maximum evidence items per category per brand (0 = no limit, default: {DEFAULT_TOP_N_EVIDENCE_PER_CATEGORY})",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://frederick-douglas-pearce.github.io",
        help="Base URL for the website (for Atom feed links)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview export without writing files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-scorecard",
        action="store_true",
        help="Exclude scorecard section from JSON export",
    )
    parser.add_argument(
        "--scorecard-period-days",
        type=int,
        default=SCORECARD_PERIOD_DAYS,
        help=f"Lookback period for scorecard calculation (default: {SCORECARD_PERIOD_DAYS})",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable article deduplication for scorecard",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=SCORECARD_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold for deduplication (default: {SCORECARD_SIMILARITY_THRESHOLD})",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Validate arguments
    if args.format == "both":
        if not args.json_output or not args.atom_output:
            logger.error("--json-output and --atom-output required when --format=both")
            return 1
    elif not args.output and not args.dry_run:
        logger.error("-o/--output required unless --dry-run is specified")
        return 1

    # Initialize database
    db.init_db()

    # Query articles and export within session context
    # (relationships require active session)
    logger.info("Querying labeled articles...")
    with db.get_session() as session:
        articles = query_labeled_articles(session, limit=args.limit)

        if not articles:
            logger.warning("No labeled articles found")
            return 0

        logger.info(f"Found {len(articles)} labeled articles")

        # Export based on format (must be within session for lazy-loaded relationships)
        json_data = None
        atom_content = None

        if args.format in ("json", "both"):
            logger.info("Generating JSON export...")
            json_data = export_to_json(
                articles,
                top_n_evidence_per_category=args.top_n_evidence,
                include_scorecard=not args.no_scorecard,
                scorecard_period_days=args.scorecard_period_days,
                scorecard_dedupe=not args.no_dedupe,
                scorecard_similarity_threshold=args.similarity_threshold,
            )

        if args.format in ("atom", "both"):
            logger.info("Generating Atom feed...")
            atom_content = export_to_atom(articles, args.base_url)

    # Write files outside session
    if json_data and not args.dry_run:
        output_path = Path(args.json_output if args.format == "both" else args.output)
        write_json(json_data, output_path)
        print(f"JSON export written to: {output_path}")

    if atom_content and not args.dry_run:
        output_path = Path(args.atom_output if args.format == "both" else args.output)
        write_atom(atom_content, output_path)
        print(f"Atom feed written to: {output_path}")

    # Print summary
    print_summary(articles, json_data)

    if args.dry_run:
        print("\n[DRY RUN] No files written")

        # Show sample JSON output
        if json_data and json_data["articles"]:
            print("\nSample article JSON:")
            print(json.dumps(json_data["articles"][0], indent=2, default=str)[:2000])

    return 0


if __name__ == "__main__":
    sys.exit(main())
