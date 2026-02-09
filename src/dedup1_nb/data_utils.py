"""Data loading and preparation utilities for deduplication analysis."""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def load_articles_from_db(
    days: int = 30,
    labeling_status: Optional[list[str]] = None,
    min_content_length: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load recent labeled articles from database for deduplication analysis.

    Args:
        days: Number of days to look back for articles
        labeling_status: List of labeling statuses to include.
                        Defaults to ['labeled'] for scored articles.
        min_content_length: Minimum content length to include
        verbose: Whether to print loading info

    Returns:
        DataFrame with article data including content, title, brands, dates
    """
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    engine = create_engine(database_url)

    if labeling_status is None:
        labeling_status = ["labeled"]

    # Build status filter
    status_placeholders = ", ".join([f":status_{i}" for i in range(len(labeling_status))])
    status_params = {f"status_{i}": s for i, s in enumerate(labeling_status)}

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

    query = f"""
    SELECT
        a.id,
        a.article_id,
        a.title,
        a.full_content,
        a.description,
        a.url,
        a.source_name,
        a.published_at,
        a.labeled_at,
        a.labeling_status,
        a.brands_mentioned
    FROM articles a
    WHERE a.labeling_status IN ({status_placeholders})
      AND a.labeled_at >= :cutoff_date
      AND LENGTH(COALESCE(a.full_content, '')) >= :min_content_length
    ORDER BY a.labeled_at DESC
    """

    with engine.connect() as conn:
        df = pd.read_sql(
            text(query),
            conn,
            params={
                **status_params,
                "cutoff_date": cutoff_date,
                "min_content_length": min_content_length,
            },
        )

    if verbose:
        print(f"Loaded {len(df):,} articles from last {days} days")
        print(f"  Labeling statuses: {labeling_status}")
        print(f"  Date range: {df['published_at'].min()} to {df['published_at'].max()}")
        print(f"  Sources: {df['source_name'].nunique()} unique")

    return df


def create_article_dataframe(
    articles: list,
    include_brand_labels: bool = True,
) -> pd.DataFrame:
    """Create a DataFrame from Article ORM objects.

    This is useful when working with articles loaded via SQLAlchemy
    relationships rather than raw SQL.

    Args:
        articles: List of Article ORM objects
        include_brand_labels: Whether to include brand label info

    Returns:
        DataFrame with article data
    """
    records = []
    for article in articles:
        record = {
            "id": article.id,
            "article_id": article.article_id,
            "title": article.title,
            "full_content": article.full_content,
            "description": article.description,
            "url": article.url,
            "source_name": article.source_name,
            "published_at": article.published_at,
            "labeled_at": article.labeled_at,
            "labeling_status": article.labeling_status,
            "brands_mentioned": article.brands_mentioned,
        }

        if include_brand_labels and hasattr(article, "brand_labels"):
            # Extract brand label summary
            brands_with_labels = []
            for bl in article.brand_labels:
                brands_with_labels.append(
                    {
                        "brand": bl.brand_name,
                        "categories": list(bl.esg_categories.keys()) if bl.esg_categories else [],
                    }
                )
            record["brand_labels"] = brands_with_labels

        records.append(record)

    return pd.DataFrame(records)


def prepare_text_representations(
    df: pd.DataFrame,
    text_col: str = "full_content",
    title_col: str = "title",
    max_content_chars: int = 1000,
    include_title: bool = True,
) -> list[str]:
    """Prepare text representations for embedding.

    Combines title and content (truncated) into a single string
    suitable for sentence embedding.

    Args:
        df: DataFrame with article data
        text_col: Column containing article content
        title_col: Column containing article title
        max_content_chars: Maximum characters to include from content
        include_title: Whether to prepend title to content

    Returns:
        List of text strings ready for embedding
    """
    texts = []
    for _, row in df.iterrows():
        content = row.get(text_col) or row.get("description") or ""
        title = row.get(title_col, "") if include_title else ""

        # Truncate content
        content_preview = content[:max_content_chars] if content else ""

        # Combine title and content
        if title and content_preview:
            text = f"{title} {content_preview}"
        elif title:
            text = title
        else:
            text = content_preview

        texts.append(text)

    return texts


def get_article_pairs_at_similarity(
    df: pd.DataFrame,
    embeddings,
    similarity_ranges: list[tuple[float, float]],
    n_pairs_per_range: int = 10,
    random_state: int = 42,
) -> list[dict]:
    """Sample article pairs at different similarity levels for manual labeling.

    Args:
        df: DataFrame with article data
        embeddings: Embedding matrix (n_articles x embedding_dim)
        similarity_ranges: List of (min_sim, max_sim) tuples to sample from
        n_pairs_per_range: Number of pairs to sample per range
        random_state: Random seed for reproducibility

    Returns:
        List of dicts with pair info and similarity score
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    np.random.seed(random_state)

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    pairs = []
    n_articles = len(df)

    for min_sim, max_sim in similarity_ranges:
        # Find all pairs in this similarity range
        candidates = []
        for i in range(n_articles):
            for j in range(i + 1, n_articles):
                sim = sim_matrix[i, j]
                if min_sim <= sim < max_sim:
                    candidates.append((i, j, sim))

        # Sample pairs
        if len(candidates) > n_pairs_per_range:
            indices = np.random.choice(len(candidates), n_pairs_per_range, replace=False)
            sampled = [candidates[idx] for idx in indices]
        else:
            sampled = candidates

        # Create pair records
        for i, j, sim in sampled:
            pairs.append(
                {
                    "article_id_1": str(df.iloc[i]["article_id"]),
                    "article_id_2": str(df.iloc[j]["article_id"]),
                    "title_1": df.iloc[i]["title"],
                    "title_2": df.iloc[j]["title"],
                    "source_1": df.iloc[i]["source_name"],
                    "source_2": df.iloc[j]["source_name"],
                    "similarity": float(sim),
                    "similarity_range": f"{min_sim:.2f}-{max_sim:.2f}",
                    "is_duplicate": None,  # To be filled by human labeler
                }
            )

    return pairs


def load_validation_pairs(file_path: str) -> list[dict]:
    """Load labeled validation pairs from JSONL file.

    Args:
        file_path: Path to the validation pairs file

    Returns:
        List of pair dicts with is_duplicate labels
    """
    import json
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Validation pairs file not found: {file_path}")

    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))

    return pairs


def save_validation_pairs(pairs: list[dict], file_path: str) -> None:
    """Save validation pairs to JSONL file.

    Args:
        pairs: List of pair dicts
        file_path: Path to save the file
    """
    import json
    from pathlib import Path

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Saved {len(pairs)} pairs to {file_path}")
