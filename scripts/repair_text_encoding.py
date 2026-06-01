#!/usr/bin/env python3
"""One-time repair of mojibake / illegal control characters in stored text.

Background
----------
The scraper historically stored Windows-1252 smart punctuation as raw C1
control bytes (mojibake) when a page's charset was misdetected. Those bytes
break the website feed's Jekyll build -- Jekyll parses ``_data/*.json`` with a
YAML parser (Ruby Psych) that rejects C1 control characters (U+0080-U+009F) --
and degrade any text-derived feature. Ingestion now normalizes text at the
scraper and at DB write (see ``src/data_collection/text_normalize.py``); this
script applies the same repair to rows that were stored before that fix.

Columns repaired:
  articles.full_content, articles.title, articles.description
  article_chunks.chunk_text
  brand_labels.reasoning
  label_evidence.excerpt

Rows whose repaired text feeds an embedding (articles, article_chunks) are
counted separately so the operator can decide whether re-embedding is warranted.
In practice the repair only swaps a handful of punctuation code points, so
embedding drift is negligible -- the count is informational.

Usage:
  uv run python scripts/repair_text_encoding.py --dry-run   # preview, no writes
  uv run python scripts/repair_text_encoding.py             # apply repairs
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.database import db
from src.data_collection.models import Article, ArticleChunk, BrandLabel, LabelEvidence
from src.data_collection.text_normalize import has_illegal_chars, normalize_text

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# (model, text column, embedding column or None). The embedding column, when
# present, marks rows whose embedding was derived from the repaired text.
TARGETS = [
    (Article, "full_content", "embedding"),
    (Article, "title", "embedding"),
    (Article, "description", "embedding"),
    (ArticleChunk, "chunk_text", "embedding"),
    (BrandLabel, "reasoning", None),
    (LabelEvidence, "excerpt", None),
]


def repair_target(session, model, column, embedding_column, dry_run, batch_size=500):
    """Scan one column for illegal control characters and repair in place.

    Returns (rows_scanned, rows_repaired, rows_repaired_with_embedding).
    """
    label = f"{model.__tablename__}.{column}"
    col = getattr(model, column)
    scanned = repaired = repaired_with_embedding = 0

    # Pull only id + text (+ whether an embedding exists) to keep memory bounded;
    # the embedding vectors themselves are never loaded.
    cols = [model.id, col]
    if embedding_column:
        cols.append(getattr(model, embedding_column).isnot(None).label("has_embedding"))

    # Collect (id, repaired_text) during the scan, then apply after, so we never
    # issue an UPDATE while a streaming server-side cursor is still open.
    updates: list[tuple[object, str]] = []
    query = session.query(*cols).filter(col.isnot(None)).execution_options(stream_results=True)

    for row in query.yield_per(batch_size):
        scanned += 1
        text = row[1]
        if not has_illegal_chars(text):
            continue
        repaired += 1
        if embedding_column and row.has_embedding:
            repaired_with_embedding += 1
        updates.append((row[0], normalize_text(text)))

    if not dry_run:
        for row_id, repaired_text in updates:
            session.query(model).filter(model.id == row_id).update(
                {column: repaired_text}, synchronize_session=False
            )

    suffix = ""
    if embedding_column and repaired_with_embedding:
        suffix = f" ({repaired_with_embedding} have embeddings derived from this text)"
    logger.info(f"  {label}: scanned {scanned}, repaired {repaired}{suffix}")
    return scanned, repaired, repaired_with_embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing")
    args = parser.parse_args()

    mode = "DRY RUN (no changes will be written)" if args.dry_run else "APPLYING REPAIRS"
    logger.info("=" * 60)
    logger.info(f"Text encoding repair - {mode}")
    logger.info("=" * 60)

    total_repaired = 0
    total_with_embedding = 0
    # get_session commits on success; in dry-run repair_target issues no writes,
    # so there is nothing to commit.
    with db.get_session() as session:
        for model, column, embedding_column in TARGETS:
            _, repaired, with_emb = repair_target(session, model, column, embedding_column, args.dry_run)
            total_repaired += repaired
            total_with_embedding += with_emb

    logger.info("-" * 60)
    verb = "would be repaired" if args.dry_run else "repaired"
    logger.info(f"Total rows {verb}: {total_repaired}")
    if total_with_embedding:
        logger.info(
            f"Of these, {total_with_embedding} have an embedding derived from the repaired text. "
            "Re-embedding is optional (repairs only swap punctuation code points)."
        )
    if args.dry_run and total_repaired:
        logger.info("Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
