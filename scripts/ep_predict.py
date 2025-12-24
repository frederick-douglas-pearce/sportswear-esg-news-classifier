#!/usr/bin/env python3
"""Predict ESG content using the trained EP classifier.

This script loads the trained EP classifier pipeline and makes predictions
on new articles to determine if they contain ESG content.

Usage:
    # Predict from command line
    python scripts/ep_predict.py --title "Nike sustainability report" --content "Nike announced..."

    # Predict from JSON file
    python scripts/ep_predict.py --input articles.json

    # Predict with verbose output
    python scripts/ep_predict.py --title "..." --content "..." --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ep1_nb.preprocessing import clean_text, create_text_features


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict ESG content using the EP classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Article title",
    )
    parser.add_argument(
        "--content",
        type=str,
        help="Article content",
    )
    parser.add_argument(
        "--brands",
        type=str,
        nargs="+",
        default=[],
        help="Brand names mentioned in article",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default="Unknown",
        help="Source/publication name",
    )
    parser.add_argument(
        "--category",
        type=str,
        nargs="+",
        default=[],
        help="Article categories",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to JSON file with articles to predict",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing model artifacts",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    return parser.parse_args()


def load_pipeline(model_dir: Path) -> tuple[Any, dict]:
    """Load the trained pipeline and configuration."""
    pipeline_path = model_dir / "ep_classifier_pipeline.joblib"
    config_path = model_dir / "ep_classifier_config.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    pipeline = joblib.load(pipeline_path)
    with open(config_path) as f:
        config = json.load(f)

    return pipeline, config


def predict_articles(
    pipeline: Any,
    articles: list[dict],
    threshold: float,
    verbose: bool = False
) -> list[dict]:
    """Predict ESG content for a list of articles."""
    # Convert to DataFrame
    df = pd.DataFrame(articles)

    # Create text features
    texts = create_text_features(
        df,
        text_col='content',
        title_col='title',
        brands_col='brands',
        source_name_col='source_name',
        category_col='category',
        include_metadata=True,
        clean_func=clean_text
    ).tolist()

    # Get predictions
    probabilities = pipeline.predict_proba(texts)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = []
    for i, (article, prob, pred) in enumerate(zip(articles, probabilities, predictions)):
        result = {
            'title': article.get('title', ''),
            'has_esg': bool(pred),
            'probability': float(prob),
            'threshold': threshold,
        }
        results.append(result)

        if verbose:
            label = "Has ESG" if pred else "No ESG"
            print(f"\n[{i+1}] {label} (prob={prob:.3f})")
            print(f"    Title: {article.get('title', '')[:60]}...")
            brands = article.get('brands', [])
            if brands:
                print(f"    Brands: {', '.join(brands)}")

    return results


def main() -> int:
    """Main entry point."""
    args = parse_args()
    model_dir = Path(args.model_dir)

    # Load pipeline
    try:
        pipeline, config = load_pipeline(model_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    threshold = config.get('threshold', 0.5)

    if args.verbose:
        print("=" * 60)
        print("EP CLASSIFIER PREDICTION")
        print("=" * 60)
        print(f"Threshold: {threshold:.4f}")
        print(f"Model: {config.get('model_name', 'Unknown')}")

    # Prepare articles
    if args.input:
        # Load from file
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return 1

        with open(input_path) as f:
            articles = json.load(f)
        if isinstance(articles, dict):
            articles = [articles]
    elif args.title and args.content:
        # Single article from command line
        articles = [{
            'title': args.title,
            'content': args.content,
            'brands': args.brands,
            'source_name': args.source_name,
            'category': args.category,
        }]
    else:
        print("Error: Either --input or both --title and --content are required")
        return 1

    # Make predictions
    results = predict_articles(pipeline, articles, threshold, verbose=args.verbose)

    # Output results
    if not args.verbose:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        has_esg_count = sum(1 for r in results if r['has_esg'])
        print(f"Total articles: {len(results)}")
        print(f"Has ESG: {has_esg_count}")
        print(f"No ESG: {len(results) - has_esg_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
