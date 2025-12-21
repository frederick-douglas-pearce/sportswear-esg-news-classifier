"""NER feature analysis for FP classifier tuning.

This module analyzes false positive examples to identify NER patterns
that can improve the classifier. Run this analysis when new labeled
data becomes available to tune the NER entity type categories.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def load_fp_data(data_path: str = 'data/fp_training_data.jsonl') -> pd.DataFrame:
    """Load FP training data from JSONL file.

    Args:
        data_path: Path to the JSONL training data file

    Returns:
        DataFrame with all training examples
    """
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def analyze_fp_by_brand(df: pd.DataFrame) -> Dict[str, int]:
    """Analyze false positives grouped by brand.

    Args:
        df: DataFrame with training data

    Returns:
        Dictionary of brand -> FP count
    """
    fp_df = df[df['is_sportswear'] == 0]

    brand_fps = Counter()
    for _, row in fp_df.iterrows():
        brands = row['brands'] if isinstance(row['brands'], list) else eval(row['brands'])
        for brand in brands:
            brand_fps[brand] += 1

    return dict(brand_fps.most_common())


def analyze_fp_reasons(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Extract and categorize FP reasons by brand.

    Args:
        df: DataFrame with training data

    Returns:
        Dictionary of brand -> list of FP reasons
    """
    fp_df = df[df['is_sportswear'] == 0]

    brand_reasons = defaultdict(list)
    for _, row in fp_df.iterrows():
        brands = row['brands'] if isinstance(row['brands'], list) else eval(row['brands'])
        reason = row.get('fp_reason', None)
        if pd.notna(reason):
            for brand in brands:
                brand_reasons[brand].append(reason)

    return dict(brand_reasons)


def categorize_fp_types(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Categorize false positives by type based on reason text.

    Categories:
    - geographic: Location/region mentions (Patagonia region, Anta constituency)
    - animal: Animal references (Puma the cat)
    - person: Person names (Vans as surname)
    - company: Different company with same name (Everlast Gyms)
    - product: Non-sportswear product (Converse shoes vs conversation)
    - event: Event names
    - other: Uncategorized

    Args:
        df: DataFrame with training data

    Returns:
        Dictionary of brand -> {category: count}
    """
    fp_df = df[df['is_sportswear'] == 0]

    # Keywords for each category
    category_keywords = {
        'geographic': ['region', 'geographic', 'location', 'area', 'province',
                      'constituency', 'district', 'territory', 'park', 'mountain',
                      'chile', 'argentina', 'indian', 'india'],
        'animal': ['animal', 'cat', 'wild', 'wildlife', 'species', 'zoo',
                  'leopard', 'panther', 'cougar', 'jaguar'],
        'person': ['person', 'name', 'surname', 'actor', 'player', 'musician',
                  'artist', 'politician', 'celebrity'],
        'company': ['gym', 'fitness center', 'hotel', 'restaurant', 'power',
                   'energy', 'mining', 'financial', 'bank', 'insurance',
                   'different company', 'unrelated company'],
        'event': ['event', 'festival', 'concert', 'tournament', 'competition',
                 'race', 'match', 'game'],
        'product': ['non-sportswear', 'jewelry', 'diamond', 'gold', 'curtain',
                   'furniture', 'equipment', 'tool'],
    }

    brand_categories = defaultdict(lambda: defaultdict(int))

    for _, row in fp_df.iterrows():
        brands = row['brands'] if isinstance(row['brands'], list) else eval(row['brands'])
        reason = str(row.get('fp_reason', '')).lower()
        title = str(row.get('title', '')).lower()
        content = str(row.get('content', ''))[:500].lower()

        combined_text = f"{reason} {title} {content}"

        for brand in brands:
            categorized = False
            for category, keywords in category_keywords.items():
                if any(kw in combined_text for kw in keywords):
                    brand_categories[brand][category] += 1
                    categorized = True
                    break

            if not categorized:
                brand_categories[brand]['other'] += 1

    return {brand: dict(cats) for brand, cats in brand_categories.items()}


def analyze_ner_entities(
    df: pd.DataFrame,
    proximity_window: int = 15,
    sample_size: Optional[int] = None
) -> Dict[str, Dict[str, int]]:
    """Analyze NER entity types appearing near brand mentions.

    Uses the same logic as FPFeatureTransformer._compute_ner_features()
    to ensure consistency between analysis and feature computation.

    Args:
        df: DataFrame with training data
        proximity_window: Word window around brand mentions (default 15)
        sample_size: Optional limit on samples to process (for speed)

    Returns:
        Dictionary with 'fp' and 'sportswear' keys, each containing
        entity_type -> count mapping
    """
    import spacy

    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')

    # Import brands list - same as feature_transformer uses
    from src.data_collection.config import BRANDS

    fp_entities = Counter()
    sw_entities = Counter()

    fp_df = df[df['is_sportswear'] == 0]
    sw_df = df[df['is_sportswear'] == 1]

    if sample_size:
        fp_df = fp_df.head(sample_size)
        sw_df = sw_df.head(sample_size)

    def extract_brands_from_text(text: str) -> List[str]:
        """Extract brands from text - mirrors feature_transformer logic."""
        text_lower = text.lower()
        found_brands = []
        for brand in BRANDS:
            if brand.lower() in text_lower:
                found_brands.append(brand)
        return found_brands

    def extract_entities_near_brands(text: str) -> Counter:
        """Extract entity types near brand mentions.

        Uses same logic as FPFeatureTransformer._compute_ner_features():
        - Finds brand character positions in text
        - Computes window in characters (proximity_window * 6 chars/word)
        - Counts entities within that window of any brand mention
        """
        entity_counts = Counter()

        if not text.strip():
            return entity_counts

        # Extract brands using same method as feature_transformer
        brands = extract_brands_from_text(text)
        if not brands:
            return entity_counts

        # Process text with spaCy (limit length for performance - same as feature_transformer)
        doc = nlp(text[:100000])
        text_lower = text.lower()

        # Find brand character positions - same logic as feature_transformer
        brand_char_positions = []
        for brand in brands:
            brand_lower = brand.lower()
            start_pos = 0
            while True:
                pos = text_lower.find(brand_lower, start_pos)
                if pos == -1:
                    break
                brand_char_positions.append((pos, pos + len(brand_lower)))
                start_pos = pos + len(brand_lower)

        if not brand_char_positions:
            return entity_counts

        # Check entities near brand mentions - same window calculation as feature_transformer
        window_chars = proximity_window * 6  # Approx chars per word

        for ent in doc.ents:
            # Check if entity is near any brand mention
            for brand_start, brand_end in brand_char_positions:
                if (abs(ent.start_char - brand_end) <= window_chars or
                    abs(ent.end_char - brand_start) <= window_chars):
                    entity_counts[ent.label_] += 1
                    break  # Count entity once per text (same as feature_transformer)

        return entity_counts

    print("Analyzing FP entities (near brand mentions only)...")
    for i, (_, row) in enumerate(fp_df.iterrows()):
        if i % 20 == 0:
            print(f"  Processing FP {i+1}/{len(fp_df)}...")
        text = f"{row.get('title', '')} {row.get('content', '')}"
        fp_entities.update(extract_entities_near_brands(text))

    print("Analyzing sportswear entities (near brand mentions only)...")
    for i, (_, row) in enumerate(sw_df.iterrows()):
        if i % 50 == 0:
            print(f"  Processing SW {i+1}/{len(sw_df)}...")
        text = f"{row.get('title', '')} {row.get('content', '')}"
        sw_entities.update(extract_entities_near_brands(text))

    return {
        'false_positive': dict(fp_entities),
        'sportswear': dict(sw_entities),
    }


def compute_entity_discrimination(
    entity_counts: Dict[str, Dict[str, int]]
) -> pd.DataFrame:
    """Compute discrimination power of each entity type.

    Higher positive values = more indicative of sportswear
    Higher negative values = more indicative of false positive

    Args:
        entity_counts: Output from analyze_ner_entities

    Returns:
        DataFrame with entity types and their discrimination scores
    """
    fp_counts = entity_counts['false_positive']
    sw_counts = entity_counts['sportswear']

    # Get all entity types
    all_entities = set(fp_counts.keys()) | set(sw_counts.keys())

    # Compute totals for normalization
    fp_total = sum(fp_counts.values()) or 1
    sw_total = sum(sw_counts.values()) or 1

    results = []
    for entity in all_entities:
        fp_count = fp_counts.get(entity, 0)
        sw_count = sw_counts.get(entity, 0)

        # Normalized frequencies
        fp_freq = fp_count / fp_total
        sw_freq = sw_count / sw_total

        # Discrimination score: positive = sportswear, negative = FP
        # Using log odds ratio approximation
        epsilon = 0.001
        discrimination = (sw_freq + epsilon) / (fp_freq + epsilon)

        results.append({
            'entity_type': entity,
            'fp_count': fp_count,
            'sw_count': sw_count,
            'fp_freq': fp_freq,
            'sw_freq': sw_freq,
            'discrimination': discrimination,
            'suggests': 'sportswear' if discrimination > 1.5 else
                       ('false_positive' if discrimination < 0.67 else 'neutral'),
        })

    df = pd.DataFrame(results)
    return df.sort_values('discrimination', ascending=True)


def generate_ner_recommendations(
    entity_analysis: pd.DataFrame,
    fp_threshold: float = 0.5,
    sw_threshold: float = 2.0
) -> Dict[str, List[str]]:
    """Generate recommendations for NER feature categories.

    Args:
        entity_analysis: Output from compute_entity_discrimination
        fp_threshold: Discrimination below this suggests FP
        sw_threshold: Discrimination above this suggests sportswear

    Returns:
        Dictionary with recommended FP and SW entity types
    """
    fp_entities = entity_analysis[
        entity_analysis['discrimination'] < fp_threshold
    ]['entity_type'].tolist()

    sw_entities = entity_analysis[
        entity_analysis['discrimination'] > sw_threshold
    ]['entity_type'].tolist()

    neutral_entities = entity_analysis[
        (entity_analysis['discrimination'] >= fp_threshold) &
        (entity_analysis['discrimination'] <= sw_threshold)
    ]['entity_type'].tolist()

    return {
        'fp_indicating': fp_entities,
        'sw_indicating': sw_entities,
        'neutral': neutral_entities,
    }


def run_full_analysis(
    data_path: str = 'data/fp_training_data.jsonl',
    sample_size: Optional[int] = None,
    save_report: bool = True,
    report_path: str = 'reports/ner_analysis.txt'
) -> Dict:
    """Run complete NER analysis and generate recommendations.

    Args:
        data_path: Path to training data
        sample_size: Optional limit on samples for NER analysis
        save_report: Whether to save text report
        report_path: Path for report file

    Returns:
        Dictionary with all analysis results
    """
    print("=" * 70)
    print("NER FEATURE ANALYSIS FOR FP CLASSIFIER")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    df = load_fp_data(data_path)
    print(f"   Total samples: {len(df)}")
    print(f"   Sportswear: {(df['is_sportswear'] == 1).sum()}")
    print(f"   False positives: {(df['is_sportswear'] == 0).sum()}")

    # Analyze FP by brand
    print("\n2. Analyzing FP distribution by brand...")
    brand_fps = analyze_fp_by_brand(df)
    print("   Top 10 brands with FPs:")
    for brand, count in list(brand_fps.items())[:10]:
        print(f"     {brand}: {count}")

    # Categorize FP types
    print("\n3. Categorizing FP types...")
    fp_categories = categorize_fp_types(df)

    # Aggregate categories across brands
    total_categories = Counter()
    for brand, cats in fp_categories.items():
        for cat, count in cats.items():
            total_categories[cat] += count

    print("   FP type distribution:")
    for cat, count in total_categories.most_common():
        print(f"     {cat}: {count}")

    # Analyze NER entities
    print("\n4. Analyzing NER entities near brand mentions...")
    entity_counts = analyze_ner_entities(df, sample_size=sample_size)

    print("\n   FP entity types (top 10):")
    for ent, count in sorted(entity_counts['false_positive'].items(),
                             key=lambda x: -x[1])[:10]:
        print(f"     {ent}: {count}")

    print("\n   Sportswear entity types (top 10):")
    for ent, count in sorted(entity_counts['sportswear'].items(),
                             key=lambda x: -x[1])[:10]:
        print(f"     {ent}: {count}")

    # Compute discrimination
    print("\n5. Computing entity discrimination scores...")
    entity_analysis = compute_entity_discrimination(entity_counts)

    print("\n   Entity discrimination (sorted by FP-indicating to SW-indicating):")
    print(entity_analysis.to_string(index=False))

    # Generate recommendations
    print("\n6. Generating recommendations...")
    recommendations = generate_ner_recommendations(entity_analysis)

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("\nFP-indicating entity types (add to FP_ENTITY_TYPES):")
    for ent in recommendations['fp_indicating']:
        print(f"  - {ent}")

    print("\nSportswear-indicating entity types (add to SW_ENTITY_TYPES):")
    for ent in recommendations['sw_indicating']:
        print(f"  - {ent}")

    print("\nNeutral entity types (not discriminative):")
    for ent in recommendations['neutral']:
        print(f"  - {ent}")

    # Current vs recommended comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH CURRENT SETTINGS")
    print("=" * 70)

    # Import current settings from feature_transformer
    from src.fp1_nb.feature_transformer import FPFeatureTransformer
    current_fp = FPFeatureTransformer.FP_ENTITY_TYPES
    current_sw = FPFeatureTransformer.SW_ENTITY_TYPES

    print("\nCurrent FP_ENTITY_TYPES:", current_fp)
    print("Recommended:", recommendations['fp_indicating'])

    print("\nCurrent SW_ENTITY_TYPES:", current_sw)
    print("Recommended:", recommendations['sw_indicating'])

    # Show what should be added/removed
    fp_to_add = set(recommendations['fp_indicating']) - set(current_fp)
    fp_to_remove = set(current_fp) - set(recommendations['fp_indicating']) - set(recommendations['neutral'])
    sw_to_add = set(recommendations['sw_indicating']) - set(current_sw)
    sw_to_remove = set(current_sw) - set(recommendations['sw_indicating']) - set(recommendations['neutral'])

    if fp_to_add:
        print(f"\nSuggested additions to FP_ENTITY_TYPES: {list(fp_to_add)}")
    if fp_to_remove:
        print(f"Consider removing from FP_ENTITY_TYPES: {list(fp_to_remove)}")
    if sw_to_add:
        print(f"\nSuggested additions to SW_ENTITY_TYPES: {list(sw_to_add)}")
    if sw_to_remove:
        print(f"Consider removing from SW_ENTITY_TYPES: {list(sw_to_remove)}")

    # Save report
    if save_report:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write("NER Feature Analysis Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Sportswear: {(df['is_sportswear'] == 1).sum()}\n")
            f.write(f"False positives: {(df['is_sportswear'] == 0).sum()}\n\n")
            f.write("Entity Discrimination Analysis:\n")
            f.write(entity_analysis.to_string(index=False))
            f.write("\n\nRecommendations:\n")
            f.write(f"FP-indicating: {recommendations['fp_indicating']}\n")
            f.write(f"SW-indicating: {recommendations['sw_indicating']}\n")
            f.write(f"Neutral: {recommendations['neutral']}\n")
        print(f"\nReport saved to: {report_path}")

    return {
        'data': df,
        'brand_fps': brand_fps,
        'fp_categories': fp_categories,
        'entity_counts': entity_counts,
        'entity_analysis': entity_analysis,
        'recommendations': recommendations,
    }


if __name__ == '__main__':
    # Run analysis when called directly
    results = run_full_analysis(sample_size=None)
