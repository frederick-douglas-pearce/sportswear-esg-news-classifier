"""Evidence summarizer for generating human-readable ESG explanations.

Generates condensed explanations linking evidence matches to ESG categories,
using the single source of truth from ESG_CATEGORIES in config.py.
"""

import logging
from dataclasses import dataclass

from .config import ESG_CATEGORIES
from .evidence_matcher import EvidenceMatch

logger = logging.getLogger(__name__)


@dataclass
class EvidenceSummary:
    """Summarized explanation for an evidence match."""

    category: str
    category_name: str
    brand: str
    excerpt: str
    detected_topics: list[str]
    explanation: str
    confidence: str


class EvidenceSummarizer:
    """Generate condensed explanations for evidence matches.

    Uses ESG_CATEGORIES from config.py as single source of truth for
    category keywords and names.
    """

    def __init__(self):
        """Initialize the summarizer with ESG category data."""
        # Extract keywords and names from single source of truth
        self.esg_keywords: dict[str, list[str]] = {
            cat: info["keywords"]
            for cat, info in ESG_CATEGORIES.items()
        }
        self.esg_names: dict[str, str] = {
            cat: info["name"]
            for cat, info in ESG_CATEGORIES.items()
        }
        self.esg_descriptions: dict[str, str] = {
            cat: info["description"]
            for cat, info in ESG_CATEGORIES.items()
        }

    def summarize_evidence(
        self,
        category: str,
        match: EvidenceMatch,
        brand: str,
    ) -> EvidenceSummary:
        """Generate explanation linking evidence to ESG category.

        Args:
            category: ESG category key (e.g., "environmental")
            match: Evidence match result
            brand: Brand name for the evidence

        Returns:
            EvidenceSummary with detected topics and explanation
        """
        detected = self._detect_topics(match.excerpt, category)
        category_name = self.esg_names.get(category, category.title())

        # Build human-readable explanation
        if detected:
            topics_str = self._format_topics(detected)
            explanation = (
                f"For {brand}: This excerpt discusses {topics_str} "
                f"which relates to {category_name}."
            )
        else:
            # No specific keywords found, use generic explanation
            explanation = (
                f"For {brand}: This excerpt is relevant to {category_name} "
                f"based on semantic similarity."
            )

        return EvidenceSummary(
            category=category,
            category_name=category_name,
            brand=brand,
            excerpt=match.excerpt,
            detected_topics=detected,
            explanation=explanation,
            confidence=match.confidence,
        )

    def _detect_topics(self, text: str, category: str) -> list[str]:
        """Find which ESG keywords from config appear in text.

        Args:
            text: Text to search for keywords
            category: ESG category key

        Returns:
            List of detected keyword topics
        """
        keywords = self.esg_keywords.get(category, [])
        text_lower = text.lower()

        detected = []
        for keyword in keywords:
            # Handle multi-word keywords (like "supply chain", "human rights")
            if keyword.lower() in text_lower:
                detected.append(keyword)

        return detected

    def _format_topics(self, topics: list[str]) -> str:
        """Format a list of topics into readable string.

        Args:
            topics: List of topic keywords

        Returns:
            Formatted string like "carbon, emissions, and sustainability"
        """
        if not topics:
            return ""
        if len(topics) == 1:
            return topics[0]
        if len(topics) == 2:
            return f"{topics[0]} and {topics[1]}"
        return ", ".join(topics[:-1]) + f", and {topics[-1]}"

    def summarize_all_evidence(
        self,
        brand: str,
        category_matches: dict[str, list[EvidenceMatch]],
    ) -> list[EvidenceSummary]:
        """Summarize all evidence for a brand across categories.

        Args:
            brand: Brand name
            category_matches: Dict mapping category to list of matches

        Returns:
            List of EvidenceSummary objects
        """
        summaries = []
        for category, matches in category_matches.items():
            for match in matches:
                # Only summarize matched evidence (not unmatched)
                if match.match_method != "none":
                    summary = self.summarize_evidence(category, match, brand)
                    summaries.append(summary)
        return summaries


def generate_article_summary(
    brand_evidence: dict[str, dict[str, list[EvidenceMatch]]],
) -> dict[str, list[EvidenceSummary]]:
    """Generate summaries for all evidence in an article.

    Args:
        brand_evidence: Nested dict {brand: {category: [matches]}}

    Returns:
        Dict mapping brand to list of summaries
    """
    summarizer = EvidenceSummarizer()
    result: dict[str, list[EvidenceSummary]] = {}

    for brand, category_matches in brand_evidence.items():
        summaries = summarizer.summarize_all_evidence(brand, category_matches)
        if summaries:
            result[brand] = summaries

    return result
