"""ESG keyword filter for pre-screening articles before labeling.

This module provides functions to identify articles that are likely to contain
ESG content based on keyword matching. Articles without ESG keywords can be
skipped to save labeling resources.
"""

import re
from dataclasses import dataclass


# ESG-related keywords organized by category
ESG_KEYWORDS = {
    "environmental": [
        "sustainab",  # sustainability, sustainable
        "carbon",
        "emission",
        "renewable",
        "recycl",  # recycle, recycled, recycling
        "climate",
        "environment",
        "eco-friendly",
        "green initiative",
        "waste reduction",
        "pollution",
        "biodegradable",
        "net-zero",
        "net zero",
        "decarboniz",
        "circular economy",
        "solar",
        "wind power",
        "clean energy",
        "plastic",
        "ocean cleanup",
        "zero waste",
        "organic cotton",
        "sustainable material",
        "environmental impact",
        "greenhouse gas",
        "carbon footprint",
        "carbon neutral",
    ],
    "social": [
        "worker right",
        "labor right",
        "labour right",
        "living wage",
        "minimum wage",
        "diversity",
        "inclusion",
        "human rights",
        "fair trade",
        "sweatshop",
        "factory condition",
        "working condition",
        "child labor",
        "child labour",
        "forced labor",
        "forced labour",
        "trade union",
        "labor union",
        "collective bargaining",
        "workplace safety",
        "occupational health",
        "employee wellbeing",
        "supply chain transparency",
        "ethical sourcing",
        "community engagement",
        "social responsibility",
        "equal opportunity",
        "gender equality",
        "racial justice",
        "DEI",  # Diversity, Equity, Inclusion
    ],
    "governance": [
        "board of director",
        "proxy fight",
        "proxy battle",
        "shareholder activist",
        "activist investor",
        "whistleblow",
        "corruption",
        "bribery",
        "ethics violation",
        "ethical violation",
        "ESG report",
        "ESG disclosure",
        "sustainability report",
        "transparency report",
        "compliance violation",
        "audit committee",
        "independent director",
        "corporate governance",
        "executive compensation",
        "anti-corruption",
        "stakeholder engagement",
        "materiality assessment",
    ],
}

# Keywords that indicate NON-ESG content (financial/business news)
NON_ESG_KEYWORDS = [
    # Stock/market patterns
    r"\bstock\s+(tumbl|surge|slip|drop|climb|fall|rise|plunge|jump|soar|slide)",
    r"\b(tumbl|surge|slip|drop|climb|fall|rise|plunge|jump|soar|slide)[sed]*\s+(stock|share)",
    r"\bstock\s+price\b",
    r"\bshare\s+price\b",
    r"\bmarket\s+cap\b",
    # Earnings/financial
    r"\b(Q[1-4]|quarterly)\s+(revenue|earnings|profit|results|beat|miss)",
    r"\b(revenue|earnings|profit)\s+(Q[1-4]|quarterly|fiscal)",
    r"\bEPS\s+(beat|miss|decline|increase)",
    r"\banalyst[s]?\s+(expect|predict|rate|upgrade|downgrade|say)",
    r"\bprice\s+target\b",
    r"\b(buy|sell|hold)\s+rating\b",
    # Investment/trading
    r"\binvest[s]?\s+\$[\d,]+",
    r"\btakes?\s+.*\bstake\b",
    r"\b(insider|institutional)\s+(buy|sell|trading)",
    r"\btrading\s+volume\b",
    # Market indices
    r"\bDow\s+Jones\b",
    r"\bS&P\s*500\b",
    r"\bNASDAQ\b",
    # Product launches (not ESG unless sustainability-focused)
    r"\b(launches?|reveals?|releases?|unveils?)\s+(new\s+)?(collection|shoe|sneaker|uniform|kit|jersey)",
    r"\b(limited\s+edition|colorway|collab)\b",
]


@dataclass
class ESGFilterResult:
    """Result of ESG keyword filtering."""

    has_esg_keywords: bool
    has_non_esg_keywords: bool
    esg_keywords_found: dict[str, list[str]]  # category -> keywords
    non_esg_patterns_matched: list[str]
    recommendation: str  # "label", "skip", "review"

    @property
    def total_esg_keywords(self) -> int:
        """Total number of ESG keywords found."""
        return sum(len(kws) for kws in self.esg_keywords_found.values())


def check_esg_keywords(text: str) -> ESGFilterResult:
    """Check if text contains ESG-related keywords.

    Args:
        text: Article text (title + content)

    Returns:
        ESGFilterResult with keyword analysis
    """
    text_lower = text.lower()

    # Find ESG keywords
    esg_found: dict[str, list[str]] = {cat: [] for cat in ESG_KEYWORDS}
    for category, keywords in ESG_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                esg_found[category].append(kw)

    has_esg = any(kws for kws in esg_found.values())

    # Find non-ESG patterns
    non_esg_matched = []
    for pattern in NON_ESG_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            non_esg_matched.append(pattern)

    has_non_esg = len(non_esg_matched) > 0

    # Determine recommendation
    if has_esg and not has_non_esg:
        recommendation = "label"  # Clear ESG content
    elif has_esg and has_non_esg:
        recommendation = "label"  # Has ESG keywords, worth labeling despite financial content
    elif not has_esg and has_non_esg:
        recommendation = "skip"  # Financial news without ESG keywords
    else:
        recommendation = "review"  # No clear signals either way

    return ESGFilterResult(
        has_esg_keywords=has_esg,
        has_non_esg_keywords=has_non_esg,
        esg_keywords_found={k: v for k, v in esg_found.items() if v},
        non_esg_patterns_matched=non_esg_matched,
        recommendation=recommendation,
    )


def should_label_article(title: str, content: str, threshold: int = 1) -> tuple[bool, str]:
    """Determine if an article should be sent for ESG labeling.

    Args:
        title: Article title
        content: Article content
        threshold: Minimum ESG keywords required (default: 1)

    Returns:
        Tuple of (should_label, reason)
    """
    text = f"{title} {content}"
    result = check_esg_keywords(text)

    if result.recommendation == "skip":
        return False, "No ESG keywords found, financial/business content detected"

    if result.total_esg_keywords >= threshold:
        categories = ", ".join(result.esg_keywords_found.keys())
        return True, f"ESG keywords found in categories: {categories}"

    if result.recommendation == "review":
        # No clear ESG keywords but also no clear non-ESG patterns
        # Could still be ESG content - let LLM decide
        return True, "No clear indicators, sending for LLM review"

    return False, "Insufficient ESG keywords"


def filter_articles_for_labeling(
    articles: list[dict],
    title_key: str = "title",
    content_key: str = "content",
) -> tuple[list[dict], list[dict]]:
    """Filter a list of articles into those to label and those to skip.

    Args:
        articles: List of article dicts
        title_key: Key for article title
        content_key: Key for article content

    Returns:
        Tuple of (articles_to_label, articles_to_skip)
    """
    to_label = []
    to_skip = []

    for article in articles:
        title = article.get(title_key, "")
        content = article.get(content_key, "")
        should_label, reason = should_label_article(title, content)

        article_with_reason = {**article, "_filter_reason": reason}

        if should_label:
            to_label.append(article_with_reason)
        else:
            to_skip.append(article_with_reason)

    return to_label, to_skip
