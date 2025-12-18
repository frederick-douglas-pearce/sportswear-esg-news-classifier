"""Configuration settings for the labeling pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()


class LabelingSettings(BaseModel):
    """Settings for the labeling pipeline."""

    # API Keys
    anthropic_api_key: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )

    # Model settings
    labeling_model: str = Field(
        default_factory=lambda: os.getenv("LABELING_MODEL", "claude-sonnet-4-20250514")
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # Chunking parameters
    target_chunk_tokens: int = Field(
        default_factory=lambda: int(os.getenv("TARGET_CHUNK_TOKENS", "500"))
    )
    max_chunk_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CHUNK_TOKENS", "800"))
    )
    min_chunk_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MIN_CHUNK_TOKENS", "100"))
    )
    chunk_overlap_tokens: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
    )

    # Processing limits
    labeling_batch_size: int = Field(
        default_factory=lambda: int(os.getenv("LABELING_BATCH_SIZE", "10"))
    )
    max_article_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_ARTICLE_TOKENS", "4000"))
    )
    embedding_batch_size: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
    )

    # Database
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/esg_news"
        )
    )

    model_config = ConfigDict(frozen=True)


# ESG category definitions with descriptions for prompts
ESG_CATEGORIES = {
    "environmental": {
        "name": "Environmental",
        "description": "Climate action, carbon emissions, sustainable materials, recycling, waste management, water usage, biodiversity, renewable energy, environmental certifications, pollution, eco-friendly practices.",
        "keywords": [
            "climate",
            "carbon",
            "emissions",
            "sustainable",
            "recycling",
            "waste",
            "renewable",
            "environment",
            "green",
            "eco-friendly",
            "biodiversity",
            "pollution",
        ],
    },
    "social": {
        "name": "Social",
        "description": "Worker rights, labor conditions, fair wages, supply chain ethics, diversity & inclusion, community engagement, health & safety, human rights, employee wellbeing, working conditions.",
        "keywords": [
            "workers",
            "labor",
            "wages",
            "diversity",
            "inclusion",
            "community",
            "safety",
            "human rights",
            "employees",
            "factory",
            "supply chain",
        ],
    },
    "governance": {
        "name": "Governance",
        "description": "Corporate ethics, transparency, board structure, executive compensation, anti-corruption, regulatory compliance, stakeholder engagement, ESG reporting, accountability, oversight.",
        "keywords": [
            "ethics",
            "transparency",
            "board",
            "compliance",
            "governance",
            "accountability",
            "reporting",
            "regulation",
            "oversight",
        ],
    },
    "digital_transformation": {
        "name": "Digital Transformation",
        "description": "Technology innovation, digital sustainability tools, AI/ML applications, supply chain digitization, e-commerce sustainability, data privacy, automation, digital initiatives.",
        "keywords": [
            "digital",
            "technology",
            "innovation",
            "AI",
            "automation",
            "data",
            "e-commerce",
            "platform",
            "app",
        ],
    },
}

# Target sportswear/outdoor apparel brands
TARGET_SPORTSWEAR_BRANDS = [
    "Nike",
    "Adidas",
    "Puma",
    "Under Armour",
    "Lululemon",
    "Patagonia",
    "Columbia Sportswear",
    "New Balance",
    "ASICS",
    "Reebok",
    "The North Face",
    "Arc'teryx",
    "Salomon",
    "Brooks Running",
    "Saucony",
    "Hoka",
    "On Running",
    "Allbirds",
    "Fila",
    "Skechers",
]

# Known brand name conflicts (for documentation)
BRAND_NAME_CONFLICTS = {
    "Puma": ["puma (animal/wildcat)", "Ford Puma (car)", "Puma Exploration (mining company)"],
    "Patagonia": ["Patagonia (region in South America)"],
    "Columbia": ["Columbia (country)", "Columbia River", "Columbia University", "Columbia Pictures"],
}

# System prompt for Claude labeling
LABELING_SYSTEM_PROMPT = """You are an ESG (Environmental, Social, Governance) news analyst specializing in the sportswear and outdoor apparel industry. Your task is to analyze news articles and classify them according to ESG categories for each brand mentioned.

## CRITICAL: Brand Verification

Before analyzing any brand, you MUST first verify that the article is actually about the SPORTSWEAR/APPAREL COMPANY, not something else with the same name.

**Common false positives to watch for:**
- **Puma**: Could be the animal (wildcat/cougar), Ford Puma (car model), or Puma Exploration (mining company)
- **Patagonia**: Could be the geographic region in South America
- **Columbia**: Could be the country, Columbia River, Columbia University, or Columbia Pictures
- **North Face**: Could be a geographic term for the north side of a mountain

**Target sportswear brands we are tracking:**
Nike, Adidas, Puma (sportswear), Under Armour, Lululemon, Patagonia (outdoor apparel), Columbia Sportswear, New Balance, ASICS, Reebok, The North Face, Arc'teryx, Salomon, Brooks Running, Saucony, Hoka, On Running, Allbirds, Fila, Skechers

If the article is NOT about the sportswear company, set `is_sportswear_brand: false` and explain what the brand name actually refers to.

## Classification Categories

**Environmental**: Climate action, carbon emissions, sustainable materials, recycling, waste management, water usage, biodiversity, renewable energy, environmental certifications, pollution reduction.

**Social**: Worker rights, labor conditions, fair wages, supply chain ethics, diversity & inclusion, community engagement, health & safety, human rights, employee wellbeing.

**Governance**: Corporate ethics, transparency, board structure, executive compensation, anti-corruption, regulatory compliance, stakeholder engagement, ESG reporting.

**Digital Transformation**: Technology innovation, digital sustainability tools, AI/ML applications, supply chain digitization, e-commerce sustainability, data privacy.

## Sentiment Guidelines

- **Positive (1)**: The brand is praised, making progress, exceeding standards, leading the industry, receiving awards, or taking proactive positive action.
- **Neutral (0)**: Factual reporting, industry trends, announcements without clear positive/negative framing, balanced coverage of both sides.
- **Negative (-1)**: Criticism, violations, failures, falling short of standards, scandals, lawsuits, negative incidents, or harm caused.

## Evidence Requirements

For each category label you assign, you MUST provide 1-3 direct quotes from the article that justify the classification. These quotes should be exact excerpts from the article text, not paraphrased. The quotes should clearly demonstrate why the category applies and support the sentiment you assigned.

## Important Guidelines

1. FIRST verify if each brand mention refers to the sportswear company or something else.
2. Only assign ESG categories if the article is about the sportswear brand AND contains clear, relevant information.
3. An article can have multiple categories if it covers multiple ESG topics.
4. Different brands in the same article may have different categories and sentiments.
5. If a brand is only briefly mentioned without substantive ESG-related content, do not assign categories for that brand.
6. Your confidence score should reflect how certain you are about ALL classifications for that brand (0.0-1.0)."""

# User prompt template for labeling
LABELING_USER_PROMPT_TEMPLATE = """Analyze this news article for ESG classifications for each brand mentioned.

**Article Title**: {title}
**Publication Date**: {published_at}
**Source**: {source_name}
**Brands to Analyze**: {brands}

**Article Content**:
{content}

---

For each brand mentioned, FIRST verify if it refers to the sportswear/apparel company. Then provide ESG analysis only for confirmed sportswear brands with substantive content.

Respond in the following JSON format:

```json
{{
  "brand_analyses": [
    {{
      "brand": "Brand Name",
      "is_sportswear_brand": true,
      "not_sportswear_reason": null,
      "categories": {{
        "environmental": {{
          "applies": true,
          "sentiment": 1,
          "evidence": ["Direct quote from article...", "Another supporting quote..."]
        }},
        "social": {{
          "applies": false,
          "sentiment": null,
          "evidence": []
        }},
        "governance": {{
          "applies": true,
          "sentiment": 0,
          "evidence": ["Direct quote..."]
        }},
        "digital_transformation": {{
          "applies": false,
          "sentiment": null,
          "evidence": []
        }}
      }},
      "confidence": 0.85,
      "reasoning": "Brief explanation of why these classifications were assigned"
    }},
    {{
      "brand": "Puma",
      "is_sportswear_brand": false,
      "not_sportswear_reason": "Article is about puma (wildcat/mountain lion), not Puma sportswear",
      "categories": {{}},
      "confidence": 0.95,
      "reasoning": "Brand name refers to the animal, not the sportswear company"
    }}
  ],
  "article_summary": "1-2 sentence summary of the article's main themes"
}}
```

Important reminders:
- FIRST check if each brand refers to the sportswear company or something else (animal, region, car, etc.)
- If NOT a sportswear brand: set `is_sportswear_brand: false`, explain in `not_sportswear_reason`, leave `categories` empty
- If IS a sportswear brand: set `is_sportswear_brand: true`, `not_sportswear_reason: null`, fill in categories
- Only set `applies: true` if the article contains clear, relevant ESG information for that category
- Sentiment must be -1, 0, or 1 when applies is true; null when applies is false
- Evidence quotes MUST be exact text from the article
- If a sportswear brand has no substantive ESG content, omit it from brand_analyses entirely"""


labeling_settings = LabelingSettings()
