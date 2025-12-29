"""Configuration settings for the labeling pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from src.data_collection.config import BRANDS

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

    # FP Classifier Pre-filter Settings
    fp_classifier_enabled: bool = Field(
        default_factory=lambda: os.getenv("FP_CLASSIFIER_ENABLED", "false").lower() == "true"
    )
    fp_classifier_url: str = Field(
        default_factory=lambda: os.getenv("FP_CLASSIFIER_URL", "http://localhost:8000")
    )
    fp_skip_llm_threshold: float = Field(
        default_factory=lambda: float(os.getenv("FP_SKIP_LLM_THRESHOLD", "0.5"))
    )
    fp_classifier_timeout: float = Field(
        default_factory=lambda: float(os.getenv("FP_CLASSIFIER_TIMEOUT", "30.0"))
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

# Target sportswear/outdoor apparel brands - imported from data_collection.config
TARGET_SPORTSWEAR_BRANDS = BRANDS

# Known brand name conflicts (for documentation and prompt guidance)
BRAND_NAME_CONFLICTS = {
    "Puma": ["puma (animal/wildcat)", "Ford Puma (car)", "Puma Exploration (mining company)"],
    "Patagonia": ["Patagonia (region in South America)"],
    "Columbia": ["Columbia (country)", "Columbia River", "Columbia University", "Columbia Pictures"],
    "Black Diamond": ["Black Diamond Corporation (power company)", "black diamond (gemstone)", "black diamond ski run"],
    "North Face": ["north face (geographic term for north side of mountain)"],
    "Anta": [
        "Anta (town/constituency in Rajasthan, India - 'Anta Assembly', 'Anta bypoll')",
        "Antalpha Platform (NASDAQ: ANTA - financial/crypto company, NOT sportswear)",
        "anta as substring in words (e.g., 'Vasundhara', 'Santa', 'advantage')",
    ],
    "Vans": [
        "vans (vehicles - 'container vans', 'camper vans', 'police vans', 'delivery vans')",
        "vans as common noun for any vehicle type",
        "electric vans (EVs) - 'electric van fleet', 'EV vans', 'battery-swapping vans'",
        "cargo/transit vans - 'cargo vans', 'transit vans', 'VW vans', 'Ford Transit vans'",
        "automotive policy context - 'ZEV mandate for vans', 'CO2 legislation for vans'",
    ],
    "Decathlon": [
        "Decathlon Capital Partners (venture capital/private equity firm)",
        "Decathlon Management (investment management company)",
        "Only Decathlon (French sporting goods retailer) articles about stores, products, or sports equipment are valid",
    ],
    "Converse": [
        "converse (verb meaning to talk/communicate)",
        "Converse, Texas (city near San Antonio)",
        "Converse County (Wyoming)",
    ],
}

# System prompt template for Claude labeling (brand list populated dynamically)
_LABELING_SYSTEM_PROMPT_TEMPLATE = """You are an ESG (Environmental, Social, Governance) news analyst specializing in the sportswear and outdoor apparel industry. Your task is to analyze news articles and classify them according to ESG categories for each brand mentioned.

## CRITICAL: Brand Verification

Before analyzing any brand, you MUST first verify that the article is actually about the SPORTSWEAR/APPAREL COMPANY, not something else with the same name.

**Common false positives to watch for:**
- **Puma**: Could be the animal (wildcat/cougar), Ford Puma (car model), or Puma Exploration (mining company)
- **Patagonia**: Could be the geographic region in South America
- **Columbia**: Could be the country, Columbia River, Columbia University, or Columbia Pictures
- **North Face**: Could be a geographic term for the north side of a mountain
- **Black Diamond**: Could be Black Diamond Corporation (power company), "black diamond" gemstone, or black diamond ski run difficulty. Note: Black Diamond Equipment (climbing/outdoor gear) IS the brand we're tracking.
- **Anta**: VERY HIGH FALSE POSITIVE RATE - carefully check context:
  - "Anta Assembly", "Anta bypoll", "Anta constituency" = political district in Rajasthan, India (NOT sportswear)
  - "Antalpha Platform" (NASDAQ: ANTA) = financial/crypto company (NOT the sportswear brand)
  - Words containing "anta" as substring (e.g., "Vasundhara", "Santa", "advantage") = NOT the brand
  - Only ANTA Sports (Chinese sportswear company) articles about shoes, apparel, athletes, or sports sponsorships are valid
- **Vans**: VERY HIGH FALSE POSITIVE RATE - check if referring to VEHICLES:
  - "container vans", "delivery vans", "camper vans", "police vans", "council vans" = vehicles (NOT sportswear)
  - "electric vans", "EV vans", "van fleet", "battery-swapping vans" = electric vehicles (NOT sportswear)
  - "cargo vans", "transit vans", "VW vans", "Ford Transit" = commercial vehicles (NOT sportswear)
  - "ZEV mandate for vans", "CO2 legislation for cars and vans" = automotive policy (NOT sportswear)
  - Any article about GM BrightDrop, Volkswagen ID. Buzz, electric fleets = vehicles (NOT sportswear)
  - Only Vans (skateboarding/footwear brand) articles about shoes, apparel, skateboarding, or retail are valid
- **Decathlon**: Check if referring to the SPORTING GOODS RETAILER:
  - "Decathlon Capital Partners", "Decathlon Management" = investment/VC firms (NOT sportswear)
  - Only Decathlon (French sporting goods retailer) articles about stores, products, or sports equipment are valid
- **Li-Ning/361 Degrees/Xtep/Peak**: Chinese sportswear brands - verify it's about the apparel company

**Target sportswear brands we are tracking:**
{brands}

If the article is NOT about the sportswear company, set `is_sportswear_brand: false` and explain what the brand name actually refers to.

## Classification Categories

**Environmental**: Climate action, carbon emissions, sustainable materials, recycling, waste management, water usage, biodiversity, renewable energy, environmental certifications, pollution reduction.

**Social**: Worker rights, labor conditions, fair wages, supply chain ethics, diversity & inclusion, community engagement, health & safety, human rights, employee wellbeing.

**Governance**: Corporate ethics, transparency, board structure, executive compensation, anti-corruption, regulatory compliance, stakeholder engagement, ESG reporting.

**Digital Transformation**: Technology innovation, digital sustainability tools, AI/ML applications, supply chain digitization, e-commerce sustainability, data privacy.

## CRITICAL: ESG News Content vs Product Features

This is the most important distinction for accurate classification. ESG categories should ONLY be assigned when the article's PRIMARY FOCUS is on ESG topics, NOT when ESG-related features are mentioned incidentally in product-focused content.

**Do NOT assign ESG categories to product sale/release/review articles, EVEN IF they mention:**
- Sustainable materials (recycled plastic, sugarcane EVA, organic cotton, recycled polyester)
- Eco-friendly product names (EcoStep, GreenRun, Sustainability Collection)
- Environmental claims ("lighter environmental footprint", "eco-conscious design")
- Digital/tech product features (app integration, smart features)
- Brief mentions of the brand's sustainability commitments in product marketing copy

**Examples of articles that should NOT receive ESG categories:**
- "Hoka Transport GTX waterproof walking shoes on sale for $139.99" - Even though it mentions "sugarcane EVA midsole" and "EcoStep outsole", this is a PRODUCT SALE article, not ESG news
- "Nike Releases New Air Jordan Retro Collection" - Product launch, no ESG categories
- "Best running shoes for 2025" - Product review/comparison, no ESG categories
- "Lululemon x Erewhon limited edition collection drops" - Product collaboration, no ESG categories
- "ASICS launches the new GEL-NIMBUS 28 running shoe" - Product launch, even if it mentions performance features

**Examples of articles that SHOULD receive ESG categories:**
- "Nike announces 50% carbon reduction target by 2030" - Corporate sustainability initiative (Environmental)
- "Adidas partners with Parley to remove ocean plastic from beaches" - Environmental program focus (Environmental)
- "Puma faces investigation over Vietnam factory working conditions" - Labor/supply chain issue (Social)
- "Under Armour publishes first ESG transparency report" - Governance/reporting focus (Governance)
- "Nike launches AI-powered supply chain tracking for sustainability" - Tech initiative for ESG goals (Digital Transformation)

**The key question to ask:** Is the PRIMARY PURPOSE of this article to report on ESG-related news (initiatives, issues, programs, reports), or is it to promote/review/sell a product that happens to have some sustainable features?

If the article is primarily about a product (sale, release, review, collection, where to buy), do NOT assign ESG categories regardless of what sustainable features are mentioned.

## CRITICAL: Governance vs Financial News

Many articles about stock prices, earnings, or investment analysis are incorrectly labeled as "Governance". ESG Governance refers to corporate ethics and oversight structures, NOT financial performance.

**Governance IS about:**
- Board structure, composition, and independence
- Executive compensation policies
- Anti-corruption measures and ethics violations
- Regulatory compliance and legal issues
- Transparency and ESG disclosure/reporting
- Shareholder rights and stakeholder engagement
- Corporate accountability and oversight mechanisms

**Governance is NOT about:**
- Stock price movements ("Nike stock tumbles 6%")
- Quarterly earnings reports and financial performance
- Investment recommendations ("Buy/Sell this stock")
- CEO/leadership changes (unless involving ethics violations or accountability)
- Market analysis and stock valuations
- Investor purchases (e.g., "Tim Cook buys Nike shares")
- Revenue growth or decline
- Financial restructuring or loans (treasury operations)

**Examples that should NOT receive Governance labels:**
- "Nike Stock Tumbles 6% After Q2 Beat" - Stock price news, not governance
- "3 Must-Know Facts Before You Buy Lululemon Stock" - Investment advice
- "Apple's Tim Cook doubles Nike stake with $3M purchase" - Stock purchase news
- "Lululemon earnings beat expectations" - Financial performance
- "PUMA SE secures €500m bridge loan" - Treasury/financing operations

**Examples that SHOULD receive Governance labels:**
- "Nike board approves new executive compensation policy" - Board decision on exec pay
- "Adidas faces shareholder lawsuit over misleading statements" - Accountability issue
- "Under Armour releases first ESG transparency report" - ESG disclosure
- "Puma CEO resigns amid accounting scandal" - Ethics/accountability
- "Lululemon adds independent directors to improve oversight" - Board structure

## CRITICAL: Athlete Endorsements & Sponsorships

Athlete endorsement deals, signature shoe releases, and sports sponsorships are typically marketing/business activities, NOT ESG content.

**Do NOT assign ESG categories to:**
- Athlete signature shoe releases ("Reebok Angel Reese 1 releases November 14")
- Sports team uniform reveals ("Nike Reveals 2025-26 City Edition Uniforms")
- Athlete endorsement announcements ("New Balance announces Cooper Flagg shoe")
- Sponsorship deal news ("Adidas sponsors World Cup event")
- Fashion collaborations ("Madhappy x Converse Chuck 70 collab")

**EXCEPTION - These MAY qualify for Social category:**
- Olympic/Paralympic partnerships that focus on athlete development programs
- Sponsorships with explicit diversity, equity, and inclusion initiatives
- Community investment programs beyond just product/uniform provision
- Articles about athlete welfare, not just endorsement announcements

**Examples that should NOT receive ESG categories:**
- "New Balance announces Cooper Flagg new shoe" - Athlete endorsement product launch
- "Oregon Ducks Orange Bowl Uniforms are a Powerful Recruitment Tool" - Sports marketing
- "Reebok Angel Reese 1 'Tiago King Reese' Release" - Signature shoe launch
- "Nike Reveals England World Cup kit" - Uniform reveal (unless focused on sustainability tech)

**Examples that MAY receive Social category:**
- "Lululemon partners with Paralympic athletes to improve adaptive clothing" - Inclusion initiative
- "Nike invests $10M in youth sports programs in underserved communities" - Community investment
- "Adidas launches scholarship program for women athletes" - Athlete development

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
6. Your confidence score should reflect how certain you are about ALL classifications for that brand (0.0-1.0).

## CRITICAL: Understanding is_sportswear_brand

The `is_sportswear_brand` field indicates whether the brand name refers to the SPORTSWEAR/APPAREL COMPANY. This is about IDENTITY, not content.

**Set `is_sportswear_brand: false` ONLY when the brand name refers to something ELSE:**
- Puma the animal (not Puma sportswear)
- Patagonia the geographic region (not Patagonia outdoor apparel)
- "Vans" meaning vehicles (not Vans footwear)
- Black Diamond Equipment for climbing gear (not sportswear)

**Set `is_sportswear_brand: true` (or omit the brand) when the article IS about the sportswear brand:**
- Product announcements, reviews, sales (→ true, but no ESG categories apply)
- Store openings, sponsorships, athlete signings (→ true, but no ESG categories apply)
- Financial articles ABOUT the sportswear company's business (→ true, check for governance content)

**Tangential Mentions (still set `is_sportswear_brand: false`):**
- Former executives now at other companies (article is about the OTHER company)
- Brand mentioned only as biographical context or comparison
- Industry reports where brand appears in a list without substantive coverage

**Key Distinction:**
- "Timberland boots on sale" → `is_sportswear_brand: true` (it's about Timberland products), but omit from brand_analyses since no ESG content
- "Timberland forests need protection" → `is_sportswear_brand: false` (referring to forests, not the brand)

**When in doubt:** If the article is genuinely ABOUT the sportswear brand's products, stores, or business activities (even without ESG content), set `is_sportswear_brand: true` and either assign no ESG categories or omit the brand from the response. Reserve `is_sportswear_brand: false` for cases where the brand name refers to something completely different."""

# Build the system prompt with the current brand list
LABELING_SYSTEM_PROMPT = _LABELING_SYSTEM_PROMPT_TEMPLATE.format(
    brands=", ".join(TARGET_SPORTSWEAR_BRANDS)
)

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
