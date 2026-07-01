"""Configuration settings for the labeling pipeline."""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from src.data_collection.config import BRANDS

load_dotenv()

logger = logging.getLogger(__name__)

# Root directory for prompts (relative to project root)
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "labeling"


def _get_prompt_version() -> str:
    """Get the prompt version to use.

    Returns version from LABELING_PROMPT_VERSION env var, or falls back to
    production version from registry.json.
    """
    # Check for explicit version override
    version = os.getenv("LABELING_PROMPT_VERSION")
    if version:
        return version

    # Load production version from registry
    registry_path = PROMPTS_DIR / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Prompt registry not found: {registry_path}")

    with open(registry_path) as f:
        registry = json.load(f)

    return registry.get("production", "v1.0.0")


def _load_prompt_template(version: str, prompt_type: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        version: Version string (e.g., "v1.3.0")
        prompt_type: Either "system_prompt" or "user_prompt"

    Returns:
        Prompt template string
    """
    version_dir = PROMPTS_DIR / version
    prompt_file = version_dir / f"{prompt_type}.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file) as f:
        return f.read()


def load_labeling_prompts(brands: list[str]) -> tuple[str, str]:
    """Load system and user prompts for labeling.

    Args:
        brands: List of brand names to include in prompts

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    version = _get_prompt_version()
    logger.info(f"Loading labeling prompts version: {version}")

    # Load templates
    system_template = _load_prompt_template(version, "system_prompt")
    user_template = _load_prompt_template(version, "user_prompt")

    # Format system prompt with brands list
    system_prompt = system_template.format(brands=", ".join(brands))

    return system_prompt, user_template


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
        default_factory=lambda: os.getenv("LABELING_MODEL", "claude-haiku-4-5-20251001")
    )

    # Prompt version (None = use production version from registry)
    prompt_version: str | None = Field(
        default_factory=lambda: os.getenv("LABELING_PROMPT_VERSION")
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # Chunking parameters (reduced for more granular evidence matching)
    target_chunk_tokens: int = Field(
        default_factory=lambda: int(os.getenv("TARGET_CHUNK_TOKENS", "200"))
    )
    max_chunk_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CHUNK_TOKENS", "350"))
    )
    min_chunk_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MIN_CHUNK_TOKENS", "75"))
    )
    chunk_overlap_tokens: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
    )

    # Evidence matching parameters
    evidence_min_confidence: float = Field(
        default_factory=lambda: float(os.getenv("EVIDENCE_MIN_CONFIDENCE", "0.50"))
    )
    evidence_use_embedding_rerank: bool = Field(
        default_factory=lambda: os.getenv("EVIDENCE_USE_EMBEDDING_RERANK", "true").lower() == "true"
    )

    # Cross-encoder reranking settings
    rerank_enabled: bool = Field(
        default_factory=lambda: os.getenv("RERANK_ENABLED", "true").lower() == "true"
    )
    rerank_model: str = Field(
        default_factory=lambda: os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    rerank_top_k: int = Field(
        default_factory=lambda: int(os.getenv("RERANK_TOP_K", "10"))
    )
    rerank_weight: float = Field(
        default_factory=lambda: float(os.getenv("RERANK_WEIGHT", "0.6"))
    )  # Weight for combining: (1-weight)*initial + weight*rerank

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

    # Novelty scoring settings (for drift detection)
    # Uses sentence-transformers (384-dim) for lower dimensionality clustering
    # Computed for ALL articles before FP classification
    novelty_enabled: bool = Field(
        default_factory=lambda: os.getenv("NOVELTY_ENABLED", "true").lower() == "true"
    )
    novelty_centroids_path: str = Field(
        default_factory=lambda: os.getenv(
            "NOVELTY_CENTROIDS_PATH",
            str(Path(__file__).parent.parent.parent / "models" / "novelty_centroids.pkl")
        )
    )
    novelty_embedding_model: str = Field(
        default_factory=lambda: os.getenv("NOVELTY_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
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


def _get_labeling_prompts() -> tuple[str, str]:
    """Load labeling prompts at module initialization.

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    try:
        return load_labeling_prompts(TARGET_SPORTSWEAR_BRANDS)
    except FileNotFoundError as e:
        logger.warning(f"Failed to load prompts from files: {e}. Using fallback.")
        # Return empty strings - will fail at runtime if prompts are needed
        return "", ""


# Load prompts dynamically from prompts/ folder
# Uses version from LABELING_PROMPT_VERSION env var or production version from registry.json
LABELING_SYSTEM_PROMPT, LABELING_USER_PROMPT_TEMPLATE = _get_labeling_prompts()


labeling_settings = LabelingSettings()

