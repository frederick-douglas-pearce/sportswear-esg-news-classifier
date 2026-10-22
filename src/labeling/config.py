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

EMBEDDING_COST_PER_1K_TOKENS = 0.00002


class LabelingModelSpec(BaseModel):
    """What the labeler needs to know about a labeling model: its prices and request surface."""

    model_config = ConfigDict(frozen=True)

    input_cost_per_mtok: float
    output_cost_per_mtok: float
    # Haiku 4.5 takes temperature; Haiku 5.5 returns a 400 for any non-default value
    accepts_sampling: bool
    # output_config.effort errors on Haiku 4.5; Haiku 5.5 takes low through max
    accepts_effort: bool
    # Thinking counts toward max_tokens, so a model that thinks by default needs more room
    default_max_tokens: int
    # Fewest characters per input token assumed for a worst-case cost bound. The v1.11.0 prompt
    # measured 3.91 on Haiku 4.5 and 2.65 on Haiku 5.5, whose tokenizer counts more tokens (#212)
    min_chars_per_token: float


# LLM prices per million tokens and request surface, by model id prefix (a dated id such as
# claude-haiku-4-5-20251001 matches claude-haiku-4-5).
#
# One table, because the report mixes sources: spend comes from
# `labeling_runs.estimated_cost_usd` (written by `complete_labeling_run`) while the FP pre-filter
# savings come from the script's stdout. When those two priced differently the same report could
# claim it saved more than it spent (issue #81). A model missing here is refused by
# `labeling_model_spec` rather than priced at another model's rates (#212; #54 covers the rest).
#
# Haiku 5.5 has a second rate card ($0.50 / $2.50) for prompts over 100K tokens. A labeling
# prompt is capped by MAX_ARTICLE_TOKENS (4000) plus the prompt template, far below that, and the
# aggregate token counts priced here cannot be split by prompt length, so only the lower card is
# recorded.
LABELING_MODELS: dict[str, LabelingModelSpec] = {
    "claude-haiku-4-5": LabelingModelSpec(
        input_cost_per_mtok=1.00,
        output_cost_per_mtok=5.00,
        accepts_sampling=True,
        accepts_effort=False,
        default_max_tokens=2000,
        min_chars_per_token=3.0,
    ),
    # The candidate default of scripts/eval_model_migration.py (#38), kept on that script's request
    # surface (temperature, no effort). Sonnet 4, its baseline, is retired and needs no entry: the
    # script builds a labeler for the candidate only.
    "claude-sonnet-4-6": LabelingModelSpec(
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        accepts_sampling=True,
        accepts_effort=False,
        default_max_tokens=2000,
        min_chars_per_token=3.0,
    ),
    "claude-haiku-5-5": LabelingModelSpec(
        input_cost_per_mtok=0.10,
        output_cost_per_mtok=0.50,
        accepts_sampling=False,
        accepts_effort=True,
        default_max_tokens=8000,
        min_chars_per_token=2.0,
    ),
}


def labeling_model_spec(model: str) -> LabelingModelSpec:
    """The spec for a model id, matched on the longest prefix; refuses an unknown model."""
    matches = [prefix for prefix in LABELING_MODELS if model.startswith(prefix)]
    if not matches:
        raise ValueError(
            f"No pricing or request spec for labeling model {model!r}; "
            f"add it to LABELING_MODELS in src/labeling/config.py (known: {sorted(LABELING_MODELS)})"
        )
    return LABELING_MODELS[max(matches, key=len)]


def llm_cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    """Cost in USD of the given token counts at the model's rates.

    No tokens cost nothing under any model, so a run that never called the LLM (every article
    skipped before it) records zero without its model being looked up.
    """
    if not input_tokens and not output_tokens:
        return 0.0
    spec = labeling_model_spec(model)
    return (
        input_tokens / 1_000_000 * spec.input_cost_per_mtok
        + output_tokens / 1_000_000 * spec.output_cost_per_mtok
    )

# Recorded as threshold_used (NOT NULL) for every article of an FP pre-filter
# batch that raises, whether the API call failed or a later step did (saving a
# prediction included), unless FP_SKIP_LLM_THRESHOLD is set (D035 §4).
FP_FAILED_BATCH_THRESHOLD = 0.5


def _fp_skip_threshold_override() -> float | None:
    """FP_SKIP_LLM_THRESHOLD as a float, or None when it is unset or blank."""
    value = os.getenv("FP_SKIP_LLM_THRESHOLD", "").strip()
    return float(value) if value else None


def check_fp_skip_threshold(value: float | None) -> None:
    """Refuse an FP_SKIP_LLM_THRESHOLD outside 0 to 1, nan and inf included.

    Called when a LabelingPipeline is built, not at import: importing
    src.labeling builds the settings, and the agent imports it, so a range check
    there would stop every agent workflow instead of labeling alone (the
    point-of-use precedent in run_audit.py, #75).
    """
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"FP_SKIP_LLM_THRESHOLD must be between 0 and 1, got {value}")


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
    # output_config.effort, sent only to a model that accepts it (Haiku 5.5's default is medium)
    labeling_effort: str = Field(
        default_factory=lambda: os.getenv("LABELING_EFFORT", "medium")
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
    # Optional override of the served model's threshold for LLM skips. Unset or blank
    # means the pipeline applies the threshold the FP API returns with each
    # prediction (D035 §4).
    fp_skip_llm_threshold: float | None = Field(default_factory=_fp_skip_threshold_override)
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

