"""Configuration settings for data collection."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    newsdata_api_key: str = Field(default_factory=lambda: os.getenv("NEWSDATA_API_KEY", ""))
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/esg_news"
        )
    )
    max_api_calls_per_day: int = Field(
        default_factory=lambda: int(os.getenv("MAX_API_CALLS_PER_DAY", "200"))
    )
    scrape_delay_seconds: float = Field(
        default_factory=lambda: float(os.getenv("SCRAPE_DELAY_SECONDS", "2"))
    )
    gdelt_timespan: str = Field(
        default_factory=lambda: os.getenv("GDELT_TIMESPAN", "3m")  # 3 months default
    )
    gdelt_max_records: int = Field(
        default_factory=lambda: int(os.getenv("GDELT_MAX_RECORDS", "250"))
    )
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )

    model_config = ConfigDict(frozen=True)


BRANDS: list[str] = [
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
    "Skechers",
    "Fila",
    "The North Face",
    "Vans",
    "Converse",
    "Salomon",
    "Mammut",
    "Umbro",
    "Anta",
    "Li-Ning",
    "Brooks Running",
    "Decathlon",
    "Deckers",
    "Yonex",
    "Mizuno",
    "K-Swiss",
    "Altra Running",
    "Hoka",
    "Saucony",
    "Merrell",
    "Timberland",
    "Spyder",
    "On Running",
    "Allbirds",
    "Gymshark",
    "Everlast",
    "Arc'teryx",
    "Jack Wolfskin",
    "Athleta",
    "Vuori",
    "Cotopaxi",
    "Prana",
    "Eddie Bauer",
    "361 Degrees",
    "Xtep",
    "Peak Sport",
    "Mountain Hardwear",
    "Black Diamond",
    "Outdoor Voices",
    "Diadora"
]

KEYWORDS: list[str] = [
    # Environmental
    "sustainability",
    "climate",
    "emissions",
    "recycling",
    "environment",
    "carbon",
    "green",
    # Social
    "labor",
    "workers",
    "factory",
    "supply chain",
    "diversity",
    # Governance
    "ESG",
    "ethics",
    "transparency",
    # Digital Transformation
    "digital",
    "technology",
    "innovation"
]

# Max query length for free tier is 100 characters
MAX_QUERY_LENGTH: int = 100

LANGUAGE: str = "en"

settings = Settings()
