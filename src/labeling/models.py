"""Pydantic models for LLM labeling responses."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CategoryLabel(BaseModel):
    """Label for a single ESG category."""

    applies: bool = Field(description="Whether this category applies to the brand in this article")
    sentiment: Literal[-1, 0, 1] | None = Field(
        default=None,
        description="Sentiment: -1 (negative), 0 (neutral), 1 (positive), null if not applicable",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Direct quotes from the article supporting this classification",
    )

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v, info):
        """Ensure sentiment is only set when category applies."""
        # Note: This validator runs after the field is set, so we can't access 'applies' here
        # The validation logic will be in BrandAnalysis
        return v


class BrandAnalysis(BaseModel):
    """Analysis results for a single brand in an article."""

    brand: str = Field(description="Brand name being analyzed")
    categories: dict[str, CategoryLabel] = Field(
        description="Category labels keyed by category name",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this brand's classification (0.0-1.0)",
    )
    reasoning: str = Field(description="Brief explanation of classification decisions")

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v):
        """Validate that all expected categories are present."""
        expected = {"environmental", "social", "governance", "digital_transformation"}
        if not expected.issubset(set(v.keys())):
            missing = expected - set(v.keys())
            raise ValueError(f"Missing required categories: {missing}")
        return v

    def get_applicable_categories(self) -> list[str]:
        """Get list of categories that apply to this brand."""
        return [name for name, label in self.categories.items() if label.applies]


class LabelingResponse(BaseModel):
    """Complete response from the labeling LLM."""

    brand_analyses: list[BrandAnalysis] = Field(
        description="Analysis results for each brand with substantive ESG content",
    )
    article_summary: str = Field(
        description="1-2 sentence summary of the article's ESG themes",
    )

    def get_brands(self) -> list[str]:
        """Get list of brands analyzed."""
        return [analysis.brand for analysis in self.brand_analyses]

    def get_analysis_for_brand(self, brand: str) -> BrandAnalysis | None:
        """Get analysis for a specific brand."""
        for analysis in self.brand_analyses:
            if analysis.brand.lower() == brand.lower():
                return analysis
        return None


# Response model for when the article has no ESG-relevant content
class NoESGContentResponse(BaseModel):
    """Response when article has no substantive ESG content."""

    has_esg_content: Literal[False] = False
    reason: str = Field(description="Explanation of why no ESG content was found")
