"""The scraper's brand test, which now fails a scrape (#227, D044)."""

import pytest

from src.data_collection.brand_match import BRAND_PATTERNS, brands_in_text
from src.data_collection.config import BRANDS


@pytest.mark.parametrize("brand", BRANDS)
def test_never_narrower_than_collection_extraction(brand):
    """Collection extracts a brand by its name, case-insensitively, at word boundaries
    (api_client._extract_brands). Text containing that string must pass, or the gate would skip
    an article that names its own brand."""
    assert brands_in_text(f"Report: {brand} said", [brand]) == [brand]
    assert brands_in_text(f"report: {brand.lower()} said", [brand]) == [brand]
    assert brands_in_text(f"REPORT: {brand.upper()} SAID", [brand]) == [brand]


def test_every_tracked_brand_has_patterns():
    assert set(BRANDS) <= set(BRAND_PATTERNS)


@pytest.mark.parametrize(
    "text, brand",
    [
        ("A new Arc’teryx jacket", "Arc'teryx"),
        ("A new Arcteryx jacket", "Arc'teryx"),
        ("Li–Ning reported results", "Li-Ning"),
        ("Li Ning reported results", "Li-Ning"),
        ("K‑Swiss sneakers", "K-Swiss"),
        ("On Holding AG raised guidance", "On Running"),
        ("The On Cloud 6 review", "On Running"),
        ("Brooks released a shoe", "Brooks Running"),
        ("The North Face store", "The North Face"),
        ("Columbia reported earnings", "Columbia Sportswear"),
    ],
)
def test_known_forms(text, brand):
    assert brands_in_text(text, [brand]) == [brand]


@pytest.mark.parametrize(
    "text, brand",
    [
        ("The punishing workout was tough.", "Puma"),
        ("She will keep going until 2028.", "On Running"),
        ("A Santa parade downtown.", "Anta"),
        ("Basic income rose.", "ASICS"),
    ],
)
def test_no_match(text, brand):
    assert brands_in_text(text, [brand]) == []


def test_returns_found_brands_in_given_order():
    assert brands_in_text("Adidas and Nike", ["Nike", "Puma", "Adidas"]) == ["Nike", "Adidas"]


def test_untracked_brand_matches_its_own_name():
    assert brands_in_text("Acme Sports grew", ["Acme Sports"]) == ["Acme Sports"]
    assert brands_in_text("Acme grew", ["Acme Sports"]) == []


@pytest.mark.parametrize("text", [None, ""])
def test_empty_text(text):
    assert brands_in_text(text, ["Nike"]) == []
