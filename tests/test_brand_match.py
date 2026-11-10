"""The scraper's brand test, which now fails a scrape (#227, D044)."""

import pytest

from src.data_collection.api_client import NewsDataClient
from src.data_collection.brand_match import BRAND_PATTERNS, brands_in_text
from src.data_collection.config import BRANDS
from src.data_collection.gdelt_client import GDELTClient

# The collection-time extractors read no client state; called unbound so no API key is needed
EXTRACTORS = [
    pytest.param(lambda text: NewsDataClient._extract_brands(None, text), id="newsdata"),
    pytest.param(lambda text: GDELTClient._extract_brands(None, text), id="gdelt"),
]


@pytest.mark.parametrize("extract", EXTRACTORS)
@pytest.mark.parametrize("brand", BRANDS)
def test_matches_whatever_collection_extracted(extract, brand):
    """Differential: every brand a collection extractor finds in a text, brands_in_text finds in
    the same text. The gate skips an article whose content names none of its extracted brands,
    so a narrower test would skip an article that names its own brand."""
    for text in (f"Report: {brand} said", f"report: {brand.lower()} said",
                 f"REPORT: {brand.upper()} SAID", f"{brand}'s results"):
        extracted = extract(text)
        assert brand in extracted
        assert set(extracted) <= set(brands_in_text(text, BRANDS))


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
        ("Anta and 361° results", "361 Degrees"),
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
