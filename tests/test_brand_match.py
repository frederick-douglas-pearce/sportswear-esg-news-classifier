"""The scraper's brand test, which now fails a scrape (#227, D044), and leaves out a cookie notice
(#273, D047)."""

import pytest

from src.data_collection.api_client import NewsDataClient
from src.data_collection.brand_match import (
    BRAND_PATTERNS,
    CONSENT_NOTICE_MARKERS,
    brands_in_article_text,
    brands_in_text,
    is_consent_notice_paragraph,
    without_consent_notice,
)
from src.data_collection.config import BRANDS
from src.data_collection.gdelt_client import GDELTClient

# The collection-time extractors read no client state; called unbound so no API key is needed
EXTRACTORS = [
    pytest.param(lambda text: NewsDataClient._extract_brands(None, text), id="newsdata"),
    pytest.param(lambda text: GDELTClient._extract_brands(None, text), id="gdelt"),
]


# The scraper's test and the pattern matcher must both be at least as wide as collection on text
# with no cookie notice
BRAND_TESTS = [
    pytest.param(brands_in_text, id="brands_in_text"),
    pytest.param(brands_in_article_text, id="brands_in_article_text"),
]


@pytest.mark.parametrize("brand_test", BRAND_TESTS)
@pytest.mark.parametrize("extract", EXTRACTORS)
@pytest.mark.parametrize("brand", BRANDS)
def test_matches_whatever_collection_extracted(extract, brand, brand_test):
    """Differential: every brand a collection extractor finds in a text, the brand test under test
    finds in the same text. The gate skips an article whose content names none of its extracted brands,
    so a narrower test would skip an article that names its own brand."""
    for text in (f"Report: {brand} said", f"report: {brand.lower()} said",
                 f"REPORT: {brand.upper()} SAID", f"{brand}'s results"):
        extracted = extract(text)
        assert brand in extracted
        assert set(extracted) <= set(brand_test(text, BRANDS))


@pytest.mark.parametrize("brand_test", BRAND_TESTS)
@pytest.mark.parametrize("brand", BRANDS)
def test_never_narrower_than_collection_extraction(brand, brand_test):
    """Collection extracts a brand by its name, case-insensitively, at word boundaries
    (api_client._extract_brands). Text containing that string must pass, or the gate would skip
    an article that names its own brand."""
    assert brand_test(f"Report: {brand} said", [brand]) == [brand]
    assert brand_test(f"report: {brand.lower()} said", [brand]) == [brand]
    assert brand_test(f"REPORT: {brand.upper()} SAID", [brand]) == [brand]


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


# reebok.com's cookie notice, transcribed for #273 from a stored reebok.com row (eadb57c8)
REEBOK_NOTICE = (
    "Your Privacy Choices\n\nOn this website, we use cookies and similar tracking technologies, and "
    "allow our advertising partners to use such technologies, so we can, among other things, show "
    "you ads promoting Reebok on other sites and services. These activities may be considered "
    "\u201csales,\u201d \u201csharing,\u201d or \u201ctargeted advertising\u201d under certain "
    "privacy laws.\n\nTo opt out, slide the toggle below to \u201coff\u201d and then click "
    "\u201cConfirm My Choices.\u201d Note that you will need to renew this opt-out choice if you "
    "visit our website from another device or browser or if you clear your cookies. You can also "
    "opt out by visiting our website with a legally-recognized opt-out preference signal, such as "
    "the Global Privacy Control."
)
ARTICLE = "Reebok opened a new store in Boston on Monday."


def test_brand_named_only_in_a_cookie_notice_does_not_count():
    assert brands_in_text(REEBOK_NOTICE, ["Reebok"]) == ["Reebok"]
    assert brands_in_article_text(REEBOK_NOTICE, ["Reebok"]) == []


@pytest.mark.parametrize(
    "text",
    [
        ARTICLE + "\n\n" + REEBOK_NOTICE,
        REEBOK_NOTICE + "\n\n" + ARTICLE,
        REEBOK_NOTICE + "\n \n" + ARTICLE,  # a blank line holding whitespace
    ],
)
def test_article_beside_a_notice_keeps_its_brand(text):
    assert brands_in_article_text(text, ["Reebok"]) == ["Reebok"]


def test_text_with_no_blank_line_is_one_paragraph():
    """An article run together with a notice in one block is one paragraph, dropped whole."""
    block = ARTICLE + " We use cookies on this website."
    assert not is_consent_notice_paragraph(ARTICLE)
    assert is_consent_notice_paragraph(block)
    assert brands_in_article_text(block, ["Reebok"]) == []
    assert brands_in_article_text(ARTICLE + "\n" + "We use cookies on this website.", ["Reebok"]) == []


@pytest.mark.parametrize("marker", CONSENT_NOTICE_MARKERS)
def test_every_marker_makes_a_notice(marker):
    paragraph = f"{marker.capitalize()} to show you ads for Reebok."
    assert is_consent_notice_paragraph(paragraph)
    assert is_consent_notice_paragraph(paragraph.upper())
    assert brands_in_article_text(paragraph, ["Reebok"]) == []


@pytest.mark.parametrize(
    "paragraph",
    [
        "On this website we use Cookies.",
        "THIS SITE USES COOKIES.",
    ],
)
def test_notice_paragraphs(paragraph):
    assert is_consent_notice_paragraph(paragraph)


@pytest.mark.parametrize(
    "paragraph, brand",
    [
        # news about cookies (ESG Digital), which a notice test must not take for a notice
        ("Nike was fined after regulators found tracking cookies were set without consent.", "Nike"),
        ("Lululemon faces a lawsuit claiming its website shared shoppers' data via cookies; the "
         "complaint says users who tried to opt out were still tracked.", "Lululemon"),
        ("The suit says Adidas changed its cookie policy and ignored users' cookie settings on "
         "our website, the plaintiffs' lawyers said.", "Adidas"),
        # a colorway named for cookies, in a post that speaks as the site does
        ("The Nike Dunk Low 'Cookies and Cream' drops Friday for $115; we use affiliate links, so "
         "this site may earn a commission.", "Nike"),
        # phrases a notice uses, without the site stating its cookie use
        ("Our site relies on cookie-based ad targeting, Puma said.", "Puma"),
        ("Click Accept All to allow cookies, Puma's banner says.", "Puma"),
        ("Manage your Cookie Settings on the Puma app.", "Puma"),
        # the site's voice, no cookie
        ("On this website you can opt out of marketing emails from Nike.", "Nike"),
    ],
)
def test_paragraphs_that_are_not_a_notice(paragraph, brand):
    assert not is_consent_notice_paragraph(paragraph)
    assert without_consent_notice(paragraph) == paragraph
    assert brands_in_article_text(paragraph + "\n\nSizes run true.", [brand]) == [brand]


def test_a_heading_alone_is_not_a_notice():
    assert not is_consent_notice_paragraph("Your Privacy Choices")


def test_a_two_brand_notice_counts_for_neither():
    notice = "On this website we use cookies to show you ads for Vans and Converse."
    assert brands_in_text(notice, ["Vans", "Converse"]) == ["Vans", "Converse"]
    assert brands_in_article_text(notice, ["Vans", "Converse"]) == []
    assert brands_in_article_text(notice + "\n\nVans sales rose.", ["Vans", "Converse"]) == ["Vans"]


@pytest.mark.parametrize("text", [None, ""])
def test_article_text_of_empty_text(text):
    assert without_consent_notice(text) == ""
    assert brands_in_article_text(text, ["Reebok"]) == []
