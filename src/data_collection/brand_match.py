"""Whether a text names a tracked brand, in any form the scraper knows (#227, D044).

The scraper fails a scrape whose content names none of the article's brands, and
``scripts/triage_no_brand_content.py`` selects stored rows by the same test, so both use this
module. It imports no scraping library, so the triage and its tests load none.

A brand's patterns must match at least what collection-time extraction matched: the brand name
itself, case-insensitively, at word boundaries (``api_client._extract_brands``). Otherwise an
article's own brand string could fail the check. ``tests/test_brand_match.py`` checks this for every
brand in ``BRANDS``.
"""

import re

# Hyphen forms a source may use where the brand name has an ASCII hyphen.
_HYPHEN = "[-‐‑–]"

# Canonical brand name -> regex patterns, matched case-insensitively. A brand not listed matches
# its own name at word boundaries.
BRAND_PATTERNS: dict[str, list[str]] = {
    "Nike": [r"\bNike\b"],
    "Adidas": [r"\bAdidas\b"],
    "Puma": [r"\bPuma\b"],
    "Under Armour": [r"\bUnder\s+Armour\b", r"\bUnderArmour\b"],
    "Lululemon": [r"\bLululemon\b"],
    "Patagonia": [r"\bPatagonia\b"],
    "Columbia Sportswear": [r"\bColumbia\s+Sportswear\b", r"\bColumbia\b"],
    "New Balance": [r"\bNew\s+Balance\b"],
    "ASICS": [r"\bASICS\b"],
    "Reebok": [r"\bReebok\b"],
    "Skechers": [r"\bSkechers\b"],
    "Fila": [r"\bFila\b"],
    "The North Face": [r"\bThe\s+North\s+Face\b", r"\bNorth\s+Face\b"],
    "Vans": [r"\bVans\b"],
    "Converse": [r"\bConverse\b"],
    "Salomon": [r"\bSalomon\b"],
    "Mammut": [r"\bMammut\b"],
    "Umbro": [r"\bUmbro\b"],
    "Anta": [r"\bAnta\s+Sports\b", r"\bAnta\b"],
    "Li-Ning": [rf"\bLi{_HYPHEN}Ning\b", r"\bLi\s*Ning\b"],
    "Brooks Running": [r"\bBrooks\s+Running\b", r"\bBrooks\b"],
    "Decathlon": [r"\bDecathlon\b"],
    "Deckers": [r"\bDeckers\b"],
    "Yonex": [r"\bYonex\b"],
    "Mizuno": [r"\bMizuno\b"],
    "K-Swiss": [rf"\bK{_HYPHEN}Swiss\b"],
    "Altra Running": [r"\bAltra\s+Running\b", r"\bAltra\b"],
    "Hoka": [r"\bHoka\b"],
    "Saucony": [r"\bSaucony\b"],
    "Merrell": [r"\bMerrell\b"],
    "Timberland": [r"\bTimberland\b"],
    "Spyder": [r"\bSpyder\b"],
    "On Running": [r"\bOn\s+Running\b", r"\bOn\s+Cloud\b", r"\bOn\s+Holding\b"],
    "Allbirds": [r"\bAllbirds\b"],
    "Gymshark": [r"\bGymshark\b"],
    "Everlast": [r"\bEverlast\b"],
    # normalize_text keeps curly quotes, so a body may write the apostrophe either way
    "Arc'teryx": [r"\bArc['’]teryx\b", r"\bArcteryx\b"],
    "Jack Wolfskin": [r"\bJack\s+Wolfskin\b"],
    "Athleta": [r"\bAthleta\b"],
    "Vuori": [r"\bVuori\b"],
    "Cotopaxi": [r"\bCotopaxi\b"],
    "Prana": [r"\bPrana\b"],
    "Eddie Bauer": [r"\bEddie\s+Bauer\b"],
    "361 Degrees": [r"\b361\s*Degrees\b", r"\b361°"],
    "Xtep": [r"\bXtep\b"],
    "Peak Sport": [r"\bPeak\s+Sport\b"],
    "Mountain Hardwear": [r"\bMountain\s+Hardwear\b"],
    "Black Diamond": [r"\bBlack\s+Diamond\b"],
    "Outdoor Voices": [r"\bOutdoor\s+Voices\b"],
    "Diadora": [r"\bDiadora\b"],
}


def brands_in_text(text: str | None, brands: list[str]) -> list[str]:
    """The brands, of those given, that ``text`` names, in the order given."""
    if not text:
        return []
    found = []
    for brand in brands:
        patterns = BRAND_PATTERNS.get(brand, [rf"\b{re.escape(brand)}\b"])
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            found.append(brand)
    return found
