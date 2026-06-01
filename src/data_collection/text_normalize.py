"""Text normalization: repair mojibake and strip parser-breaking characters.

Used both at scrape/ingest time (so the database stores clean text) and at
feed-export time (a defense-in-depth guard). The website feed is committed to a
Jekyll site that parses ``_data/*.json`` with the YAML parser (Ruby Psych),
which rejects C1 control characters (U+0080-U+009F). Those most often appear as
Windows-1252 *mojibake* -- smart quotes and dashes (' " " --) mis-decoded as raw
C1 bytes by the scraper. ``ftfy`` repairs that mojibake (e.g. U+0092 -> U+2019);
a final pass strips any control character that is genuinely illegal in YAML.

Normalization here is **value-preserving**: it repairs encoding damage and drops
illegal control characters, but does not discard otherwise-valid content. Lossy
display policies (e.g. stripping emoji) belong at the feed boundary, not on the
stored data. The functions are idempotent, so they are safe to run at every
layer (scrape, DB write, DB repair, export).
"""

import re
from collections.abc import Iterable

import ftfy

# Control characters that are illegal in a YAML 1.1 scalar (and therefore break
# Jekyll's Psych parser): the C0 range except tab/newline/carriage-return, the
# DEL character, and the entire C1 range. ftfy normally repairs/removes these,
# but we strip explicitly as a deterministic backstop for pathological input.
_ILLEGAL_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def repair_mojibake(text: str) -> str:
    """Repair encoding damage (mojibake) in ``text`` using ftfy.

    ``uncurl_quotes=False`` preserves the source's curly vs. straight quote
    distinction; ``fix_character_width=False`` leaves CJK fullwidth characters
    alone (relevant for brand names such as 361 Degrees / Li-Ning).
    """
    return ftfy.fix_text(text, uncurl_quotes=False, fix_character_width=False)


def has_illegal_chars(text: str) -> bool:
    """Fast presence check for YAML-illegal control characters (early-exit).

    Cheaper than :func:`find_illegal_chars` for gate checks that only need a
    yes/no answer, since it stops at the first match instead of collecting all.
    """
    return _ILLEGAL_CONTROL_RE.search(text) is not None


def find_illegal_chars(text: str) -> set[str]:
    """Return the set of YAML-illegal control characters present in ``text``.

    Used by the export guard and validator to detect (and report) characters
    that would break Jekyll's parser before/after normalization.
    """
    return set(_ILLEGAL_CONTROL_RE.findall(text))


def format_codepoints(chars: Iterable[str]) -> str:
    """Render characters as a sorted, comma-separated ``U+XXXX`` list for logs."""
    return ", ".join(f"U+{ord(c):04X}" for c in sorted(chars))


def normalize_text(text: str | None) -> str | None:
    """Normalize ``text`` for safe storage and parsing (value-preserving).

    Repairs mojibake and removes YAML-illegal control characters. Does NOT strip
    emoji or other valid content -- that is a feed-display policy applied at
    export time. Returns ``None`` unchanged so callers can pass through nullable
    database columns. Idempotent.
    """
    if text is None:
        return None
    text = repair_mojibake(text)
    text = _ILLEGAL_CONTROL_RE.sub("", text)
    return text
