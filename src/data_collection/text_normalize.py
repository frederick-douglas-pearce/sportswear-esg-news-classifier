"""Text normalization: repair mojibake and strip parser-breaking characters.

Used both at scrape/ingest time (so the database stores clean text) and at
feed-export time (a defense-in-depth guard). The website feed is committed to a
Jekyll site that parses ``_data/*.json`` with the YAML parser (Ruby Psych),
which rejects C1 control characters (U+0080-U+009F). Those most often appear as
Windows-1252 *mojibake* -- smart quotes and dashes (' " " --) mis-decoded as raw
C1 bytes by the scraper. ``ftfy`` repairs that mojibake (e.g. U+0092 -> U+2019);
a final pass strips any control character that is genuinely illegal in YAML.

The functions here are idempotent: applying them to already-clean text returns
it unchanged, so they are safe to run at every layer (scrape, DB repair, export).
"""

import re

import ftfy

# Control characters that are illegal in a YAML 1.1 scalar (and therefore break
# Jekyll's Psych parser): the C0 range except tab/newline/carriage-return, the
# DEL character, and the entire C1 range. ftfy normally repairs/removes these,
# but we strip explicitly as a deterministic backstop for pathological input.
_ILLEGAL_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Emoji and pictographic symbols. Preserved from the original export-side
# ``sanitize_text``: these render poorly in the feed and have caused YAML issues.
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags (iOS)
    "\U00002702-\U000027b0"  # dingbats
    "\U000024c2-\U0001f251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "♀-♂"
    "☀-⭕"
    "‍"
    "⏏"
    "⏩"
    "⌚"
    "️"  # variation selectors
    "〰"
    "]+",
    flags=re.UNICODE,
)


def repair_mojibake(text: str) -> str:
    """Repair encoding damage (mojibake) in ``text`` using ftfy.

    ``uncurl_quotes=False`` preserves the source's curly vs. straight quote
    distinction; ``fix_character_width=False`` leaves CJK fullwidth characters
    alone (relevant for brand names such as 361 Degrees / Li-Ning).
    """
    return ftfy.fix_text(text, uncurl_quotes=False, fix_character_width=False)


def find_illegal_chars(text: str) -> set[str]:
    """Return the set of YAML-illegal control characters present in ``text``.

    Used by the export guard and validator to detect (and report) characters
    that would break Jekyll's parser before/after normalization.
    """
    return set(_ILLEGAL_CONTROL_RE.findall(text))


def normalize_text(text: str | None) -> str | None:
    """Normalize ``text`` for safe storage and feed export.

    Repairs mojibake, strips emoji/pictographs, and removes any remaining
    YAML-illegal control characters. Returns ``None`` unchanged so callers can
    pass through nullable database columns. Idempotent.
    """
    if text is None:
        return None
    text = repair_mojibake(text)
    text = _EMOJI_RE.sub("", text)
    text = _ILLEGAL_CONTROL_RE.sub("", text)
    return text
