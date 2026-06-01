"""Tests for shared text normalization (mojibake repair + control stripping)."""

from src.data_collection.text_normalize import (
    find_illegal_chars,
    normalize_text,
    repair_mojibake,
)


class TestRepairMojibake:
    """ftfy-based repair of Windows-1252 mojibake stored as raw C1 bytes."""

    def test_repairs_smart_apostrophe(self):
        assert repair_mojibake("McDonald\x92s") == "McDonald’s"

    def test_repairs_smart_double_quotes(self):
        assert repair_mojibake("\x93Laurent\x94") == "“Laurent”"

    def test_repairs_em_dash(self):
        assert repair_mojibake("tour \x97 here") == "tour — here"

    def test_leaves_clean_text_unchanged(self):
        clean = "Nike announced new sustainability goals."
        assert repair_mojibake(clean) == clean

    def test_preserves_cjk_characters(self):
        # fix_character_width=False keeps CJK / fullwidth characters intact.
        assert repair_mojibake("安踏 Anta") == "安踏 Anta"


class TestFindIllegalChars:
    """Detection of YAML-illegal control characters."""

    def test_detects_c1_control(self):
        assert find_illegal_chars("McDonald\x92s") == {"\x92"}

    def test_detects_c0_control_and_del(self):
        assert find_illegal_chars("a\x00b\x7fc") == {"\x00", "\x7f"}

    def test_allows_tab_newline_carriage_return(self):
        assert find_illegal_chars("a\tb\nc\r") == set()

    def test_clean_text_has_none(self):
        assert find_illegal_chars("Plain ASCII and ’ curly quote") == set()


class TestNormalizeText:
    """End-to-end normalization used at ingest and export."""

    def test_none_passthrough(self):
        assert normalize_text(None) is None

    def test_repairs_mojibake(self):
        assert normalize_text("McDonald\x92s") == "McDonald’s"

    def test_strips_emoji(self):
        assert normalize_text("Nike \U0001f680 wins").strip() == "Nike  wins".strip()

    def test_removes_residual_control_chars_but_keeps_whitespace(self):
        assert normalize_text("a\x00b\tc\nd") == "ab\tc\nd"

    def test_output_has_no_illegal_chars(self):
        result = normalize_text("McDonald\x92s \x93quote\x94 tour\x97end")
        assert find_illegal_chars(result) == set()

    def test_idempotent(self):
        once = normalize_text("McDonald\x92s \x93q\x94 \U0001f680 tour\x97")
        assert normalize_text(once) == once
