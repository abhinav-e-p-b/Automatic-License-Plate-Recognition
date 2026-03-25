"""
tests/test_ocr.py — Unit tests for OCR post-processing logic.

Run with:
  python -m pytest tests/ -v
  python -m pytest tests/test_ocr.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from utils.ocr import (
    fix_characters,
    normalise_raw,
    validate_plate,
    merge_multiline,
    VALID_STATES,
)


# ---------------------------------------------------------------------------
# normalise_raw
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_lowercase_to_upper(self):
        assert normalise_raw("kl07bb1234") == "KL07BB1234"

    def test_strips_spaces(self):
        assert normalise_raw("KL 07 BB 1234") == "KL07BB1234"

    def test_strips_hyphens(self):
        assert normalise_raw("KL-07-BB-1234") == "KL07BB1234"

    def test_strips_dots(self):
        assert normalise_raw("KL.07.BB.1234") == "KL07BB1234"

    def test_already_clean(self):
        assert normalise_raw("MH12AB3456") == "MH12AB3456"


# ---------------------------------------------------------------------------
# fix_characters
# ---------------------------------------------------------------------------

class TestFixCharacters:
    # Zero / O
    def test_zero_in_letter_position_becomes_O(self):
        # Position 0 and 1 are letter positions
        assert fix_characters("0L07BB1234")[0] == "O"
        assert fix_characters("K007BB1234")[1] == "O"

    def test_O_in_digit_position_becomes_zero(self):
        # Positions 2,3 are digit positions
        assert fix_characters("KLO7BB1234")[2] == "0"
        assert fix_characters("KL0OBB1234")[3] == "0"

    # One / I
    def test_one_in_letter_position_becomes_I(self):
        assert fix_characters("1L07BB1234")[0] == "I"

    def test_I_in_digit_position_becomes_one(self):
        assert fix_characters("KLI7BB1234")[2] == "1"

    # Eight / B
    def test_eight_in_letter_position_becomes_B(self):
        assert fix_characters("KL078B1234")[4] == "B"

    def test_B_in_digit_position_becomes_eight(self):
        assert fix_characters("KL0B7B1234")[3] == "8"   # pos 3 = digit

    # Five / S
    def test_five_in_letter_position_becomes_S(self):
        assert fix_characters("KL075B1234")[4] == "S"

    def test_S_in_digit_position_becomes_five(self):
        assert fix_characters("KLS7BB1234")[2] == "5"

    # No change when correct
    def test_no_change_on_correct_plate(self):
        plate = "KL07BB1234"
        assert fix_characters(plate) == plate

    def test_mixed_errors(self):
        # KL 07 B0 1234 — position 5 is 0 (letter slot → should be O)
        raw = "KL07B01234"
        fixed = fix_characters(raw)
        assert fixed[5] == "O"


# ---------------------------------------------------------------------------
# validate_plate
# ---------------------------------------------------------------------------

class TestValidatePlate:
    # Valid standard plates
    def test_valid_2_letter_series(self):
        assert validate_plate("KL07BB1234") == "KL07BB1234"

    def test_valid_1_letter_series(self):
        assert validate_plate("DL01B1234") is None  # 9 chars only — invalid
        # Correct: DL 01 B 1234 → DL01B1234 is only 9 chars — real plates have 10
        # Actually valid: DL01AB1234
        assert validate_plate("DL01AB1234") == "DL01AB1234"

    def test_valid_various_states(self):
        valid_plates = [
            "MH12AB3456",
            "KA04CD5678",
            "TN09EF1234",
            "WB20GH9876",
            "UP32IJ5432",
        ]
        for p in valid_plates:
            assert validate_plate(p) == p, f"Should be valid: {p}"

    # Invalid state codes
    def test_invalid_state_code(self):
        assert validate_plate("ZZ07BB1234") is None
        assert validate_plate("XX00AA0000") is None

    def test_too_short(self):
        assert validate_plate("KL07") is None
        assert validate_plate("") is None

    # BH series
    def test_valid_bh_series(self):
        assert validate_plate("22BH1234AA") == "22BH1234AA"
        assert validate_plate("23BH9999AB") == "23BH9999AB"

    # Format mismatches
    def test_letters_in_digit_positions(self):
        assert validate_plate("KLABBB1234") is None   # positions 2-3 must be digits

    def test_digits_in_letter_positions(self):
        assert validate_plate("12345B1234") is None


# ---------------------------------------------------------------------------
# State codes coverage
# ---------------------------------------------------------------------------

class TestStateCodes:
    def test_all_known_states_present(self):
        expected = {
            "KL", "MH", "DL", "KA", "TN", "AP", "TS", "GJ",
            "RJ", "UP", "WB", "MP", "HR", "PB", "BR", "OD",
        }
        for state in expected:
            assert state in VALID_STATES, f"{state} missing from VALID_STATES"

    def test_invalid_states_absent(self):
        invalid = {"ZZ", "XX", "YY", "AA", "BB", "00"}
        for state in invalid:
            assert state not in VALID_STATES, f"{state} should not be in VALID_STATES"


# ---------------------------------------------------------------------------
# merge_multiline
# ---------------------------------------------------------------------------

class TestMergeMultiline:
    def _make_result(self, text: str, top_y: float):
        """Create a fake EasyOCR result tuple."""
        bbox = [[0, top_y], [100, top_y], [100, top_y + 20], [0, top_y + 20]]
        return (bbox, text, 0.9)

    def test_single_line(self):
        results = [self._make_result("KL07BB1234", 50)]
        assert merge_multiline(results) == "KL07BB1234"

    def test_two_lines_sorted_by_y(self):
        # Second line has lower y (further down) — should come second
        results = [
            self._make_result("BB1234", 80),   # bottom row
            self._make_result("KL07", 20),      # top row
        ]
        merged = merge_multiline(results)
        assert merged == "KL07BB1234"

    def test_already_in_order(self):
        results = [
            self._make_result("KL07", 20),
            self._make_result("BB1234", 80),
        ]
        assert merge_multiline(results) == "KL07BB1234"


# ---------------------------------------------------------------------------
# End-to-end string pipeline
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    def _process(self, raw: str):
        normalised = normalise_raw(raw)
        fixed = fix_characters(normalised)
        return validate_plate(fixed)

    def test_common_errors_corrected(self):
        assert self._process("KL 07 B0 1234") == "KL07BO1234"  # 0→O at pos 5

    def test_fully_correct_plate_passes(self):
        assert self._process("MH 12 AB 3456") == "MH12AB3456"

    def test_invalid_state_rejected(self):
        assert self._process("ZZ 07 BB 1234") is None

    def test_lowercase_input_handled(self):
        assert self._process("kl 07 bb 1234") == "KL07BB1234"
