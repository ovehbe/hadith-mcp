"""Tests for Sunnah.com-style reference numbering."""

from __future__ import annotations

from hadith_mcp.pipeline.canonical_numbering import canonical_id_in_book_by_hadith_id


def _row(hid: int, id_in_book: int, chapter_id: int) -> dict[str, int]:
    return {"id": hid, "idInBook": id_in_book, "chapterId": chapter_id}


def test_default_numbering_is_unchanged_without_intro_rows() -> None:
    rows = [_row(10, 1, 1), _row(11, 2, 1)]

    assert canonical_id_in_book_by_hadith_id("nasai", rows) == {10: 1, 11: 2}


def test_intro_rows_already_first_are_unchanged() -> None:
    rows = [_row(10, 1, 0), _row(11, 2, 0), _row(12, 3, 1)]

    assert canonical_id_in_book_by_hadith_id("example", rows) == {10: 1, 11: 2, 12: 3}


def test_trailing_intro_rows_are_numbered_before_main_rows() -> None:
    rows = [
        _row(10, 1, 1),
        _row(11, 2, 1),
        _row(12, 3, 0),
        _row(13, 4, 0),
    ]

    assert canonical_id_in_book_by_hadith_id("ibnmajah", rows) == {
        10: 3,
        11: 4,
        12: 1,
        13: 2,
    }


def test_bukhari_uses_explicit_sunnah_reference_ranges() -> None:
    rows = [
        _row(1, 1, 1),
        _row(273, 273, 4),
        _row(397, 397, 8),
        _row(398, 398, 8),
        _row(7277, 7277, 97),
    ]

    assert canonical_id_in_book_by_hadith_id("bukhari", rows) == {
        1: 1,
        273: 274,
        397: 402,
        398: 402,
        7277: 7563,
    }
