"""Checks for the checked-in SQLite database when Git LFS data is available."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _db_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "hadith.db"


def _skip_if_lfs_pointer(path: Path) -> None:
    if not path.exists():
        pytest.skip("data/hadith.db is not present")
    with path.open("rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs.github.com/spec"):
        pytest.skip("data/hadith.db Git LFS object has not been fetched")


def test_checked_in_db_uses_canonical_sunnah_refs() -> None:
    db = _db_path()
    _skip_if_lfs_pointer(db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        ibnmajah_1727 = conn.execute(
            """
            SELECT h.id, h.english
            FROM hadiths h
            JOIN collections c ON c.id = h.collection_id
            WHERE c.slug = 'ibnmajah' AND h.id_in_book = 1727
            """
        ).fetchone()
        assert ibnmajah_1727 is not None
        assert "Dhul" in ibnmajah_1727["english"]
        assert "Hijjah" in ibnmajah_1727["english"]

        bukhari_7563 = conn.execute(
            """
            SELECT h.id, h.english
            FROM hadiths h
            JOIN collections c ON c.id = h.collection_id
            WHERE c.slug = 'bukhari' AND h.id_in_book = 7563
            """
        ).fetchone()
        assert bukhari_7563 is not None
        assert bukhari_7563["id"] == 7277
        assert "two words" in bukhari_7563["english"]

        bukhari_7277 = conn.execute(
            """
            SELECT h.id
            FROM hadiths h
            JOIN collections c ON c.id = h.collection_id
            WHERE c.slug = 'bukhari' AND h.id_in_book = 7277
            """
        ).fetchone()
        assert bukhari_7277 is not None
        assert bukhari_7277["id"] != 7277

        global_bukhari_final = conn.execute(
            "SELECT id_in_book FROM hadiths WHERE id = 7277"
        ).fetchone()
        assert global_bukhari_final is not None
        assert global_bukhari_final["id_in_book"] == 7563
    finally:
        conn.close()
