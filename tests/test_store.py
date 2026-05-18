"""Tests for read-only hadith store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from hadith_mcp.pipeline.schema import apply_schema
from hadith_mcp.store import HadithStore


def _minimal_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    apply_schema(conn)
    conn.execute(
        """
        INSERT INTO collections (id, slug, name_english, name_arabic, hadith_count)
        VALUES (1, 'bukhari', 'Bukhari', 'البخاري', 2)
        """
    )
    conn.execute(
        """
        INSERT INTO hadiths (id, id_in_book, collection_id, chapter_id, arabic, narrator, english, provenance)
        VALUES (1, 9, 1, NULL, 'السلام', 'n1', 'Peace be upon you', NULL)
        """
    )
    conn.execute(
        """
        INSERT INTO hadiths (id, id_in_book, collection_id, chapter_id, arabic, narrator, english, provenance)
        VALUES (2, 10, 1, NULL, '٢', 'n2', 'Charity is encouraged', NULL)
        """
    )
    conn.execute(
        """
        INSERT INTO hadiths (id, id_in_book, collection_id, chapter_id, arabic, narrator, english, provenance)
        VALUES (3, 9, 1, NULL, 'three', 'n3', 'Another row with the same reference', NULL)
        """
    )
    conn.execute(
        """
        INSERT INTO cross_references (hadith_id, matched_hadith_id, similarity, narrator_match)
        VALUES (1, 2, 0.9, 1)
        """
    )
    conn.commit()
    conn.close()


def test_store_fetch_search_crossref(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    _minimal_db(db)
    s = HadithStore(db)
    try:
        assert s.fetch_hadith(hadith_id=1)["english"] == "Peace be upon you"
        row = s.fetch_hadith(collection_slug="bukhari", id_in_book=9)
        assert row is not None and row["id"] == 1
        found = s.search_hadith("charity", limit=5)
        assert len(found) == 1 and found[0]["id"] == 2
        xr = s.fetch_cross_references(1)
        assert len(xr) == 1
        assert xr[0]["matched_hadith_id"] == 2
        assert xr[0]["similarity"] == 0.9
        cols = s.list_collections()
        assert len(cols) == 1 and cols[0]["slug"] == "bukhari"
        assert s.resolve_collection_slug("Bukhari") == "bukhari"
        assert s.resolve_hadith_id("bukhari", 9) == 1
        rng = s.fetch_hadiths_in_range("bukhari", 9, 10)
        assert len(rng) == 3 and [r["id"] for r in rng] == [1, 3, 2]
        by_ids = s.fetch_hadiths_by_ids([2, 1])
        assert [r["id"] for r in by_ids] == [2, 1]
    finally:
        s.close()
