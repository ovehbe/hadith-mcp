"""Tests for SQLite schema migrations."""

from __future__ import annotations

import sqlite3

from hadith_mcp.pipeline.schema import apply_schema


def test_apply_schema_removes_legacy_hadith_reference_unique_constraint() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE collections (
            id INTEGER PRIMARY KEY,
            slug TEXT NOT NULL UNIQUE,
            name_english TEXT NOT NULL,
            name_arabic TEXT NOT NULL,
            author_english TEXT,
            author_arabic TEXT,
            hadith_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE chapters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_id INTEGER NOT NULL REFERENCES collections(id),
            source_chapter_id INTEGER NOT NULL,
            name_english TEXT,
            name_arabic TEXT,
            UNIQUE(collection_id, source_chapter_id)
        );

        CREATE TABLE hadiths (
            id INTEGER PRIMARY KEY,
            id_in_book INTEGER NOT NULL,
            collection_id INTEGER NOT NULL REFERENCES collections(id),
            chapter_id INTEGER REFERENCES chapters(id),
            arabic TEXT NOT NULL,
            narrator TEXT,
            english TEXT NOT NULL,
            provenance TEXT,
            embedding BLOB,
            UNIQUE(collection_id, id_in_book)
        );

        CREATE TABLE cross_references (
            hadith_id INTEGER NOT NULL REFERENCES hadiths(id),
            matched_hadith_id INTEGER NOT NULL REFERENCES hadiths(id),
            similarity REAL NOT NULL,
            narrator_match INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (hadith_id, matched_hadith_id)
        );

        INSERT INTO collections (id, slug, name_english, name_arabic)
        VALUES (1, 'bukhari', 'Bukhari', 'x');

        INSERT INTO hadiths (id, id_in_book, collection_id, arabic, english)
        VALUES (397, 397, 1, 'a', 'first');
        """
    )

    apply_schema(conn)

    conn.execute(
        """
        INSERT OR REPLACE INTO hadiths (id, id_in_book, collection_id, arabic, english)
        VALUES (397, 402, 1, 'a', 'first')
        """
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO hadiths (id, id_in_book, collection_id, arabic, english)
        VALUES (398, 402, 1, 'b', 'second')
        """
    )

    rows = conn.execute("SELECT id, id_in_book FROM hadiths ORDER BY id").fetchall()
    crossref_schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cross_references'"
    ).fetchone()[0]

    assert rows == [(397, 402), (398, 402)]
    assert "REFERENCES hadiths(id)" in crossref_schema
