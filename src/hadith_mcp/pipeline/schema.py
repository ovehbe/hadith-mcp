"""SQLite schema for hadith.db."""

from __future__ import annotations

import sqlite3

HADITHS_COLUMNS_SQL = """
    id INTEGER PRIMARY KEY,
    id_in_book INTEGER NOT NULL,
    collection_id INTEGER NOT NULL REFERENCES collections(id),
    chapter_id INTEGER REFERENCES chapters(id),
    arabic TEXT NOT NULL,
    narrator TEXT,
    english TEXT NOT NULL,
    provenance TEXT,
    embedding BLOB
"""

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    name_english TEXT NOT NULL,
    name_arabic TEXT NOT NULL,
    author_english TEXT,
    author_arabic TEXT,
    hadith_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id),
    source_chapter_id INTEGER NOT NULL,
    name_english TEXT,
    name_arabic TEXT,
    UNIQUE(collection_id, source_chapter_id)
);

CREATE TABLE IF NOT EXISTS hadiths (
{hadiths_columns}
);

CREATE TABLE IF NOT EXISTS cross_references (
    hadith_id INTEGER NOT NULL REFERENCES hadiths(id),
    matched_hadith_id INTEGER NOT NULL REFERENCES hadiths(id),
    similarity REAL NOT NULL,
    narrator_match INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (hadith_id, matched_hadith_id)
);

CREATE INDEX IF NOT EXISTS idx_hadiths_collection ON hadiths(collection_id);
CREATE INDEX IF NOT EXISTS idx_hadiths_collection_ref ON hadiths(collection_id, id_in_book);
CREATE INDEX IF NOT EXISTS idx_hadiths_provenance ON hadiths(provenance);
CREATE INDEX IF NOT EXISTS idx_hadiths_chapter ON hadiths(chapter_id);
CREATE INDEX IF NOT EXISTS idx_crossref_hadith ON cross_references(hadith_id);
CREATE INDEX IF NOT EXISTS idx_crossref_matched ON cross_references(matched_hadith_id);
""".format(hadiths_columns=HADITHS_COLUMNS_SQL.strip())

HADITHS_COLUMN_NAMES = (
    "id",
    "id_in_book",
    "collection_id",
    "chapter_id",
    "arabic",
    "narrator",
    "english",
    "provenance",
    "embedding",
)


def apply_schema(conn: sqlite3.Connection) -> None:
    _migrate_legacy_hadith_reference_unique(conn)
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def _migrate_legacy_hadith_reference_unique(conn: sqlite3.Connection) -> None:
    if not _has_legacy_hadith_reference_unique(conn):
        return

    foreign_keys_enabled = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    columns = ", ".join(HADITHS_COLUMN_NAMES)
    migration_table = "__hadiths_no_reference_unique"

    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN")
        conn.execute(f"DROP TABLE IF EXISTS {migration_table}")
        conn.execute(f"CREATE TABLE {migration_table} ({HADITHS_COLUMNS_SQL})")
        conn.execute(f"INSERT INTO {migration_table} ({columns}) SELECT {columns} FROM hadiths")
        conn.execute("DROP TABLE hadiths")
        conn.execute(f"ALTER TABLE {migration_table} RENAME TO hadiths")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute(f"PRAGMA foreign_keys = {1 if foreign_keys_enabled else 0}")


def _has_legacy_hadith_reference_unique(conn: sqlite3.Connection) -> bool:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'hadiths'"
    ).fetchone()
    if not exists:
        return False

    for index in conn.execute("PRAGMA index_list('hadiths')").fetchall():
        index_name = str(index[1])
        is_unique = bool(index[2])
        if not is_unique:
            continue
        if _index_columns(conn, index_name) == ["collection_id", "id_in_book"]:
            return True
    return False


def _index_columns(conn: sqlite3.Connection, index_name: str) -> list[str]:
    quoted = index_name.replace('"', '""')
    return [
        str(row[2])
        for row in conn.execute(f'PRAGMA index_info("{quoted}")').fetchall()
    ]
