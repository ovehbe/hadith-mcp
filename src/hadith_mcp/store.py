"""Read-only SQLite access for MCP tools."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


def _like_term(term: str) -> str:
    esc = (
        term.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )
    return f"%{esc}%"


# ─── Collection-slug normalization ─────────────────────────
#
# LLMs and users write hadith collection names many different ways
# (``sahih-bukhari`` vs ``Sahih al-Bukhari`` vs ``bukhari``). The functions
# below reduce every one of those surface forms to a single comparison
# key so ``resolve_collection_slug`` can match them against the canonical
# short slugs stored in ``collections.slug``.

# Collection-type prefixes to strip from the front of an input. Order
# matters only in that multi-word prefixes must come before their shorter
# substrings. Everything here is matched case-insensitively.
_COLLECTION_PREFIX_WORDS: tuple[str, ...] = (
    "40 hadith",
    "forty hadith",
    "sahih",
    "sunan",
    "jami",
    "jami'",
    "musnad",
    "muwatta",
)

# Arabic definite-article prefixes (transliterated) to strip after the
# collection-type prefix has been peeled off.
_ARABIC_ARTICLES: tuple[str, ...] = (
    "al", "an", "at", "ad", "as", "ar", "az",
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_collection_key(raw: str) -> str:
    """Reduce an arbitrary collection reference to a comparison key.

    >>> _normalize_collection_key("Sahih al-Bukhari")
    'bukhari'
    >>> _normalize_collection_key("sunan-abu-dawud")
    'abudawud'
    >>> _normalize_collection_key("jami at-tirmidhi")
    'tirmidhi'
    >>> _normalize_collection_key("40 hadith Nawawi")
    'nawawi'
    """
    s = raw.strip().lower()
    # Peel off collection-type prefix words repeatedly (e.g. an input like
    # "sahih sahih-bukhari" would normalize to "bukhari").
    changed = True
    while changed:
        changed = False
        for word in _COLLECTION_PREFIX_WORDS:
            for sep in (" ", "-", "_"):
                pref = word + sep
                if s.startswith(pref):
                    s = s[len(pref):].lstrip(" -_")
                    changed = True
        for art in _ARABIC_ARTICLES:
            for sep in ("-", " ", "_"):
                pref = art + sep
                if s.startswith(pref):
                    s = s[len(pref):].lstrip(" -_")
                    changed = True
    return _NON_ALNUM_RE.sub("", s)


# Orthographic variants that normalization alone cannot collapse. Keys are
# already-normalized forms of the input; values are canonical slugs.
_COLLECTION_ALIAS_MAP: dict[str, str] = {
    # "Musnad Ahmad" → normalizes to "ahmad", but canonical slug is "ahmed".
    "ahmad": "ahmed",
    "ahmadibnhanbal": "ahmed",
    # "40 Hadith Nawawi" / "Nawawi 40" → normalize to "nawawi" / "40nawawi";
    # canonical slug is "nawawi40".
    "nawawi": "nawawi40",
    "40nawawi": "nawawi40",
    "fortynawawi": "nawawi40",
    "qudsi": "qudsi40",
    "40qudsi": "qudsi40",
    "fortyqudsi": "qudsi40",
    "shahwaliullah": "shahwaliullah40",
    "40shahwaliullah": "shahwaliullah40",
    # Common transliteration variants.
    "abudawood": "abudawud",
    "nasai": "nasai",
    "nasa": "nasai",
    "nasa'i": "nasai",
    "ibnmajah": "ibnmajah",
    "ibnemajah": "ibnmajah",
    "ibnmaja": "ibnmajah",
}


class HadithStore:
    def __init__(self, db_path: Path) -> None:
        if not db_path.is_file():
            raise FileNotFoundError(f"hadith database not found: {db_path}")
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA query_only = ON")
        self._conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self._conn.close()

    def list_collections(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT id, slug, name_english, name_arabic, author_english, author_arabic, hadith_count
            FROM collections
            ORDER BY id
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def resolve_collection_slug(self, name_or_slug: str) -> str | None:
        """Resolve an English name or slug to the canonical ``collections.slug``.

        Handles the common ways LLMs and users write these collection names:

        * canonical short slugs (``bukhari``, ``ibnmajah``, ``nawawi40``)
        * ``name_english`` verbatim (``"Sahih al-Bukhari"``, ``"Sunan Abu Dawud"``)
        * full qualified forms (``"sahih-bukhari"``, ``"sunan-an-nasai"``,
          ``"jami-at-tirmidhi"``, ``"musnad-ahmad"``, ``"40 hadith nawawi"``)

        The matching strategy: try the exact slug / ``name_english`` first
        (fastest path, handles the common case), then fall back to a
        normalized comparison that strips collection-type prefix words
        (``sahih``, ``sunan``, ``jami``, ``musnad``, ``muwatta``,
        ``40 hadith``), Arabic definite-article prefixes (``al-``, ``an-``,
        ``at-``, ``ad-``, ``as-``, ``ar-``, ``az-``), and all
        non-alphanumerics. A small alias table handles orthographic
        variants that normalization alone cannot (``ahmad`` vs ``ahmed``,
        ``nawawi`` vs ``nawawi40``).
        """
        raw = name_or_slug.strip()
        if not raw:
            return None
        row = self._conn.execute(
            "SELECT slug FROM collections WHERE slug = ? COLLATE NOCASE",
            (raw,),
        ).fetchone()
        if row:
            return str(row[0])
        row = self._conn.execute(
            "SELECT slug FROM collections WHERE lower(name_english) = lower(?)",
            (raw,),
        ).fetchone()
        if row:
            return str(row[0])
        key = _normalize_collection_key(raw)
        if not key:
            return None
        alias = _COLLECTION_ALIAS_MAP.get(key)
        if alias:
            return alias
        # Finally, compare normalized forms of stored slugs / names.
        cur = self._conn.execute(
            "SELECT slug, name_english FROM collections"
        )
        for slug, name_english in cur.fetchall():
            if _normalize_collection_key(slug) == key:
                return str(slug)
            if name_english and _normalize_collection_key(name_english) == key:
                return str(slug)
        return None

    def get_collection_id(self, slug: str) -> int | None:
        row = self._conn.execute(
            "SELECT id FROM collections WHERE slug = ? COLLATE NOCASE",
            (slug.strip(),),
        ).fetchone()
        return int(row[0]) if row else None

    def resolve_hadith_id(self, collection_slug: str, id_in_book: int) -> int | None:
        row = self._conn.execute(
            """
            SELECT h.id FROM hadiths h
            JOIN collections c ON c.id = h.collection_id
            WHERE c.slug = ? COLLATE NOCASE AND h.id_in_book = ?
            ORDER BY h.id
            """,
            (collection_slug.strip(), id_in_book),
        ).fetchone()
        return int(row[0]) if row else None

    def fetch_hadiths_in_range(
        self,
        collection_slug: str,
        start_in_book: int,
        end_in_book: int,
    ) -> list[dict[str, Any]]:
        """Inclusive range of ``id_in_book`` within one collection (ordered by id_in_book)."""
        lo, hi = (start_in_book, end_in_book) if start_in_book <= end_in_book else (end_in_book, start_in_book)
        cur = self._conn.execute(
            """
            SELECT h.id, h.id_in_book, h.arabic, h.narrator, h.english, h.provenance, h.chapter_id,
                   c.slug AS collection_slug, c.name_english AS collection_name_english,
                   c.name_arabic AS collection_name_arabic,
                   ch.name_english AS chapter_name_english, ch.name_arabic AS chapter_name_arabic
            FROM hadiths h
            JOIN collections c ON c.id = h.collection_id
            LEFT JOIN chapters ch ON ch.id = h.chapter_id
            WHERE c.slug = ? COLLATE NOCASE AND h.id_in_book BETWEEN ? AND ?
            ORDER BY h.id_in_book, h.id
            """,
            (collection_slug.strip(), lo, hi),
        )
        return [dict(r) for r in cur.fetchall()]

    def fetch_hadiths_by_ids(self, hadith_ids: list[int]) -> list[dict[str, Any]]:
        """Return full hadith rows for ids (arbitrary order; caller may reorder)."""
        if not hadith_ids:
            return []
        uniq = list(dict.fromkeys(int(i) for i in hadith_ids))
        qs = ",".join("?" * len(uniq))
        cur = self._conn.execute(
            f"""
            SELECT h.id, h.id_in_book, h.arabic, h.narrator, h.english, h.provenance, h.chapter_id,
                   c.slug AS collection_slug, c.name_english AS collection_name_english,
                   c.name_arabic AS collection_name_arabic,
                   ch.name_english AS chapter_name_english, ch.name_arabic AS chapter_name_arabic
            FROM hadiths h
            JOIN collections c ON c.id = h.collection_id
            LEFT JOIN chapters ch ON ch.id = h.chapter_id
            WHERE h.id IN ({qs})
            """,
            uniq,
        )
        by_id = {int(r["id"]): dict(r) for r in cur.fetchall()}
        return [by_id[i] for i in uniq if i in by_id]

    def fetch_hadith(
        self,
        *,
        hadith_id: int | None = None,
        collection_slug: str | None = None,
        id_in_book: int | None = None,
    ) -> dict[str, Any] | None:
        if hadith_id is not None:
            cur = self._conn.execute(
                """
                SELECT h.id, h.id_in_book, h.arabic, h.narrator, h.english, h.provenance, h.chapter_id,
                       c.slug AS collection_slug, c.name_english AS collection_name_english,
                       c.name_arabic AS collection_name_arabic,
                       ch.name_english AS chapter_name_english, ch.name_arabic AS chapter_name_arabic
                FROM hadiths h
                JOIN collections c ON c.id = h.collection_id
                LEFT JOIN chapters ch ON ch.id = h.chapter_id
                WHERE h.id = ?
                """,
                (hadith_id,),
            )
        elif collection_slug is not None and id_in_book is not None:
            cur = self._conn.execute(
                """
                SELECT h.id, h.id_in_book, h.arabic, h.narrator, h.english, h.provenance, h.chapter_id,
                       c.slug AS collection_slug, c.name_english AS collection_name_english,
                       c.name_arabic AS collection_name_arabic,
                       ch.name_english AS chapter_name_english, ch.name_arabic AS chapter_name_arabic
                FROM hadiths h
                JOIN collections c ON c.id = h.collection_id
                LEFT JOIN chapters ch ON ch.id = h.chapter_id
                WHERE c.slug = ? AND h.id_in_book = ?
                ORDER BY h.id
                """,
                (collection_slug.strip(), id_in_book),
            )
        else:
            raise ValueError(
                "Provide either hadith_id, or both collection_slug and id_in_book"
            )
        row = cur.fetchone()
        return dict(row) if row else None

    def search_hadith(
        self,
        query: str,
        *,
        limit: int = 20,
        collection_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        q = query.strip()
        if len(q) < 2:
            return []
        terms = [t for t in q.split() if len(t) >= 1]
        if not terms:
            return []
        limit = max(1, min(limit, 100))
        where_parts: list[str] = []
        params: list[str | int] = []
        for t in terms:
            pat = _like_term(t)
            where_parts.append(
                "("
                "(h.english LIKE ? ESCAPE '\\' OR COALESCE(h.narrator,'') LIKE ? ESCAPE '\\' "
                "OR h.arabic LIKE ? ESCAPE '\\')"
                ")"
            )
            params.extend([pat, pat, pat])
        where_sql = " AND ".join(where_parts)
        sql = f"""
            SELECT h.id, h.id_in_book, c.slug AS collection_slug,
                   substr(h.english, 1, 280) AS english_excerpt
            FROM hadiths h
            JOIN collections c ON c.id = h.collection_id
            WHERE {where_sql}
        """
        if collection_slug:
            sql += " AND c.slug = ?"
            params.append(collection_slug.strip())
        sql += " ORDER BY h.id LIMIT ?"
        params.append(limit)
        cur = self._conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def fetch_cross_references(
        self, hadith_id: int, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        cur = self._conn.execute(
            """
            SELECT cr.matched_hadith_id, cr.similarity, cr.narrator_match,
                   c.slug AS collection_slug, h.id_in_book,
                   substr(h.english, 1, 200) AS english_excerpt
            FROM cross_references cr
            JOIN hadiths h ON h.id = cr.matched_hadith_id
            JOIN collections c ON c.id = h.collection_id
            WHERE cr.hadith_id = ?
            ORDER BY cr.similarity DESC
            LIMIT ?
            """,
            (hadith_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]
