"""Load and normalize hadith-json by_book/*.json files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hadith_mcp.pipeline.canonical_numbering import canonical_id_in_book_by_hadith_id
from hadith_mcp.pipeline.collections_meta import BOOK_FILES, resolve_book_path


@dataclass
class LoadedHadith:
    id: int
    id_in_book: int
    collection_id: int
    slug: str
    chapter_source_id: int
    arabic: str
    narrator: str
    english: str


@dataclass
class LoadedChapter:
    collection_id: int
    source_chapter_id: int
    name_arabic: str
    name_english: str


@dataclass
class LoadedCollection:
    id: int
    slug: str
    name_english: str
    name_arabic: str
    author_english: str
    author_arabic: str
    hadith_count: int


def _meta_str(meta: dict[str, Any], lang: str, key: str) -> str:
    block = meta.get(lang) or {}
    val = block.get(key)
    return val if isinstance(val, str) else ""


def load_all(by_book_root: Path) -> tuple[list[LoadedCollection], list[LoadedChapter], list[LoadedHadith]]:
    collections: list[LoadedCollection] = []
    chapters: list[LoadedChapter] = []
    hadiths: list[LoadedHadith] = []
    seen_ids: set[int] = set()

    for bp in BOOK_FILES:
        path = resolve_book_path(by_book_root, bp)
        if not path.is_file():
            raise FileNotFoundError(f"Missing book JSON: {path}")
        with path.open(encoding="utf-8") as f:
            book = json.load(f)

        top_id = int(book["id"])
        if top_id != bp.collection_id:
            raise ValueError(f"Book id mismatch {path}: json id={top_id} expected={bp.collection_id}")

        meta = book["metadata"]
        collections.append(
            LoadedCollection(
                id=top_id,
                slug=bp.slug,
                name_english=_meta_str(meta, "english", "title"),
                name_arabic=_meta_str(meta, "arabic", "title"),
                author_english=_meta_str(meta, "english", "author"),
                author_arabic=_meta_str(meta, "arabic", "author"),
                hadith_count=len(book["hadiths"]),
            )
        )

        for ch in book.get("chapters") or []:
            chapters.append(
                LoadedChapter(
                    collection_id=top_id,
                    source_chapter_id=int(ch["id"]),
                    name_arabic=str(ch.get("arabic") or ""),
                    name_english=str(ch.get("english") or ""),
                )
            )

        canonical_numbers = canonical_id_in_book_by_hadith_id(bp.slug, book["hadiths"])

        for h in book["hadiths"]:
            hid = int(h["id"])
            if hid in seen_ids:
                raise ValueError(f"Duplicate hadith id {hid} in {path}")
            seen_ids.add(hid)
            eng = h.get("english") or {}
            narrator = str(eng.get("narrator") or "")
            english = str(eng.get("text") or "")
            hadiths.append(
                LoadedHadith(
                    id=hid,
                    id_in_book=canonical_numbers[hid],
                    collection_id=int(h["bookId"]),
                    slug=bp.slug,
                    chapter_source_id=int(h["chapterId"]),
                    arabic=str(h.get("arabic") or ""),
                    narrator=narrator,
                    english=english,
                )
            )

    return collections, chapters, hadiths


def embedding_input(narrator: str, english: str, *, arabic: str = "") -> str:
    """Text used for OpenAI embedding: prefer English narrator + matn, else Arabic matn.

    Some collections (e.g. short forties) have empty English fields; Arabic is still present.
    """
    n = narrator.strip()
    e = english.strip()
    if n and e:
        return f"{n}\n{e}"
    if e or n:
        return e or n
    return (arabic or "").strip()
