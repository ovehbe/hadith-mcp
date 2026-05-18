"""FastMCP server: tools over ``hadith.db``."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import json

import anyio
import numpy as np
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from fastmcp.resources import ResourceContent
from fastmcp.server.lifespan import lifespan
from fastmcp.tools import ToolResult
from mcp.types import Icon, TextContent
from openai import OpenAI
from starlette.requests import Request
from starlette.responses import Response

from hadith_mcp.embeddings_index import EmbeddingIndex
from hadith_mcp.grounding import GROUNDING_RULES
from hadith_mcp.grounding_state import GroundingState
from hadith_mcp.middleware_logging import ToolCallLoggingMiddleware
from hadith_mcp.openai_fallback import should_fallback_to_keyword
from hadith_mcp.query_cache import SearchResponseCache
from hadith_mcp.rate_limit import RateLimiter
from hadith_mcp.settings import AppConfig, load_app_config
from hadith_mcp.stats import StatsTracker
from hadith_mcp.store import HadithStore

logger = logging.getLogger("hadith_mcp.server")

_MAX_HADITH_RANGE = 25
_SEARCH_APP_BASE_URL = os.environ.get("HADITH_SEARCH_APP_URL", "https://search.hadith-mcp.org").strip().rstrip("/")

_HADITH_APP_MIME = "text/html;profile=mcp-app"
_HADITH_APP_ASSETS_DIR = Path(__file__).parent / "assets"
_HADITH_APP_HTML_PATH = _HADITH_APP_ASSETS_DIR / "hadith_app.html"
_HADITH_APP_SDK_PATH = _HADITH_APP_ASSETS_DIR / "ext-apps.bundle.js"
_HADITH_APP_SDK_PLACEHOLDER = "/*__SDK_BUNDLE__*/"


def _search_client_key(ctx: Context) -> str:
    """Coarse client bucket for search rate limits (HTTP client IP when available)."""
    try:
        from fastmcp.server.dependencies import get_http_request

        req = get_http_request()
        if req.client and req.client.host:
            return f"ip:{req.client.host}"
        return "ip:unknown"
    except Exception:
        pass
    rc = ctx.request_context
    if rc is not None:
        return f"mcp:{id(rc)}"
    return "stdio:default"


def _session_key(ctx: Context) -> str:
    rc = ctx.request_context
    sess = getattr(rc, "session", None) if rc is not None else None
    return hex(id(sess)) if sess is not None else "default"


def _record_mcp(ctx: Context, kind: str) -> None:
    """Count MCP tool usage (search / lookup); client key is coarse (IP when HTTP)."""
    lc = ctx.lifespan_context
    if not isinstance(lc, dict):
        return
    st = lc.get("stats")
    if st is not None:
        st.record("mcp", kind, _search_client_key(ctx))


def _hadith_url(hadith_id: int) -> str:
    return f"{_SEARCH_APP_BASE_URL}/?id={hadith_id}"


def _add_hadith_url(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["url"] = _hadith_url(int(row["id"]))
    return out


def _add_match_url(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["url"] = _hadith_url(int(row["matched_hadith_id"]))
    return out


def _parse_hadith_span(
    hadith_number: int | str | None,
    hadith_number_end: int | None,
    id_in_book: int | None,
) -> tuple[int, int | None]:
    """Return ``(start_in_book, end_in_book_or_none)`` for a single id or inclusive range."""
    if hadith_number is None:
        if id_in_book is None:
            raise ValueError("Provide hadith_number or id_in_book with collection")
        hadith_number = id_in_book
    if isinstance(hadith_number, str):
        s = hadith_number.strip().replace(" ", "")
        if "-" in s:
            a, _, b = s.partition("-")
            return int(a), int(b)
        start = int(s)
        return start, hadith_number_end
    start = int(hadith_number)
    return start, hadith_number_end


@lifespan
async def _lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    cfg: AppConfig = getattr(server, "_hadith_cfg", None) or load_app_config()
    store = HadithStore(cfg.db_path)
    emb_index: EmbeddingIndex | None = None
    try:
        emb_index = await anyio.to_thread.run_sync(EmbeddingIndex.load, cfg.db_path)
        logger.info(
            "loaded embedding index rows=%s dim=%s",
            emb_index.mat.shape[0],
            emb_index.mat.shape[1],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("embedding index unavailable: %s", exc)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    openai_client = OpenAI(api_key=api_key) if api_key else None
    if openai_client and emb_index is None:
        logger.warning("OPENAI_API_KEY set but embedding index missing; semantic search disabled.")
    if emb_index is not None and openai_client is None:
        logger.warning("Embedding index present but OPENAI_API_KEY missing; semantic search disabled.")
    rate_limiter = RateLimiter(cfg.rate_limit_search_per_minute)
    search_cache = (
        SearchResponseCache(cfg.search_cache_max_entries)
        if cfg.search_cache_max_entries > 0
        else None
    )
    logger.info(
        "search: query_model=%s rate_limit_rpm=%s cache_max=%s",
        cfg.query_embedding_model,
        cfg.rate_limit_search_per_minute,
        cfg.search_cache_max_entries,
    )
    grounding = GroundingState()
    stats_tracker = StatsTracker()
    state = {
        "store": store,
        "config": cfg,
        "embeddings": emb_index,
        "openai": openai_client,
        "grounding": grounding,
        "search_rate_limiter": rate_limiter,
        "search_cache": search_cache,
        "stats": stats_tracker,
        "stats_boot_mono": time.monotonic(),
    }
    server._hadith_state = state  # type: ignore[attr-defined]
    try:
        yield state
    finally:
        try:
            server._hadith_state = None  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            stats_tracker.close()
        except Exception:
            pass
        store.close()
        logger.info("closed database connection")


def _format_narrator_line(raw: str | None) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    if s.lower().startswith(("narrated", "reported", "it was narrated")):
        return s if s.endswith(":") else f"{s}:"
    return f"Narrated {s}:"


def _format_detail_fallback(hadith: dict[str, Any], cross_refs: list[dict[str, Any]]) -> str:
    coll = hadith.get("collection_name_english") or hadith.get("collection_slug") or ""
    header = f"{coll} #{hadith.get('id_in_book', '?')}"
    chapter = hadith.get("chapter_name_english")
    if chapter:
        header += f" — {chapter}"
    parts: list[str] = [header]
    narr = _format_narrator_line(hadith.get("narrator"))
    if narr:
        parts += ["", narr]
    english = (hadith.get("english") or "").strip()
    if english:
        parts += ["", english]
    arabic = (hadith.get("arabic") or "").strip()
    if arabic:
        parts += ["", arabic]
    url = hadith.get("url")
    if url:
        parts += ["", f"URL: {url}"]
    if cross_refs:
        parts += ["", f"Cross-references ({len(cross_refs)}):"]
        for m in cross_refs[:5]:
            sim = m.get("similarity")
            sim_str = f" ({float(sim):.2f})" if isinstance(sim, (int, float)) else ""
            parts.append(
                f"- {m.get('collection_slug')} #{m.get('id_in_book')}"
                f"{sim_str} {m.get('url', '')}".rstrip()
            )
    return "\n".join(parts)


def _format_search_fallback(
    query: str, results: list[dict[str, Any]], note: str | None
) -> str:
    if not results:
        return f'No results for "{query}".' + (f"\n{note}" if note else "")
    lines = [f'Search: "{query}" — {len(results)} result(s)']
    if note:
        lines.append(note)
    lines.append("")
    for i, r in enumerate(results[:10], 1):
        sim = r.get("similarity")
        sim_str = (
            f" — {int(round(float(sim) * 100))}%"
            if isinstance(sim, (int, float))
            else ""
        )
        lines.append(
            f"{i}. {r.get('collection_slug')} #{r.get('id_in_book')}{sim_str}"
        )
        excerpt = (r.get("english_excerpt") or "").strip()
        if excerpt:
            lines.append(f"   {excerpt[:200]}")
        if r.get("url"):
            lines.append(f"   {r['url']}")
    return "\n".join(lines)


_EMPTY_FALLBACK_TEXT = (
    "Hadith Reader — no query given.\n\n"
    "Call show_hadith(hadith_id=…) to open a specific hadith, "
    "show_hadith(collection=…, hadith_number=…), "
    "or show_hadith(query=…) to run a search in the reader."
)


def build_server(*, config_yaml: Path | None = None) -> FastMCP:
    cfg = load_app_config(config_yaml=config_yaml)
    mcp = FastMCP(
        "hadith-mcp",
        instructions=(
            "Hadith corpus in SQLite. Call fetch_grounding_rules first when citing hadith. "
            "Never quote hadith from memory: use fetch_hadith or search_hadith. "
            "Cite with collection slug/name and id_in_book "
            "(Sunnah.com-style collection reference number). "
            "ALWAYS include the 'url' field from tool responses next to every hadith citation you "
            "surface to the user (e.g. append it in parentheses or as a markdown link). This is the "
            "canonical verification link for this corpus. "
            "Do NOT fabricate or append links to other hadith sites (sunnah.com, quran.com, "
            "hadithcollection.com, etc.); the only hadith URL you may emit is the 'url' returned by "
            "these tools. If a tool response has no 'url' field for a given row, cite without a URL "
            "rather than inventing one. "
            "search_hadith defaults to semantic (embeddings); use mode='keyword' for substring search. "
            "Semantic search falls back to keyword on rate limits, quota/billing errors, or model/index mismatch. "
            "Cross-references are algorithmic, not scholarly isnad proof. "
            "When the user asks to open, read, browse, or 'show' a hadith (or a search result set) "
            "interactively, call show_hadith so the Hadith Reader App opens for them. Prefer "
            "show_hadith(hadith_id=<id>) — the 'id' field on any fetch_hadith / search_hadith / "
            "fetch_cross_references row. If you do not already know that canonical id from an "
            "earlier tool call in this conversation, first look the hadith up with fetch_hadith "
            "(collection + hadith_number) or search_hadith (free text), THEN call show_hadith "
            "with the hadith_id from the response. Never guess hadith_id values from memory. "
            "Keep using fetch_hadith / search_hadith for raw text you need to quote or reason "
            "over in your answer. show_hadith also returns a text fallback with the same 'url' "
            "field, so cite from that output just like any other tool response."
        ),
        icons=[Icon(src="https://hadith-mcp.org/logo.png")],
        lifespan=_lifespan,
    )
    mcp._hadith_cfg = cfg  # type: ignore[attr-defined]
    mcp.add_middleware(ToolCallLoggingMiddleware())

    async def _semantic_search(
        ctx: Context,
        query: str,
        limit: int,
        coll_filter: str | None,
    ) -> dict[str, Any]:
        cfg: AppConfig = ctx.lifespan_context["config"]
        store: HadithStore = ctx.lifespan_context["store"]
        idx = ctx.lifespan_context.get("embeddings")
        client = ctx.lifespan_context.get("openai")
        if idx is None or client is None:
            return {"ok": False, "reason": "semantic_unavailable", "results": [], "fallback": False}

        rl = ctx.lifespan_context.get("search_rate_limiter")
        if rl is not None and not rl.allow(_search_client_key(ctx)):
            return {"ok": False, "reason": "rate_limited", "results": [], "fallback": True}

        cache = ctx.lifespan_context.get("search_cache")
        cache_key = (query.strip().lower(), limit, coll_filter or "", cfg.query_embedding_model)
        if cache is not None:
            hit = cache.get(cache_key)
            if hit is not None:
                return {"ok": True, "results": hit, "cache_hit": True, "fallback": False}

        coll_id: int | None = None
        if coll_filter:
            slug = store.resolve_collection_slug(coll_filter) or coll_filter.strip()
            coll_id = store.get_collection_id(slug)

        model = cfg.query_embedding_model

        def _embed() -> np.ndarray:
            r = client.embeddings.create(model=model, input=query)
            return np.asarray(r.data[0].embedding, dtype=np.float32)

        try:
            qv = await anyio.to_thread.run_sync(_embed)
        except Exception as exc:  # noqa: BLE001
            if should_fallback_to_keyword(exc):
                logger.warning("semantic search falling back to keyword (OpenAI): %s", exc)
                return {
                    "ok": False,
                    "reason": "openai_error",
                    "fallback": True,
                    "openai_message": str(exc),
                    "results": [],
                }
            raise

        if int(qv.shape[0]) != int(idx.mat.shape[1]):
            logger.warning(
                "query embedding dim %s != index dim %s (model=%s); use a query model that matches hadith.db",
                qv.shape[0],
                idx.mat.shape[1],
                model,
            )
            return {
                "ok": False,
                "reason": "dimension_mismatch",
                "fallback": True,
                "results": [],
            }

        top = idx.topk(qv, limit, collection_id=coll_id)
        ids = [i for i, _ in top]
        scores = {i: s for i, s in top}
        rows = store.fetch_hadiths_by_ids(ids)
        results: list[dict[str, Any]] = []
        for r in rows:
            hid = int(r["id"])
            results.append(
                {
                    "hadith_id": hid,
                    "similarity": float(scores[hid]),
                    "collection_slug": r["collection_slug"],
                    "id_in_book": r["id_in_book"],
                    "english_excerpt": (r.get("english") or "")[:280],
                    "url": _hadith_url(hid),
                }
            )
        if cache is not None:
            cache.set(cache_key, results)
        return {"ok": True, "results": results, "cache_hit": False, "fallback": False}

    def _keyword_search(
        ctx: Context,
        query: str,
        limit: int,
        coll_filter: str | None,
    ) -> dict[str, Any]:
        store: HadithStore = ctx.lifespan_context["store"]
        kw_slug: str | None = None
        if coll_filter:
            kw_slug = store.resolve_collection_slug(coll_filter) or coll_filter.strip()
        rows = store.search_hadith(query, limit=limit, collection_slug=kw_slug)
        results = [
            {
                "hadith_id": int(r["id"]),
                "similarity": None,
                "collection_slug": r["collection_slug"],
                "id_in_book": r["id_in_book"],
                "english_excerpt": r.get("english_excerpt"),
                "url": _hadith_url(int(r["id"])),
            }
            for r in rows
        ]
        return {"ok": True, "results": results}

    @mcp.tool()
    def list_collections(ctx: Context) -> list[dict[str, Any]]:
        """List all collections (slug, English/Arabic names, hadith counts)."""
        store: HadithStore = ctx.lifespan_context["store"]
        return store.list_collections()

    @mcp.tool()
    def fetch_hadith(
        ctx: Context,
        hadith_id: int | None = None,
        collection: str | None = None,
        collection_slug: str | None = None,
        hadith_number: int | str | None = None,
        hadith_number_end: int | None = None,
        id_in_book: int | None = None,
        include_cross_references: bool = False,
    ) -> dict[str, Any]:
        """Fetch hadith text by global ``hadith_id`` or by ``collection`` + ``hadith_number`` (or ``id_in_book``).

        ``hadith_number`` may be an int, or a string range like ``\"1-5\"`` (inclusive). Ranges are
        capped (see ``hadiths`` list length). Set ``include_cross_references`` to attach algorithmic
        matches per returned hadith.
        """
        store: HadithStore = ctx.lifespan_context["store"]
        coll_raw = (collection or collection_slug or "").strip() or None

        def _attach_cross(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
            out: dict[str, list[dict[str, Any]]] = {}
            for r in rows:
                hid = int(r["id"])
                out[str(hid)] = [_add_match_url(m) for m in store.fetch_cross_references(hid, limit=40)]
            return out

        if hadith_id is not None and coll_raw is not None:
            return {
                "error": "Use either hadith_id alone, or collection + hadith_number — not both",
                "hadith": None,
                "hadiths": None,
                "cross_references": None,
            }
        if hadith_id is not None:
            row = store.fetch_hadith(hadith_id=hadith_id)
            if row is None:
                return {"error": "not_found", "hadith": None, "hadiths": None, "cross_references": None}
            row_out = _add_hadith_url(row)
            crs = _attach_cross([row_out]) if include_cross_references else None
            _record_mcp(ctx, "lookup")
            return {"error": None, "hadith": row_out, "hadiths": None, "cross_references": crs}

        if not coll_raw:
            return {
                "error": "Provide hadith_id, or collection + hadith_number / id_in_book",
                "hadith": None,
                "hadiths": None,
                "cross_references": None,
            }

        slug = store.resolve_collection_slug(coll_raw) or coll_raw
        try:
            start, end = _parse_hadith_span(hadith_number, hadith_number_end, id_in_book)
        except (TypeError, ValueError) as e:
            return {
                "error": f"invalid_hadith_number: {e}",
                "hadith": None,
                "hadiths": None,
                "cross_references": None,
            }

        if end is None:
            row = store.fetch_hadith(collection_slug=slug, id_in_book=start)
            if row is None:
                return {"error": "not_found", "hadith": None, "hadiths": None, "cross_references": None}
            row_out = _add_hadith_url(row)
            crs = _attach_cross([row_out]) if include_cross_references else None
            _record_mcp(ctx, "lookup")
            return {"error": None, "hadith": row_out, "hadiths": None, "cross_references": crs}

        span = abs(end - start) + 1
        if span > _MAX_HADITH_RANGE:
            return {
                "error": f"range too large (max {_MAX_HADITH_RANGE} hadiths)",
                "hadith": None,
                "hadiths": None,
                "cross_references": None,
            }
        rows = store.fetch_hadiths_in_range(slug, start, end)
        if not rows:
            return {"error": "not_found", "hadith": None, "hadiths": [], "cross_references": None}
        rows_out = [_add_hadith_url(r) for r in rows]
        crs = _attach_cross(rows_out) if include_cross_references else None
        _record_mcp(ctx, "lookup")
        return {"error": None, "hadith": None, "hadiths": rows_out, "cross_references": crs}

    @mcp.tool()
    async def search_hadith(
        ctx: Context,
        query: str,
        limit: int = 20,
        collection: str | None = None,
        collection_slug: str | None = None,
        mode: str = "semantic",
    ) -> dict[str, Any]:
        """Search hadiths: ``mode='semantic'`` (default) uses embeddings; ``keyword`` uses SQL LIKE; ``both`` runs both."""
        limit = max(1, min(int(limit), 100))
        coll_f = (collection or collection_slug or "").strip() or None
        mode_l = mode.strip().lower()
        if mode_l not in {"semantic", "keyword", "both"}:
            return {
                "mode": mode_l,
                "error": "mode must be semantic, keyword, or both",
                "results": [],
            }

        if mode_l == "keyword":
            kw = _keyword_search(ctx, query, limit, coll_f)
            _record_mcp(ctx, "search")
            return {"mode": "keyword", "results": kw["results"], "note": None}

        if mode_l == "semantic":
            sem = await _semantic_search(ctx, query, limit, coll_f)
            if sem["ok"]:
                note = "cached_response" if sem.get("cache_hit") else None
                _record_mcp(ctx, "search")
                return {"mode": "semantic", "results": sem["results"], "note": note}
            kw = _keyword_search(ctx, query, limit, coll_f)
            reason = sem.get("reason")
            if reason == "semantic_unavailable":
                msg = "Semantic search unavailable (missing index or OPENAI_API_KEY); used keyword search."
            elif reason == "rate_limited":
                msg = "Search rate limit exceeded; used keyword search."
            elif reason == "dimension_mismatch":
                msg = (
                    "Query embedding size does not match database vectors; "
                    "fix embedding.query_model / HADITH_MCP_QUERY_EMBEDDING_MODEL to match hadith.db. "
                    "Used keyword search."
                )
            elif reason == "openai_error":
                msg = f"OpenAI embedding failed; used keyword search. ({sem.get('openai_message', '')})"
            else:
                msg = "Semantic search failed; used keyword search."
            _record_mcp(ctx, "search")
            return {"mode": "keyword_fallback", "results": kw["results"], "note": msg}

        sem = await _semantic_search(ctx, query, limit, coll_f)
        kw = _keyword_search(ctx, query, limit, coll_f)
        note = None
        if not sem["ok"]:
            note = (
                f"Semantic leg incomplete ({sem.get('reason', 'unknown')}); "
                "keyword leg still returned."
            )
        _record_mcp(ctx, "search")
        return {
            "mode": "both",
            "semantic": sem,
            "keyword": kw,
            "note": note,
        }

    @mcp.tool()
    def fetch_cross_references(
        ctx: Context,
        hadith_id: int | None = None,
        collection: str | None = None,
        collection_slug: str | None = None,
        hadith_number: int | None = None,
        id_in_book: int | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Cross-collection similarity matches for a hadith (by ``hadith_id`` or ``collection`` + number)."""
        store: HadithStore = ctx.lifespan_context["store"]
        hid = hadith_id
        if hid is None:
            coll_raw = (collection or collection_slug or "").strip() or None
            inn = hadith_number if hadith_number is not None else id_in_book
            if not coll_raw or inn is None:
                return {
                    "error": "Provide hadith_id, or collection + hadith_number / id_in_book",
                    "hadith_id": None,
                    "matches": [],
                }
            slug = store.resolve_collection_slug(coll_raw) or coll_raw
            hid = store.resolve_hadith_id(slug, int(inn))
            if hid is None:
                return {"error": "not_found", "hadith_id": None, "matches": []}
        matches = [_add_match_url(m) for m in store.fetch_cross_references(hid, limit=limit)]
        return {"error": None, "hadith_id": hid, "matches": matches}

    @mcp.tool()
    def fetch_grounding_rules(
        ctx: Context,
        nonce: str | None = None,
        force_full: bool = False,
    ) -> dict[str, Any]:
        """Citation and limitation guidance. Re-calls without ``force_full`` return a short repeat message."""
        grounding: GroundingState = ctx.lifespan_context["grounding"]
        return grounding.fetch(
            _session_key(ctx),
            nonce=nonce,
            force_full=force_full,
            full_text=GROUNDING_RULES,
        )

    @mcp.tool(
        name="show_hadith",
        title="Show Hadith Reader",
        description=(
            "Open the interactive Hadith Reader UI for a user. Renders a card-based reader with "
            "collection/number/chapter chips, Arabic typography, and cross-references.\n\n"
            "PREFERRED ENTRY POINT: global 'hadith_id' (the database id returned by "
            "fetch_hadith / search_hadith as the 'id' field). This is the most reliable form "
            "because it skips all name / slug / number resolution.\n\n"
            "Recommended flow when the user asks to 'show / open / read / view a hadith':\n"
            "  1. If you do not already know the canonical 'hadith_id' from an earlier tool "
            "     call in this conversation, first call fetch_hadith (for a known "
            "     collection+number) or search_hadith (for free text) to find it.\n"
            "  2. Then call show_hadith(hadith_id=<that id>) to open the reader on that exact "
            "     hadith. Do NOT rely on the model's memory for hadith_id values — always get "
            "     them from a prior tool response.\n\n"
            "Secondary entry points (use only if step 1 is not practical):\n"
            "  - 'collection' (slug or English name; sahih-bukhari / Sahih al-Bukhari / bukhari "
            "    all resolve) + 'hadith_number' (same as the Sunnah.com-style id_in_book).\n"
            "  - 'query' — free-text search rendered inside the reader (semantic with keyword "
            "    fallback).\n"
            "  - no arguments — opens an empty reader for the user to browse.\n\n"
            "Do not follow show_hadith with fetch_hadith / search_hadith for the same hadith "
            "just to re-read text — the structured response already contains the full row. "
            "Always surface the returned 'url' alongside any hadith you cite."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        meta={"ui": {"resourceUri": "ui://hadith.html"}},
        tags={"preview", "app", "hadith"},
    )
    async def show_hadith(
        ctx: Context,
        hadith_id: int | None = None,
        collection: str | None = None,
        collection_slug: str | None = None,
        hadith_number: int | None = None,
        id_in_book: int | None = None,
        query: str | None = None,
    ) -> ToolResult:
        store: HadithStore = ctx.lifespan_context["store"]
        coll_raw = (collection or collection_slug or "").strip() or None
        num = hadith_number if hadith_number is not None else id_in_book
        q = (query or "").strip() or None

        structured: dict[str, Any] = {
            "kind": "empty",
            "hadith": None,
            "cross_references": None,
            "query": None,
            "search_results": None,
            "search_mode": None,
            "search_note": None,
            "collection_filter": coll_raw,
            "collections": store.list_collections(),
            "search_app_url": _SEARCH_APP_BASE_URL,
            "interactive": True,
        }

        if hadith_id is not None or (coll_raw and num is not None):
            row = None
            if hadith_id is not None:
                row = store.fetch_hadith(hadith_id=int(hadith_id))
            else:
                slug = store.resolve_collection_slug(coll_raw) or coll_raw
                try:
                    row = store.fetch_hadith(collection_slug=slug, id_in_book=int(num))
                except (TypeError, ValueError):
                    row = None
            if row is None:
                structured["kind"] = "empty"
                return ToolResult(
                    content=[TextContent(type="text", text="Hadith not found.")],
                    structured_content=structured,
                )
            row_out = _add_hadith_url(row)
            cross_refs = [
                _add_match_url(m)
                for m in store.fetch_cross_references(int(row["id"]), limit=40)
            ]
            structured["kind"] = "detail"
            structured["hadith"] = row_out
            structured["cross_references"] = cross_refs
            _record_mcp(ctx, "lookup")
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=_format_detail_fallback(row_out, cross_refs),
                    )
                ],
                structured_content=structured,
            )

        if q:
            sem = await _semantic_search(ctx, q, 30, coll_raw)
            if sem.get("ok"):
                results = sem["results"]
                mode = "semantic"
                note = "cached_response" if sem.get("cache_hit") else None
            else:
                kw = _keyword_search(ctx, q, 30, coll_raw)
                results = kw["results"]
                reason = sem.get("reason")
                if reason == "semantic_unavailable":
                    note = "Semantic search unavailable; used keyword search."
                elif reason == "rate_limited":
                    note = "Search rate limit exceeded; used keyword search."
                elif reason == "dimension_mismatch":
                    note = (
                        "Query embedding size does not match database vectors; "
                        "used keyword search."
                    )
                elif reason == "openai_error":
                    note = (
                        "OpenAI embedding failed; used keyword search."
                        f" ({sem.get('openai_message', '')})"
                    )
                else:
                    note = "Semantic search failed; used keyword search."
                mode = "keyword_fallback"

            structured["kind"] = "search"
            structured["query"] = q
            structured["search_results"] = results
            structured["search_mode"] = mode
            structured["search_note"] = note
            _record_mcp(ctx, "search")
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=_format_search_fallback(q, results, note),
                    )
                ],
                structured_content=structured,
            )

        return ToolResult(
            content=[TextContent(type="text", text=_EMPTY_FALLBACK_TEXT)],
            structured_content=structured,
        )

    _HADITH_APP_HTML: str | None = None
    if _HADITH_APP_HTML_PATH.is_file():
        raw_html = _HADITH_APP_HTML_PATH.read_text(encoding="utf-8")
        if _HADITH_APP_SDK_PATH.is_file():
            sdk_js = _HADITH_APP_SDK_PATH.read_text(encoding="utf-8")
            if _HADITH_APP_SDK_PLACEHOLDER not in raw_html:
                logger.warning(
                    "hadith app html is missing the SDK placeholder %r; "
                    "the reader will render with a missing-SDK error",
                    _HADITH_APP_SDK_PLACEHOLDER,
                )
                _HADITH_APP_HTML = raw_html
            else:
                # str.replace is a literal substitution (no regex / backref
                # handling), so it's safe to splice the JS body verbatim.
                _HADITH_APP_HTML = raw_html.replace(
                    _HADITH_APP_SDK_PLACEHOLDER, sdk_js, 1
                )
                logger.info(
                    "loaded hadith app html (%d bytes, sdk=%d bytes) from %s",
                    len(_HADITH_APP_HTML),
                    len(sdk_js),
                    _HADITH_APP_HTML_PATH,
                )
        else:
            logger.warning(
                "hadith app SDK bundle missing at %s; reader will show a "
                "missing-SDK error. Run scripts/fetch_ext_apps.py to vendor it.",
                _HADITH_APP_SDK_PATH,
            )
            _HADITH_APP_HTML = raw_html
    else:
        logger.warning("hadith app html missing at %s", _HADITH_APP_HTML_PATH)

    @mcp.resource(
        "ui://hadith.html",
        name="Hadith Reader App",
        description=(
            "Interactive hadith reader with search, detail view, cross-references, "
            "Arabic typography, and citation URLs."
        ),
        mime_type=_HADITH_APP_MIME,
        tags={"preview", "app"},
    )
    async def hadith_app() -> list[ResourceContent]:
        if _HADITH_APP_HTML is None:
            raise FileNotFoundError(
                f"Hadith app HTML not found at {_HADITH_APP_HTML_PATH}."
            )
        return [
            ResourceContent(
                _HADITH_APP_HTML,
                mime_type=_HADITH_APP_MIME,
                meta={
                    "ui": {
                        # NOTE: we intentionally do NOT set ui.domain here.
                        # ChatGPT and Claude require incompatible formats
                        # for that field (ChatGPT wants any https URL;
                        # Claude requires a sha256-derived subdomain of
                        # claudemcpcontent.com and errors with "App domain
                        # configuration is invalid" on anything else). Both
                        # hosts work correctly when the field is simply
                        # omitted — Claude auto-generates its sandbox
                        # identity and ChatGPT just shows a non-blocking
                        # "widget domain not set" warning in the registry.
                        "csp": {
                            # Self-contained app: CSS, JS, and the ext-apps
                            # SDK are all inlined. No external origins are
                            # needed at runtime.
                            "resourceDomains": [],
                        },
                    },
                },
            )
        ]

    _ICON_PATH = Path(__file__).parent / "assets" / "icon.png"
    _icon_bytes: bytes | None = None
    if _ICON_PATH.is_file():
        _icon_bytes = _ICON_PATH.read_bytes()
        logger.info("loaded icon asset (%d bytes)", len(_icon_bytes))

    @mcp.custom_route("/icon.png", methods=["GET", "HEAD"])
    async def serve_icon(request: Request) -> Response:
        if _icon_bytes is None:
            return Response(status_code=404)
        return Response(
            content=_icon_bytes,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=604800, immutable"},
        )

    # --- public REST API backing the search.hadith-mcp.org frontend ---
    #
    # Lives on the same FastMCP process; exposed at api.hadith-mcp.org via nginx
    # proxying /api/* to this backend. Browsers need ACAO to read JSON from
    # hadith-mcp.org and search.hadith-mcp.org; include here so it works even if
    # nginx is not adding CORS for these routes.

    def _api_cors_headers() -> dict[str, str]:
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        }

    def _api_json(data: Any, status: int = 200) -> Response:
        return Response(
            content=json.dumps(data).encode("utf-8"),
            status_code=status,
            media_type="application/json; charset=utf-8",
            headers=_api_cors_headers(),
        )

    def _api_state() -> dict[str, Any] | None:
        return getattr(mcp, "_hadith_state", None)

    def _api_client_key(request: Request) -> str:
        host = None
        if request.client is not None:
            host = request.client.host
        xff = request.headers.get("x-forwarded-for")
        if xff:
            host = xff.split(",")[0].strip() or host
        return f"ip:{host or 'unknown'}"

    def _record_api(state: dict[str, Any], request: Request, kind: str) -> None:
        if request.method == "HEAD":
            return
        st = state.get("stats")
        if st is not None:
            st.record("api", kind, _api_client_key(request))

    def _api_hadith_item(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        if not out.get("english_excerpt"):
            out["english_excerpt"] = (out.get("english") or "")[:280]
        out["url"] = _hadith_url(int(out["id"]))
        return out

    def _api_keyword_payload(
        store: HadithStore,
        q: str,
        limit: int,
        coll_filter: str | None,
        note: str | None,
    ) -> Response:
        kw_slug: str | None = None
        if coll_filter:
            kw_slug = store.resolve_collection_slug(coll_filter) or coll_filter
        rows_small = store.search_hadith(q, limit=limit, collection_slug=kw_slug)
        ids = [int(r["id"]) for r in rows_small]
        rows_full = store.fetch_hadiths_by_ids(ids) if ids else []
        full_by_id = {int(r["id"]): r for r in rows_full}
        results: list[dict[str, Any]] = []
        for r in rows_small:
            rid = int(r["id"])
            merged = dict(full_by_id.get(rid, {}))
            merged.update({k: v for k, v in r.items() if v is not None})
            item = _api_hadith_item(merged)
            item["similarity"] = None
            results.append(item)
        return _api_json({"results": results, "mode": "keyword", "note": note})

    @mcp.custom_route("/api/collections", methods=["GET", "HEAD"])
    async def api_collections(request: Request) -> Response:
        state = _api_state()
        if state is None:
            return _api_json({"error": "server starting"}, status=503)
        store: HadithStore = state["store"]
        return _api_json({"collections": store.list_collections()})

    async def _api_stats_get(request: Request) -> Response:
        state = _api_state()
        if state is None:
            body = {
                "total_searches": 0,
                "total_lookups": 0,
                "unique_visitors": 0,
                "uptime_seconds": 0,
                "mcp": {"searches": 0, "lookups": 0},
                "api": {"searches": 0, "lookups": 0},
            }
            return Response(
                content=json.dumps(body).encode("utf-8"),
                status_code=503,
                media_type="application/json; charset=utf-8",
                headers={**_api_cors_headers(), "Cache-Control": "public, max-age=5"},
            )
        boot = state.get("stats_boot_mono")
        uptime = int(max(0.0, time.monotonic() - float(boot))) if isinstance(boot, (int, float)) else 0
        st = state.get("stats")
        if st is None:
            data: dict[str, Any] = {
                "total_searches": 0,
                "total_lookups": 0,
                "unique_visitors": 0,
                "mcp": {"searches": 0, "lookups": 0},
                "api": {"searches": 0, "lookups": 0},
            }
        else:
            data = dict(st.get_stats())
        data["uptime_seconds"] = uptime
        return Response(
            content=json.dumps(data).encode("utf-8"),
            status_code=200,
            media_type="application/json; charset=utf-8",
            headers={**_api_cors_headers(), "Cache-Control": "public, max-age=30"},
        )

    @mcp.custom_route("/api/stats", methods=["GET", "HEAD"])
    async def api_stats(request: Request) -> Response:
        return await _api_stats_get(request)

    @mcp.custom_route("/api/stats/", methods=["GET", "HEAD"])
    async def api_stats_slash(request: Request) -> Response:
        return await _api_stats_get(request)

    @mcp.custom_route("/api/stats", methods=["OPTIONS"])
    async def api_stats_options(_request: Request) -> Response:
        return Response(
            status_code=204,
            headers=_api_cors_headers(),
        )

    @mcp.custom_route("/api/stats/", methods=["OPTIONS"])
    async def api_stats_slash_options(_request: Request) -> Response:
        return Response(
            status_code=204,
            headers=_api_cors_headers(),
        )

    @mcp.custom_route("/api/hadith/{hadith_id:int}", methods=["GET", "HEAD"])
    async def api_hadith_by_id(request: Request) -> Response:
        state = _api_state()
        if state is None:
            return _api_json({"error": "server starting", "hadith": None}, status=503)
        store: HadithStore = state["store"]
        try:
            hid = int(request.path_params["hadith_id"])
        except (KeyError, ValueError, TypeError):
            return _api_json({"error": "invalid_id", "hadith": None}, status=400)
        row = store.fetch_hadith(hadith_id=hid)
        if row is None:
            return _api_json({"error": "not_found", "hadith": None}, status=404)
        _record_api(state, request, "lookup")
        return _api_json({"hadith": _api_hadith_item(row)})

    @mcp.custom_route("/api/hadith/{hadith_id:int}/cross-references", methods=["GET", "HEAD"])
    async def api_cross_references(request: Request) -> Response:
        state = _api_state()
        if state is None:
            return _api_json({"error": "server starting", "cross_references": []}, status=503)
        store: HadithStore = state["store"]
        try:
            hid = int(request.path_params["hadith_id"])
        except (KeyError, ValueError, TypeError):
            return _api_json({"error": "invalid_id", "cross_references": []}, status=400)
        try:
            lim = min(int(request.query_params.get("limit", 40)), 100)
        except (ValueError, TypeError):
            lim = 40
        matches = [_add_match_url(m) for m in store.fetch_cross_references(hid, limit=lim)]
        return _api_json({"cross_references": matches})

    @mcp.custom_route("/api/hadith/{slug:str}/{id_in_book:int}", methods=["GET", "HEAD"])
    async def api_hadith_by_collection(request: Request) -> Response:
        state = _api_state()
        if state is None:
            return _api_json({"error": "server starting", "hadith": None}, status=503)
        store: HadithStore = state["store"]
        slug_raw = str(request.path_params.get("slug") or "").strip()
        try:
            id_in_book = int(request.path_params["id_in_book"])
        except (KeyError, ValueError, TypeError):
            return _api_json({"error": "invalid_id", "hadith": None}, status=400)
        if not slug_raw:
            return _api_json({"error": "invalid_slug", "hadith": None}, status=400)
        slug = store.resolve_collection_slug(slug_raw) or slug_raw
        row = store.fetch_hadith(collection_slug=slug, id_in_book=id_in_book)
        if row is None:
            return _api_json({"error": "not_found", "hadith": None}, status=404)
        _record_api(state, request, "lookup")
        return _api_json({"hadith": _api_hadith_item(row)})

    @mcp.custom_route("/api/search", methods=["GET", "HEAD"])
    async def api_search(request: Request) -> Response:
        state = _api_state()
        if state is None:
            return _api_json({"results": [], "mode": "none", "note": "server starting"}, status=503)
        store: HadithStore = state["store"]
        cfg_local: AppConfig = state["config"]
        idx = state.get("embeddings")
        client = state.get("openai")
        cache = state.get("search_cache")
        rl = state.get("search_rate_limiter")

        q = (request.query_params.get("q") or "").strip()
        if len(q) < 2:
            return _api_json(
                {"results": [], "mode": "none", "note": "query too short"},
                status=400,
            )
        try:
            limit = max(1, min(int(request.query_params.get("limit") or 20), 100))
        except (TypeError, ValueError):
            limit = 20
        coll_filter = (request.query_params.get("collection") or "").strip() or None

        client_key = _api_client_key(request)

        def _api_search_recorded(resp: Response) -> Response:
            _record_api(state, request, "search")
            return resp

        # Semantic path (if configured and allowed by rate limit)
        if idx is not None and client is not None:
            if rl is not None and not rl.allow(client_key):
                return _api_search_recorded(
                    _api_keyword_payload(
                        store, q, limit, coll_filter, note="rate_limited; keyword fallback"
                    )
                )

            cache_key = (q.lower(), limit, coll_filter or "", cfg_local.query_embedding_model)
            if cache is not None:
                hit = cache.get(cache_key)
                if hit is not None:
                    return _api_search_recorded(
                        _api_json(
                            {"results": hit, "mode": "semantic", "note": "cache"},
                        )
                    )

            coll_id: int | None = None
            if coll_filter:
                slug = store.resolve_collection_slug(coll_filter) or coll_filter
                coll_id = store.get_collection_id(slug)
            model = cfg_local.query_embedding_model

            def _embed() -> np.ndarray:
                r = client.embeddings.create(model=model, input=q)
                return np.asarray(r.data[0].embedding, dtype=np.float32)

            try:
                qv = await anyio.to_thread.run_sync(_embed)
            except Exception as exc:  # noqa: BLE001
                if should_fallback_to_keyword(exc):
                    logger.warning("api /search OpenAI fallback: %s", exc)
                    return _api_search_recorded(
                        _api_keyword_payload(
                            store, q, limit, coll_filter, note=f"openai_error; keyword fallback ({exc})"
                        )
                    )
                raise

            if int(qv.shape[0]) != int(idx.mat.shape[1]):
                logger.warning(
                    "api /search dim mismatch qv=%s idx=%s model=%s",
                    qv.shape[0],
                    idx.mat.shape[1],
                    model,
                )
                return _api_search_recorded(
                    _api_keyword_payload(
                        store, q, limit, coll_filter, note="dim_mismatch; keyword fallback"
                    )
                )

            top = idx.topk(qv, limit, collection_id=coll_id)
            ids = [i for i, _ in top]
            scores = {i: s for i, s in top}
            rows = store.fetch_hadiths_by_ids(ids)
            results: list[dict[str, Any]] = []
            for r in rows:
                item = _api_hadith_item(r)
                item["similarity"] = float(scores[int(r["id"])])
                results.append(item)
            if cache is not None:
                cache.set(cache_key, results)
            return _api_search_recorded(_api_json({"results": results, "mode": "semantic", "note": None}))

        return _api_search_recorded(
            _api_keyword_payload(
                store,
                q,
                limit,
                coll_filter,
                note="semantic unavailable (missing index or OPENAI_API_KEY); used keyword",
            )
        )

    return mcp


load_dotenv()
mcp = build_server()
