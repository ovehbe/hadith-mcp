# hadith-mcp

**Model Context Protocol (MCP) server and data pipeline** for serving **canonical hadith text** (Arabic and English) to assistants in a **citation-safe** way—similar in spirit to [quran-mcp](https://github.com/quran/quran-mcp): fetch from a real corpus instead of quoting from model memory.

This repository provides a **FastMCP** server over **`data/hadith.db`** plus a **data pipeline** to build that database: normalized SQLite, **OpenAI embeddings** (`text-embedding-3-large`), **cross-collection references** (cosine similarity + narrator-aware scoring), and **provenance-style tags** (e.g. muttafaq-style links between Sahih al-Bukhari and Sahih Muslim).

## Data sources and credits

- **Hadith text** comes from the community **[hadith-json](https://github.com/AhmedBaset/hadith-json)** dataset (scraped from [Sunnah.com](https://sunnah.com/)), which aligns with the broader **[sunnah-com](https://github.com/sunnah-com/api)** / Quran Foundation ecosystem—the same family of sources behind [quran-mcp](https://github.com/quran/quran-mcp).
- **Architecture and patterns** are inspired by **[quran-mcp](https://github.com/quran/quran-mcp)** (FastMCP, grounding mindset, tooling layout).

If you ship a product or paper, keep upstream attribution visible (dataset authors, Sunnah.com, and the scholarly collections themselves).

## Repository layout


| Path                                     | Purpose                                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `scripts/build_db.py`                    | Load `hadith-json` `db/by_book` JSON → SQLite schema, optional embed, cross-ref, provenance      |
| `scripts/embed_hadith.py`                | **Resume-only** embedding for rows with `embedding IS NULL` (slow, checkpoint-friendly)          |
| `scripts/merge_embedding_checkpoints.py` | Replay JSONL embedding checkpoints into `hadith.db` after crashes or restores                    |
| `scripts/compute_crossref.py`            | Recompute `cross_references` + `provenance` only (does **not** re-import JSON; safe after embed) |
| `scripts/fetch_ext_apps.py`              | Vendor / refresh `@modelcontextprotocol/ext-apps` as a classic script (sets `window.__hadithMcpSdk`) used by the interactive reader |
| `scripts/generate_search_sitemap.py`     | Regenerate `search/sitemap.xml` (index) + `search/sitemaps/*.xml` (~50k `?id=` URLs) from `data/hadith.db` for SEO after DB changes |
| `src/hadith_mcp/pipeline/`               | Loaders, schema, embed, cross-reference, provenance logic                                        |
| `src/hadith_mcp/server.py`               | FastMCP app: MCP tools + a small REST surface (`/api/collections`, `/api/hadith/{id}`, `/api/hadith/{slug}/{n}`, `/api/search`) reusing the same store and embedding index |
| `src/hadith_mcp/assets/hadith_app.html`  | Self-contained MCP App UI template (inline CSS + app logic, system fonts only) served at `ui://hadith.html` for the `show_hadith` tool |
| `src/hadith_mcp/assets/ext-apps.bundle.js` | Vendored ext-apps SDK (zero external imports) inlined into `hadith_app.html` at resource-render time |
| `search/`                                | Static search frontend (HTML/CSS/JS) deployed standalone (e.g. `search.hadith-mcp.org`)          |
| `site/`                                  | Static landing page for the main domain                                                          |
| `config.yml`                             | Optional default DB path (overridden by `HADITH_MCP_DB_PATH`)                                    |

Large reference trees **`hadith-json-main/`** and **`quran-mcp-master/`** are listed in `.gitignore`. Clone or unpack **[hadith-json](https://github.com/AhmedBaset/hadith-json)** locally (for example as `hadith-json-main/`) or pass **`--data-dir`** to `build_db.py`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set OPENAI_API_KEY for embedding steps
```

### 1) Build the database (without calling OpenAI)

Point `--data-dir` at your local `hadith-json` **`db/by_book`** directory.

```bash
python scripts/build_db.py --fresh --skip-embed --skip-cross --skip-provenance \
  --data-dir ./hadith-json-main/db/by_book
```

### 2) Embeddings (long run; use a separate machine if you prefer)

Safe defaults: **batch size 1**, **commit every 10 rows**, **sleep between calls**, optional **JSONL checkpoint** for safety.

```bash
python scripts/embed_hadith.py \
  --db-path ./data/hadith.db \
  --checkpoint ./data/embeddings_checkpoint.jsonl \
  --batch-size 1 \
  --commit-every 10 \
  --sleep-between-batches 0.15
```

Replay checkpoints into the DB when needed:

```bash
python scripts/merge_embedding_checkpoints.py --db-path ./data/hadith.db \
  ./data/embeddings_checkpoint.jsonl --only-missing
```

### 3) Cross-references and provenance (local CPU)

Do **not** re-run `build_db.py` without `--fresh` after embedding unless you intend to re-import JSON (that path can **overwrite** rows and clear `embedding`). Instead:

```bash
python scripts/compute_crossref.py --db-path ./data/hadith.db
```

For a **single-machine** full build (import + embed + cross + provenance), run `build_db.py` once **without** `--skip-embed` / `--skip-cross` / `--skip-provenance`, and pass embedding pacing flags as needed (`python scripts/build_db.py --help`).

### 4) MCP server (stdio for Cursor / Claude Desktop)

From the repo root with `data/hadith.db` present (or set `HADITH_MCP_DB_PATH`):

```bash
hadith-mcp --transport stdio
# or: python -m hadith_mcp --transport stdio
# or: fastmcp run hadith_mcp.server:mcp
```

Optional **`--config config.yml`** sets `database.path` relative to the config file. **`HADITH_MCP_DB_PATH`** overrides both.

HTTP / SSE / streamable HTTP (see FastMCP docs for host/port env vars):

```bash
hadith-mcp --transport http
```

**Tools (summary):** `fetch_grounding_rules` returns full text once per MCP session (then a short repeat unless `force_full=True`); pass returned `nonce` only when you need to disambiguate errors. `fetch_hadith` accepts global `hadith_id` **or** `collection` + `hadith_number` (int, or string range like `1-5`) with optional `include_cross_references`; `hadith_number` / `id_in_book` is the Sunnah.com-style collection reference number (for example `bukhari:7563` or `ibnmajah:1727`), while global `hadith_id` is the stable SQLite row id. The `collection` argument is resolved through a **forgiving slug matcher** — canonical slugs (`bukhari`), common variants (`sahih-bukhari`, `Sahih al-Bukhari`, `sahih_bukhari`), and human names (`Sunan Abu Dawud`, `Musnad Ahmad`, `40 Hadith Nawawi`) all map to the same row. `search_hadith` defaults to **semantic** search (loads all embeddings at startup, embeds the query with the configured **query embedding model**); use `mode=keyword` for SQL substring search, or `mode=both`. Semantic search needs **`OPENAI_API_KEY`** and a database whose rows include embeddings. If OpenAI returns quota/billing/rate-limit errors (or the query vector size does not match the DB), the server **falls back to keyword search** instead of failing. `fetch_cross_references` returns algorithmic similarity matches across collections for a given hadith. `show_hadith` opens an **interactive Hadith Reader** MCP App in supported hosts (ChatGPT Developer Apps, Claude with app support, etc.); **prefer calling it with the canonical `hadith_id`** returned by `fetch_hadith` / `search_hadith` / `fetch_cross_references` — `collection` + `hadith_number` and free-text `query` are supported fallbacks, and the tool always returns a plain-text fallback so non-App hosts still get a readable answer with the same citation URLs. The top-level MCP instructions nudge assistants toward the two-step flow (look up first, then `show_hadith(hadith_id=…)`) to avoid guessing slugs or numbers from memory. Optional **per-client rate limits** and an **LRU query cache** reduce cost and abuse (see `config.yml` / `.env.example`).

**Citation URLs.** `fetch_hadith`, `search_hadith`, `fetch_cross_references`, and `show_hadith` attach a **`url`** field to each hadith or cross-reference row pointing at the search frontend (`https://search.hadith-mcp.org/?id=<db_id>` by default, overridable via **`HADITH_SEARCH_APP_URL`**). The server's MCP instructions tell assistants to always surface this link alongside citations and to never fabricate links to external hadith sites (sunnah.com, etc.).

**Interactive reader (`show_hadith`).** The tool binds to a `ui://hadith.html` resource served as `text/html;profile=mcp-app`. The HTML template lives at `src/hadith_mcp/assets/hadith_app.html` and is **fully self-contained** — inline CSS, inline app logic, system fonts only, zero CDN fetches, zero cross-origin iframes. The `@modelcontextprotocol/ext-apps` SDK is **vendored** under `src/hadith_mcp/assets/ext-apps.bundle.js` (rewritten to a classic script that attaches to `window.__hadithMcpSdk`) and spliced into the template at startup, so the whole widget ships as a single HTML document. Refresh the pinned SDK version with `python3 scripts/fetch_ext_apps.py --version <x.y.z>`. The resource meta sets `ui.csp.resourceDomains = []` (no external origins at runtime) and intentionally **omits `ui.domain`** because ChatGPT and Claude require incompatible formats for that field (ChatGPT wants any `https://…` URL; Claude requires a sha256-derived `*.claudemcpcontent.com` subdomain and errors with "App domain configuration is invalid" on anything else) — both hosts work correctly when the field is omitted. Once mounted, the embedded app calls `fetch_hadith` and `search_hadith` over the MCP bridge (no extra HTTPS) to let users open cross-references and switch between detail and search views without LLM round-trips. A single-hadith `show_hadith` call renders pure card chrome (no search bar); calls with a `query` or no arguments render the search-bar + results UI.

### 5) Public search frontend and REST API

The repo ships a small static search app in **`search/`** and an HTTP REST surface on the same FastMCP process, intended to be deployed as two subdomains (e.g. `search.hadith-mcp.org` and `api.hadith-mcp.org`) with nginx / Caddy proxying `/api/*` to the FastMCP port.

- **Frontend (`search/`):** plain HTML/CSS/JS, no build step. Bootstraps from `?id=<db_id>` or `?q=<query>` on load, so the URLs MCP tools emit resolve directly. The API base defaults to `https://api.hadith-mcp.org`; override in the browser via `window.HADITH_API_BASE` (set before `script.js` loads) for local or staging deployments. **`search/sitemap.xml`** is a sitemap index; per-collection URL lists live under **`search/sitemaps/`** — regenerate with `python3 scripts/generate_search_sitemap.py` after rebuilding the database.
- **REST endpoints** (same process, mounted via `@mcp.custom_route`):
  - `GET /api/collections` → `{collections: [...]}`
  - `GET /api/hadith/{hadith_id}` → `{hadith: {...}}`
  - `GET /api/hadith/{slug}/{id_in_book}` → `{hadith: {...}}`, where `id_in_book` is the Sunnah.com-style collection reference number
  - `GET /api/search?q=&limit=&collection=` → `{results, mode, note}`; semantic by default with the same keyword fallback behavior as the MCP tool. Shares `HADITH_MCP_RATE_LIMIT_SEARCH_RPM` and the query cache with MCP clients, so one budget covers both surfaces.
  - `GET /api/stats` (optional trailing slash) → aggregate search/lookup counts, unique visitors, uptime. Landing + search UIs try **same-origin** `GET /api/stats` first, then the public API host; re-copy **`site/`** and **`search/`** when you update those pages, or the browser will run old HTML/JS.

## Configuration

- **Secrets:** `.env` is gitignored; see `.env.example` for `OPENAI_API_KEY` and MCP tuning (`HADITH_MCP_QUERY_EMBEDDING_MODEL`, `HADITH_MCP_RATE_LIMIT_SEARCH_RPM`, `HADITH_MCP_SEARCH_CACHE_MAX`).
- **Hosted MCP:** Put **`OPENAI_API_KEY`** on the server only if you accept paying for query embeddings; tune **`HADITH_MCP_RATE_LIMIT_SEARCH_RPM`** (e.g. `30`–`120`) and cache size. The query model **must match the dimension of vectors stored in `hadith.db`** (this repo’s build uses **`text-embedding-3-large`** / 3072). A cheaper OpenAI model generally means **rebuilding the database** with that model so dimensions align.
- **Artifacts:** Other `data/*.db` files and embedding checkpoint globs are gitignored; this repo tracks **`data/hadith.db`** (Git LFS) plus **`data/SHA256SUMS`** for verification (`cd data && sha256sum -c SHA256SUMS`).
- **Embeddings:** Rows with empty English narrator and text still embed using **Arabic** text when present. Long inputs are clipped with **tiktoken** (`cl100k_base`) to stay under the **8192-token** API limit, with a further shrink ladder if a row still hits length errors.
- **Count rows without the `sqlite3` CLI:**  
`python -c "import sqlite3; c=sqlite3.connect('data/hadith.db'); print(c.execute('SELECT COUNT(*) FROM hadiths WHERE embedding IS NULL').fetchone()[0])"`

## License

- **Software in this repository** (Python, scripts, and documentation we added) is licensed under **[GNU General Public License v3.0 only](LICENSE)** (SPDX: `GPL-3.0-only`).
- **Hadith text and other upstream material** remain under their original terms (**[hadith-json](https://github.com/AhmedBaset/hadith-json)**, **[Sunnah.com](https://sunnah.com/)**). Our GPL applies to our code, not to a relicensing of that content; keep attribution and follow upstream rules when you redistribute data or excerpts.

### Releases and data integrity

- **Checksum:** `data/SHA256SUMS` lists `hadith.db`. After cloning or downloading the database, run `cd data && sha256sum -c SHA256SUMS`.
- **Signing (optional):** A detached **GPG** or **[Sigstore](https://www.sigstore.dev/)** signature over the checksum file or the database proves who published the bytes and that they were not altered afterward. Signing does not certify scholarly accuracy of every narration or automated cross-reference.
- **Reproducibility:** For audits or rebuilds, record the **hadith-json** revision, this repo’s **git** revision, the embedding **model id**, and script versions you used.

## Contributing

Issues and PRs welcome. Please keep diffs focused and match existing style (`ruff` / `pytest` when present).
