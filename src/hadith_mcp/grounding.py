"""Citation and limitation text returned by ``fetch_grounding_rules``."""

GROUNDING_RULES = """
## Using this corpus

- **Source of truth:** Hadith text, collection metadata, embeddings, and cross-references come from
  this project's SQLite database (`hadith.db`), derived from the community **hadith-json** dataset
  (Sunnah.com). Prefer tool output over model paraphrase when quoting wording or numbering.

- **Citations:** When you cite a hadith, include **collection** (slug or English name) and the
  Sunnah.com-style collection reference (`id_in_book`) or the stable **row id** (`id`) returned by
  tools. Do not invent numbering.

- **Cross-references:** `cross_references` are **algorithmic** (embedding similarity plus narrator hints).
  They suggest textual overlap across collections; they are **not** a substitute for classical
  `mutaba`ah` scholarship or critical edition work.

- **Provenance field:** Values such as `muttafaq_alayh`, `bukhari`, `muslim`, `corroborated`, and
  `cross_referenced` are **heuristic tags** from the pipeline graph. Treat them as orientation, not
  a legal or theological ruling.

- **Arabic / English:** English translations may be incomplete or missing narrators; Arabic strings
  are authoritative for the stored row. When English is empty, rely on Arabic.

- **Limits:** Search is keyword-based (substring match). It does not rank by scholarly importance.
  Narrow queries with collection slugs when possible.
""".strip()
