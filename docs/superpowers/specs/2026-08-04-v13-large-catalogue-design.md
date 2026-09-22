# V13 Large-Catalogue RAG Design

## Goal

Make the catalogue RAG scale from short product brochures to large, heterogeneous industrial catalogues without introducing product-family-specific rules.

## Design

V13 builds a catalogue map before retrieval. It uses the PDF outline when available, otherwise layout headings and bounded page windows. Every chunk receives section, kind, marker, page and extraction-quality metadata.

Retrieval remains global for recall, then adds section-focused and structural lanes for precision. Lexical, exact-code, dense, section and ordering/selection rankings are fused with weighted reciprocal-rank fusion. Near-duplicate prose is removed while distinct variant columns and order-code maps are preserved. A diversified shortlist is reranked in bounded batches before the generation LLM receives the final context.

Search depth is based on pages, chunks and section count, with `standard`, `large`, `huge` and automatic profiles. Exact-reference retrieval remains global even when hierarchy routing is active.

Native PDF extraction is always attempted first. OCR is optional and page-scoped: `auto` only considers image pages with insufficient native text; `force` remains bounded by `--ocr-max-pages`.

## Compatibility and safety

The complete source specification remains the primary input. A provenance JSON may supplement it but never replaces it. V12 output fields, multi-candidate behavior, reasoning modes, diagnostics, JSON repair and reference-proof validation remain compatible.

No production rule names a manufacturer, product family or expected reference. Structural validation uses only information extracted from the supplied catalogue.
