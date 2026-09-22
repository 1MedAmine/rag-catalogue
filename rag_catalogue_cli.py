#!/usr/bin/env python3
"""CLI du RAG hybride NVIDIA pour retrouver et classer des references catalogue."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from rag_catalogue.llm_client import (
    DEFAULT_NVIDIA_API_BASE,
    DEFAULT_REASONING_MODE,
    REASONING_PROFILES,
    NvidiaChatClient,
    ResilientNvidiaChatClient,
)
from rag_catalogue.retrieval_clients import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_URL,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_URL,
    NvidiaEmbeddingClient,
    NvidiaRerankClient,
)
from rag_catalogue.pipeline import index_catalogue, inspect_catalogue, retrieve_catalogue_chunks, run_search


DEFAULT_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
DEFAULT_FALLBACK_MODEL = "nvidia/nemotron-3-super-120b-a12b"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("un entier strictement positif est requis")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("un entier positif ou nul est requis")
    return parsed


def _add_pdf_processing_options(parser: argparse.ArgumentParser, *, hierarchy: bool = True) -> None:
    parser.add_argument(
        "--ocr",
        choices=("off", "auto", "force"),
        default=os.environ.get("RAG_OCR_MODE", "auto"),
        help="OCR page par page: off, auto sur pages image sans texte, ou force",
    )
    parser.add_argument(
        "--ocr-lang",
        default=os.environ.get("RAG_OCR_LANG", os.environ.get("RAG_OCR_LANGUAGE", "eng")),
        help="langue Tesseract/PyMuPDF, par exemple eng ou fra+eng",
    )
    parser.add_argument(
        "--ocr-max-pages",
        type=_nonnegative_int,
        default=int(os.environ.get("RAG_OCR_MAX_PAGES", "12")),
        help="nombre maximal de pages OCRisees par catalogue",
    )
    parser.add_argument(
        "--min-native-chars",
        type=_nonnegative_int,
        default=int(os.environ.get("RAG_MIN_NATIVE_CHARS", "40")),
        help="seuil de texte natif sous lequel une page image devient candidate OCR",
    )
    if hierarchy:
        parser.add_argument(
            "--profil-catalogue",
            choices=("auto", "standard", "large", "huge"),
            default=os.environ.get("RAG_CATALOGUE_PROFILE", "auto"),
            help="profondeur de recherche standard, large, huge ou automatique",
        )
        parser.add_argument(
            "--section-k",
            type=_positive_int,
            help="nombre maximal de sections routees; automatique par defaut",
        )
        parser.add_argument(
            "--sans-hierarchie",
            action="store_true",
            help="force le RAG global normal et desactive la navigation hierarchique V14",
        )


def _add_embedding_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--modele-embedding",
        default=os.environ.get("NVIDIA_EMBED_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="modele NVIDIA d'embedding",
    )
    parser.add_argument(
        "--embedding-url",
        default=os.environ.get("NVIDIA_EMBED_API_URL", DEFAULT_EMBEDDING_URL),
        help="endpoint complet /embeddings",
    )
    parser.add_argument("--sans-embedding", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("RAG_CACHE_DIR", ".rag_cache")),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V14: RAG catalogue a navigation humaine, avec repli global automatique.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("chercher", help="chercher et classer les references avec le LLM")
    search.add_argument("--fiche", required=True, type=Path, help="cahier produit complet PDF, TXT, MD ou JSON")
    search.add_argument(
        "--provenance",
        type=Path,
        help="JSON compagnon de provenance; sinon decouverte automatique a cote du cahier",
    )
    search.add_argument(
        "--sans-provenance-auto",
        action="store_true",
        help="desactive la decouverte automatique du JSON compagnon",
    )
    search.add_argument("--catalogue", required=True, type=Path)
    search.add_argument(
        "--modele",
        default=os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL),
        help="modele NVIDIA de generation",
    )
    search.add_argument(
        "--modele-secours",
        default=os.environ.get("NVIDIA_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL),
        help="modele NVIDIA de secours; chaîne vide pour désactiver",
    )
    search.add_argument(
        "--raisonnement",
        choices=tuple(REASONING_PROFILES),
        default=os.environ.get("NVIDIA_REASONING_MODE", DEFAULT_REASONING_MODE),
        help="rapide, normal (defaut) ou approfondi",
    )
    search.add_argument("--api-base", default=os.environ.get("NVIDIA_API_BASE", DEFAULT_NVIDIA_API_BASE))
    _add_embedding_options(search)
    search.add_argument(
        "--modele-reranking",
        default=os.environ.get("NVIDIA_RERANK_MODEL", DEFAULT_RERANK_MODEL),
        help="modele NVIDIA de reranking",
    )
    search.add_argument(
        "--rerank-url",
        default=os.environ.get("NVIDIA_RERANK_API_URL", DEFAULT_RERANK_URL),
        help="endpoint complet de reranking",
    )
    search.add_argument("--sans-reranking", action="store_true")
    search.add_argument(
        "--rerank-batch-size",
        type=_positive_int,
        default=int(os.environ.get("RAG_RERANK_BATCH_SIZE", "64")),
        help="taille maximale d'un appel de reranking",
    )
    validation = search.add_mutually_exclusive_group()
    validation.add_argument(
        "--validation-preuves",
        dest="validation_preuves",
        action="store_true",
        help="active le controle structurel de la reference (defaut)",
    )
    validation.add_argument(
        "--sans-validation-preuves",
        dest="validation_preuves",
        action="store_false",
        help="desactive le controle structurel de la reference",
    )
    search.set_defaults(validation_preuves=True)
    search.add_argument("--top-k", type=_positive_int, default=8)
    search.add_argument("--candidate-k", type=_positive_int, default=32)
    _add_pdf_processing_options(search, hierarchy=True)
    search.add_argument(
        "--diagnostic",
        action="store_true",
        help="enregistre le diagnostic RAG dans un fichier JSON separe",
    )
    search.add_argument("--sortie-diagnostic", type=Path)
    search.add_argument(
        "--details",
        action="store_true",
        help="conserve les preuves et segments dans le resultat principal",
    )
    search.add_argument("--sortie", type=Path)

    retrieve = subparsers.add_parser("retrouver", help="inspecter la recuperation sans LLM")
    retrieve.add_argument("--catalogue", required=True, type=Path)
    retrieve.add_argument("--requete", required=True)
    retrieve.add_argument("--top-k", type=_positive_int, default=8)
    retrieve.add_argument("--candidate-k", type=_positive_int, default=32)
    _add_pdf_processing_options(retrieve, hierarchy=True)

    inspect_command = subparsers.add_parser("inspecter", help="inspecter le PDF, l'OCR et les sections sans API")
    inspect_command.add_argument("--catalogue", required=True, type=Path)
    _add_pdf_processing_options(inspect_command, hierarchy=False)
    inspect_command.add_argument("--sortie", type=Path)

    index_command = subparsers.add_parser("indexer", help="preconstruire la carte, les chunks et le cache dense")
    index_command.add_argument("--catalogue", required=True, type=Path)
    _add_embedding_options(index_command)
    _add_pdf_processing_options(index_command, hierarchy=False)
    index_command.add_argument("--sortie", type=Path)

    return parser

def _write_json(value: object, output: Path | None = None) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if output is None:
        print(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(output.resolve())


def _default_diagnostic_path(output: Path | None) -> Path:
    if output is None:
        return Path("diagnostic_rag.json")
    return output.with_name(f"{output.stem}_diagnostic.json")


def _compact_result(result: dict[str, object]) -> dict[str, object]:
    reference = result.get("reference")
    validation_errors = result.get("validation_errors")
    raw_candidates = result.get("candidates")
    candidates = raw_candidates if isinstance(raw_candidates, list) else []
    if len(candidates) > 1:
        status = "multiple_candidates"
    elif reference is not None:
        status = "found"
    elif isinstance(validation_errors, list) and validation_errors:
        status = "rejected"
    else:
        status = "not_found"
    compact = {
        "status": status,
        "reference": reference,
        "catalogue_pages": result.get("catalogue_pages", []),
        "explanation": result.get("explanation", ""),
        "validation_errors": validation_errors if isinstance(validation_errors, list) else [],
        "model": result.get("model", ""),
    }
    if candidates:
        compact["candidates"] = [
            {
                "rank": item.get("rank"),
                "reference": item.get("reference"),
                "variant": item.get("variant"),
                "match_level": item.get("match_level"),
                "equivalence_status": item.get("equivalence_status"),
                "catalogue_pages": item.get("catalogue_pages", []),
                "reason": item.get("reason", ""),
                "differences": item.get("differences", []),
                "mandatory_differences": item.get("mandatory_differences", []),
            }
            for item in candidates
            if isinstance(item, dict)
        ]
    models = result.get("models")
    if isinstance(models, dict):
        compact["models"] = models
    reasoning = result.get("reasoning")
    if isinstance(reasoning, dict):
        compact["reasoning"] = reasoning
    model_routing = result.get("model_routing")
    if isinstance(model_routing, dict):
        compact["model_routing"] = model_routing
    diagnostic_file = result.get("diagnostic_file")
    if isinstance(diagnostic_file, str) and diagnostic_file:
        compact["diagnostic_file"] = diagnostic_file
    return compact


def _run_search(args: argparse.Namespace) -> int:
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("NVIDIA_API_KEY est absente du fichier .env ou de l'environnement")
    primary_llm = NvidiaChatClient(
        api_key=api_key,
        model=args.modele,
        base_url=args.api_base,
        reasoning_mode=args.raisonnement,
    )
    fallback_name = str(args.modele_secours or "").strip()
    fallback_llm = (
        NvidiaChatClient(
            api_key=api_key,
            model=fallback_name,
            base_url=args.api_base,
            reasoning_mode=args.raisonnement,
        )
        if fallback_name and fallback_name != args.modele
        else None
    )
    llm = ResilientNvidiaChatClient(primary_llm, fallback_llm)
    embedding_client = (
        None
        if args.sans_embedding
        else NvidiaEmbeddingClient(
            api_key=api_key,
            model=args.modele_embedding,
            api_url=args.embedding_url,
        )
    )
    rerank_client = (
        None
        if args.sans_reranking
        else NvidiaRerankClient(
            api_key=api_key,
            model=args.modele_reranking,
            api_url=args.rerank_url,
        )
    )
    result = run_search(
        args.fiche,
        args.catalogue,
        llm,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        embedding_client=embedding_client,
        rerank_client=rerank_client,
        cache_dir=args.cache_dir,
        include_diagnostics=args.diagnostic,
        validate_result=args.validation_preuves,
        provenance_file=args.provenance,
        auto_discover_provenance=not args.sans_provenance_auto,
        hierarchy_enabled=not args.sans_hierarchie,
        catalogue_profile=args.profil_catalogue,
        section_k=args.section_k,
        ocr_mode=args.ocr,
        ocr_language=args.ocr_lang,
        ocr_max_pages=args.ocr_max_pages,
        min_native_chars=args.min_native_chars,
        rerank_batch_size=args.rerank_batch_size,
    )
    if args.diagnostic:
        diagnostic = result.pop("diagnostic", None)
        if not isinstance(diagnostic, dict):
            raise RuntimeError("diagnostic RAG absent du resultat interne")
        diagnostic_path = args.sortie_diagnostic or _default_diagnostic_path(args.sortie)
        _write_json(diagnostic, diagnostic_path)
        result["diagnostic_file"] = str(diagnostic_path.resolve())
    output_result = result if args.details else _compact_result(result)
    _write_json(output_result, args.sortie)
    return 0


def _run_retrieval(args: argparse.Namespace) -> int:
    metadata: dict[str, object] = {}
    ranked = retrieve_catalogue_chunks(
        args.catalogue,
        args.requete,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        retrieval_metadata=metadata,
        hierarchical=not args.sans_hierarchie,
        catalogue_profile=args.profil_catalogue,
        section_k=args.section_k,
        ocr_mode=args.ocr,
        ocr_language=args.ocr_lang,
        ocr_max_pages=args.ocr_max_pages,
        min_native_chars=args.min_native_chars,
    )
    payload = {
        "retrieval_metadata": metadata,
        "results": [
            {
                "rank": rank,
                "page": item.chunk.page_number,
                "chunk_id": item.chunk.chunk_id,
                "section_id": item.chunk.section_id,
                "section_title": item.chunk.section_title,
                "kind": item.chunk.kind,
                "score": round(item.score, 6),
                "excerpt": item.chunk.text[:700],
            }
            for rank, item in enumerate(ranked, start=1)
        ],
    }
    _write_json(payload)
    return 0


def _run_inspection(args: argparse.Namespace) -> int:
    result = inspect_catalogue(
        args.catalogue,
        ocr_mode=args.ocr,
        ocr_language=args.ocr_lang,
        ocr_max_pages=args.ocr_max_pages,
        min_native_chars=args.min_native_chars,
    )
    _write_json(result, args.sortie)
    return 0


def _run_index(args: argparse.Namespace) -> int:
    embedding_client = None
    if not args.sans_embedding:
        api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY est absente; utilise --sans-embedding pour un index local seulement"
            )
        embedding_client = NvidiaEmbeddingClient(
            api_key=api_key,
            model=args.modele_embedding,
            api_url=args.embedding_url,
        )
    result = index_catalogue(
        args.catalogue,
        embedding_client=embedding_client,
        cache_dir=args.cache_dir,
        ocr_mode=args.ocr,
        ocr_language=args.ocr_lang,
        ocr_max_pages=args.ocr_max_pages,
        min_native_chars=args.min_native_chars,
    )
    _write_json(result, args.sortie)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "chercher":
            return _run_search(args)
        if args.command == "retrouver":
            return _run_retrieval(args)
        if args.command == "inspecter":
            return _run_inspection(args)
        return _run_index(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
