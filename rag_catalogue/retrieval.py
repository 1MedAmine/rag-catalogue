"""Domain-neutral local hybrid lexical retrieval for industrial catalogues."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .pdf_text import CatalogueChunk


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk: CatalogueChunk
    score: float
    word_score: float
    char_score: float
    exact_score: float
    code_score: float = 0.0
    dense_score: float = 0.0
    lexical_rank: int | None = None
    dense_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", without_marks.casefold()).strip()


def _exact_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"(?u)\b[\w][\w./+\-]*\b", _fold(text))
        if any(char.isdigit() for char in token) or len(token) >= 4
    }


def _code_tokens(text: str) -> set[str]:
    return {
        token
        for token in _exact_tokens(text)
        if any(char.isalpha() for char in token)
        and any(char.isdigit() for char in token)
        and len(token) >= 4
    }


class HybridRetriever:
    """Combine word, character, exact-value, and exact-code relevance."""

    def __init__(self, chunks: list[CatalogueChunk]) -> None:
        if not chunks:
            raise ValueError("au moins un chunk catalogue est requis")
        self._chunks = list(chunks)
        texts = [chunk.text for chunk in self._chunks]
        self._word_vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w./+\-]+\b",
            sublinear_tf=True,
        )
        self._char_vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 6),
            sublinear_tf=True,
        )
        self._word_matrix = self._word_vectorizer.fit_transform(texts)
        self._char_matrix = self._char_vectorizer.fit_transform(texts)
        self._chunk_tokens = [_exact_tokens(text) for text in texts]
        self._chunk_code_tokens = [_code_tokens(text) for text in texts]
        self._chunk_index = {chunk.chunk_id: index for index, chunk in enumerate(self._chunks)}

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        weights: tuple[float, float, float, float] | None = None,
        allowed_chunk_ids: set[str] | None = None,
    ) -> list[RankedChunk]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("la requete ne peut pas etre vide")
        if top_k <= 0:
            raise ValueError("top_k doit etre strictement positif")

        word_query = self._word_vectorizer.transform([query])
        char_query = self._char_vectorizer.transform([query])
        word_scores = (self._word_matrix @ word_query.T).toarray().ravel()
        char_scores = (self._char_matrix @ char_query.T).toarray().ravel()

        query_tokens = _exact_tokens(query)
        exact_scores = np.array(
            [
                len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
                for chunk_tokens in self._chunk_tokens
            ],
            dtype=float,
        )
        query_code_tokens = _code_tokens(query)
        code_scores = np.array(
            [
                len(query_code_tokens & chunk_tokens) / max(1, len(query_code_tokens))
                for chunk_tokens in self._chunk_code_tokens
            ],
            dtype=float,
        )
        if weights is None:
            weights = (
                (0.25, 0.20, 0.15, 0.40)
                if query_code_tokens
                else (0.45, 0.35, 0.20, 0.0)
            )
        if len(weights) != 4 or any(weight < 0 for weight in weights):
            raise ValueError("weights doit contenir quatre poids positifs ou nuls")
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("la somme des poids doit etre strictement positive")
        word_weight, char_weight, exact_weight, code_weight = (
            weight / total_weight for weight in weights
        )
        combined = (
            word_weight * word_scores
            + char_weight * char_scores
            + exact_weight * exact_scores
            + code_weight * code_scores
        )
        if allowed_chunk_ids is None:
            eligible = np.arange(len(self._chunks), dtype=int)
        else:
            # Preserve catalogue order even when callers pass an unordered set.
            # Stable score sorting can otherwise make tied passages depend on
            # PYTHONHASHSEED and produce different retrieval contexts.
            eligible = np.asarray(
                sorted(
                    self._chunk_index[chunk_id]
                    for chunk_id in allowed_chunk_ids
                    if chunk_id in self._chunk_index
                ),
                dtype=int,
            )
        if eligible.size == 0:
            return []
        local_order = np.argsort(-combined[eligible], kind="stable")[: min(top_k, eligible.size)]
        ordered = eligible[local_order]
        return [
            RankedChunk(
                chunk=self._chunks[index],
                score=float(combined[index]),
                word_score=float(word_scores[index]),
                char_score=float(char_scores[index]),
                exact_score=float(exact_scores[index]),
                code_score=float(code_scores[index]),
                lexical_rank=rank,
            )
            for rank, index in enumerate(ordered, start=1)
        ]


def reciprocal_rank_fusion(
    lexical: list[RankedChunk],
    dense: list[RankedChunk],
    *,
    top_k: int,
    rrf_k: int = 60,
    lexical_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> list[RankedChunk]:
    """Fuse independent lexical and dense rankings without score calibration."""

    if top_k <= 0:
        raise ValueError("top_k doit etre strictement positif")
    if rrf_k < 0:
        raise ValueError("rrf_k doit etre positif ou nul")
    if lexical_weight < 0 or dense_weight < 0 or lexical_weight + dense_weight <= 0:
        raise ValueError("poids de fusion invalides")

    entries: dict[str, dict[str, object]] = {}

    def add(items: list[RankedChunk], *, kind: str, weight: float) -> None:
        for rank, item in enumerate(items, start=1):
            entry = entries.setdefault(
                item.chunk.chunk_id,
                {
                    "chunk": item.chunk,
                    "fused": 0.0,
                    "word": 0.0,
                    "char": 0.0,
                    "exact": 0.0,
                    "code": 0.0,
                    "dense": 0.0,
                    "lexical_rank": None,
                    "dense_rank": None,
                },
            )
            entry["fused"] = float(entry["fused"]) + weight / (rrf_k + rank)
            entry["word"] = max(float(entry["word"]), item.word_score)
            entry["char"] = max(float(entry["char"]), item.char_score)
            entry["exact"] = max(float(entry["exact"]), item.exact_score)
            entry["code"] = max(float(entry["code"]), item.code_score)
            entry["dense"] = max(float(entry["dense"]), item.dense_score)
            entry[f"{kind}_rank"] = rank

    add(lexical, kind="lexical", weight=lexical_weight)
    add(dense, kind="dense", weight=dense_weight)
    ordered = sorted(
        entries.values(),
        key=lambda entry: (
            -float(entry["fused"]),
            entry["chunk"].page_number,
            entry["chunk"].chunk_id,
        ),
    )[: min(top_k, len(entries))]
    return [
        RankedChunk(
            chunk=entry["chunk"],
            score=float(entry["fused"]),
            word_score=float(entry["word"]),
            char_score=float(entry["char"]),
            exact_score=float(entry["exact"]),
            code_score=float(entry["code"]),
            dense_score=float(entry["dense"]),
            lexical_rank=entry["lexical_rank"],
            dense_rank=entry["dense_rank"],
            fused_score=float(entry["fused"]),
        )
        for entry in ordered
    ]



def weighted_rank_fusion(
    lanes: list[tuple[str, list[RankedChunk], float]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[RankedChunk]:
    """Fuse any number of ranked retrieval lanes with explicit weights."""

    if top_k <= 0:
        raise ValueError("top_k doit etre strictement positif")
    if rrf_k < 0:
        raise ValueError("rrf_k doit etre positif ou nul")
    if not lanes or all(weight <= 0 or not items for _name, items, weight in lanes):
        return []
    entries: dict[str, dict[str, object]] = {}
    for _name, items, weight in lanes:
        if weight < 0:
            raise ValueError("les poids de fusion ne peuvent pas etre negatifs")
        if weight == 0:
            continue
        for rank, item in enumerate(items, start=1):
            entry = entries.setdefault(
                item.chunk.chunk_id,
                {
                    "chunk": item.chunk,
                    "fused": 0.0,
                    "word": 0.0,
                    "char": 0.0,
                    "exact": 0.0,
                    "code": 0.0,
                    "dense": 0.0,
                    "lexical_rank": None,
                    "dense_rank": None,
                },
            )
            entry["fused"] = float(entry["fused"]) + weight / (rrf_k + rank)
            entry["word"] = max(float(entry["word"]), item.word_score)
            entry["char"] = max(float(entry["char"]), item.char_score)
            entry["exact"] = max(float(entry["exact"]), item.exact_score)
            entry["code"] = max(float(entry["code"]), item.code_score)
            entry["dense"] = max(float(entry["dense"]), item.dense_score)
            if item.lexical_rank is not None:
                current = entry["lexical_rank"]
                entry["lexical_rank"] = (
                    item.lexical_rank
                    if current is None
                    else min(int(current), item.lexical_rank)
                )
            if item.dense_rank is not None:
                current = entry["dense_rank"]
                entry["dense_rank"] = (
                    item.dense_rank
                    if current is None
                    else min(int(current), item.dense_rank)
                )
    ordered = sorted(
        entries.values(),
        key=lambda entry: (
            -float(entry["fused"]),
            entry["chunk"].page_number,
            entry["chunk"].chunk_id,
        ),
    )[: min(top_k, len(entries))]
    return [
        RankedChunk(
            chunk=entry["chunk"],
            score=float(entry["fused"]),
            word_score=float(entry["word"]),
            char_score=float(entry["char"]),
            exact_score=float(entry["exact"]),
            code_score=float(entry["code"]),
            dense_score=float(entry["dense"]),
            lexical_rank=entry["lexical_rank"],
            dense_rank=entry["dense_rank"],
            fused_score=float(entry["fused"]),
        )
        for entry in ordered
    ]

def rerank_ranked_chunks(
    query: str,
    candidates: list[RankedChunk],
    rerank_client: object,
    *,
    top_k: int,
) -> list[RankedChunk]:
    """Apply a cross-encoder score to a bounded candidate pool."""

    if top_k <= 0:
        raise ValueError("top_k doit etre strictement positif")
    if not candidates:
        return []
    scores = rerank_client.rerank(query, [item.chunk.text for item in candidates])
    if not isinstance(scores, list) or len(scores) != len(candidates):
        raise ValueError("le reranker doit retourner un score par passage")
    output: list[RankedChunk] = []
    for base, score in zip(candidates, scores):
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("score de reranking invalide")
        output.append(
            RankedChunk(
                chunk=base.chunk,
                score=base.score,
                word_score=base.word_score,
                char_score=base.char_score,
                exact_score=base.exact_score,
                code_score=base.code_score,
                dense_score=base.dense_score,
                lexical_rank=base.lexical_rank,
                dense_rank=base.dense_rank,
                fused_score=base.fused_score,
                rerank_score=float(score),
            )
        )
    output.sort(
        key=lambda item: (
            -(item.rerank_score if item.rerank_score is not None else float("-inf")),
            -item.fused_score,
            item.chunk.page_number,
            item.chunk.chunk_id,
        )
    )
    return output[: min(top_k, len(output))]


def rerank_ranked_chunks_batched(
    query: str,
    candidates: list[RankedChunk],
    rerank_client: object,
    *,
    top_k: int,
    batch_size: int = 48,
    metadata: dict[str, object] | None = None,
) -> list[RankedChunk]:
    """Rerank large pools with bounded batches and one final tournament pass."""

    if top_k <= 0:
        raise ValueError("top_k doit etre strictement positif")
    if batch_size <= 1:
        raise ValueError("batch_size doit etre superieur a 1")
    if not candidates:
        if metadata is not None:
            metadata["rerank_calls"] = int(metadata.get("rerank_calls", 0))
            metadata["rerank_batch_count"] = 0
            metadata["rerank_batches"] = 0
            metadata["rerank_rounds"] = 0
            metadata["rerank_batch_size"] = batch_size
        return []
    if len(candidates) <= batch_size:
        result = rerank_ranked_chunks(
            query, candidates, rerank_client, top_k=min(top_k, len(candidates))
        )
        if metadata is not None:
            metadata["rerank_calls"] = int(metadata.get("rerank_calls", 0)) + 1
            metadata["rerank_batch_count"] = 1
            metadata["rerank_batches"] = 1
            metadata["rerank_rounds"] = 1
            metadata["rerank_batch_size"] = batch_size
            metadata["rerank_input_candidates"] = len(candidates)
            metadata["rerank_tournament"] = False
        return result

    batches = [
        candidates[index:index + batch_size]
        for index in range(0, len(candidates), batch_size)
    ]
    per_batch_keep = max(
        top_k, min(batch_size, max(4, batch_size // len(batches)))
    )
    semifinalists: list[RankedChunk] = []
    calls = 0
    for batch in batches:
        semifinalists.extend(
            rerank_ranked_chunks(
                query,
                batch,
                rerank_client,
                top_k=min(per_batch_keep, len(batch)),
            )
        )
        calls += 1

    semifinalists.sort(
        key=lambda item: (
            -(item.rerank_score if item.rerank_score is not None else float("-inf")),
            -item.fused_score,
            item.chunk.page_number,
            item.chunk.chunk_id,
        )
    )
    tournament = semifinalists[:batch_size]
    final = rerank_ranked_chunks(
        query, tournament, rerank_client, top_k=min(top_k, len(tournament))
    )
    calls += 1
    if metadata is not None:
        metadata["rerank_calls"] = int(metadata.get("rerank_calls", 0)) + calls
        metadata["rerank_batch_count"] = len(batches)
        metadata["rerank_batches"] = calls
        metadata["rerank_rounds"] = 2
        metadata["rerank_batch_size"] = batch_size
        metadata["rerank_input_candidates"] = len(candidates)
        metadata["rerank_tournament"] = True
        metadata["rerank_semifinalists"] = len(semifinalists)
    return final
