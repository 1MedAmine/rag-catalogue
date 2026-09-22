"""Dense vector index and persistent cache for catalogue chunks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .pdf_text import CatalogueChunk


@dataclass(frozen=True, slots=True)
class DenseHit:
    chunk: CatalogueChunk
    score: float
    rank: int


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("matrice d'embeddings invalide")
    if not np.isfinite(values).all():
        raise ValueError("matrice d'embeddings non finie")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("les embeddings ne peuvent pas etre nuls")
    return values / norms


class DenseIndex:
    """Cosine-similarity index over a fixed list of catalogue chunks."""

    def __init__(self, chunks: Sequence[CatalogueChunk], vectors: np.ndarray) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("au moins un chunk catalogue est requis")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.chunks):
            raise ValueError("un embedding est requis pour chaque chunk")
        self.matrix = _normalise_rows(matrix)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        allowed_chunk_ids: set[str] | None = None,
    ) -> list[DenseHit]:
        if top_k <= 0:
            raise ValueError("top_k doit etre strictement positif")
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.matrix.shape[1]:
            raise ValueError("dimension de requete incompatible")
        if not np.isfinite(query).all():
            raise ValueError("embedding de requete invalide")
        norm = float(np.linalg.norm(query))
        if norm <= 0:
            raise ValueError("embedding de requete nul")
        scores = self.matrix @ (query / norm)
        if allowed_chunk_ids is None:
            eligible = np.arange(len(self.chunks), dtype=int)
        else:
            eligible = np.asarray(
                [index for index, chunk in enumerate(self.chunks) if chunk.chunk_id in allowed_chunk_ids],
                dtype=int,
            )
        if eligible.size == 0:
            return []
        local_order = np.argsort(-scores[eligible], kind="stable")[: min(top_k, eligible.size)]
        ordered = eligible[local_order]
        return [
            DenseHit(
                chunk=self.chunks[index],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(ordered, start=1)
        ]


def _cache_key(
    catalogue: Path,
    chunks: Sequence[CatalogueChunk],
    model: str,
    *,
    max_words: int,
    overlap_words: int,
) -> str:
    stat = catalogue.stat()
    digest = hashlib.sha256()
    digest.update(str(catalogue.resolve()).encode("utf-8"))
    digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode("ascii"))
    digest.update(model.encode("utf-8"))
    digest.update(f"\0{max_words}\0{overlap_words}\0dense-v13\0".encode("ascii"))
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.section_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.kind.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_or_create_dense_index(
    catalogue_pdf: Path | str,
    chunks: Sequence[CatalogueChunk],
    embedding_client: Any,
    *,
    cache_dir: Path | str,
    max_words: int,
    overlap_words: int,
) -> tuple[DenseIndex, bool]:
    """Load cached passage vectors or call the embedding service once."""

    catalogue = Path(catalogue_pdf)
    if not catalogue.is_file():
        raise FileNotFoundError(f"PDF introuvable: {catalogue}")
    values = list(chunks)
    if not values:
        raise ValueError("au moins un chunk catalogue est requis")
    model = str(getattr(embedding_client, "model", "")).strip()
    if not model:
        raise ValueError("le client embedding doit exposer son modele")
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        "dense_"
        + _cache_key(
            catalogue,
            values,
            model,
            max_words=max_words,
            overlap_words=overlap_words,
        )
        + ".npz"
    )
    expected_ids = [chunk.chunk_id for chunk in values]
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                ids = [str(value) for value in data["chunk_ids"].tolist()]
                matrix = np.asarray(data["vectors"], dtype=np.float32)
            if ids == expected_ids and matrix.shape[0] == len(values):
                return DenseIndex(values, matrix), True
        except (OSError, ValueError, KeyError):
            pass

    vectors = embedding_client.embed_passages([chunk.text for chunk in values])
    index = DenseIndex(values, np.asarray(vectors, dtype=np.float32))
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        chunk_ids=np.asarray(expected_ids),
        vectors=index.matrix.astype(np.float32),
    )
    temporary.replace(path)
    return index, False
