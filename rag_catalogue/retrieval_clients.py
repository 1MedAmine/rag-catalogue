"""NVIDIA NIM clients for text embeddings and passage reranking."""

from __future__ import annotations

from math import isfinite
from typing import Any, Iterable

import requests


DEFAULT_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
DEFAULT_RERANK_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
DEFAULT_EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"
DEFAULT_RERANK_URL = (
    "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
    "llama-nemotron-rerank-1b-v2/reranking"
)


class RetrievalAPIError(RuntimeError):
    """A NVIDIA retrieval response cannot be used safely."""


def _clean_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} ne peut pas etre vide")
    return value.strip()


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _response_detail(response: object | None) -> str:
    if response is None:
        return ""
    try:
        body = response.json()  # type: ignore[attr-defined]
    except Exception:
        body = None
    if isinstance(body, dict):
        for key in ("message", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        if isinstance(error, str) and error.strip():
            return error.strip()
    text = getattr(response, "text", "")
    return " ".join(str(text or "").split())[:500]


def _numeric_vector(raw: object, *, expected_dimension: int | None = None) -> list[float]:
    if not isinstance(raw, list) or not raw:
        raise RetrievalAPIError("embedding NVIDIA invalide")
    vector: list[float] = []
    for value in raw:
        if isinstance(value, bool) or type(value) not in (int, float):
            raise RetrievalAPIError("embedding NVIDIA invalide")
        number = float(value)
        if not isfinite(number):
            raise RetrievalAPIError("embedding NVIDIA invalide")
        vector.append(number)
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise RetrievalAPIError("dimensions d'embedding NVIDIA incoherentes")
    return vector


class NvidiaEmbeddingClient:
    """Client for the hosted or self-hosted NVIDIA embeddings endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
        api_url: str = DEFAULT_EMBEDDING_URL,
        batch_size: int = 32,
        timeout: float = 180.0,
        session: Any | None = None,
    ) -> None:
        self.api_key = _clean_text(api_key, label="api_key")
        self.model = _clean_text(model, label="modele d'embedding")
        self.api_url = _clean_text(api_url, label="URL embedding").rstrip("/")
        if batch_size <= 0:
            raise ValueError("batch_size doit etre strictement positif")
        if timeout <= 0:
            raise ValueError("timeout doit etre strictement positif")
        self.batch_size = int(batch_size)
        self.timeout = float(timeout)
        self._session = requests.Session() if session is None else session

    def _embed(self, input_value: str | list[str], *, input_type: str) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": input_value,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        }
        try:
            response = self._session.post(
                self.api_url,
                headers=_headers(self.api_key),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            response = getattr(exc, "response", None)
            detail = _response_detail(response)
            raise RetrievalAPIError(
                "appel embedding NVIDIA impossible: "
                f"modele={self.model}, url={self.api_url}"
                + (f", detail={detail}" if detail else "")
            ) from exc
        except ValueError as exc:
            raise RetrievalAPIError("reponse embedding NVIDIA non JSON") from exc

        expected_count = len(input_value) if isinstance(input_value, list) else 1
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != expected_count:
            raise RetrievalAPIError("reponse embedding NVIDIA incomplete")
        indexed: dict[int, list[float]] = {}
        dimension: int | None = None
        for item in data:
            if not isinstance(item, dict):
                raise RetrievalAPIError("reponse embedding NVIDIA invalide")
            index = item.get("index")
            if type(index) is not int or not (0 <= index < expected_count) or index in indexed:
                raise RetrievalAPIError("index d'embedding NVIDIA invalide")
            vector = _numeric_vector(item.get("embedding"), expected_dimension=dimension)
            dimension = len(vector) if dimension is None else dimension
            indexed[index] = vector
        if set(indexed) != set(range(expected_count)):
            raise RetrievalAPIError("ordre d'embedding NVIDIA incomplet")
        return [indexed[index] for index in range(expected_count)]

    def embed_passages(self, texts: Iterable[str]) -> list[list[float]]:
        passages = list(texts)
        if not passages:
            raise ValueError("au moins un passage est requis")
        cleaned = [_clean_text(text, label="passage") for text in passages]
        vectors: list[list[float]] = []
        dimension: int | None = None
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start:start + self.batch_size]
            batch_vectors = self._embed(batch, input_type="passage")
            for vector in batch_vectors:
                if dimension is None:
                    dimension = len(vector)
                elif len(vector) != dimension:
                    raise RetrievalAPIError("dimensions d'embedding NVIDIA incoherentes")
                vectors.append(vector)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed(_clean_text(text, label="requete"), input_type="query")[0]


class NvidiaRerankClient:
    """Client for NVIDIA's ranking endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_RERANK_MODEL,
        api_url: str = DEFAULT_RERANK_URL,
        timeout: float = 180.0,
        session: Any | None = None,
    ) -> None:
        self.api_key = _clean_text(api_key, label="api_key")
        self.model = _clean_text(model, label="modele de reranking")
        self.api_url = _clean_text(api_url, label="URL reranking").rstrip("/")
        if timeout <= 0:
            raise ValueError("timeout doit etre strictement positif")
        self.timeout = float(timeout)
        self._session = requests.Session() if session is None else session

    def rerank(self, query: str, passages: Iterable[str]) -> list[float]:
        cleaned_query = _clean_text(query, label="requete")
        values = list(passages)
        if not values:
            raise ValueError("au moins un passage est requis pour le reranking")
        cleaned = [_clean_text(value, label="passage") for value in values]
        payload = {
            "model": self.model,
            "query": {"text": cleaned_query},
            "passages": [{"text": value} for value in cleaned],
            "truncate": "END",
        }
        try:
            response = self._session.post(
                self.api_url,
                headers=_headers(self.api_key),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            response = getattr(exc, "response", None)
            detail = _response_detail(response)
            raise RetrievalAPIError(
                "appel reranking NVIDIA impossible: "
                f"modele={self.model}, url={self.api_url}"
                + (f", detail={detail}" if detail else "")
            ) from exc
        except ValueError as exc:
            raise RetrievalAPIError("reponse de reranking NVIDIA non JSON") from exc

        rankings = body.get("rankings") if isinstance(body, dict) else None
        if not isinstance(rankings, list) or len(rankings) != len(cleaned):
            raise RetrievalAPIError("reponse de reranking NVIDIA incomplete")
        scores: dict[int, float] = {}
        for item in rankings:
            if not isinstance(item, dict):
                raise RetrievalAPIError("reponse de reranking NVIDIA invalide")
            index = item.get("index")
            score = item.get("logit", item.get("score"))
            if (
                type(index) is not int
                or not (0 <= index < len(cleaned))
                or index in scores
                or isinstance(score, bool)
                or type(score) not in (int, float)
                or not isfinite(float(score))
            ):
                raise RetrievalAPIError("reponse de reranking NVIDIA invalide")
            scores[index] = float(score)
        if set(scores) != set(range(len(cleaned))):
            raise RetrievalAPIError("reponse de reranking NVIDIA incomplete")
        return [scores[index] for index in range(len(cleaned))]
