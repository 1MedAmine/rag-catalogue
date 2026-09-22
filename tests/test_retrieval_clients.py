from __future__ import annotations

from dataclasses import dataclass

import pytest
import requests

from rag_catalogue.retrieval_clients import (
    DEFAULT_EMBEDDING_URL,
    DEFAULT_RERANK_URL,
    NvidiaEmbeddingClient,
    NvidiaRerankClient,
    RetrievalAPIError,
)


@dataclass
class _Response:
    body: object
    status_code: int = 200
    text: str = ""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = self  # type: ignore[assignment]
            raise error

    def json(self) -> object:
        return self.body


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def test_embedding_client_uses_query_and_passage_contracts() -> None:
    session = _Session([
        _Response({"data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]}),
        _Response({"data": [{"index": 0, "embedding": [0.5, 0.5]}]}),
    ])
    client = NvidiaEmbeddingClient(api_key="secret", session=session)

    passages = client.embed_passages(["alpha", "beta"])
    query = client.embed_query("question")

    assert passages == [[1.0, 0.0], [0.0, 1.0]]
    assert query == [0.5, 0.5]
    assert session.calls[0]["url"] == DEFAULT_EMBEDDING_URL
    assert session.calls[0]["json"] == {
        "model": "nvidia/nemotron-3-embed-1b",
        "input": ["alpha", "beta"],
        "input_type": "passage",
        "encoding_format": "float",
        "truncate": "END",
    }
    assert session.calls[1]["json"] == {
        "model": "nvidia/nemotron-3-embed-1b",
        "input": "question",
        "input_type": "query",
        "encoding_format": "float",
        "truncate": "END",
    }


def test_embedding_client_batches_passages() -> None:
    session = _Session([
        _Response({"data": [
            {"index": 0, "embedding": [1.0]},
            {"index": 1, "embedding": [2.0]},
        ]}),
        _Response({"data": [{"index": 0, "embedding": [3.0]}]}),
    ])
    client = NvidiaEmbeddingClient(api_key="secret", session=session, batch_size=2)

    result = client.embed_passages(["a", "b", "c"])

    assert result == [[1.0], [2.0], [3.0]]
    assert [call["json"]["input"] for call in session.calls] == [["a", "b"], ["c"]]


def test_embedding_error_includes_nvidia_message() -> None:
    session = _Session([
        _Response(
            {"object": "error", "message": "model not found"},
            status_code=404,
            text='{"message":"model not found"}',
        )
    ])
    client = NvidiaEmbeddingClient(api_key="secret", model="wrong", session=session)

    with pytest.raises(RetrievalAPIError, match="model not found"):
        client.embed_query("question")


def test_rerank_client_uses_hosted_model_specific_endpoint_and_aligns_scores() -> None:
    session = _Session([
        _Response({"rankings": [
            {"index": 2, "logit": 4.5},
            {"index": 0, "logit": 1.25},
            {"index": 1, "logit": -2.0},
        ]})
    ])
    client = NvidiaRerankClient(api_key="secret", session=session)

    scores = client.rerank("query", ["first", "second", "third"])

    assert scores == [1.25, -2.0, 4.5]
    assert session.calls[0]["url"] == DEFAULT_RERANK_URL
    assert session.calls[0]["json"] == {
        "model": "nvidia/llama-nemotron-rerank-1b-v2",
        "query": {"text": "query"},
        "passages": [
            {"text": "first"},
            {"text": "second"},
            {"text": "third"},
        ],
        "truncate": "END",
    }


def test_rerank_client_rejects_duplicate_indexes() -> None:
    client = NvidiaRerankClient(
        api_key="secret",
        session=_Session([_Response({"rankings": [
            {"index": 0, "logit": 1.0},
            {"index": 0, "logit": 0.5},
        ]})]),
    )

    with pytest.raises(RetrievalAPIError, match="reranking"):
        client.rerank("query", ["first", "second"])
