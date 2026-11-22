"""Model indirmeden embedding sözleşmesini doğrular."""

from dataclasses import replace

import pytest

from app.rag.embeddings import E5Embedder, EmbeddingSpec, validate_vectors

SPEC = EmbeddingSpec("test-only", "v1", dimension=2, max_tokens=20)


def test_valid_vectors_and_empty_batch() -> None:
    validate_vectors([[0.6, 0.8], [1.0, 0.0]], 2, SPEC)
    validate_vectors([], 0, SPEC)


@pytest.mark.parametrize(
    "vectors,count",
    [
        ([], 1),
        ([[1.0]], 1),
        ([[float("nan"), 0.0]], 1),
        ([[float("inf"), 0.0]], 1),
        ([[0.0, 0.0]], 1),
        ([[1.0, 1.0]], 1),
    ],
)
def test_invalid_vectors(vectors: list[list[float]], count: int) -> None:
    with pytest.raises(ValueError):
        validate_vectors(vectors, count, SPEC)


def test_spec_signature_includes_model_revision_and_query_convention() -> None:
    assert SPEC.signature == replace(SPEC).signature
    assert SPEC.signature != replace(SPEC, revision="v2").signature
    assert SPEC.signature != replace(SPEC, query_prefix="other: ").signature
    assert SPEC.signature != replace(SPEC, normalized=False).signature


def test_empty_input_does_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_load() -> None:
        pytest.fail("Boş metin için model yüklenmemelidir.")

    embedder = E5Embedder()
    monkeypatch.setattr(embedder, "_load", unexpected_load)
    assert embedder.embed_documents([]) == []
    with pytest.raises(ValueError, match="boş olmayan"):
        embedder.embed_documents(["  "])
    with pytest.raises(ValueError, match="boş olmayan"):
        embedder.embed_query("")
    with pytest.raises(ValueError, match="boş olmayan"):
        embedder.embed_documents(["\ufeff"])


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_invalid_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError):
        E5Embedder(batch_size=batch_size)


def test_mutable_model_revision_is_rejected() -> None:
    with pytest.raises(ValueError, match="commit"):
        E5Embedder(revision="main")
