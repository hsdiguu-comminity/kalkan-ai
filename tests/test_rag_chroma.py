"""Chroma kurulu ortamlarda gerçek kalıcı depo entegrasyonu."""

from pathlib import Path

import pytest

from app.rag.chunking import SourceDocument
from app.rag.ingest import ChromaIngestWriter, ingest_document
from tests.test_rag_ingest import RecordingEmbedder

pytest.importorskip("chromadb")


def test_chroma_persists_updates_and_rejects_foreign_collections(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    root.mkdir()
    embedder = RecordingEmbedder()
    path = tmp_path / "chroma"
    writer = ChromaIngestWriter.open(path, "test-knowledge", embedder.spec, root)
    document = SourceDocument("bank.md", "İlk belge", "Başlık")
    assert ingest_document(document, embedder, writer).status == "updated"
    reopened = ChromaIngestWriter.open(path, "test-knowledge", embedder.spec, root)
    assert ingest_document(document, embedder, reopened).status == "unchanged"
    assert reopened.collection.count() == 1
    updated = SourceDocument("bank.md", "Değişen belge", "Başlık")
    assert ingest_document(updated, embedder, reopened).removed_chunks == 1
    result = reopened.collection.get(include=["documents", "embeddings", "metadatas"])
    assert result["documents"] == [updated.text]
    assert result["embeddings"].shape == (1, 2)
    assert result["metadatas"][0]["source"] == "bank.md"
    ingest_document(SourceDocument("bank.md", ""), embedder, reopened)
    assert reopened.collection.count() == 0

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    client.create_collection("foreign-collection", embedding_function=None)
    with pytest.raises(ValueError, match="sözleşmesi"):
        ChromaIngestWriter.open(path, "foreign-collection", embedder.spec, root)
