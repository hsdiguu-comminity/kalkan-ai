"""Tekrarsız yazma, güncelleme ve hata sonrası kurtarma davranışları."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from app.rag.chunking import ChunkingConfig, SourceDocument
from app.rag.embeddings import EmbeddingSpec
from app.rag.ingest import (
    ChromaIngestWriter,
    ingest_directory,
    ingest_document,
    main,
    read_document,
)


class RecordingEmbedder:
    """Yalnızca akış testi içindir; anlamsal embedding modeli değildir."""

    spec = EmbeddingSpec("test-only", "v1", dimension=2, max_tokens=20)

    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("Test modeli hatası")
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class MemoryCollection:
    """Chroma hatalarını kontrollü üretmek için bellek içi test ikizi."""

    def __init__(self, metadata: dict[str, str]) -> None:
        self.metadata = metadata
        self.rows: dict[str, dict[str, Any]] = {}
        self.fail_after_upsert = False
        self.fail_delete = False

    def get(self, *, where: dict[str, str], include: list[str]) -> dict[str, list[str]]:
        return {
            "ids": [
                key
                for key, row in self.rows.items()
                if row["metadata"]["source"] == where["source"]
            ]
        }

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for key, vector, text, metadata in zip(ids, embeddings, documents, metadatas, strict=True):
            self.rows[key] = {"vector": vector, "text": text, "metadata": metadata}
        if self.fail_after_upsert:
            raise RuntimeError("Yazma sonrası bağlantı kesintisi")

    def delete(self, *, ids: list[str]) -> None:
        if self.fail_delete:
            raise RuntimeError("Silme sırasında bağlantı kesintisi")
        for key in ids:
            self.rows.pop(key, None)


def make_writer(root: Path, embedder: RecordingEmbedder) -> ChromaIngestWriter:
    metadata = ChromaIngestWriter.collection_metadata(embedder.spec, root)
    return ChromaIngestWriter(MemoryCollection(metadata), embedder.spec, root)


def test_second_run_skips_embedding_and_does_not_duplicate(tmp_path: Path) -> None:
    embedder = RecordingEmbedder()
    writer = make_writer(tmp_path, embedder)
    document = SourceDocument("a.md", "Örnek bilgi. " * 30, "Başlık")
    first = ingest_document(document, embedder, writer, ChunkingConfig(70, 10))
    second = ingest_document(document, embedder, writer, ChunkingConfig(70, 10))
    assert first.status == "updated"
    assert second.status == "unchanged"
    assert first.chunks == second.chunks == len(writer.collection.rows)
    assert embedder.calls == 1
    assert all(row["metadata"]["title"] == "Başlık" for row in writer.collection.rows.values())


def test_update_removes_obsolete_chunks_without_touching_other_sources(tmp_path: Path) -> None:
    embedder = RecordingEmbedder()
    writer = make_writer(tmp_path, embedder)
    ingest_document(SourceDocument("a.md", "Önceki metin " * 50), embedder, writer)
    ingest_document(SourceDocument("b.md", "Diğer kaynak"), embedder, writer)
    previous = writer.ids_for_source("a.md")
    unrelated = writer.ids_for_source("b.md")
    result = ingest_document(SourceDocument("a.md", "Yeni kısa metin"), embedder, writer)
    assert result.removed_chunks == len(previous)
    assert not previous.intersection(writer.collection.rows)
    assert writer.ids_for_source("b.md") == unrelated
    assert len(writer.ids_for_source("a.md")) == 1


def test_empty_document_clears_only_its_previous_chunks(tmp_path: Path) -> None:
    embedder = RecordingEmbedder()
    writer = make_writer(tmp_path, embedder)
    ingest_document(SourceDocument("a.md", "Eski metin"), embedder, writer)
    result = ingest_document(SourceDocument("a.md", " \n"), embedder, writer)
    assert result.removed_chunks == 1
    assert writer.ids_for_source("a.md") == set()
    assert embedder.calls == 1


def test_embedding_failure_preserves_previous_document(tmp_path: Path) -> None:
    embedder = RecordingEmbedder()
    writer = make_writer(tmp_path, embedder)
    ingest_document(SourceDocument("a.md", "Eski metin"), embedder, writer)
    previous = dict(writer.collection.rows)
    embedder.fail = True
    with pytest.raises(RuntimeError):
        ingest_document(SourceDocument("a.md", "Yeni metin"), embedder, writer)
    assert writer.collection.rows == previous


@pytest.mark.parametrize("failure", ["fail_after_upsert", "fail_delete"])
def test_retry_repairs_interrupted_update(tmp_path: Path, failure: str) -> None:
    embedder = RecordingEmbedder()
    writer = make_writer(tmp_path, embedder)
    ingest_document(SourceDocument("a.md", "Eski metin"), embedder, writer)
    old_ids = writer.ids_for_source("a.md")
    updated = SourceDocument("a.md", "Yeni metin")
    setattr(writer.collection, failure, True)
    with pytest.raises(RuntimeError):
        ingest_document(updated, embedder, writer)
    assert old_ids.issubset(writer.ids_for_source("a.md"))
    setattr(writer.collection, failure, False)
    result = ingest_document(updated, embedder, writer)
    assert result.status == "updated"
    assert len(writer.collection.rows) == 1
    assert not old_ids.intersection(writer.collection.rows)


def test_invalid_vectors_cannot_modify_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = RecordingEmbedder()
    writer = make_writer(tmp_path, embedder)
    ingest_document(SourceDocument("a.md", "Eski metin"), embedder, writer)
    previous = dict(writer.collection.rows)
    monkeypatch.setattr(embedder, "embed_documents", lambda texts: [[float("nan"), 0.0]])
    with pytest.raises(ValueError):
        ingest_document(SourceDocument("a.md", "Yeni metin"), embedder, writer)
    assert writer.collection.rows == previous


def test_incompatible_model_or_root_is_rejected(tmp_path: Path) -> None:
    embedder = RecordingEmbedder()
    writer = make_writer(tmp_path, embedder)
    with pytest.raises(ValueError, match="sözleşmesi"):
        ChromaIngestWriter(writer.collection, replace(embedder.spec, revision="v2"), tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="sözleşmesi"):
        ChromaIngestWriter(writer.collection, embedder.spec, other)
    embedder.spec = replace(embedder.spec, revision="v2")
    with pytest.raises(ValueError, match="sözleşmesi"):
        ingest_document(SourceDocument("a.md", "Metin"), embedder, writer)


def test_directory_reads_utf8_and_preserves_relative_paths(tmp_path: Path) -> None:
    folder = tmp_path / "sms"
    folder.mkdir()
    (folder / "banka.MD").write_text("\ufeff# Başlık\r\nİşlem şüphesi", encoding="utf-8")
    (tmp_path / "notlar.txt").write_text("İkinci belge", encoding="utf-8")
    (tmp_path / "ignored.pdf").write_bytes(b"not a PDF")
    embedder = RecordingEmbedder()
    writer = make_writer(tmp_path, embedder)
    results = ingest_directory(tmp_path, embedder, writer)
    assert [result.source for result in results] == ["notlar.txt", "sms/banka.MD"]
    document = read_document(folder / "banka.MD", tmp_path)
    assert document.title == "Başlık"
    assert not document.text.startswith("\ufeff")


def test_missing_oversized_outside_and_non_utf8_sources(tmp_path: Path) -> None:
    root = tmp_path / "kb"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("Dış belge", encoding="utf-8")
    with pytest.raises(ValueError, match="kökün"):
        read_document(outside, root)
    with pytest.raises(FileNotFoundError):
        read_document(root / "missing.txt", root)
    inside = root / "inside.txt"
    inside.write_text("uzun metin", encoding="utf-8")
    with pytest.raises(ValueError, match="boyut"):
        read_document(inside, root, max_file_bytes=3)
    inside.write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeError):
        read_document(inside, root)


def test_directory_failure_reports_source_without_raw_content(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(b"private-data\xff")
    embedder = RecordingEmbedder()
    writer = make_writer(tmp_path, embedder)
    with pytest.raises(RuntimeError, match="a.txt") as failure:
        ingest_directory(tmp_path, embedder, writer)
    assert "private-data" not in str(failure.value)
    assert writer.collection.rows == {}


def test_cli_rejects_invalid_configuration(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--source", "missing", "--chunk-size", "0"]) == 1
    assert "başarısız" in capsys.readouterr().err
