"""RAG parçalama sözleşmesi ve veri kaybı kontrolleri."""

from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import pytest

from app.rag.chunking import ChunkingConfig, SourceDocument, chunk_document, normalize_text

FIXTURES = Path(__file__).parent / "fixtures" / "rag"


@pytest.mark.parametrize("text", ["", " \t\n\r\n", "\ufeff"])
def test_empty_documents_have_no_chunks(text: str) -> None:
    assert chunk_document(SourceDocument("empty.txt", text)) == []


def test_turkish_text_normalization_and_offsets() -> None:
    raw = "\ufeff  I İ ı i ş ç ğ ö ü\r\n\r\nI\u0307şlem.  "
    normalized = normalize_text(raw)
    chunk = chunk_document(SourceDocument("turkce.md", raw, "Türkçe"))[0]
    assert chunk.text == "I İ ı i ş ç ğ ö ü\n\nİşlem."
    assert normalized[chunk.char_start : chunk.char_end] == chunk.text
    assert chunk.source == "turkce.md"
    assert chunk.title == "Türkçe"
    assert chunk.chunk_index == 0
    assert asdict(chunk)["text"] == chunk.text


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)])
def test_invalid_configuration(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(size, overlap)


def test_configuration_rejects_boolean_sizes() -> None:
    with pytest.raises(TypeError):
        ChunkingConfig(True, 0)


def test_missing_source_is_rejected() -> None:
    with pytest.raises(ValueError):
        SourceDocument("  ", "Örnek")


def test_exact_overlap_for_text_without_boundaries() -> None:
    chunks = chunk_document(SourceDocument("long.txt", "abcdefghij" * 4), ChunkingConfig(12, 4))
    assert [chunk.char_start for chunk in chunks] == [0, 8, 16, 24, 32]
    assert all(left.text[-4:] == right.text[:4] for left, right in pairwise(chunks))
    assert chunks[-1].char_end == 40


@pytest.mark.parametrize(
    "text,expected",
    [
        ("a" * 23 + "\n\n" + "b" * 14 + " " + "c" * 30, "a" * 23),
        ("a" * 24 + ". " + "b" * 13 + " " + "c" * 30, "a" * 24 + "."),
        ("a" * 30 + " " + "b" * 40, "a" * 30),
    ],
)
def test_prefers_natural_boundaries(text: str, expected: str) -> None:
    chunks = chunk_document(SourceDocument("boundaries.txt", text), ChunkingConfig(40, 5))
    assert chunks[0].text == expected


@pytest.mark.parametrize("size,overlap", [(1, 0), (12, 0), (12, 11), (100, 15), (1000, 150)])
@pytest.mark.parametrize("filename", ["synthetic_bank_sms.md", "synthetic_delivery.txt"])
def test_samples_preserve_all_non_whitespace_content(
    size: int, overlap: int, filename: str
) -> None:
    text = (FIXTURES / filename).read_text(encoding="utf-8")
    normalized = normalize_text(text)
    chunks = chunk_document(SourceDocument(filename, text), ChunkingConfig(size, overlap))
    covered: set[int] = set()
    for index, chunk in enumerate(chunks):
        assert 0 < len(chunk.text) <= size
        assert chunk.text == chunk.text.strip()
        assert chunk.text == normalized[chunk.char_start : chunk.char_end]
        assert chunk.chunk_index == index
        assert chunk.source == filename
        covered.update(range(chunk.char_start, chunk.char_end))
    assert all(index in covered for index, char in enumerate(normalized) if not char.isspace())
    assert all(left.char_start < right.char_start for left, right in pairwise(chunks))
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_ids_are_repeatable_and_source_aware() -> None:
    document = SourceDocument("bank.md", "Örnek belge.")
    original = chunk_document(document)
    assert original == chunk_document(document)
    other = chunk_document(SourceDocument("other.md", document.text))
    assert original[0].chunk_id != other[0].chunk_id
    assert original[0].document_id != other[0].document_id
    changed = chunk_document(SourceDocument(document.source, "Güncellenmiş belge."))
    assert original[0].document_id == changed[0].document_id
    assert original[0].chunk_id != changed[0].chunk_id


def test_equivalent_line_endings_produce_same_ids() -> None:
    assert chunk_document(SourceDocument("a.txt", "İlk\r\nİkinci")) == chunk_document(
        SourceDocument("a.txt", "İlk\nİkinci")
    )
