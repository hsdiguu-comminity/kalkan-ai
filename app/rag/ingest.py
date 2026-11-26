"""UTF-8 bilgi tabanını parçalar, embedding üretir ve yerel Chroma'ya yazar.

Bu yazma adaptörü Hasti'nin ingest akışına aittir; Seyit'in arama katmanından
bağımsızdır. Aynı koleksiyona aynı anda yalnızca bir ingest işlemi yazmalıdır.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.rag.chunking import ChunkingConfig, SourceDocument, TextChunk, chunk_document
from app.rag.embeddings import E5Embedder, EmbeddingBackend, EmbeddingSpec, validate_vectors

MAX_FILE_BYTES = 5 * 1024 * 1024
STORE_BATCH_SIZE = 128


@dataclass(frozen=True, slots=True)
class IngestResult:
    source: str
    status: str
    chunks: int
    removed_chunks: int = 0


def read_document(path: Path, root: Path, max_file_bytes: int = MAX_FILE_BYTES) -> SourceDocument:
    """Yalnızca kök içindeki normal Markdown/metin dosyalarını okur."""
    root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(root):
        raise ValueError("Kaynak dosyası sembolik bağlantı olamaz ve kökün içinde olmalıdır.")
    if not resolved.is_file() or resolved.suffix.lower() not in {".md", ".txt"}:
        raise ValueError("Yalnızca .md ve .txt dosyaları desteklenir.")
    if max_file_bytes < 1:
        raise ValueError("Dosya boyutu sınırı pozitif olmalıdır.")
    with resolved.open("rb") as stream:
        raw = stream.read(max_file_bytes + 1)
    if len(raw) > max_file_bytes:
        raise ValueError("Kaynak dosyası boyut sınırını aşıyor.")
    text = raw.decode("utf-8-sig")
    title = resolved.stem
    if resolved.suffix.lower() == ".md":
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()[:256]
                break
    return SourceDocument(resolved.relative_to(root).as_posix(), text, title)


class ChromaIngestWriter:
    """Hazır vektörleri kaydeder; Chroma'nın varsayılan modelini kullanmaz."""

    def __init__(self, collection: Any, spec: EmbeddingSpec, source_root: Path) -> None:
        self.collection = collection
        self.spec = spec
        self.source_root = source_root.resolve(strict=True)
        expected = self.collection_metadata(spec, self.source_root)
        actual = collection.metadata or {}
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Koleksiyonun model veya kaynak sözleşmesi farklı; yeni koleksiyon adı kullanın."
            )

    @staticmethod
    def collection_metadata(spec: EmbeddingSpec, source_root: Path) -> dict[str, str]:
        return {
            "rag_schema": "kalkan-ingest-v1",
            "embedding_signature": spec.signature,
            "embedding_spec": json.dumps(asdict(spec), sort_keys=True, ensure_ascii=False),
            "source_root": str(source_root.resolve(strict=True)),
            "hnsw:space": "cosine",
        }

    @classmethod
    def open(
        cls, path: Path, name: str, spec: EmbeddingSpec, source_root: Path
    ) -> "ChromaIngestWriter":
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Yerel vektör deposu için chromadb kurulmalıdır.") from exc
        from chromadb.config import Settings

        source_root = source_root.resolve(strict=True)
        if not source_root.is_dir():
            raise ValueError("Bilgi tabanı kökü bir klasör olmalıdır.")
        client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_or_create_collection(
            name=name,
            metadata=cls.collection_metadata(spec, source_root),
            embedding_function=None,
        )
        return cls(collection, spec, source_root)

    def ids_for_source(self, source: str) -> set[str]:
        result = self.collection.get(where={"source": source}, include=[])
        return set(result["ids"])

    def replace_document(
        self, source: str, chunks: Sequence[TextChunk], vectors: list[list[float]]
    ) -> int:
        validate_vectors(vectors, len(chunks), self.spec)
        if any(chunk.source != source for chunk in chunks):
            raise ValueError("Parçaların kaynakları yazılan belgeyle eşleşmiyor.")
        new_ids = {chunk.chunk_id for chunk in chunks}
        if len(new_ids) != len(chunks):
            raise ValueError("Aynı belge içinde parça kimlikleri benzersiz olmalıdır.")
        previous_ids = self.ids_for_source(source)
        # Önce yeni veriyi yazar. Başarısız embedding/upsert eski veriyi silmez.
        for offset in range(0, len(chunks), STORE_BATCH_SIZE):
            batch = chunks[offset : offset + STORE_BATCH_SIZE]
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=vectors[offset : offset + STORE_BATCH_SIZE],
                documents=[chunk.text for chunk in batch],
                metadatas=[
                    {
                        "source": chunk.source,
                        "title": chunk.title,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "embedding_signature": self.spec.signature,
                    }
                    for chunk in batch
                ],
            )
        stale_ids = sorted(previous_ids - new_ids)
        for offset in range(0, len(stale_ids), STORE_BATCH_SIZE):
            self.collection.delete(ids=stale_ids[offset : offset + STORE_BATCH_SIZE])
        return len(stale_ids)


def ingest_document(
    document: SourceDocument,
    embedder: EmbeddingBackend,
    writer: ChromaIngestWriter,
    config: ChunkingConfig | None = None,
) -> IngestResult:
    """Aynı içerikte modeli çağırmaz; değişen belgede eski parçaları temizler."""
    if embedder.spec != writer.spec:
        raise ValueError("Embedding modeli depo sözleşmesiyle eşleşmiyor.")
    chunks = chunk_document(document, config)
    if writer.ids_for_source(document.source) == {chunk.chunk_id for chunk in chunks}:
        return IngestResult(document.source, "unchanged", len(chunks))
    vectors = embedder.embed_documents([chunk.text for chunk in chunks]) if chunks else []
    removed = writer.replace_document(document.source, chunks, vectors)
    return IngestResult(document.source, "updated", len(chunks), removed)


def ingest_directory(
    root: Path,
    embedder: EmbeddingBackend,
    writer: ChromaIngestWriter,
    config: ChunkingConfig | None = None,
    *,
    max_file_bytes: int = MAX_FILE_BYTES,
) -> list[IngestResult]:
    root = root.resolve(strict=True)
    if root != writer.source_root or not root.is_dir():
        raise ValueError("Kaynak kökü koleksiyonun kaynak köküyle eşleşmiyor.")
    results = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or path.is_dir():
            continue
        try:
            document = read_document(path, root, max_file_bytes)
            results.append(ingest_document(document, embedder, writer, config))
        except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
            source = path.relative_to(root).as_posix()
            raise RuntimeError(f"Belge işlenemedi: {source}. {type(exc).__name__}") from exc
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kalkan bilgi tabanını yerel Chroma'ya yükler.")
    parser.add_argument("--source", type=Path, required=True, help="UTF-8 belge klasörü")
    parser.add_argument("--store", type=Path, default=Path("data/rag/chroma"))
    parser.add_argument("--collection", default="kalkan-knowledge-e5-v1")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument(
        "--offline", action="store_true", help="Yalnızca önbellekteki modeli kullan"
    )
    args = parser.parse_args(argv)
    try:
        config = ChunkingConfig(args.chunk_size, args.chunk_overlap)
        embedder = E5Embedder(
            batch_size=args.batch_size,
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=args.offline,
        )
        writer = ChromaIngestWriter.open(args.store, args.collection, embedder.spec, args.source)
        results = ingest_directory(args.source, embedder, writer, config)
    except Exception as exc:
        # Üçüncü taraf istisnalar ham içerik taşıyabilir; CLI bunları yazdırmaz.
        print(f"İçe aktarma başarısız ({type(exc).__name__}).", file=sys.stderr)
        if isinstance(exc, (ValueError, RuntimeError)):
            print(str(exc) if type(exc) in {ValueError, RuntimeError} else "", file=sys.stderr)
        return 1
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
