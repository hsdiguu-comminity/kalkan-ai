"""Gerçek model/depo ile teslim kontrolü; Seyit'in arama modülü değildir.

Proje kökünde: python -m scripts.rag_acceptance --source ... --store ... --report ...
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from app.rag.chunking import ChunkingConfig, normalize_text
from app.rag.embeddings import E5Embedder, validate_vectors
from app.rag.ingest import ChromaIngestWriter, ingest_directory, read_document


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run_acceptance(
    *,
    source: Path,
    store: Path,
    collection_name: str,
    cache_dir: Path,
    query: str,
    config: ChunkingConfig,
    expected_source: str | None = None,
    offline: bool = True,
) -> dict[str, Any]:
    """İndeksleme, tekrar, bağımsız okuma ve kaynak atfını uçtan uca doğrular."""
    import chromadb
    from chromadb.config import Settings

    model = E5Embedder(cache_dir=cache_dir, local_files_only=offline)
    writer = ChromaIngestWriter.open(store, collection_name, model.spec, source)
    first = ingest_directory(source, model, writer, config)
    expected_count = sum(result.chunks for result in first)
    check(expected_count > 0, "Kontrol için en az bir boş olmayan belge gerekir.")
    check(writer.collection.count() == expected_count, "Koleksiyonda beklenmeyen kayıt var.")
    first_ids = set(writer.collection.get(include=[])["ids"])
    repeated = ingest_directory(source, model, writer, config)
    check(
        all(result.status == "unchanged" for result in repeated), "Tekrar giriş değişiklik yaptı."
    )
    check(set(writer.collection.get(include=[])["ids"]) == first_ids, "Parça kimlikleri değişti.")

    # Okuyucu get_collection kullanır; eksik koleksiyonu sessizce oluşturmaz.
    client = chromadb.PersistentClient(
        path=str(store), settings=Settings(anonymized_telemetry=False)
    )
    collection: Any = client.get_collection(collection_name, embedding_function=None)
    check(
        (collection.metadata or {}).get("embedding_signature") == model.spec.signature,
        "Okuyucu ve indeks model sözleşmeleri farklı.",
    )
    stored = collection.get(include=["documents", "metadatas", "embeddings"])
    check(stored["embeddings"] is not None, "Kayıtlı vektör bulunamadı.")
    vectors = [[float(value) for value in row] for row in stored["embeddings"]]
    validate_vectors(vectors, expected_count, model.spec)
    source_texts = {
        result.source: normalize_text(read_document(source / result.source, source).text)
        for result in first
    }
    for text, metadata in zip(stored["documents"], stored["metadatas"], strict=True):
        check(metadata["source"] in source_texts, "Parçanın kaynağı bulunamadı.")
        original = source_texts[metadata["source"]]
        check(
            original[metadata["char_start"] : metadata["char_end"]] == text,
            "Kayıtlı metin kaynak konumlarıyla eşleşmiyor.",
        )
        check(
            metadata["embedding_signature"] == model.spec.signature,
            "Parçanın model kimliği uyuşmuyor.",
        )

    result = collection.query(
        query_embeddings=[model.embed_query(query)],
        n_results=min(3, expected_count),
        include=["metadatas", "distances"],
    )
    hits = [
        {"chunk_id": key, "source": metadata["source"], "cosine_distance": distance}
        for key, metadata, distance in zip(
            result["ids"][0], result["metadatas"][0], result["distances"][0], strict=True
        )
    ]
    check(bool(hits), "Sorgu sonucu boş.")
    if expected_source is not None:
        check(
            any(hit["source"] == expected_source for hit in hits),
            "Beklenen kaynak ilk üç sonuçta bulunamadı.",
        )
    return {
        "status": "passed",
        "checked_at": datetime.now(UTC).isoformat(),
        "source_root": str(source.resolve()),
        "store": str(store.resolve()),
        "collection": collection_name,
        "model": asdict(model.spec),
        "embedding_signature": model.spec.signature,
        "chunking": asdict(config),
        "documents": len(first),
        "chunks": expected_count,
        "first_run": [asdict(item) for item in first],
        "second_run": [asdict(item) for item in repeated],
        "checks": {
            "real_embeddings": True,
            "no_duplicate_on_repeat": True,
            "reopened_collection_readable": True,
            "source_offsets_match": True,
            "query_returns_sources": True,
        },
        "sample_hits": hits,
        "versions": {name: version(name) for name in ("torch", "transformers", "chromadb")},
        "limitations": [
            "Bu kontrol Seyit'in kodunu çalıştırmaz.",
            "Kaynakların ekip onayı ve gerçek soru setinde kalite ölçümü ayrıca gerekir.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG teslim ve okuyucu uyumluluk kontrolü")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--collection", default="kalkan-acceptance")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument(
        "--query", default="Şüpheli banka mesajındaki bağlantı nasıl değerlendirilir?"
    )
    parser.add_argument("--expected-source")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_acceptance(
            source=args.source,
            store=args.store,
            collection_name=args.collection,
            cache_dir=args.cache_dir,
            query=args.query,
            config=ChunkingConfig(args.chunk_size, args.chunk_overlap),
            expected_source=args.expected_source,
            offline=not args.allow_download,
        )
    except Exception as exc:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "checked_at": datetime.now(UTC).isoformat(),
                    "error_type": type(exc).__name__,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Teslim kontrolü başarısız: {type(exc).__name__}", file=sys.stderr)
        return 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Kontrol başarılı: {report['documents']} belge, {report['chunks']} parça.")
    print(f"Rapor: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
