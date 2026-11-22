"""Önbellek hazırsa gerçek E5 modeliyle çevrimdışı entegrasyon; indirme yapmaz."""

import importlib.util
import math
from pathlib import Path

import pytest

from app.rag.embeddings import E5_MODEL, E5_REVISION, E5Embedder, validate_vectors

CACHE = Path(__file__).resolve().parents[1] / "models" / "huggingface"
WEIGHTS = CACHE / ("models--" + E5_MODEL.replace("/", "--")) / "snapshots" / E5_REVISION
pytestmark = pytest.mark.skipif(
    not (WEIGHTS / "model.safetensors").is_file()
    or importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None,
    reason="Gerçek model testi için yerel E5 önbelleği ve ML paketleri gerekir.",
)


@pytest.fixture(scope="module")
def model() -> E5Embedder:
    return E5Embedder(batch_size=2, cache_dir=CACHE, local_files_only=True)


def test_real_model_produces_normalized_vectors_and_padding_is_consistent(
    model: E5Embedder,
) -> None:
    texts = ["Banka mesajı", "Kargo teslimatında adres bilgisi kontrol edilir. " * 4]
    batch = model.embed_documents(texts)
    validate_vectors(batch, 2, model.spec)
    single = model.embed_documents([texts[0]])[0]
    assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(batch[0], single, strict=True))
    query = model.embed_query(texts[0])
    validate_vectors([query], 1, model.spec)
    assert query != batch[0]


def test_real_model_rejects_overlong_text_without_truncation(model: E5Embedder) -> None:
    with pytest.raises(ValueError, match="512 token"):
        model.embed_documents(["bankacılık " * 600])


def test_real_ingest_to_independent_reader_contract(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    from app.rag.chunking import ChunkingConfig
    from scripts.rag_acceptance import run_acceptance

    report = run_acceptance(
        source=Path(__file__).parent / "fixtures" / "rag",
        store=tmp_path / "chroma",
        collection_name="test-reader-contract",
        cache_dir=CACHE,
        query="Örnek Kargo teslimat mesajı",
        config=ChunkingConfig(300, 50),
        expected_source="synthetic_delivery.txt",
    )
    assert report["status"] == "passed"
    assert report["documents"] == 2
    assert report["chunks"] == 7
    assert all(report["checks"].values())
