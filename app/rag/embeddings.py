"""Yerel E5 embedding üretimi ve depoyla paylaşılan model sözleşmesi."""

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from app.rag.chunking import normalize_text

E5_MODEL = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    """Aynı koleksiyondaki belge ve sorgular bu sözleşmeyi kullanmalıdır."""

    model: str
    revision: str
    dimension: int
    max_tokens: int
    document_prefix: str = "passage: "
    query_prefix: str = "query: "
    pooling: str = "masked-mean"
    normalized: bool = True
    version: str = "embedding-v1"

    @property
    def signature(self) -> str:
        value = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


class EmbeddingBackend(Protocol):
    @property
    def spec(self) -> EmbeddingSpec: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def validate_vectors(vectors: list[list[float]], expected_count: int, spec: EmbeddingSpec) -> None:
    """Eksik, sıfır, sonsuz ve uyumsuz vektörleri yazmadan önce reddeder."""
    if len(vectors) != expected_count:
        raise ValueError("Embedding sayısı metin sayısıyla eşleşmiyor.")
    for vector in vectors:
        if len(vector) != spec.dimension:
            raise ValueError("Embedding boyutu model sözleşmesiyle eşleşmiyor.")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding yalnızca sonlu sayılar içermelidir.")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Embedding normu geçersiz.")
        if spec.normalized and not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError("Embedding birim uzunlukta olmalıdır.")


class E5Embedder:
    """Model ilk kullanımda yüklenir; belgeler harici API'ye gönderilmez.

    Ağırlıklar ilk kullanımda indirilebilir. Token sınırı aşılırsa metin sessizce
    kesilmez; daha küçük parçalarla tekrar çalıştırılmasını isteyen hata üretilir.
    """

    def __init__(
        self,
        *,
        batch_size: int = 16,
        device: str = "cpu",
        cache_dir: Path | None = None,
        local_files_only: bool = False,
        revision: str = E5_REVISION,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Embedding batch_size pozitif tam sayı olmalıdır.")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Model sürümü sabit bir 40 karakterli commit kimliği olmalıdır.")
        self.batch_size = batch_size
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._spec = EmbeddingSpec(E5_MODEL, revision, dimension=384, max_tokens=512)
        self._tokenizer: Any = None
        self._model: Any = None

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Embedding için torch ve transformers kurulmalıdır.") from exc

        options = {
            "revision": self.spec.revision,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "local_files_only": self.local_files_only,
            "trust_remote_code": False,
        }
        tokenizer = AutoTokenizer.from_pretrained(self.spec.model, **options)
        model = AutoModel.from_pretrained(self.spec.model, use_safetensors=True, **options)
        model.to(self.device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def _embed(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        prepared = []
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("Embedding girdisi boş olmayan metin olmalıdır.")
            normalized_text = normalize_text(text).strip()
            if not normalized_text:
                raise ValueError("Embedding girdisi boş olmayan metin olmalıdır.")
            prepared.append(prefix + normalized_text)
        self._load()
        import torch

        vectors: list[list[float]] = []
        for offset in range(0, len(prepared), self.batch_size):
            encoded = self._tokenizer(
                prepared[offset : offset + self.batch_size],
                padding=False,
                truncation=False,
                return_attention_mask=True,
            )
            if any(len(ids) > self.spec.max_tokens for ids in encoded["input_ids"]):
                raise ValueError(
                    "Metin modelin 512 token sınırını aşıyor; chunk_size değerini küçültün."
                )
            batch = self._tokenizer.pad(encoded, padding=True, return_tensors="pt")
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.inference_mode():
                hidden = self._model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).bool()
                pooled = hidden.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(normalized.cpu().tolist())
        validate_vectors(vectors, len(texts), self.spec)
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, self.spec.document_prefix)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self.spec.query_prefix)[0]
