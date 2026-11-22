"""Bilgi tabanı metinlerini kaynak bilgisi korunarak parçalara ayırır.

Boyutlar token değil, NFC normalizasyonundan sonraki Unicode karakter sayısıdır.
Bu modül dosya okumaz, ağ isteği yapmaz ve vektör veritabanına bağlanmaz.
"""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass

CHUNKING_VERSION = "character-v1"
_BOUNDARIES = (
    re.compile(r"\n[ \t]*\n"),
    re.compile(r"[.!?…][\"'”’»)]*\s+"),
    re.compile(r"\s+"),
)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """source: bilgi tabanı köküne göre sabit yol; text: çözülmüş belge içeriği."""

    source: str
    text: str
    title: str = ""

    def __post_init__(self) -> None:
        for name in ("source", "text", "title"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} bir metin olmalıdır.")
        if not self.source.strip():
            raise ValueError("Belge kaynağı boş olamaz.")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Örtüşme üst sınırdır; boşlukların atlanması örtüşmeyi azaltabilir."""

    chunk_size: int = 1000
    chunk_overlap: int = 150

    def __post_init__(self) -> None:
        for value in (self.chunk_size, self.chunk_overlap):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("Parça boyutu ve örtüşme tam sayı olmalıdır.")
        if self.chunk_size <= 0:
            raise ValueError("Parça boyutu sıfırdan büyük olmalıdır.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Örtüşme sıfır veya daha büyük ve parça boyutundan küçük olmalıdır.")


@dataclass(frozen=True, slots=True)
class TextChunk:
    """char_start dahil, char_end hariç; konumlar normalize edilmiş metne aittir."""

    chunk_id: str
    document_id: str
    source: str
    title: str
    chunk_index: int
    char_start: int
    char_end: int
    text: str


def normalize_text(text: str) -> str:
    """Türkçe harfleri ve paragraf yapısını korur; satır sonlarını eşitler."""
    return (
        unicodedata.normalize("NFC", text.removeprefix("\ufeff"))
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def _stable_id(parts: list[str | int]) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _chunk_end(text: str, start: int, config: ChunkingConfig) -> int:
    limit = min(start + config.chunk_size, len(text))
    if limit == len(text):
        return limit

    # Çok kısa parçaları önler; yüksek örtüşmede de ilerleme garanti edilir.
    minimum = start + max(config.chunk_size // 2, config.chunk_overlap + 1)
    for pattern in _BOUNDARIES:
        candidates = [
            match.end() for match in pattern.finditer(text, start, limit) if match.end() >= minimum
        ]
        if candidates:
            return candidates[-1]
    return limit


def chunk_document(
    document: SourceDocument, config: ChunkingConfig | None = None
) -> list[TextChunk]:
    """Belgeyi sıralı, deterministik parçalara ayırır; boş içerik için [] döner.

    Öncelik paragraf, cümle ve boşluk sınırlarıdır; uygun sınır yoksa sert kesilir.
    Örtüşme karakter tabanlıdır; bir parçanın başlangıcı sözcük içine gelebilir.
    """
    config = config if config is not None else ChunkingConfig()
    text = normalize_text(document.text)
    document_id = _stable_id([document.source])
    revision = _stable_id([text, document.title])
    chunks: list[TextChunk] = []
    start = 0

    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start == len(text):
            break

        boundary = _chunk_end(text, start, config)
        end = boundary
        while end > start and text[end - 1].isspace():
            end -= 1

        chunks.append(
            TextChunk(
                chunk_id=_stable_id([CHUNKING_VERSION, document_id, revision, start, end]),
                document_id=document_id,
                source=document.source,
                title=document.title,
                chunk_index=len(chunks),
                char_start=start,
                char_end=end,
                text=text[start:end],
            )
        )
        if boundary == len(text):
            break
        start = max(start + 1, boundary - config.chunk_overlap)

    return chunks
