"""Kalkan AI servisi.

Asistan, sınıflandırıcı, OCR ve URL modeli uç noktaları app/routers/ altında yazılır ve
burada include_router ile eklenir. Ana API bu servise HTTP üzerinden bağlanır.
"""

from fastapi import FastAPI

app = FastAPI(title="Kalkan AI Service", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "kalkan-ai"}


# Örnek: app.include_router(classifier.router, prefix="/classify", tags=["classifier"])
