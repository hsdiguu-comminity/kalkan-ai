# Kalkan – Yapay Zeka

Kalkan Asistanı (LLM + RAG), Türkçe dolandırıcılık metin sınıflandırıcı, OCR ve URL risk modeli; FastAPI servisi olarak yayına alınır.

**Sorumlu:** Furkan Erden · **Dil:** Python · **Stack:** PyTorch, Hugging Face, FastAPI

## Klasör yapısı

```
app/                    AI servisi (FastAPI)
├── assistant/          LLM asistanı, araç çağrıları, insana devir
├── rag/                Bilgi tabanı indeksleme & arama
├── classifier/         güvenli / şüpheli / dolandırıcılık + kategori (BERTurk)
├── ocr/                Ekran görüntüsünden metin çıkarma
├── url_model/          Alan adı özelliklerinden risk skoru
└── schemas/            İstek/yanıt şemaları (risk_level + reasons)
training/               Eğitim betikleri
evaluation/             Test setleri ve metrikler
notebooks/              Deneyler
models/  data/          Yerel – repoya eklenmez
tests/                  pytest
```

**MVP hedefleri:** F1 ≥ 0.85 · URL ROC-AUC ≥ 0.90 · asistan doğruluğu ≥ %85 · sınıflandırma < 1 sn

## Çalışma kuralları

- `main` korumalıdır; her değişiklik ayrı bir branch ve pull request ile gelir, en az 1 onay gerekir.
- Branch adları: `feature/<alan>-<kısa-açıklama>`, `fix/<alan>-<kısa-açıklama>`
- Commit mesajları: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- `.env` dosyaları repoya eklenmez; örnek değerler `.env.example` içinde tutulur.
- Projenin tamamı için bkz. [`CLAUDE.md`](CLAUDE.md).
