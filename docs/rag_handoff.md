# Hasti — RAG indeksleme teslim dosyası

## Teslim durumu

Hasti / Kullanıcı-3 kapsamındaki metin parçalama, gerçek embedding üretimi ve
Chroma'ya veri yükleme çalışır durumdadır. Test, kullanım rehberi, okuyucu sözleşmesi
ve yeniden çalıştırılabilir kabul kontrolü hazırlanmıştır.

**Seyit'in gerçek arama koduyla entegrasyon tamamlanmadı.** 22 Kasım 2026 kontrolünde
AI deposunun `main` ve `feature/ai-assistant-tools` dallarında `app/rag/` yalnızca
`.gitkeep` içeriyordu. Yerelde de `vector_store.py`, `retriever.py`, `reranker.py`
yoktu. Aşağıdaki okuyucu kontrolü bu kodların yerini almaz.

Güvenlik ekibinin gerçek belgeleriyle teknik giriş testi yapıldı. Kullanılan
[PR #1](https://github.com/hsdiguu-comminity/kalkan-security/pull/1) açık ve
`REVIEW_REQUIRED` durumundaydı; içerik onayı verilmiş sayılmaz. Kaynak sürümü:
`711cf79a4661c9f0c68662f8083863aec898d30e`, klasör:
`tool-engine/knowledge-base/`. Dokuz Markdown dosyası (README dahil) yalnızca yerel
deneme için alındı. Asıl güvenlik deposuna değişiklik gönderilmedi.

## Dosyalar ve sorumluluklar

| Dosya | İşlev |
|---|---|
| `app/rag/chunking.py` | Metin, kaynak ve başlığı alır; konum ve kararlı kimlik taşıyan parçalar üretir. |
| `app/rag/embeddings.py` | Sabit E5 sürümüyle belge/sorgu vektörleri üretir, token ve vektör geçerliliğini kontrol eder. |
| `app/rag/ingest.py` | UTF-8 dosyaları okur, değişmeyen belgeleri atlar, güncellenen parçaları Chroma'ya yazar. |
| `scripts/rag_acceptance.py` | Gerçek modelle yükleme, tekrar giriş, yeniden açılmış koleksiyon ve sorgu/kaynak uyumunu doğrular. |
| `tests/test_rag_*.py` | Birim, hata kurtarma, gerçek Chroma ve yerel E5 entegrasyon testleri. |
| `tests/fixtures/rag/` | İki açıkça kurgusal Türkçe test belgesi. |
| `requirements-rag.txt` | RAG için çalışma bağımlılıkları. |
| `docs/rag_ingestion.md` | Kurulum, tüm ayarlar ve veri güncelleme sınırları. |

## Çalıştırma ve yeniden üretme

Komutlar proje kökünde çalıştırılır. Ortam kurulumu ve ilk model indirmesi için
[RAG kullanım rehberini](rag_ingestion.md) izleyin. Kabul kontrolü varsayılan olarak
çevrimdışıdır; önbellek yoksa `--allow-download` açıkça eklenmelidir.

Kurgusal örnekler üzerinde tam kontrol:

```powershell
.\.venv\rag-runtime\Scripts\python.exe -m scripts.rag_acceptance --source tests/fixtures/rag --store data/rag-synthetic-acceptance/chroma --collection kalkan-synthetic-acceptance --report data/rag-synthetic-acceptance/report.json --chunk-size 300 --chunk-overlap 50 --query "Örnek Kargo teslimat mesajı" --expected-source synthetic_delivery.txt
```

Ekip kaynaklarının alınmış sabit yerel kopyası üzerinde kontrol:

```powershell
.\.venv\rag-runtime\Scripts\python.exe -m scripts.rag_acceptance --source data/security-kb-review/knowledge-base --store data/rag-security-acceptance/chroma --collection kalkan-security-review --report data/rag-security-acceptance/report.json --expected-source dolandiricilik-turleri/oltalama-phishing.md
```

Bu ikinci komut için kaynak kopyası gerekir. Belgeler teslim ZIP'ine dahil değildir;
aynı Git commit'indeki bilgi tabanı alınarak yeniden üretilebilir. Yerel kaynak
izleme kaydı `data/security-kb-review/provenance.json` içindedir.

Bir raporun `status: passed` değeri yalnızca listedeki teknik kontrolleri ifade eder.
Beklenen örnek kaynağın ilk üç sonuçta bulunması tek sorguluk kontroldür; doğruluk
yüzdesi veya tüm dolandırıcılık türlerinde başarı ölçümü değildir. Raporlarda ham
belge metni yer almaz.

## Doğrulama kanıtı

22 Kasım 2026 tarihinde Windows / Python 3.14 ortamında:

- 58 pytest testi geçti; Ruff lint/format ve dört kaynak dosyasında mypy kontrolü başarılı.
- Güvenlik PR'ındaki 9 belge, varsayılan 1000 karakter / 150 örtüşmeyle 29 parçaya
  dönüştürüldü; gerçek E5 vektörleriyle kalıcı Chroma deposuna yazıldı.
- Aynı giriş ikinci kez çalıştırıldı; bütün belgeler `unchanged` oldu.
- Ayrı Chroma istemcisi koleksiyonu açıp vektörleri ve metadataları okudu.
- Her parçanın metni, kaynak belgedeki normalize karakter konumlarıyla eşleşti.
- Örnek banka/bağlantı sorgusunun ilk sonucu
  `dolandiricilik-turleri/oltalama-phishing.md` oldu.
- Model/depo sürümleri: torch `2.11.0+cu128`, transformers `5.19.0`, Chroma `1.5.9`.
  Çıkarım CPU üzerinde yapıldı.

Makine tarafından üretilmiş yerel rapor:
`data/rag-security-acceptance/report.json`. Model ağırlıkları ve indeksler Git'e
eklenmez. Projenin Python 3.12 GitHub CI çalıştırması bu yerel teslimde yapılmadı.

## Seyit'in okuyucu sözleşmesi

| Alan | Beklenen değer |
|---|---|
| Embedding | `intfloat/multilingual-e5-small` |
| Model sürümü | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| Boyut / normalizasyon | 384 / L2 |
| Sorgu öneki | `query: `; `embed_query` otomatik uygular. |
| Belge öneki | `passage: `; depodaki ham parça metnine eklenmez. |
| Koleksiyon | Bu gerçek kaynak denemesinde `kalkan-security-review` |
| Benzerlik | Cosine distance; daha küçük değer daha yakındır, olasılık değildir. |
| Kaynak atfı | `source`, `title`, `document_id`, `chunk_index`, `char_start`, `char_end` |
| Model kontrolü | Koleksiyon ve parçalardaki `embedding_signature`, `embedder.spec.signature` ile aynı olmalıdır. |

Bağımsız okuyucuda gereken bağlantı örneği:

```python
from pathlib import Path

import chromadb
from chromadb.config import Settings

from app.rag.embeddings import E5Embedder

embedder = E5Embedder(cache_dir=Path("models/huggingface"), local_files_only=True)
client = chromadb.PersistentClient(
    path="data/rag-security-acceptance/chroma",
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_collection("kalkan-security-review", embedding_function=None)
if (collection.metadata or {}).get("embedding_signature") != embedder.spec.signature:
    raise ValueError("Sorgu modeli indeksle uyumlu değil.")
results = collection.query(
    query_embeddings=[embedder.embed_query("Şüpheli banka mesajını nasıl değerlendiririm?")],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)
```

Bu örnek bir chatbot yanıtı veya `risk_level` üretmez. Kaynak parçalarını döndürür;
arama/reranking ve asistanın yanıt oluşturma sorumluluğu ilgili ekip üyelerindedir.
İndeks güncellemesi atomik değildir; aynı koleksiyona tek yazıcı kullanılmalıdır.
Silinmiş dosyalar için otomatik prune yoktur. Diğer sınırlar kullanım rehberindedir.

## Tam ekip entegrasyonunun açık kalan kısmı

1. Seyit'in gerçek arama modülü sağlanmalı; yukarıdaki sözleşmeyle bağlantısı ve
   mevcut asistan çağrı noktası üzerinde test yapılmalıdır.
2. Güvenlik ekibi PR içeriğini onaylamalı veya kabul edilen kaynak sürümünü bildirmelidir.
3. Ortak model/depo tercihi farklıysa uyarlama ve yeniden indeksleme gerekir.
4. Onaylı bir soru setiyle arama kalitesi ölçülmeli; bu teknik demo o ölçüm değildir.

Bu maddeler için ekip üyelerine mesaj gönderilmedi ve onların onayı varsayılmadı.

## Hasti için kısa sunum metni

“Ben bilgi tabanının indekslenmesini hazırladım. Sistem Türkçe Markdown ve metin
dosyalarını okuyor. Uzun metinleri kaynaklarını kaybetmeden küçük parçalara ayırıyor.
Her parça için yerel E5 modeliyle 384 boyutlu vektör üretiyor ve Chroma'ya kaydediyor.
Aynı belgeler tekrar geldiğinde kayıt çoğalmıyor; değişen belgenin eski parçaları
yeni parçalar yazıldıktan sonra kaldırılıyor. Gerçek güvenlik ekibi kaynaklarında
9 belge ve 29 parça ile yükleme ve okuma testini yaptım. Seyit aynı modelle sorgu
vektörü üretip bu parçaları arayabilecek. Onun gerçek koduyla bağlantı ve kaynak
içeriğinin ekip onayı henüz bekliyor.”

Koddaki akış: `read_document` → `chunk_document` → `embed_documents` →
`replace_document`. Değişmeyen belge kontrolü `ingest_document` içindedir.
Model eğitilmez; hazır model çıkarım için kullanılır. Metinler bir embedding
API'sine gönderilmez; ilk model indirmesi dışında çalışma yereldir.

## İncelemeye sunma

Çalışma klasörü Git geçmişi içermeyen bir ZIP kopyasıdır. Yayın için asıl
`kalkan-ai` deposu ayrıca klonlanıp `feature/ai-rag-ingest` dalı hazırlanmıştır.
Teslim ZIP'i mevcut depoya uygulanacak dosyaları içerir; tek başına tüm uygulama
deposu değildir. Ağırlıklar, veri tabanları, güvenlik PR belgeleri ve `.env` içermez.
Yukarıdaki test kanıtı yerel doğrulamaya aittir; güncel GitHub CI sonucu PR üzerinden
izlenmelidir.

Önerilen PR başlığı: `feat: add knowledge-base ingestion and local E5 embeddings`

PR açıklama taslağı:

> Kullanıcı-3 / Hasti: Türkçe bilgi tabanı dosyaları kararlı parça kimlikleriyle
> indekslenir, yerel E5 vektörleri Chroma'ya yazılır. Tekrar giriş veri çoğaltmaz;
> değişen belgelerin eski parçaları yeni yazma tamamlandıktan sonra temizlenir.
> Birim ve gerçek model/depo testleri ile bağımsız okuyucu kabul kontrolü eklendi.
> Güvenlik PR #1 kaynaklarıyla 9 belge / 29 parça kontrolü başarılıdır. Seyit'in
> gerçek arama modülüyle entegrasyon, içerik onayı ve GitHub CI doğrulaması bekliyor.
