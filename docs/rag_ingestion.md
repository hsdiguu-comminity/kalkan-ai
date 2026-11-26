# RAG veri hazırlama — Hasti

Bu belge giriş/çıkış sözleşmesi, örnekler, metin parçalama, embedding üretimi ve
yerel Chroma'ya yüklemeyi kapsar. Güvenlik ekibinin açık PR'ındaki kaynaklarla
teknik yükleme ve bağımsız okuyucu kontrolü yapıldı; içerik onayı ve Seyit'in gerçek
arama katmanıyla entegrasyon bekliyor. [Teslim dosyası](rag_handoff.md) doğrulama
kanıtını, sunum metnini ve açık kalan bağlantıyı açıklar. Aşağıdaki model/depo
seçimi çalışan yerel uygulamanın sözleşmesidir; ekibin nihai onayı değildir.

## Girdi

Kaynak, güvenlik ekibinin `kalkan-security/tool-engine/knowledge-base` içeriğidir.
İlk kapsam UTF-8 `.md` ve `.txt` belgeleridir. Dosyayı `ingest.py` okur;
`chunking.py` yalnızca çözülmüş metin alır. PDF, HTML ve OCR
ayrıştırma bu ilk kapsamda yoktur. Markdown başlıkları ve işaretleri metinde korunur;
bu aşama Markdown yapısını ayrıştırmaz.

`SourceDocument` alanları:

| Alan | Anlam |
|---|---|
| `source` | Boş olamaz. Kaynak köküne göre sabit göreli yol; örnek: `sms/banka.md`. Okuyucu `/` ayırıcısını kullanır. |
| `text` | Belgenin Unicode metni; boş olabilir. |
| `title` | İsteğe bağlı belge başlığı; varsayılan boş metin. |

Kaynak yolu bir kimliktir; parçalama modülü bu yolu açmaz veya doğrulamaz.
Gerçek belgelerin sağlanması ve içerik onayı güvenlik ekibine bağlıdır.

## Parçalama ve çıktı

`chunk_document(document, config)` sıralı `TextChunk` listesi döndürür.
Varsayılan `chunk_size=1000`, `chunk_overlap=150` değerleri başlangıç ayarlarıdır;
gerçek belge ve embedding modeliyle ölçüm yapıldıktan sonra değiştirilebilir.
Her iki değer de **token değil Unicode karakter** sayısıdır. Modelin token sınırını
embedding katmanı ayrıca kontrol etmelidir.

- Başlangıç BOM işareti kaldırılır; Unicode NFC ve LF satır sonları uygulanır.
- Türkçe büyük/küçük harfler değiştirilmez. İç boşluklar ve paragraf yapısı korunur.
- Pencerenin son yarısında, ilerlemeye izin veren paragraf sınırı tercih edilir;
  yoksa cümle, ardından boşluk sınırı kullanılır. Sınır bulunamazsa sert kesilir.
- Cümle sınırları basit noktalama sezgisidir; kısaltmaları anlayan dil modeli değildir.
- Parçanın başındaki/sonundaki boşluklar çıkarılır. Örtüşme en fazla belirtilen
  karakter sayısıdır; boşluklardan dolayı daha az olabilir. Örtüşme başlangıcı veya
  sert kesim bir sözcüğün içine gelebilir.
- Boş/yalnızca boşluk içeren belge `[]` döndürür. Geçersiz boyutlar hata üretir.

| Çıktı alanı | Anlam |
|---|---|
| `chunk_id` | Algoritma sürümü, belge kimliği, normalize içerik/başlık ve konumlardan türetilen SHA-256 kimliği. |
| `document_id` | Kaynak yolundan türetilen sabit kimlik; belge güncellenince değişmez. |
| `source`, `title` | Kaynak belgeden taşınan bilgiler. |
| `chunk_index` | Belge içindeki sıfırdan başlayan sıra. |
| `char_start`, `char_end` | Normalize metindeki başlangıç dahil / bitiş hariç konumlar. Ham dosyanın byte konumları değildir. |
| `text` | Embedding için hazırlanmış metin parçası. |

`dataclasses.asdict(chunk)` ile çıktı JSON'a uygun sözlüğe çevrilebilir. Aynı girdi
aynı kimlikleri üretir. `ingest.py` bunları mevcut kaynak kimlikleriyle karşılaştırır;
değişmemiş belge için embedding ve yazma atlanır. Değişen belgenin yeni parçaları
`upsert` ile yazıldıktan sonra, aynı `source` altındaki eski kimlikler silinir.

Bu çıktı bir risk analizi değildir; bu nedenle `risk_level` içermez.

## Yerel örnek

Proje kökünde Python ile:

```python
from dataclasses import asdict
from pathlib import Path

from app.rag.chunking import ChunkingConfig, SourceDocument, chunk_document

path = Path("tests/fixtures/rag/synthetic_bank_sms.md")
document = SourceDocument(
    source="samples/synthetic_bank_sms.md",
    text=path.read_text(encoding="utf-8"),
    title="TEST VERİSİ — Kurgusal banka mesajı",
)
chunks = chunk_document(document, ChunkingConfig(chunk_size=300, chunk_overlap=50))
for chunk in chunks:
    print(asdict(chunk))
```

`tests/fixtures/rag/` altında banka mesajı ve kargo bildirimi için iki küçük,
tamamen kurgusal test belgesi bulunur. Bunlar eğitim veri seti veya onaylı güvenlik
içeriği değildir; kişisel veri içermez ve gerçek bilgi tabanına otomatik eklenmez.
Uzun metin, boş içerik ve farklı satır sonları testlerde ayrıca oluşturulur.

## Embedding sözleşmesi

`embeddings.py` içindeki `E5Embedder` gerçek yerel modeli kullanır; üretim kodunda
sahte vektör veya ağ hatasında sessiz bir alternatif modele geçiş yoktur.

| Ayar | Değer |
|---|---|
| Model | `intfloat/multilingual-e5-small` |
| Sabit sürüm | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| Boyut | 384 |
| Belge / sorgu öneki | `passage: ` / `query: ` |
| Havuzlama | Attention mask ile ortalama |
| Normalizasyon | L2; cosine uzaklığı ile kullanım |
| Token sınırı | Önek ve özel tokenlar dahil 512 |
| Varsayılan cihaz / batch | CPU / 16 |

Önekler Türkçe metinlerde de uygulanır. Bu yapı
[modelin resmi kullanım tarifine](https://huggingface.co/intfloat/multilingual-e5-small)
dayanır. Token sınırı aşılırsa içerik sessizce kesilmez: işlem hata verir ve
`--chunk-size` küçültülmelidir. Karakter sayısı token sınırının garantisi değildir.
Sorgu vektörü `embedder.embed_query(soru)` ile üretilir; arama/reranking bu modülün
kapsamında değildir.

Model ilk embedding isteğinde yüklenir; dosyaları yoksa Hugging Face'ten indirilir.
Belgeler embedding API'sine gönderilmez. İndirilen ağırlıklar `models/huggingface`
altındadır. Sonraki çalıştırmalarda `--offline` ile yalnızca önbellek kullanılabilir.
Bu modelin gerçek Kalkan soruları üzerindeki kalite ölçümü henüz yapılmamıştır.

## Çalıştırma

Yeni ortamda yalnızca gerekli RAG bağımlılıklarını kurmak için:

```powershell
py -m venv .venv/rag-runtime
.\.venv\rag-runtime\Scripts\python.exe -m pip install -r requirements-rag.txt
```

Geliştirme sırasında mevcut torch/transformers kurulumunu kullanmak için ortam
`--system-site-packages` ile de oluşturulabilir. RAG çalışma bağımlılıkları zaten
projenin ML bağımlılıkları arasındadır; bu küçük dosya OCR paketlerini kurmadan
çalıştırmayı sağlar. Yerelde doğrulanan sürümler: Python 3.14, torch 2.11,
transformers 5.19 ve Chroma 1.5.9. Projenin Python 3.12 CI ortamı ayrıca doğrulanmalıdır.

Proje kökünde, yalnızca kurgusal örnekleri ayrı demo koleksiyonuna yüklemek için:

```powershell
.\.venv\rag-runtime\Scripts\python.exe -m app.rag.ingest --source tests/fixtures/rag --store data/rag-demo/chroma --collection kalkan-synthetic-demo --chunk-size 300 --chunk-overlap 50
```

Aynı komutu tekrar çalıştırınca her belge için `status: unchanged` görülür.
İlk yüklemede veya değişiklikte `status: updated` görülür. `chunks` belgenin güncel
parça sayısı, `removed_chunks` bu çalıştırmada kaldırılan eski parça sayısıdır.
Başarılı CLI çıkış kodu 0, hata çıkış kodu 1'dir. Ayrıntılı yardım:

```powershell
.\.venv\rag-runtime\Scripts\python.exe -m app.rag.ingest --help
```

Gerçek kaynaklar gelince ayrı bir koleksiyon ve veri dizini kullanın. Kaynak klasörü
zorunludur; demo içerikleri gerçek kaynaklara otomatik karıştırılmaz. Varsayılan
depo `data/rag/chroma`, koleksiyon `kalkan-knowledge-e5-v1` olur. `data/` ve `models/`
Git dışında kalır. Yeni API anahtarı veya ortam değişkeni gerekmez.

## Kalıcılık, tekrar ve hata davranışı

- Chroma `PersistentClient` kullanılır; metin, vektör ve kaynak metadatası saklanır.
  Vektörler açıkça verilir; Chroma'nın varsayılan embedding işlevi devre dışıdır.
  [Chroma ekleme](https://docs.trychroma.com/docs/collections/add-data) ve
  [upsert](https://docs.trychroma.com/docs/collections/update-data) sözleşmeleri kullanılır.
- Koleksiyon model adı/sürümü, boyutu, önekleri, normalizasyonu ve kaynak köküne
  bağlanır. Uyumsuz veya başka amaçla oluşturulmuş koleksiyona yazma reddedilir.
  Model/kök değişirse yeni koleksiyon gerekir. Klasörü başka yere taşıdıktan sonra
  da yeni koleksiyonla tekrar indeksleyin.
- Aynı kaynağın tekrar çalıştırılması kayıt çoğaltmaz. Farklı kaynaklardaki aynı
  metinler kaynak atfını korumak için ayrı tutulur. Parçalama ayarı değiştiğinde
  yeni parçalar yazılır ve eski parçalar temizlenir.
- Boşaltılmış belge eski parçalarını kaldırır. Diskten silinmiş veya adı değişmiş
  dosyaların eski kayıtları **otomatik silinmez**; tam eşitleme/prune bu sürümde yoktur.
- Belgeler en fazla 5 MiB olabilir. UTF-8 dışı metin, desteklenmeyen dosya türü,
  kök dışı dosya veya dosya sembolik bağlantısı reddedilir. Klasör taramasında
  `.md`/`.txt` dışındaki dosyalar atlanır.
- Her belgenin embedding'i ve vektör doğrulaması yazmadan önce tamamlanır.
  Model hatası veya geçersiz vektör önceki belgeyi değiştirmez.
- Güncelleme, çok adımlı `upsert` ardından eski kimliklerin silinmesidir;
  **atomik bir işlem değildir**. Depo kesintisinde geçici olarak iki sürüm bir arada
  kalabilir; aynı komutu yeniden çalıştırmak eksik temizliği tamamlar. Daha önce
  başarıyla işlenen belgeler sonraki belge hata verince geri alınmaz.
- Aynı koleksiyona eşzamanlı yazma desteklenmez. Tek ingest süreci çalıştırın;
  bu sürüm süreçler arası kilit veya canlı arama için atomik sürüm geçişi sağlamaz.

## Doğrulama ve sonraki adım

```powershell
.\.venv\rag-runtime\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\rag-runtime\Scripts\python.exe -m pytest -q
```

Testler Türkçe karakterleri, kimliklerin kararlılığını, kaynak ayrımını, doğal
sınırları, örtüşmeyi, boyutları ve boşluk dışındaki tüm karakterlerin korunmasını
kontrol eder. Yeni testler tekrar girişini, belge güncellemeyi, boşaltmayı, bozuk
vektörleri, model uyuşmazlığını, dosya hatalarını ve kesintiden sonra tekrar çalışmayı
kontrol eder. `test_rag_chroma.py` gerçek kalıcı Chroma deposunu kullanır; Chroma
kurulu değilse atlanır. Bu testte model yerine açıkça işaretlenmiş bir test ikizi
kullanılır; modelin anlamsal kalitesini ölçmez. `test_rag_model.py`, E5 ağırlıkları
yerelde hazırsa gerçek modelle normalizasyon, padding tutarlılığı, sorgu öneki ve
token taşması davranışını kontrol eder; önbellek yoksa atlanır. Hiçbir pytest testi
model indirmez. İlk indirme için yukarıdaki demo komutu kullanılabilir.

Yerel gerçek model denemesinde iki örnek belge 4 ve 3 parçaya ayrılıp 7 vektör olarak
kaydedildi. Aynı komutun `--offline` ile ikinci çalıştırılmasında iki belge de
`unchanged` döndü. Bu, yükleme akışının doğrulamasıdır; arama kalitesi veya gerçek
güvenlik içeriklerinin doğruluğu hakkında bir başarı metriği değildir.

Seyit'e model sözleşmesi, koleksiyon adı, metadatalar ve `embed_query` arayüzü
aktarılmalıdır. Ekip pgvector veya farklı model seçerse yazma adaptörü/model
sözleşmesi uyarlanmalı ve yeni indeks oluşturulmalıdır. `vector_store.py`,
`retriever.py` ve `reranker.py` bu çalışmada oluşturulmamıştır.
