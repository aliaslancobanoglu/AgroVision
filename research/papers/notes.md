# Makale notları — karar etkisi olanlar

Tarih: 2026-10-15. PDF'ler bu klasörde (`<arxiv_id>.pdf`).
Bizim kurulum: 16 m kol, 8 kamera, 1280-1920 px, 15 km/s,
git/dur kriteri %98 **strip-level** ot recall'ünde spray rate.

> **Düzeltme (2026-10-15):** İlk sürümde "45 ms bütçe = 8 kare 45 ms'de"
> varsayımı vardı; yanlıştı. Doğru bütçe (the project notes):
> - **Gecikme ≈ 150-200 ms.** Kamera nozulun 1 m önünde; 15 km/s'de bu
>   240 ms'lik pencere, eksi valf tepkisi ve pay.
> - **Asıl kısıt throughput.** Boşluksuz kapsama için kamera başına ~6 FPS
>   gerekiyor. 8 kamera tek Jetson'u paylaşırsa bu, kare başına ~20 ms
>   (uçtan uca) demek.
>
> Aşağıdaki hesaplar bu bütçeye göre yeniden yapıldı.

---

## Özet: planı değiştiren 5 bulgu

1. **Tek AGX Orin'de 8 kamera × 1280 px sınırda sığabilir, ama bu kanıtlanmış
   değil.** Ölçülen en hızlı değer (AGX Orin, YOLO11n INT8, 640 px) yalnızca
   model motoru için 2.43 ms. 1280 px için kabaca ×4 alınırsa (tahmin, ölçülmedi)
   kare başına ~9.7 ms çıkıyor. Bu, ~20 ms/kare bütçesinin yarısı. Kalan ~10 ms
   ön işlem, son işlem ve kamera yakalamaya kalıyor, yani pay dar.
   Nano Super için YOLO11n ölçümü yok. YOLO11s 640'ta 6.99 ms; ×4 ile ~28 ms
   ediyor, bütçenin üstünde.
   → Jetson sayısı, gerçek bir kartta uçtan uca ölçüm yapılmadan belirlenmemeli.
2. **Yayınlanan benchmark'larda motor süresi ile uçtan uca süre arasında devasa
   fark var.** AgriJetsonBench'te aynı model motor düzeyinde 12 ms iken,
   optimize edilmemiş Python boru hattında 915 ms. → ~20 ms/kare throughput
   bütçesi uçtan uca ölçülmeli. Bizim 2.1 ms (4070) değeri de yalnızca motor
   süresi.
3. **SAHI / dilimleme bize göre değil.** İki makale de masaüstü GPU'da kare
   başına 190-300 ms veriyor. ROI-Gated sürümü ise recall kaybettiriyor; bu da
   %98 kuralıyla çelişir. → Planımızdaki "doğrudan imgsz=1280 ile eğit ve çıkarım
   yap" yaklaşımı doğru, dilimleme eklenmemeli.
4. **Bize en çok benzeyen veri seti (ACRE: robot, aşağı bakan kamera, mısır) en
   zoru.** Başka veri setlerini eklemek ona yardım etmedi. Tek sınıflı aynı
   ayarda mAP50:95 0.334 → 0.324, yani neredeyse nötr.
   → Kaggle → Konya zincirinin kazanacağı garanti değil. the project notes'deki a/b
   kıyası zorunlu kalmalı. (Kanıt ilk sürümde yazdığımdan zayıf; aşağıdaki
   düzeltmeye bakın.)
5. **Yayınlanan hiçbir in-domain sonuçta ot recall'ü %98'e ulaşmıyor.** Recall
   değerleri 0.63 ile 0.97 arasında; bunlar varsayılan conf eşiğinde. Bizim 0.901
   tavanımız bu dağılımın tam ortasında. → %98 recall kriteri literatürün çok
   ötesinde. Bu, git/dur kararının en büyük riski olarak kayda geçmeli.

---

## 2608.00927 — AgriJetsonBench (TensorRT, Jetson AGX Orin vs Orin Nano Super)

**Jetson rakamları** (batch=1, 640 px, motor süresi: ön işlem, NMS ve son işlem
hariç):

| Kart / mod | Model | Hassasiyet | Medyan | p95 | FPS | J/çıkarım |
|---|---|---|---|---|---|---|
| AGX Orin (en iyi mod) | YOLO11n | INT8 | 2.43 ms | — | — | 0.043 |
| AGX Orin (en iyi mod) | YOLO11s | INT8 | 3.39 ms | — | — | 0.069 |
| AGX Orin 50 W | YOLO11s | INT8 @416 | — | — | 518 | 0.044 |
| AGX Orin 15 W | YOLO11s | INT8 | 11.83 ms | 11.90 ms | 80.9 | 0.181 |
| Orin Nano Super 15 W | YOLO11s | INT8 | 6.99 ms | 7.02 ms | 145.8 | 0.077 |
| AGX Orin 50 W | RT-DETR-R18 | FP16 | — | ~9.7 ms | ~105 | 0.311 |

- Bellek: YOLO11s FP16 her iki kartta **1536 px'e kadar** güvenle çalışıyor.
  1280 px bellek açısından sorun değil. 1280 px için **gecikme rakamı yok.**
- Uçtan uca (AGX, YOLO11s INT8, optimize edilmemiş Python): **915 ms medyan,
  1.07 FPS**. Motor süresinin ~75 katı.
- Boşta çekilen güç: AGX ~10.9 W, Nano Super ~5.3 W.

**Bize uygunluk:**
- Ölçüm protokolü doğrudan uygulanabilir. Batch=1 kullanılmış; 8 kamera için
  batch=8 hiç test edilmemiş (makale de bunu sınırlama olarak yazıyor).
- Doğruluk: CottonWeedDet12 (yakın plan, bizim Kaggle verisine benzer) üzerinde
  YOLO11n ile YOLO11s neredeyse aynı (mAP50-95 0.904 / 0.902, recall
  0.909 / 0.907). → YOLO11s'e geçmek recall kazandırmıyor, n'de kalmak mantıklı.

**Planla çelişen:**
- "Büyük kart = daha hızlı" doğru değil: 15 W INT8 ayarında Nano Super, AGX'ten
  1.8× hızlı. Kart seçimi, kartın sınıfına bakılarak değil, hedef güç modunda
  ölçülerek yapılmalı.
- INT8 kabul kriterleri mAP düşüşü ≤ 0.03 ve AR100. Yüksek recall bölgesi ayrıca
  test edilmemiş. → Biz INT8'e geçersek evaluate.py'nin recall-vs-spray eğrisi
  INT8 motorla **yeniden çıkarılmalı**. mAP'nin korunması, %98 recall'deki
  noktanın da korunduğu anlamına gelmez.

---

## 2604.19233 — Adaptive SAHI (ASAHI)

**Jetson rakamı:** Yok. RTX 3080 / 2080 Ti, TPH-YOLOv5: VisDrone'da 5.26 img/s
(**~190 ms/kare**). SAHI'den %20-25 hızlı.

**Bize uygunluk: Uygulanamaz.** Yöntem her kare için 6 ya da 12 dilim üzerine
bir de tam görüntü çıkarımı yapıyor. Eşik T = 1818 px olduğu için 1280 px'lik
bir kare 6 dilime bölünüyor; tam görüntüyle birlikte kare başına 7 ileri
geçiş eder.
- **Hesaplama yükü:** 1280 px'te dilimler ~475 px ve 512'ye büyütülüyor.
  Altı dilimin toplam piksel yükü 6 × 512² ≈ 1.6 Mpx, bu da kabaca tek bir
  1280² geçişe denk. Tam görüntü geçişi buna eklenince toplam, tek 1280 px
  geçişin ~1.1-2 katı oluyor (tam geçişin boyutuna göre; makalede net değil).
  Yani saf FLOP olarak engelleyici değil. Ama 7 ayrı çağrı, dilimleri
  birleştirme adımı ve CDN-NMS ek gecikme getiriyor. Tek 1280 px geçişte
  bunların hiçbiri yok.
- **Asıl gerekçe:** 512'ye büyütülmüş ~475 px'lik dilimler, doğrudan 1280 px
  çıkarımla hemen hemen aynı piksel yoğunluğunu veriyor. Dilimleme bize ek
  çözünürlük kazandırmıyor, sadece ek adım ve karmaşıklık getiriyor.
- Büyük nesne mAP'si düşüyor (dilimler büyük nesneleri parçalıyor).
- Masaüstü GPU'da bile ölçülen ~190 ms/kare, ağır bir TPH-YOLOv5 modeline ait;
  bizim modelimize doğrudan taşınmaz.

**Kullanılabilecek tek parça:** SAF, yani eğitimde tam görüntüyle birlikte
dilimleri de kullanmak. Çıkarım maliyeti yok, sadece veri artırma. Ama biz zaten
1280 px'te doğrudan eğitiyoruz, bu yüzden muhtemelen gereksiz.

**Güvenilirlik uyarısı:** Kurum adı tutarsız ("College of Educational Science,
Polytechnic University of Turin"). Yöntem, Zhang ve ark. 2023'te (Remote
Sensing 15(5)) yayımlanan aynı adlı ASAHI ile örtüşüyor. Kaynakçada konuyla
ilgisiz çok sayıda doküman anlama makalesi var. Rakamlar bağımsız
doğrulanmadan kullanılmamalı.

---

## 2608.23923 — ROI-Gated SAHI

**Jetson rakamı:** Yok. Donanım belirtilmemiş, PyTorch. COCO128'de:
Full SAHI 264 ms, statik ROI-Gated 298 ms, adaptif (τ = 0.40) 258 ms.

**Bize uygunluk: Uygulanamaz; %98 recall kuralıyla doğrudan çelişiyor.**
- Önce hafif bir model ilgi bölgelerini öneriyor. Bu modelin kaçırdığı bölge
  **bir daha kurtarılamıyor** (yazarlar da bunu sınırlama olarak yazıyor).
- Etkisi: mAP@0.5 0.757 → 0.660. Full SAHI'ye göre recall uyumu 0.31-0.50.
  Yani tam SAHI'nin bulduğu nesnelerin yarısı ya da daha fazlası kayboluyor.
- Hız kazancı yalnızca ilgi bölgesi görüntünün %30'undan azını kapladığında
  var (en iyi durumda 6.9×). Ortalamada kazanç neredeyse yok (1.02×).
- Kanıt zayıf: 128 görsel artı 3 görsellik bir vaka çalışması. Sonuç bölümünde
  mAP rakamları ters yazılmış.

**Dolaylı ders:** Bu, "ucuz bir model önce tarasın, pahalı model sadece şüpheli
yerlere baksın" kademeli yapısının tam karşı-örneği. Bu yapı spray rate'i
düşürür ama recall'ü öldürür. Gelecekte gecikme baskısıyla böyle bir fikir
gelirse bu makale referans alınmalı.

---

## 2608.21254 — Cross-Field Distribution Shift (pamuk ↔ soya, İHA)

**Jetson rakamı:** Yok (Faster R-CNN, gecikme ölçülmemiş).

**Bize uygunluk: Kısmen.** Geometri farklı: İHA ile 7.6 m yükseklik, 2.13 cm/px.
Bizde 1 m yükseklik ve ~1-2 mm/px. Başlık "cross-field" dese de aslında
ekin/veri seti arası transferi ölçüyor; tarla bazlı bölme sızıntısını test
etmiyor.

**Karar etkisi olan bulgular:**
- Sıfır-atış transfer çoğunlukla tek haneli mAP50'ye çöküyor. Kaynak verideki
  yüksek skor, hedefte ne olacağını öngörmüyor. → the project notes'deki "Kaggle
  sonucunu yanlış okuma" uyarısını doğruluyor.
- **Yakın plan (yer seviyesi) kaynakla ön eğitim, tam veriyle ince ayarda
  yine de faydalı:**
  - CottonWeedDet12 → soya: 84.9 (sıfırdan eğitim 76.8)
  - CottonWeedDet12 → pamuk: 43.2 (sıfırdan 37.4)
  - CottonWeedDet12, bizim Kaggle setine benzer yakın plan bir veri seti.
    → Konya'da (b) seçeneğinin, yani Kaggle best.pt → Konya, kazanması için
    makul bir önsel var. Ama tek modelle ve tek metrikle (mAP50) ölçülmüş;
    a/b kıyası yine yapılmalı.
- 25 etiketli görselle ince ayar, öğrenilebilir hedefte (soya) neredeyse
  sıfırdan eğitim seviyesine çıkıyor (80.0 vs 76.8). Zor hedefte (pamuk) hiçbir
  şey değiştirmiyor (9.4). Denetimsiz alan uyarlaması (DAOD) ise tutarsız.
  → Önerim: Konya'da ilk tarladan **~25-50 karelik erken bir pilot etiketleme**
  yapılsın. Bu, tam etiketleme maliyetine girmeden önce görevin öğrenilebilir
  olup olmadığını haftalar içinde gösterir.

**Planla kısmi çelişki:** Pamuk veri setinde 5706 karo var ama sıfırdan eğitimle
bile mAP50 yalnızca 37.4. Yani "veri miktarı kritik" dersi eksik: **ot boyutu /
piksel çözünürlüğü** en az veri miktarı kadar belirleyici. 2.13 cm/px'te sorun
çözülemiyor. Bu bulgu bizim 1-1.5 mm/px hedefimizi ve 8 kamera kararını
destekliyor.

---

## 2609.33991 — Multi-Dataset YOLO Weed Benchmark (YOLO26 n/s/m, 7 veri seti)

**Jetson rakamı:** Yok. GTX 1080 Ti, 640 px, ortalama gecikme: YOLO26n
3.21 ms, YOLO26s 5.79 ms, YOLO26m 12.16 ms. Ölçüm sınırı belirtilmemiş.

**Karar etkisi olan bulgular:**
- **YOLO26'nın NMS'siz çıkarımı.** Jetson benchmark'larındaki motor süreleri
  NMS'yi içermiyor ve NMS genelde CPU'da çalışan gizli bir maliyet. NMS'siz bir
  model bu kalemi tamamen siliyor. YOLO26n'nin maliyeti 5.4 GFLOPs; aynı
  640 px'te YOLO11n için AgriJetsonBench ~7.5 GFLOPs veriyor.
  → **Plan değişikliği önerisi:** Konya kıyasına YOLO26n de eklenmeli.
  the project notes şu an yolo11n.pt'yi sabit tutuyor.
- Veri setleri arası ortalama düşüş: mAP50:95 0.603 → 0.148 (%−75.6).
  Bütün veri setlerini birleştirerek eğitmek bu açığı kapatmıyor.
- **ACRE** (robot, aşağı bakan Basler kamera, mısır/fasulye; bize en yakın
  kurulum):
  - En zor veri seti. İki sınıflı in-domain sonuç: mAP50:95 0.45-0.49,
    recall 0.69-0.74.
  - **Düzeltme (2026-10-15):** İlk sürümde 0.487 → 0.324 ("kötüleşiyor")
    yazmıştım. Bu karşılaştırma yanlıştı: 0.487 iki sınıflı (ekin + ot)
    in-domain sonuç, 0.324 ise tek sınıflı (sadece ot) çok kaynaklı sonuç.
    Doğru tek sınıflı karşılaştırma: ACRE → ACRE 0.334 (Şekil 2'nin
    köşegeni), 7 veri setiyle birlikte 0.324. Fark neredeyse nötr.
    Tek sınıflı recall'ün in-domain karşılığı makalede yok; 0.632'yi iki
    sınıflı 0.69-0.74 ile karşılaştırmak da aynı hatayı tekrarlar.
  - → Sonuç değişmedi ama kanıt zayıfladı. Ek kaynak veri bize en yakın
    kurulumda **yardım etmiyor** (zarar da vermiyor). Kaggle → Konya
    zincirinin kazanacağı garanti değil; a/b kıyası şart.
- Recall tavanı: in-domain recall'ler 0.63 ile 0.97 arasında (YOLO26n/s/m).
  Hiçbiri 0.98'e ulaşmıyor.
- Resmi bölmesi olmayan veri setleri **rastgele** 70/15/15 bölünmüş. Bu
  yüzden in-domain skorlar muhtemelen şişkin. → Tarla bazlı bölme kuralımızın
  neden gerekli olduğunun bir örneği daha.

---

## Önerilen aksiyonlar

> 2026-10-15 durumu: Beş aksiyonun hepsi the project notes'ye işlendi. 1, 3, 4 ve 5:
> plan madde 6a-c ve "Reddedilen yaklaşımlar" bölümü. 2: YOLO26n, plan madde
> 5c olarak eklendi. YOLO26n'in Kaggle'daki avantajı zayıf: strip recall'de
> +3 puan, ama SPILL_TOLERANCE 0.1 referansında bu fark kayboluyor.

1. Donanım bütçesi: Jetson üzerinde 1280 px'te **uçtan uca** throughput
   (hedef ≤ ~20 ms/kare, 8 kamera × 6 FPS) ve gecikme (hedef ≤ 150-200 ms)
   ölçülmeli: yakalama → ön işlem → motor → NMS → nozul kararı. Bunun için
   batch=8 ile "kamera başına batch=1" karşılaştırılmalı. Bu ölçüm yapılmadan
   Jetson sayısı belirlenmemeli. (the project notes plan madde 6c)
2. Konya kıyası (a)/(b)'ye bir (c) eklenmeli: yolo26n.pt (COCO) → Konya.
3. INT8'e geçilirse recall-vs-spray eğrisi INT8 motorla yeniden üretilmeli.
4. Konya ilkbaharı başında 25-50 karelik pilot etiketleme yapılıp hızlı bir
   öğrenilebilirlik testi koşulmalı.
5. SAHI / dilimleme / ROI kademeli yapı: reddedildi. Gerekçe yukarıda.
