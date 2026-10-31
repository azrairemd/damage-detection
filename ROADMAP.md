#  Uydu Görüntülerinden Afet Sonrası Bina Hasar Tespiti — Yol Haritası
---

##  İçindekiler
1. [Proje Özeti](#1-proje-özeti)
2. [Fazlar](#2-fazlar)
3. [Repo Yapısı](#3-repo-yapısı)
---

## 1. Proje Özeti

Afet öncesi (pre) ve sonrası (post) uydu görüntü çiftlerinden, piksel düzeyinde bina hasar haritası üreten bir derin öğrenme modeli geliştirilecektir.

- **Girdi:** Pre + post RGB görüntü çifti
- **Model:** Paylaşımlı ağırlıklı (Siamese) encoder → her ölçekte pre/post özellik füzyonu → U-Net decoder
- **Çıktı:** 5 sınıflı maske — `0: arka plan`, `1: hasarsız`, `2: az hasarlı`, `3: ağır hasarlı`, `4: yıkılmış` (`255: sınıflandırılmamış → kayba dahil edilmez`)
- **Kayıp:** `L = λ·Focal + (1−λ)·Dice`
- **Değerlendirme:** xView2 skoru, sınıf bazında F1/IoU, confusion matrix
- **Açıklanabilirlik:** Grad-CAM ısı haritaları ve yorumları
- **Genişleme (Faz 9):** xBD'de eğitilen modelin Türkiye 2023 depremi verisinde zero-shot ve fine-tune performansının karşılaştırılması

---

## 2. Fazlar

### Faz 0 — Kurulum ve Veri Temini

**Hedef:** Repo, ortam ve xBD'ye erişim hazır.

- [x] Klasör iskeletini oluştur
- [x] `.gitignore` (veri, ağırlıklar, `__pycache__`, `.ipynb_checkpoints`, `wandb/`)
- [x] `requirements.txt`: `torch`, `segmentation-models-pytorch`, `albumentations`, `shapely`, `opencv-python`, `scikit-learn`, `grad-cam`, `pyyaml`, `pandas`, `matplotlib`
- [x] `README.md`: proje özeti, kurulum, ilerleme günlüğü bölümü
- [x] xBD'ye Kaggle'da eriş (mevcut dataset veya kendi yüklemen)
- [x] Kaggle notebook'ta repoyu `git clone` + `pip install -e .` ile çalıştırmayı test et

**Çıktı:** Çalışan boş iskelet, veri erişimi, `docs/data_sources.md`
**Bitirme kriteri:** Kaggle notebook'u repodan bir fonksiyon import edip xBD'den bir görüntü çifti gösterebiliyor.

---

### Faz 1 — Literatür Taraması

**Hedef:** Yöntem seçimlerini kaynaklarla gerekçelendirmek.

- [x] xBD makalesi (Gupta ve ark., 2019) — veri seti, etiket şeması, xView2 metriği
- [x] xView2 yarışmasının kazanan çözümleri (Siamese U-Net, iki aşamalı yaklaşımlar)
- [x] Focal Loss (Lin ve ark., 2017), Dice Loss
- [ ] ~~Grad-CAM (Selvaraju ve ark., 2017), segmentasyon için Grad-CAM uyarlamaları~~ → Faz 8'e taşındı
- [x] Her kaynak için 3–5 satırlık özet

**Çıktı:** `docs/literature.md` (+ BibTeX: `docs/references.bib`)
**Bitirme kriteri:** Her tasarım kararı (mimari, kayıp, metrik, XAI) en az bir kaynakla eşleşiyor.

---

### Faz 2 — Keşifsel Veri Analizi (EDA)

**Hedef:** Veriyi tanımak; dengesizliği sayılarla göstermek.

- [x] Olay (event) ve afet türü bazında görüntü sayıları
- [x] Sınıf bazında **bina** ve **piksel** sayıları (genel + afet türü bazında)
- [x] Görüntü başına bina sayısı dağılımı; hiç bina içermeyen görüntü oranı
- [x] Örnek pre/post/maske görselleri (her afet türünden)
- [x] Bulguları özetle → Focal + Dice ve sampler kararlarının gerekçesi

> Atlanan analizler: bina boyutları (karar Weber & Kané, 2020'ye dayandırıldı), çekim koşulları (off-nadir, çözünürlük), no-data alanları ve renk istatistikleri (ImageNet normalizasyonu kullanılacak). Ayrıntılar: `02_eda.ipynb`, bölüm 8.

**Çıktı:** `notebooks/02_eda.ipynb` (çıktılarıyla commit'lenir), `reports/figures/eda_*`
**Bitirme kriteri:** "Sınıf dengesizliği" ve "afet türü dengesizliği" grafiklerle gösterilmiş.

---

### Faz 3 — Ön İşleme ve Split

**Hedef:** Eğitime hazır, tekrarlanabilir veri.

- [x] **Maske üretimi:** post JSON'lardaki poligonları rasterize et (`subtype` → sınıf id). `un-classified` → `255`
  - [x] Maskeler **post** etiketlerinden çiziliyor (EDA: pre ve post'ta binalar aynı)
  - [x] Koordinatlar görüntü sınırlarına kırpılıyor (clip); çakışmalarda ağır hasar üste çiziliyor
  - [x] **Doğrulama:** hazır maskelerle karşılaştırma (100 görüntü, un-classified hariç %99,996 uyum)
- [x] **Split (tile'lamadan ÖNCE, görüntü seviyesinde):**
  - [x] Tüm tier'ları tek tabloda birleştir (`image_id, folder, disaster, disaster_type, ...`)
  - [x] `train_test_split(stratify=disaster, seed=42)` → %60 / %40, ardından %40 → %20 / %20
  - [x] `splits/train.csv`, `splits/val.csv`, `splits/test.csv` → **repoya commit**
  - [x] Split dağılım tablosu: her split'te olay oranları (stratification'ın kanıtı)
- [x] **Sızıntı kontrolü:** `tests/test_split_leakage.py` (çakışma yok, 11.034 çiftin hepsi atanmış, 19 olayın hepsi her split'te)
- [x] 11.034 maske kaydedildi ve notebook çıktısı olarak saklandı (Faz 4'te input olarak eklenecek)
- [ ] ~~Tile'lama~~ → Faz 4'e taşındı: parçalar diske kaydedilmiyor, eğitim sırasında kesiliyor

**Çıktı:** `src/data/labels.py`, `src/data/metadata.py`, `src/data/masks.py`, `src/data/split.py`, `tests/test_split_leakage.py`, `notebooks/03_preprocessing_split.ipynb`, `splits/*.csv`
**Bitirme kriteri:** Split dosyaları sabit, sızıntı testi geçiyor, split tabloları raporlanmış.

---

### Faz 4 — Eğitim Altyapısı

**Hedef:** Config ile yönetilen, kesintiye dayanıklı eğitim döngüsü.

- [x] **Dataset** (`XBDDataset`): pre + post + maske döndürür; pre, post ve maskeye **birebir aynı** dönüşüm
  - [x] Maskeler `03_preprocessing_split` notebook çıktısından okunur
  - [x] **Tile'lama:** train'de 1024×1024 görüntüden rastgele 512×512 parça kesilir; val/test'te tam 1024×1024 görüntü kullanılır (küçültme değil; gerekçe: Weber & Kané, 2020)
  - [x] ImageNet normalizasyonu
- [x] Augmentation: yatay/dikey çevirme ve 90° döndürme (NumPy ile; albumentations kullanılmadı)
- [x] **Kayıplar** (`src/losses/`):
  - [x] `FocalLoss(alpha, gamma=2)`, `ignore_index=255`
  - [x] `DiceLoss`: sadece hasar sınıfları (1–4), **arka plan dahil değil** (EDA: piksellerin %96,65'i arka plan)
  - [x] `CombinedLoss(λ)`: `λ·Focal + (1−λ)·Dice`
  - [x] **Sınıf ağırlıkları:** train maskelerinden, yumuşatılmış ters orantı (1/√f, ortalaması 1): `[0.080, 0.479, 1.382, 1.334, 1.725]` (destroyed/arka plan ≈ 22×, ham ters orantıda ≈ 500×)
- [x] **Metrik:** xView2 skoru (`0.3·F1_loc + 0.7·F1_dmg`, F1_dmg = 4 hasar sınıfının harmonik ortalaması), sınıf bazında F1; sayaçlar bütün val seti boyunca toplanır
- [x] **Model:** `BaselineUNet` (ResNet34, ImageNet, sadece post); bütün modeller aynı arayüzü kullanır: `model(pre, post)`
- [x] **Eğitim döngüsü** (`src/train.py`): `train_one_epoch`, `validate`, `fit`
  - [x] **Early stopping:** `patience = max(1, int(0.2 * max_epochs))`; izlenen metrik: val xView2 skoru
  - [x] `best.pt` (en iyi model) + `last.pt` (her epoch, tam durum) + `resume` desteği
  - [x] Loglama: her epoch `log.csv`'ye (kayıp, xView2, F1'ler)
- [x] `configs/baseline.yaml`: bütün hiperparametreler; notebook'ta `load_config` ile okunur (`run` fonksiyonu yazılmadı)
- [x] **İlk deneme:** 64/16 örnek, 3 epoch → kayıp azalıyor, val skoru artıyor
- [x] **Sağlamlık testi:** tek batch 200 adımda ezberleniyor (gruptaki sınıflarda F1 ≈ 0,97–0,98)
- [ ] ~~Seed sabitleme~~ → kullanılmadı (raporda limitasyon olarak belirtilecek)
- [ ] ~~Weights & Biases~~ → kullanılmadı, CSV log yeterli
- [ ] Mixed precision (AMP) → Faz 5'e taşındı (eğitim süresine göre)
- [ ] Binasız parçaların azaltılması → Faz 5'e taşındı

**Çıktı:** `src/data/dataset.py`, `src/losses/{focal_loss, dice_loss, combined_loss, weights}.py`, `src/metrics/xview2.py`, `src/models/baseline_unet.py`, `src/train.py`, `src/utils/config.py`, `configs/baseline.yaml`, `notebooks/04_training_pipeline.ipynb`
**Bitirme kriteri:** Tek batch overfit başarılı; eğitim uçtan uca çalışıyor; `best.pt`, `last.pt` ve `log.csv` oluşuyor.

---

### Faz 5 — Baseline Model

**Hedef:** Karşılaştırma noktası oluşturmak.

- [ ] **Baseline:** sadece post görüntü alan tek encoder'lı U-Net (ResNet34, ImageNet pretrained) → model Faz 4'te hazır
- [ ] Bir epoch'un süresini ölç; gerekirse **mixed precision (AMP)** ekle
- [ ] Binasız parçaların bir kısmını atla (EDA: görüntülerin %35'i boş)
- [ ] Kaggle oturumu kesilirse `last.pt` ile devam (`resume=True`)
- [ ] Aynı split, aynı kayıp, aynı early stopping
- [ ] Val skorlarını kaydet; `log.csv`'den eğitim eğrilerini çiz
- [ ] **Grad-CAM prototipi:** baseline üzerinde ilk ısı haritası (wrapper sorunlarını erkenden görmek için)

**Çıktı:** `src/models/baseline_unet.py`, `notebooks/05_train_baseline.ipynb`, sonuç tablosunda ilk satır
**Bitirme kriteri:** Baseline val xView2 skoru kayıtlı; Grad-CAM en az bir görüntüde çalışıyor.

---

### Faz 6 — Siamese Model ve Ablation

**Hedef:** Ana modeli geliştirmek ve her tasarım kararını deneyle desteklemek.

- [ ] **Siamese U-Net:** paylaşımlı encoder (pre ve post aynı ağırlıklarla), her ölçekte füzyon, U-Net decoder
- [ ] (Opsiyonel) Bina lokalizasyonu için yardımcı binary head
- [ ] **Ablation deneyleri** (hepsi val setinde):

| # | Değişken | Seçenekler |
|---|---|---|
| A1 | Girdi | Post-only (baseline) vs. Siamese |
| A2 | Füzyon | Concat vs. Fark `\|f_post − f_pre\|` vs. Concat + Fark |
| A3 | Kayıp | CE vs. Focal vs. Focal + Dice |
| A4 | λ | 0.3 / 0.5 / 0.7 |
| A5 | Örnekleme | Normal vs. hasarlı bina içeren görüntüleri ağırlıklı örnekleme (EDA: deprem görüntülerinin %99'u hasarsız, deprem ağırlıklı örnekleme ters etki yapar) |
| A6 | Encoder | ResNet34 vs. EfficientNet-B3 (kota izin verirse) |

- [ ] Her deney için config + skor + eğitim süresi kaydı
- [ ] En iyi konfigürasyonu seç → **final xBD modeli**

**Çıktı:** `src/models/siamese_unet.py`, `notebooks/06_train_siamese.ipynb`, `reports/results.md` (ablation tablosu)
**Bitirme kriteri:** Final model seçilmiş; seçim gerekçesi tabloyla belgelenmiş.

> ⚠️ Model seçimi **yalnızca val** setiyle yapılır. Test setine bu fazda dokunulmaz.

---

### Faz 7 — xBD Test Değerlendirmesi

**Hedef:** Final modelin tarafsız performansı.

- [ ] Final model ile test setinde **tek seferlik** değerlendirme
- [ ] xView2 skoru, F1_loc, F1_dmg, sınıf bazında F1/IoU
- [ ] Normalize confusion matrix
- [ ] Olay bazında ve afet türü bazında skor kırılımı
- [ ] Deprem sonuçları sadece lokalizasyon ve no-damage için yorumlanacak (EDA: tüm depremde sadece 3 yıkılmış bina)
- [ ] Baseline'ın test skoru (karşılaştırma için)
- [ ] Nitel sonuçlar: iyi ve kötü tahmin örnekleri

**Çıktı:** `src/evaluate.py`, `notebooks/07_evaluate_xbd_test.ipynb`, `reports/results.md` güncel
**Bitirme kriteri:** Test sonuçları tablo ve figürlerle raporlanmış; `best.pt` GitHub Release'e yüklenmiş.

---

### Faz 8 — Grad-CAM / XAI

**Hedef:** "Model kararını görüntünün neresine bakarak veriyor?" sorusunu görsel ve yorumla cevaplamak.

- [ ] Literatür: Grad-CAM, Grad-CAM++, segmentasyon için Grad-CAM, Sanity Checks (Faz 1'den taşındı) → `docs/literature.md` E bölümü
- [ ] **Wrapper:** Siamese modeli tek girdi (6 kanal: pre+post) alacak şekilde sar
- [ ] **Hedef:** `SemanticSegmentationTarget` — c sınıfının tahmin maskesi bölgesindeki logit toplamı
- [ ] **Katman karşılaştırması:** encoder son bloğu vs. decoder katmanı
- [ ] **Görsel şablonu:** Pre | Post | Gerçek maske | Tahmin | Grad-CAM (sınıf bazında)
- [ ] Her hasar sınıfı için örnekler
- [ ] **Doğru vs. yanlış** tahmin örnekleri + kısa yorumlar
- [ ] **Baseline vs. Siamese** Grad-CAM karşılaştırması (pre görüntünün modele ne kattığını göstermek için)
- [ ] **Kısayol öğrenme kontrolü:** Model hasarlı binaya mı, yoksa afet izine (sel suyu, kül) mı bakıyor? (EDA: minor ← %70 rüzgâr, major ← %68 sel)
- [ ] (Bonus) Grad-CAM++ ve deletion testi (önemli bölge maskelenince skor düşüşü)

**Çıktı:** `src/xai/gradcam.py`, `src/xai/wrappers.py`, `notebooks/08_gradcam_xai.ipynb`, `reports/figures/xai_*`
**Bitirme kriteri:** Raporun XAI bölümü için en az 8–10 yorumlanmış figür hazır.

> 🏁 **Ana gövde tamam.** Faz 8 bitince danışmanın tüm şartları karşılanmış olur ve `v0.8-core` tag'i atılır.
> **Karar noktası:** Teslime yeterli süre kaldıysa → Faz 9. Kalmadıysa → doğrudan Faz 10; Türkiye çalışması raporda "Gelecek Çalışmalar" bölümüne yazılır.

---

### Faz 9 — Genişleme: Türkiye Fine-Tune

**Hedef:** xBD modelinin Kahramanmaraş 2023 depremine transferini ölçmek.

> EDA motivasyonu: xBD'deki tek deprem olayında (Mexico City) binaların %99,2'si hasarsız; model xBD'den deprem yıkımını öğrenemiyor.

- [ ] Literatür: Türkiye 2023 depremi hasar tespiti çalışmaları (KATE-CD, TUE-CD, xBD→Türkiye transfer çalışmaları)
- [ ] Veri setini temin et; lisansını ve etiket şemasını kontrol et (aday: KATE-CD, yedek: TUE-CD)
- [ ] Etiket şemasını xBD sınıflarıyla eşle (ör. binary ise: `major + destroyed → hasarlı`)
- [ ] Çözünürlüğü xBD'ye yakın olacak şekilde yeniden örnekle
- [ ] **Split:** 60/20/20, il/şehir bazında stratified → `splits/turkey_*.csv`
- [ ] **Deneyler:**

| # | Deney | Açıklama |
|---|---|---|
| T1 | Zero-shot | xBD final modeli, Türkiye test setinde (eğitimsiz) |
| T2 | Fine-tune | xBD modeli + Türkiye train (düşük LR; ilk N epoch encoder dondurulmuş) |
| T3 | Sıfırdan | Aynı mimari, sadece Türkiye verisiyle |

- [ ] Aynı kurallar: Focal + Dice, patience = %20, val ile seçim, test tek sefer
- [ ] T1 / T2 / T3 karşılaştırma tablosu + nitel örnekler
- [ ] **XAI:** aynı görüntüde fine-tune öncesi vs. sonrası Grad-CAM karşılaştırması (Faz 8 kodu yeniden kullanılır)

**Çıktı:** `notebooks/09_turkey_finetune.ipynb`, `configs/turkey_*.yaml`, `splits/turkey_*.csv`, `reports/results.md`
**Bitirme kriteri:** Üç deneyin test skorları karşılaştırmalı olarak raporlanmış.

---

### Faz 10 — Rapor ve Sunum

- [ ] Rapor iskeleti: Giriş · Literatür · Veri · Yöntem · Deneyler · Sonuçlar · XAI · *(Faz 9 yapıldıysa)* Türkiye Transferi · Tartışma ve Limitasyonlar · Sonuç ve Gelecek Çalışmalar
- [ ] Tüm tablo ve figürlerin son hali
- [ ] Limitasyonlar: coğrafi korelasyon, Kaggle hesaplama kısıtları, piksel tabanlı modelde aynı binanın farklı piksellerinin farklı sınıf alabilmesi; *(Faz 9 yapıldıysa)* etiket uyumsuzluğu, nadir görüntüde katlı bina çöküşünün zor görünmesi
- [ ] README'yi son haline getir: sonuçlar, nasıl çalıştırılır, model ağırlığı linki
- [ ] Sunum slaytları + (opsiyonel) demo notebook'u
- [ ] `v1.0` tag'i + GitHub Release

**Bitirme kriteri:** Rapor teslim edildi; repo baştan sona tekrarlanabilir.

---

## 3. Repo Yapısı

```
damage-detection/
├── configs/
│   ├── baseline.yaml
│   ├── siamese_r34.yaml
│   └── turkey_finetune.yaml       # Faz 9 (opsiyonel)
├── splits/                        # commit'lenir — tekrarlanabilirliğin kanıtı
│   ├── train.csv  val.csv  test.csv
│   └── turkey_*.csv               # Faz 9 (opsiyonel)
├── src/
│   ├── data/                      # xbd.py, labels.py, metadata.py, masks.py, split.py, dataset.py, transforms.py
│   ├── models/                    # baseline_unet.py, siamese_unet.py
│   ├── losses/                    # focal_loss.py, dice_loss.py, combined_loss.py, weights.py
│   ├── metrics/                   # xview2.py
│   ├── xai/                       # gradcam.py, wrappers.py, deletion_test.py
│   ├── utils/                     # config.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_setup_check.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_split.ipynb
│   ├── 04_training_pipeline.ipynb
│   ├── 05_train_baseline.ipynb
│   ├── 06_train_siamese.ipynb
│   ├── 07_evaluate_xbd_test.ipynb
│   ├── 08_gradcam_xai.ipynb
│   └── 09_turkey_finetune.ipynb   # Faz 9 (opsiyonel)
├── tests/                         # test_split_leakage.py
├── docs/                          # literature.md, data_sources.md, references.bib
├── reports/
│   ├── figures/
│   └── results.md                 # tüm deney sonuç tabloları
├── ROADMAP.md
├── README.md
├── requirements.txt
├── pyproject.toml                 # pip install -e . için
└── .gitignore
```

**Repoya girmeyenler:** ham/işlenmiş veri, maskeler, model ağırlıkları (`*.pt`), `wandb/`, büyük çıktılar. Maskeler → Kaggle notebook çıktısı. Ağırlıklar → GitHub Releases veya Kaggle Models.

---
*Son güncelleme: 31 Ekim 2026*