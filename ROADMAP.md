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

- [ ] Klasör iskeletini oluştur
- [ ] `.gitignore` (veri, ağırlıklar, `__pycache__`, `.ipynb_checkpoints`, `wandb/`)
- [ ] `requirements.txt`: `torch`, `segmentation-models-pytorch`, `albumentations`, `rasterio`/`shapely`, `opencv-python`, `scikit-learn`, `grad-cam`, `pyyaml`, `pandas`, `matplotlib`
- [ ] `README.md`: proje özeti, kurulum, ilerleme günlüğü bölümü
- [ ] xBD'ye Kaggle'da eriş (mevcut dataset veya kendi yüklemen)
- [ ] Kaggle notebook'ta repoyu `git clone` + `pip install -e .` ile çalıştırmayı test et

**Çıktı:** Çalışan boş iskelet, veri erişimi, `docs/data_sources.md`
**Bitirme kriteri:** Kaggle notebook'u repodan bir fonksiyon import edip xBD'den bir görüntü çifti gösterebiliyor.

---

### Faz 1 — Literatür Taraması

**Hedef:** Yöntem seçimlerini kaynaklarla gerekçelendirmek.

- [ ] xBD makalesi (Gupta ve ark., 2019) — veri seti, etiket şeması, xView2 metriği
- [ ] xView2 yarışmasının kazanan çözümleri (Siamese U-Net, iki aşamalı yaklaşımlar)
- [ ] Focal Loss (Lin ve ark., 2017), Dice Loss
- [ ] Grad-CAM (Selvaraju ve ark., 2017), segmentasyon için Grad-CAM uyarlamaları
- [ ] Her kaynak için 3–5 satırlık özet

**Çıktı:** `docs/literature.md` (+ BibTeX: `docs/references.bib`)
**Bitirme kriteri:** Her tasarım kararı (mimari, kayıp, metrik, XAI) en az bir kaynakla eşleşiyor.

---

### Faz 2 — Keşifsel Veri Analizi (EDA)

**Hedef:** Veriyi tanımak; dengesizliği sayılarla göstermek.

- [ ] Olay (event) ve afet türü bazında görüntü sayıları
- [ ] Sınıf bazında **bina** ve **piksel** sayıları (genel + olay bazında)
- [ ] Görüntü başına bina sayısı dağılımı; hiç bina içermeyen görüntü oranı
- [ ] Örnek pre/post/maske görselleri (her afet türünden)
- [ ] Pre/post hizalama kalitesi, bulut/gölge gibi sorunlara göz at
- [ ] Bulguları özetle → Focal + Dice ve sampler kararlarının gerekçesi

**Çıktı:** `notebooks/02_eda.ipynb` (çıktılarıyla commit'lenir), `reports/figures/eda_*`
**Bitirme kriteri:** "Sınıf dengesizliği" ve "afet türü dengesizliği" grafiklerle gösterilmiş.

---

### Faz 3 — Ön İşleme ve Split

**Hedef:** Eğitime hazır, tekrarlanabilir veri.

- [ ] **Maske üretimi:** post JSON'lardaki poligonları rasterize et (`subtype` → sınıf id). `un-classified` → `255`
- [ ] **Split (tile'lamadan ÖNCE, görüntü seviyesinde):**
  - [ ] Tüm tier'ları tek tabloda birleştir (`image_id, event, disaster_type, tier, n_buildings, ...`)
  - [ ] `train_test_split(stratify=event, seed=42)` → %60 / %40, ardından %40 → %20 / %20
  - [ ] Çok az örneği olan olaylar için kontrol/elle dağıtım
  - [ ] `splits/train.csv`, `splits/val.csv`, `splits/test.csv` → **repoya commit**
  - [ ] Split dağılım tablosu: her split'te olay ve sınıf oranları (stratification'ın kanıtı)
- [ ] **Sızıntı kontrolü:** aynı `image_id` iki split'te yok (assert/test)
- [ ] **Tile'lama:** 1024×1024 → 512×512 (veya yeniden boyutlandırma; karar EDA'ya göre)
- [ ] İşlenmiş veriyi Kaggle Dataset olarak kaydet (versiyonlu)

**Çıktı:** `src/data/rasterize.py`, `src/data/split.py`, `src/data/tile.py`, `notebooks/03_preprocessing_split.ipynb`, `splits/*.csv`
**Bitirme kriteri:** Split dosyaları sabit, sızıntı testi geçiyor, split tabloları raporlanmış.

---

### Faz 4 — Eğitim Altyapısı

**Hedef:** Config ile yönetilen, kesintiye dayanıklı eğitim döngüsü.

- [ ] **Dataset:** pre + post + maske döndürür; **pre ve post'a birebir aynı augmentation** (albumentations `additional_targets`)
- [ ] Augmentation: flip, rotate90, hafif ölçek/kırpma, renk oynaması (pre/post için ayrı ayrı küçük renk farkı opsiyonel)
- [ ] **Kayıplar:**
  - [ ] `FocalLoss(alpha=class_weights, gamma=2, ignore_index=255)`
  - [ ] `DiceLoss(ignore_index=255)`
  - [ ] `CombinedLoss(λ)`
- [ ] **Metrikler:** xView2 skoru (`0.3·F1_loc + 0.7·F1_dmg`, F1_dmg = 4 hasar sınıfının harmonik ortalaması), sınıf bazında F1/IoU, confusion matrix
- [ ] **Early stopping:** `patience = max(1, int(0.2 * max_epochs))`; izlenen metrik: val xView2 skoru
- [ ] Her epoch sonunda checkpoint (`last.pt`) + en iyi model (`best.pt`); `--resume` desteği
- [ ] Seed sabitleme (`torch`, `numpy`, `random`, `cudnn.deterministic`)
- [ ] Mixed precision (AMP)
- [ ] Loglama: CSV + (opsiyonel) Weights & Biases
- [ ] `configs/*.yaml` ile tüm hiperparametreler; config dosyası her deneyin çıktısına kopyalanır
- [ ] **Sağlamlık testi:** tek bir batch'i overfit edebiliyor mu?

**Çıktı:** `src/train.py`, `src/losses/`, `src/metrics/`, `src/data/dataset.py`, `configs/`
**Bitirme kriteri:** Tek batch overfit başarılı; eğitim kesilip `--resume` ile devam edebiliyor.

---

### Faz 5 — Baseline Model

**Hedef:** Karşılaştırma noktası oluşturmak.

- [ ] **Baseline:** sadece post görüntü alan tek encoder'lı U-Net (ResNet34, ImageNet pretrained)
- [ ] Aynı split, aynı kayıp, aynı early stopping
- [ ] Val skorlarını kaydet
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
| A5 | Örnekleme | Normal vs. WeightedRandomSampler (deprem ağırlıklı) |
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
- [ ] Olay bazında ve afet türü bazında skor kırılımı (özellikle deprem olayları)
- [ ] Baseline'ın test skoru (karşılaştırma için)
- [ ] Nitel sonuçlar: iyi ve kötü tahmin örnekleri

**Çıktı:** `src/evaluate.py`, `notebooks/07_evaluate_xbd_test.ipynb`, `reports/results.md` güncel
**Bitirme kriteri:** Test sonuçları tablo ve figürlerle raporlanmış; `best.pt` GitHub Release'e yüklenmiş.

---

### Faz 8 — Grad-CAM / XAI

**Hedef:** "Model kararını görüntünün neresine bakarak veriyor?" sorusunu görsel ve yorumla cevaplamak.

- [ ] **Wrapper:** Siamese modeli tek girdi (6 kanal: pre+post) alacak şekilde sar
- [ ] **Hedef:** `SemanticSegmentationTarget` — c sınıfının tahmin maskesi bölgesindeki logit toplamı
- [ ] **Katman karşılaştırması:** encoder son bloğu vs. decoder katmanı
- [ ] **Görsel şablonu:** Pre | Post | Gerçek maske | Tahmin | Grad-CAM (sınıf bazında)
- [ ] Her hasar sınıfı için örnekler
- [ ] **Doğru vs. yanlış** tahmin örnekleri + kısa yorumlar
- [ ] **Baseline vs. Siamese** Grad-CAM karşılaştırması (pre görüntünün modele ne kattığını göstermek için)
- [ ] (Bonus) Grad-CAM++ ve deletion testi (önemli bölge maskelenince skor düşüşü)

**Çıktı:** `src/xai/gradcam.py`, `src/xai/wrappers.py`, `notebooks/08_gradcam_xai.ipynb`, `reports/figures/xai_*`
**Bitirme kriteri:** Raporun XAI bölümü için en az 8–10 yorumlanmış figür hazır.

> 🏁 **Ana gövde tamam.** Faz 8 bitince danışmanın tüm şartları karşılanmış olur ve `v0.8-core` tag'i atılır.
> **Karar noktası:** Teslime yeterli süre kaldıysa → Faz 9. Kalmadıysa → doğrudan Faz 10; Türkiye çalışması raporda "Gelecek Çalışmalar" bölümüne yazılır.

---

### Faz 9 — Genişleme: Türkiye Fine-Tune 

**Hedef:** xBD modelinin Kahramanmaraş 2023 depremine transferini ölçmek.

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
- [ ] Limitasyonlar: coğrafi korelasyon, Kaggle hesaplama kısıtları; *(Faz 9 yapıldıysa)* etiket uyumsuzluğu, nadir görüntüde katlı bina çöküşünün zor görünmesi
- [ ] README'yi son haline getir: sonuçlar, nasıl çalıştırılır, model ağırlığı linki
- [ ] Sunum slaytları + (opsiyonel) demo notebook'u
- [ ] `v1.0` tag'i + GitHub Release

**Bitirme kriteri:** Rapor teslim edildi; repo baştan sona tekrarlanabilir.

---

## 3. Repo Yapısı

```
damage-assessment/
├── configs/
│   ├── baseline.yaml
│   ├── siamese_r34.yaml
│   └── turkey_finetune.yaml       # Faz 9 (opsiyonel)
├── splits/                        # commit'lenir — tekrarlanabilirliğin kanıtı
│   ├── train.csv  val.csv  test.csv
│   └── turkey_*.csv               # Faz 9 (opsiyonel)
├── src/
│   ├── data/                      # rasterize.py, split.py, tile.py, dataset.py, transforms.py
│   ├── models/                    # baseline_unet.py, siamese_unet.py
│   ├── losses/                    # focal.py, dice.py, combined.py
│   ├── metrics/                   # xview2.py, confusion.py
│   ├── xai/                       # gradcam.py, wrappers.py, deletion_test.py
│   ├── utils/                     # seed.py, checkpoint.py, logger.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_setup_check.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_split.ipynb
│   ├── 05_train_baseline.ipynb
│   ├── 06_train_siamese.ipynb
│   ├── 07_evaluate_xbd_test.ipynb
│   ├── 08_gradcam_xai.ipynb
│   └── 09_turkey_finetune.ipynb   # Faz 9 (opsiyonel)
├── tests/                         # test_split_leakage.py, test_losses.py, test_metrics.py
├── docs/                          # literature.md, data_sources.md, references.bib
├── reports/
│   ├── figures/
│   └── results.md                 # tüm deney sonuç tabloları
├── ROADMAP.md
├── README.md
├── requirements.txt
├── setup.py / pyproject.toml      # pip install -e . için
└── .gitignore
```

**Repoya girmeyenler:** ham/işlenmiş veri, model ağırlıkları (`*.pt`), `wandb/`, büyük çıktılar. Ağırlıklar → GitHub Releases veya Kaggle Models.

---
*Son güncelleme: 28 Ekim 2026*
