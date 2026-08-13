<div align="center">

# Türkiye İç Göç Atlası

**2008–2025 TÜİK verileriyle Türkiye'nin iç göç dinamiklerinin veri bilimi analizi**

Python · Pandas · Scikit-learn · Plotly · NetworkX · Streamlit

</div>

![Yıllara göre iller arası göç](reports/figures/yearly_migration.png)

## Project Overview

Türkiye İç Göç Atlası, 2008–2025 dönemindeki 116.640 kaynak–hedef akışını ve 2025 demografi tablolarını uçtan uca bir veri ürününde birleştirir. Proje; ham TÜİK dosyalarının doğrulanması, keşifçi analiz, yönlü ağ analizi, uzun dönemli feature engineering, gözetimsiz öğrenme, model persistence ve Streamlit dashboard adımlarını tekrar üretilebilir biçimde kapsar.

## Problem

İllerin yalnızca tek yıllık net göç sıralamasına bakmak; hareket hacmini, oynaklığı, uzun dönem eğilimini ve kaynak–hedef bağlantılarını gizler. Bu proje üç sorunu birlikte ele alır:

- Aynı göç hareketini iki kez saymadan akış semantiğini doğrulamak.
- 81 ilin 18 yıllık davranışını karşılaştırılabilir oran ve trendlerle temsil etmek.
- Analiz ve model sonuçlarını kullanılabilir, interaktif bir veri ürününe dönüştürmek.

## Dataset

| Kaynak | Kapsam | Gözlem seviyesi | Analizdeki rolü |
|---|---:|---|---|
| `iller_arasi_goc.xlsx` | 2008–2025 | Yıl × kaynak il × hedef il | Trend, koridor, ağ ve model özellikleri |
| `illerin_goc_ozeti.xls` | 2025 | İl | 81-il resmi uzlaştırma |
| `yas_cinsiyet_goc.xls` | 2025 | Yaş grubu | Yaş ve cinsiyet profili |
| `yas_cinsiyet_neden.xls` | 2025 | Yaş × cinsiyet × neden | Demografik motivasyonlar |
| `egitim_goc_nedeni.xls` | 2025 | Eğitim × neden | Eğitim-motivasyon ilişkisi, 6+ yaş |

Ham dosyalar reproducibility için `data/raw/` altında korunur. Processed tablolar toplam, kaynak, dipnot ve metadata satırlarından arındırılmıştır.

## Research Questions

- Türkiye'deki toplam iller arası göç 2008–2025 arasında nasıl değişti?
- Hangi iller mutlak ve nüfusa oranlı net göçte ayrışıyor?
- En güçlü yönlü göç koridorları hangileri?
- Yaş, cinsiyet, eğitim ve göç nedenleri nasıl dağılıyor?
- İller uzun dönemli hareketlilik, net göç, volatilite, trend ve nüfus özelliklerine göre hangi profillerde gruplanıyor?

## Key Insights

- Yıllık iller arası göç 2023'te **3.450.953 kişiyle** dönem zirvesine ulaştı.
- 2025 toplamı **2.475.019 kişi**; 2024'e göre **%7,7 daha düşük**.
- İstanbul 2025'te en yüksek giriş ve çıkış hacmine karşın **41.346 kişilik net kayıp** verdi.
- Ankara **31.172 kişiyle** en yüksek mutlak net kazancı kaydetti.
- **İstanbul → Kocaeli**, 23.723 kişiyle 2025'in en büyük yönlü koridoru oldu.
- **20–24 yaş** grubu 480.185 kişiyle en hareketli yaş grubu; hane/aile ferdine bağımlı göç 564.114 kişiyle en yaygın kaydedilen neden oldu.

Ayrıntılı metodoloji ve bulgular [analitik raporda](reports/project_report.md) yer alır.

## Machine Learning

Production modeli bir Scikit-learn Pipeline içinde preprocessing ve K-Means adımlarını birlikte saklar. Model girdileri:

- 2008–2025 ortalama hareketlilik hızı
- 2008–2025 ortalama net göç hızı
- net göç hızı volatilitesi
- net göç hızı doğrusal trendi
- 2023–2025 ortalama net göç hızı
- log-dönüşümlü 2025 nüfusu

`k=2…8` aralığı inertia, Silhouette, Davies–Bouldin, Calinski–Harabasz ve 10 seed arası Adjusted Rand Index ile karşılaştırıldı. En az 3 il içeren ve tek kümede illerin %75'inden fazlasını toplamayan adaylar arasında metrik dengesi **k=4** sonucunu verdi.

| Metrik | Sonuç |
|---|---:|
| Silhouette | 0,298 |
| Davies–Bouldin | 1,031 |
| Calinski–Harabasz | 36,82 |
| Seed stability ARI | 0,993 |

HDBSCAN karşılaştırması 2 küme ve 25 noise il üretti. Yeni gözlemlere tekrar profil atayabilen K-Means production modeli; HDBSCAN keşifsel karşılaştırma olarak tutuldu. PCA yalnızca boyut indirgeme ve görselleştirme için kullanıldı; ilk iki bileşen varyansın %70,5'ini açıklıyor.

## Cluster Profiles

| Profil | İl sayısı | Uzun dönem karakteristiği |
|---|---:|---|
| Büyük Ölçekli Çekim Merkezleri | 22 | Yüksek ortalama nüfus ve pozitif net göç hızı |
| Dengeli Orta Hareketlilik | 41 | Orta hareketlilik, uzun dönemde dengeye yakın profil |
| Küçük Ölçekli Yüksek Hareketlilik | 4 | Düşük nüfusa karşın yüksek kişi başı hareketlilik |
| Süregelen Net Göç Veren İller | 14 | Uzun ve yakın dönemde belirgin negatif net göç hızı |

![PCA düzleminde cluster profilleri](reports/figures/cluster_profiles_pca.png)

Profil adları sabit/keyfi eşiklerden değil, cluster centroidlerinin nüfus, hareketlilik ve net göç sıralamalarından üretilir. İl listeleri ve centroid değerleri `models/cluster_metadata.json` içinde sürümlenir.

## Dashboard

Streamlit veri ürünü modeli açılışta yeniden eğitmez; kaydedilmiş artefact'ları cache ile diskten yükler.

- **Genel Bakış:** yıla göre KPI'lar, uzun dönem trendi, net göç ve koridorlar
- **İl Profili:** 81 il için yıllık metrikler, 18 yıllık trend, en güçlü bağlantılar, profil ve CSV indirme
- **Göç Akışları:** yıl, kaynak, hedef ve minimum hacim filtreli sade sıralama; isteğe bağlı gelişmiş akış diyagramı
- **Demografi:** yaş, cinsiyet, neden ve eğitim görselleri
- **Şehir Profilleri:** model karşılaştırması, profil özetleri, interaktif PCA ve yeni gözlem profil ataması
- **Veri & Metodoloji:** veri sözlüğü, hesaplama yöntemi, sınırlamalar ve indirilebilir temiz tablolar

Ürün arayüzü veri/ürün sürüm rozetleri, kullanıcı dostu açıklamalar, otomatik kısa yorumlar,
sayfa bazlı dışa aktarma ve TÜİK kaynak beyanı içerir.

Profil atama aracı geleceği tahmin etmez; girilen özellikleri eğitilmiş clustering uzayındaki en yakın profile atar.

## Project Structure

```text
turkiye-ic-goc-atlasi/
├── app.py
├── data/
│   ├── raw/                         # Orijinal TÜİK Excel dosyaları
│   └── processed/                   # Temiz tablolar ve model profilleri
├── models/
│   ├── migration_cluster_pipeline.joblib
│   ├── pca_model.joblib
│   └── cluster_metadata.json
├── notebooks/
│   └── turkiye_ic_goc_atlasi.ipynb
├── reports/
│   ├── project_report.md
│   └── figures/
├── src/
│   ├── config.py
│   ├── data_cleaning.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── network_analysis.py
│   └── reporting.py
└── .streamlit/config.toml
```

## Tech Stack

- **Data:** Pandas, NumPy, OpenPyXL, xlrd
- **Machine Learning:** Scikit-learn, Joblib
- **Visualization:** Matplotlib, Plotly
- **Network Analysis:** NetworkX
- **Data Product:** Streamlit
- **Notebook:** Jupyter

## Installation

Python 3.9 veya üzeri önerilir.

```bash
git clone <repository-url>
cd turkiye-ic-goc-atlasi
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ham veriden processed tabloları, model artefact'larını ve raporu yeniden üretmek için:

```bash
python -m src.data_cleaning
python -m src.modeling
python -m src.reporting
```

## Run Notebook

```bash
jupyter lab notebooks/turkiye_ic_goc_atlasi.ipynb
```

Notebook proje kökünden veya `notebooks/` klasöründen çalıştırılabilir ve restart + run all akışıyla processed veri/model artefact'larını yeniden üretir.

## Run Streamlit

```bash
streamlit run app.py
```

## Methodology

1. Ham tablolar başlık yapılarına göre okunur; metadata, dipnot ve toplam satırları kaldırılır.
2. 81 il, 2008–2025 yıl aralığı, duplicate, missing, negatif alan, self-loop ve yıllık 6.480 akış assertionlarla doğrulanır.
3. Ters yön simetrisi ve `net = gelen - giden` kontrol edilir; 2025 il agregasyonları resmi özetle uzlaştırılır.
4. İl-yıl panelinden nüfusa oranlı hızlar, hareketlilik, uzun dönem ortalaması, volatilite ve trend üretilir.
5. StandardScaler + K-Means pipeline eğitilir; HDBSCAN keşifsel karşılaştırma, PCA görselleştirme olarak kullanılır.
6. Model diskten yeniden yüklenir ve cluster atamalarının eğitim sonucu ile aynı olduğu test edilir.

## Results

- 81 il × 18 yıl = **1.458** il-yıl gözlemi
- Her yıl **6.480** self-loop içermeyen yönlü akış
- **4** yorumlanabilir uzun dönemli il profili
- Model reload sonucu eğitim atamalarıyla **birebir tutarlı**
- Ana notebookta **21 statik grafik**, **1 interaktif PCA grafiği** ve grafiklerle birlikte sunulan veri-temelli yorum blokları
- GitHub'da görüntülenebilir ana notebook boyutu yaklaşık **1,8 MB**

## Limitations

- Demografi ve göç nedeni tabloları yalnızca 2025 kesitidir.
- Kaynak tablodaki `-` işareti bilgi-yok/magnitude-null olarak korunur; sıfır kabul edilmez.
- 81 il, clustering için sınırlı bir örneklemdir; içsel metrikler tek bir evrensel segmentasyon garanti etmez.
- Cluster'lar nedensellik veya gelecek tahmini sunmaz.
- Gelir, istihdam, konut maliyeti ve afet etkisi gibi dış değişkenler modelde bulunmaz.

## Future Improvements

- Kaynaklı sosyoekonomik ve mekânsal değişkenleri eklemek
- Bootstrap tabanlı cluster stability güven aralıkları hesaplamak
- Akışları coğrafi harita üzerinde göstermek
- Yeni TÜİK yılları için veri ve model sürümleme mekanizması kurmak

## Data Source

Veriler Türkiye İstatistik Kurumu (TÜİK) **İç Göç İstatistikleri** tablolarından alınmıştır. Ham dosyalardaki TÜİK kaynak ve açıklama satırları `data/raw/` altında korunur.

## License

Bu repository için henüz ayrı bir yazılım lisansı tanımlanmamıştır. Kaynak TÜİK verilerinin kullanım koşulları ayrıca geçerlidir.
