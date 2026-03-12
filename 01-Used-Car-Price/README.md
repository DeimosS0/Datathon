# 🏎️ Used Car Price Prediction (Craigslist Dataset)

Bu proje, Amerika'daki Craigslist ilanlarından alınan devasa bir veri setini kullanarak ikinci el araç fiyatlarını tahmin etmeyi amaçlar.

## 📊 Model Başarı Metrikleri
Modelimiz **RandomForestRegressor** algoritması kullanılarak eğitilmiştir ve aşağıdaki sonuçları vermiştir:

- **R2 Skoru:** `%91.2` (Yüksek açıklanabilirlik oranı)
- **Ortalama Mutlak Hata (MAE):** `1947 Dolar`

---

## 🔍 Tahmin Örnekleri
Modelin test verisi üzerindeki rastgele tahmin performansları:

| Gerçek Fiyat | Model Tahmini | Fark |
| :--- | :--- | :--- |
| 12,900 $| 13,154$ | 254 $ |
| 27,990 $| 27,990$ | 0 $ |
| 13,980 $| 13,928$ | 52 $ |

---

## 🛠️ Yapılan İşlemler
1. **Veri Temizliği:** `id`, `url`, `image_url` gibi tahminleme için anlamsız olan sütunlar kaldırıldı.
2. **Filtreleme:** Hatalı verileri elemek için 500$ altı ve 100,000$ üstü ilanlar temizlendi.
3. **Kategorik Dönüşüm:** `LabelEncoder` kullanılarak metin tabanlı veriler (marka, model vb.) sayısal formata çevrildi.
4. **Model Eğitimi:** Veri seti %80 eğitim, %20 test olarak ayrıldı ve çok çekirdekli (`n_jobs=-1`) işlem yapıldı.

## 📂 Veri Seti
Projeye ait veri setine [Kaggle - Craigslist Car/Trucks Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) üzerinden ulaşabilirsiniz.
