# BIL475-Pattern-Recognition-Projects

Bu depo, BIL475 Örüntü Tanıma dersi kapsamında gerçekleştirilen iki ana çalışmayı içermektedir. Proje; **Sınıflandırma (Classification)** ve **Regresyon (Regression)** problemlerine odaklanarak, farklı modellerin başarısını belirli metrikler üzerinden karşılaştırmaktadır.

## 👥 Ekip Üyeleri
* **Üye 1:** Azra Öykü Ulukan
* **Üye 2:** Begüm Karabaş
* **Üye 3:** Emre Veriş

---

## 📂 Proje İçerikleri ve Teknik Detaylar

### 1. Banknot Kimlik Doğrulama (Sınıflandırma)
* **Veri Seti:** `data_banknote_authentication.mat`
* **Problem:** İkili Sınıflandırma (Sahte/Gerçek Banknot Ayrımı)
* **Kullanılan Modeller:** [Model 1: kNN] ve [Model 2: SVM]
* **Zorunlu Metrikler:**
    * **ACC** (Doğruluk)
    * **F-score**
* **Görselleştirme:** Hata Matrisi (Confusion Matrix)

### 2. Gaz Türbini Emisyon Tahmini (Regresyon)
* **Veri Seti:** `Gas_Turbine_Co_NoX_2015.mat`
* **Problem:** NOx / CO Emisyon Tahmini (Sürekli Değişken)
* **Kullanılan Modeller:** [Model 1: XGBoost] ve [Model 2: ANN]
* **Zorunlu Metrikler:**
    * **MAE** (Ortalama Mutlak Hata)
    * **SMAPE** (Simetrik Ortalama Mutlak Yüzde Hata)
* **Görselleştirme:** x = y (Gerçek vs Tahmin) Grafiği


> **⚠️ Not:** Regresyon görselleştirmelerinde veri seti 1000'den fazla örnek içerdiği için rastgele 1000 örnek üzerinden analiz yapılmıştır.

---

## 🚀 Projeyi Çalıştırma

1. Repoyu klonlayın:
   ```bash
   git clone https://github.com/azraoykulukan/BIL475-Pattern-Recognition-Projects.git
