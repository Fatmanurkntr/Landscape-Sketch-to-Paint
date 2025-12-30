# **🎨 Landscape Sketch to Paint: U-Net ve Pix2Pix GAN ile Görüntü Sentezi**

**Ders:** Derin Öğrenme (Deep Learning) Dönem Projesi

**Konu:** Image-to-Image Translation (Görüntüden Görüntüye Çeviri)

**Model Yaklaşımı:** Aşamalı Geliştirme (Baseline U-Net \-\> Final Pix2Pix GAN)

**Eğitim Donanımı:** NVIDIA Tesla T4 GPU

🔗 **Canlı Demo Uygulaması:** [Streamlit Üzerinde Görüntüle](https://landscape-sketch-to-paint-8jikjxxn4lxsqcxfcebwpr.streamlit.app/)

🔗 **Veri Seti:** [Kaggle \- Landscape Pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)

## 📄 Proje Raporu

Projenin teknik detaylarını, model mimarisini ve deneysel sonuçlarını içeren
detaylı raporu incelemek için aşağıdaki bağlantıya tıklayabilirsiniz:

[👉 Proje Raporunu Görüntüle (PDF)](Rapor.pdf)

<img width="1635" height="929" alt="Ekran görüntüsü 2025-12-30 015604" src="https://github.com/user-attachments/assets/656cf571-01f4-4fe8-accc-230ceeed0a1f" />


## 1. Proje Konusu ve Seçilme Gerekçesi 

### **Problem Tanımı**

Bu proje, bilgisayarlı görü (Computer Vision) alanında **"Image Synthesis"** problemini ele alır. Temel amaç, kullanıcının girdiği basit siyah-beyaz çizimleri (eskiz/sketch); anlamsal bütünlüğü koruyarak gerçekçi doku, renk ve ışıklandırmaya sahip manzara fotoğraflarına dönüştürmektir.

### **Projenin Önemi ve Literatürdeki Yeri**

Dijital sanat üretimi, oyun geliştirme (prosedürel içerik üretimi) ve mimari görselleştirme alanlarında konsept tasarımların nihai görsele dönüştürülmesi büyük bir iş yüküdür. Geleneksel Konvolüsyonel Sinir Ağları (CNN), piksel hatalarını minimize etmeye çalışırken (L1/L2 Loss) genellikle **bulanık (blurry)** sonuçlar üretir.

Bu proje, **Generative Adversarial Networks (GAN)** yapılarının bu bulanıklık sorununu nasıl çözdüğünü göstermek ve literatürde devrim yaratan **Pix2Pix** (Isola et al., 2017\) mimarisinin uçtan uca bir uygulamasını gerçekleştirmek amacıyla seçilmiştir.

## 2. Veri Seti ve Ön İşleme Süreçleri 

Projede Kaggle platformunda bulunan **Landscape Pictures** veri seti kullanılmıştır. Ancak veri seti doğrudan kullanılmamış, **dinamik bir ön işleme hattından (preprocessing pipeline)** geçirilmiştir.

### 2.1. Dinamik Veri Üretimi (Runtime Sketch Generation)

Projede hazır "sketch" verileri yerine, renkli fotoğraflardan çalışma zamanında taslak üreten bir yapı kurulmuştur. Bu işlem için **OpenCV Canny Edge Detection** algoritması kullanılmıştır.

* **Avantajı:** Modelin farklı kalem kalınlıklarına ve çizim stillerine karşı daha dayanıklı (robust) olmasını sağlar.

\# Proje kodundan örnek (Sketch Üretimi):  
gray \= cv2.cvtColor(img, cv2.COLOR\_RGB2GRAY)  
edges \= cv2.Canny(gray, 100, 200\) \# Girdi (Input) çalışma anında üretilir

### 2.2. Teknik Kısıtlamalar ve Optimizasyon

Eğitim süreci Kaggle Kernel ortamında (Tesla T4 GPU \- 16GB VRAM) gerçekleştirilmiştir. GAN eğitimi, aynı anda iki modelin (Generator \+ Discriminator) ağırlıklarını ve gradyanlarını bellekte tuttuğu için standart CNN'lere göre 2 kat daha fazla VRAM gerektirir. ResourceExhaustedError sorununu aşmak için şu optimizasyonlar uygulanmıştır:

1. **Veri Kısıtlaması (Data Culling):** Toplam veri seti içerisinden rastgele seçim yapılarak eğitim 2.500 görüntü ile sınırlandırılmıştır.  
2. **RAM Yönetimi (Garbage Collection):** Python gc modülü kullanılarak, işlenen ham veriler (del X\_full) bellekten manuel olarak temizlenmiştir.  
3. **Normalizasyon:** Görüntü pikselleri, Generator çıkışındaki Sigmoid aktivasyonuna uygun olarak $\[0, 1\]$ aralığına normalize edilmiştir (img / 255.0).  
4. **Veri Ayrımı:** Veri seti karıştırılarak (shuffle) %90 Eğitim, %10 Doğrulama olarak ayrılmıştır.

## 3. Yöntem Seçimi ve Karşılaştırmalı Analiz 

Proje kapsamında problem çözümüne aşamalı bir yaklaşım izlenmiş ve iki farklı deney gerçekleştirilmiştir.

### 3.1. Deneysel Süreç 1: Baseline Model (Sadece U-Net)

* **İlgili Dosya:** notebooks/Training\_UNet.ipynb

İlk aşamada, problemin sadece piksel tabanlı bir regresyon problemi olarak çözülüp çözülemeyeceği test edilmiştir.

* **Yöntem:** Standart U-Net mimarisine ek olarak her katmanda Batch Normalization kullanılarak eğitim stabilize edilmiştir.  
* **Kayıp Fonksiyonu:** L1 Loss (Mean Absolute Error).  
* **Sonuç Analizi:** Model nesnelerin yerini doğru öğrense de, dokular (çimen, kaya yüzeyi) pürüzsüz ve bulanık (blurry) çıkmıştır.  
* **Nedeni:** L1 kaybı, belirsizlik durumunda olası tüm renklerin "ortalamasını" almayı tercih eder.

### 3.2. Deneysel Süreç 2: Final Model (Pix2Pix GAN)

* **İlgili Dosya:** notebooks/Training\_Pix2Pix\_GAN.ipynb

Bulanıklık sorununu çözmek için sisteme Adversarial Learning (Çekişmeli Öğrenme) eklenmiştir.

* **Yöntem:** Koşullu GAN (cGAN) yapısı kurulmuştur.  
* **Discriminator (Eleştirmen):** Görüntünün tamamına tek puan vermek yerine, resmi $30 \\times 30$ boyutunda yamalara bölen **PatchGAN** kullanılmıştır. Bu, modelin yüksek frekanslı detayları (keskinliği) öğrenmesini zorunlu kılar.  
* **Sonuç Analizi:** Sonuçlar çok daha keskin, detaylı ve gerçekçidir.

## 4. Model Eğitimi ve Mimari Detaylar 

Aşağıdaki tablo, iki aşama arasındaki teknik farkları özetlemektedir:

| Özellik | 1\. Aşama (Baseline U-Net) | 2\. Aşama (Pix2Pix GAN) |
| :---- | :---- | :---- |
| **Model Yapısı** | U-Net \+ Batch Norm | U-Net (Gen) \+ PatchGAN (Disc) |
| **Parametre Sayısı** | \~31 Milyon | \~54 Milyon (Gen) \+ \~2.7 Milyon (Disc) |
| **Kayıp Fonksiyonu** | L1 Loss (MAE) | Adversarial Loss \+ (100 \* L1 Loss) |
| **Optimizer** | Adam (LR=0.001) | Adam (LR=0.0002, Beta1=0.5) |
| **Batch Size** | 32 | 4 (VRAM Optimizasyonu) |
| **Epochs** | 37 (Early Stopping) | 30 |
| **Aktivasyon (Çıkış)** | Sigmoid | Sigmoid |
| **Özel Teknikler** | ReduceLROnPlateau | Custom Training Loop, GANMonitor |

### 4.1. Generator Mimarisi (Ortak)

Her iki deneyde de Generator olarak **U-Net** kullanılmıştır.

* **Encoder:** Görüntüyü sıkıştırarak öznitelikleri çıkarır.  
* **Decoder:** Görüntüyü tekrar genişletir.  
* **Skip Connections:** Encoder'daki kenar bilgilerini doğrudan Decoder'a taşıyarak taslağın şeklinin korunmasını sağlar.

### 4.2. Discriminator Mimarisi (PatchGAN)

Sadece GAN aşamasında kullanılmıştır. Görüntüyü $30 \\times 30$ boyutunda yamalara böler ve her yama için "Gerçek" veya "Sahte" kararı verir.

### 4.3. Kayıp Fonksiyonları

$$ Total Loss \= Loss\_{GAN} \+ (\\lambda \\times Loss\_{L1}) $$

* **Adversarial Loss:** Discriminator'ı kandırma başarısı (Gerçekçilik).  
* **L1 Loss:** Piksel bazlı benzerlik (Renk Doğruluğu). $\\lambda \= 100$ katsayısı ile ağırlıklandırılmıştır.

## 5. Sonuçların Değerlendirilmesi

### 5.1. Sayısal Analiz (Metrics)

Modelin başarısı test seti üzerinde SSIM ve PSNR metrikleri ile ölçülmüştür:

* **Ortalama PSNR:** 17.93 dB  
* **Ortalama SSIM:** 0.5333

**Yorum:** Bu değerlerin "mükemmel" (SSIM \> 0.8) sınırının altında kalmasının temel nedeni **Mevsimsel Belirsizliktir (Multimodality)**. Siyah-beyaz bir ağaç çizimi, "Sonbahar (Turuncu)" veya "İlkbahar (Yeşil)" olarak yorumlanabilir. Model görsel olarak başarılı olsa bile, orijinal fotoğraftan farklı bir mevsim/renk seçtiğinde piksel tabanlı metrikler matematiksel olarak düşük çıkmaktadır.

### 5.2. Görsel Analiz (Visual Inspection)

* **U-Net Sonuçları:** Yapısal olarak doğru ancak "sulu boya" etkisi yaratan bulanık sonuçlar.  
* **GAN Sonuçları:** Nehir yansımaları, bulut dokuları ve dağ yüzeylerinde belirgin keskinlik artışı. Ayrıca Eğitim sırasında GANMonitor callback'i ile her epoch sonunda üretilen görsellerdeki gelişim net bir şekilde gözlemlenmiştir.

## 6. Proje Dokümantasyonu ve Dosya Yapısı 

Proje dosyaları, yeniden üretilebilirlik (reproducibility) ilkesine uygun olarak, kodun modülerliğini ve okunabilirliğini artıracak şekilde organize edilmiştir. Aşağıda dizin yapısı ve dosyaların işlevleri detaylandırılmıştır:

```text
Landscape-Sketch-to-Paint/
├── app.py                     # Streamlit web arayüzü ana çalıştırma dosyası
├── requirements.txt           # Proje için gerekli Python kütüphaneleri ve sürümleri
├── style_utils.py             # Arayüz için özel CSS ve HTML tasarım kodları
├── src/                       # Kaynak Kodlar (Modüler Mimari)
│   ├── model.py               # U-Net ve GAN (Generator/Discriminator) mimari tanımları
│   └── __init__.py            # Klasörün Python paketi olarak tanınmasını sağlar
├── notebooks/                 # Model Eğitim Süreçleri (Kanıt Dosyaları)
│   ├── Training_UNet.ipynb        # 1. Aşama: Baseline U-Net deneyleri ve sonuçları
│   └── Training_Pix2Pix_GAN.ipynb # 2. Aşama: Final Pix2Pix GAN modelinin eğitimi
├── examples/                  # Test ve demo için kullanılan örnek taslak görselleri
├── models/                    # (Otomatik oluşturulur) Eğitilmiş ağırlık dosyalarının indiği klasör
└── README.md                  # Proje teknik raporu ve kurulum kılavuzu

```
## 7. Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.  **Depoyu klonlayın:**
    ```bash
    git clone https://github.com/Fatmanurkntr/Landscape-Sketch-to-Paint.git
    cd Landscape-Sketch-to-Paint
    ```

2.  **Gerekli kütüphaneleri yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Uygulamayı başlatın:**
    ```bash
    streamlit run app.py
    ```
    *(Not: Uygulama ilk açılışta Google Drive entegrasyonu sayesinde eğitilmiş model dosyalarını otomatik olarak indirecektir. Bu işlem internet hızınıza bağlı olarak birkaç dakika sürebilir.)*

    
## **8\. Referanslar**

1. Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. *Proceedings of the IEEE conference on computer vision and pattern recognition*.  
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. *MICCAI*.  
3. TensorFlow Core Tutorials: Pix2Pix.
