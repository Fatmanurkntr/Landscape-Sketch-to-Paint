# 🎨 Landscape Sketch to Paint: U-Net ve Pix2Pix GAN ile Görüntü Sentezi

**Ders:** Derin Öğrenme (Deep Learning) Dönem Projesi


**Konu:** Image-to-Image Translation (Görüntüden Görüntüye Çeviri)


**Model Yaklaşımı:** Aşamalı Geliştirme (Baseline U-Net -> Final Pix2Pix GAN)

🔗 **Canlı Demo Uygulaması:** [Streamlit Üzerinde Görüntüle](https://landscape-sketch-to-paint-8jikjxxn4lxsqcxfcebwpr.streamlit.app/)

🔗 **Veri Seti:** [Kaggle - Landscape Pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)

---

## 1. Proje Konusu ve Seçilme Gerekçesi

### Problem Tanımı
Bu proje, bilgisayarlı görü (Computer Vision) alanında "Image Synthesis" olarak bilinen problemi ele alır. Temel amaç, az bilgi içeren giriş verilerini (siyah-beyaz taslaklar/kenar haritaları), yüksek frekanslı detaylara sahip fotorealistik görüntülere (manzara fotoğrafları) dönüştürmektir. Proje, kullanıcının çizdiği basit dağ, nehir veya ağaç sınırlarını algılayarak; bu alanları anlamsal bütünlüğe uygun doku, renk ve ışıklandırma ile doldurmayı hedefler.

### Projenin Önemi ve Literatürdeki Yeri
Dijital sanat üretimi, oyun geliştirme (prosedürel içerik üretimi) ve mimari görselleştirme alanlarında, konsept tasarımların nihai görsele dönüştürülmesi ciddi bir iş yükü oluşturur. Geleneksel yöntemler manuel boyama gerektirirken, **Generative Adversarial Networks (GAN)** tabanlı yaklaşımlar bu süreci otomatize eder.
Bu proje, literatürde devrim yaratan **Pix2Pix** (Isola et al., 2017) mimarisinin pratik bir uygulamasını sunmak, geleneksel Konvolüsyonel Sinir Ağları (CNN) ile GAN tabanlı yaklaşımları karşılaştırmalı olarak analiz etmek amacıyla seçilmiştir.

---

## 2. Veri Seti ve Ön İşleme Süreçleri

Projede Kaggle platformunda bulunan **Landscape Pictures** veri seti kullanılmıştır. Veri seti, gerçek doğa fotoğraflarını ve bu fotoğraflardan algoritmik yöntemlerle (Canny Edge Detection vb.) türetilmiş kenar haritalarını (sketch) içerir.

### Teknik Kısıtlamalar ve Optimizasyon
Eğitim süreci **Kaggle Kernel** ortamında (Tesla P100 GPU - 16GB VRAM) gerçekleştirilmiştir. Orijinal veri setinde 3.500'den fazla görüntü çifti bulunmaktadır. Ancak GAN mimarisinin, standart bir CNN'e göre yaklaşık 2 kat daha fazla bellek gerektirmesi (Generator + Discriminator + Gradient Tape hesaplamalarının VRAM üzerinde tutulması) nedeniyle `ResourceExhaustedError` (RAM Taşması) sorunu yaşanmıştır.

Eğitimi stabilize etmek ve donanım limitleri dahilinde en iyi sonucu almak için aşağıdaki optimizasyon stratejileri uygulanmıştır:

1.  **Veri Seti Alt Örnekleme (Random Subsampling):** Bellek yönetimini sağlamak amacıyla veri seti içerisinden rastgele seçim yapılarak eğitim seti **2.500 görüntü çiftine** indirilmiştir.
2.  **Yeniden Boyutlandırma (Resizing):** Tüm giriş (sketch) ve çıkış (photo) görüntüleri $256 \times 256$ piksel boyutuna sabitlenmiştir.
3.  **Normalizasyon:** Görüntü pikselleri, Generator modelinin çıkış katmanındaki `Tanh` aktivasyon fonksiyonunun çalışma aralığına uygun olması için $[0, 255]$ aralığından $[-1, 1]$ aralığına normalize edilmiştir.
4.  **Veri Ayrımı:** Veri seti %80 Eğitim, %20 Test olacak şekilde ayrılmıştır.

---

## 3. Yöntem Seçimi ve Karşılaştırmalı Analiz (Deneysel Süreç)

Proje kapsamında problem çözümüne **aşamalı** bir yaklaşım izlenmiş ve iki farklı deney gerçekleştirilmiştir. Bu deneyler, kayıp fonksiyonlarının görüntü kalitesi üzerindeki etkisini göstermektedir.

### 3.1. Deneysel Süreç 1: Baseline Model (Sadece U-Net)
**İlgili Dosya:** `notebooks/Training_UNet.ipynb`

İlk aşamada, problemin sadece piksel tabanlı bir regresyon problemi olarak çözülüp çözülemeyeceği test edilmiştir.
* **Yöntem:** Standart bir U-Net mimarisi (Encoder-Decoder + Skip Connections) kurulmuştur.
* **Kayıp Fonksiyonu:** L1 Loss (Mean Absolute Error). Model, `|Gerçek - Tahmin|` farkını minimize etmeye odaklanmıştır.
* **Sonuç Analizi:** Model, taslağın sınırlarını (dağların şeklini, nehrin yolunu) öğrenmede başarılı olmuştur. Ancak üretilen görseller **bulanık (blurry)** ve dokusuzdur.
* **Nedeni:** L1 kaybı, belirsizlik durumunda olası tüm renklerin "ortalamasını" almaya meyillidir. Bu durum, çim veya kaya gibi yüksek frekanslı detayların kaybolmasına neden olur.

### 3.2. Deneysel Süreç 2: Final Model (Pix2Pix GAN)
**İlgili Dosya:** `notebooks/Training_Pix2Pix_GAN.ipynb`

Bulanıklık sorununu çözmek için sisteme "Adversarial Learning" (Çekişmeli Öğrenme) mekanizması eklenmiştir.
* **Yöntem:** Koşullu GAN (cGAN) yapısı kurulmuştur.
* **Discriminator (Ayırt Edici):** Görüntünün tamamına tek bir puan vermek yerine, görüntüyü küçük yamalara (patch) bölerek inceleyen **PatchGAN** kullanılmıştır.
* **Sonuç Analizi:** PatchGAN, modelin sadece renkleri değil, yerel doku tutarlılığını (keskinliği) da öğrenmesini zorunlu kılmıştır. Sonuçlar çok daha gerçekçi, keskin ve detaylıdır.

---

## 4. Model Eğitimi ve Mimari Detaylar

### 4.1. Generator Mimarisi (Ortak)
Her iki deneyde de Generator olarak **U-Net** kullanılmıştır. U-Net, görüntüyü sıkıştırıp (Encoder) tekrar genişletirken (Decoder), aradaki detay kaybını önlemek için özel bir yapı kullanır.
* **Encoder (Downsampling):** `Conv2D`, `BatchNormalization` ve `LeakyReLU` katmanları ile görüntü 256x256 boyutundan 1x1 boyutuna sıkıştırılır (Feature Extraction).
* **Decoder (Upsampling):** `Conv2DTranspose` ile görüntü tekrar genişletilir. İlk 3 katmanda Dropout uygulanarak overfitting engellenir.
* **Skip Connections (Atlamalı Bağlantılar):** Encoder katmanındaki yapısal detaylar (kenarlar), darboğaz (bottleneck) katmanında kaybolmamaları için doğrudan Decoder katmanına kopyalanır (`Concatenate`). Bu, taslağın şeklinin korunmasını sağlar.

### 4.2. Discriminator Mimarisi (PatchGAN)
Sadece GAN aşamasında kullanılmıştır.
* Giriş olarak hem "Hedef Resim" hem de "Üretilen/Gerçek Resim" çiftini alır.
* Görüntüyü $30 \times 30$ boyutunda yamalara böler.
* Her yama için "Gerçek" veya "Sahte" kararı verir. Bu, modelin resmin geneline değil, ince detaylarına odaklanmasını sağlar.

### 4.3. Kayıp Fonksiyonları (Loss Functions)
GAN eğitimi sırasında karma bir kayıp fonksiyonu minimize edilmiştir:

$$Total Loss = Loss_{GAN} + (\lambda \times Loss_{L1})$$

1.  **Adversarial Loss:** Generator'ın Discriminator'ı kandırma başarısı. (Gerçekçilik sağlar).
2.  **L1 Loss:** Üretilen resmin orijinal fotoğrafla piksel bazında eşleşmesi. (Renk ve içerik doğruluğu sağlar).
    * $\lambda$ (Lambda) katsayısı, L1 kaybının etkisini artırmak için 100 olarak belirlenmiştir.

### 4.4. Eğitim Parametreleri
* **Platform:** Kaggle (Tesla P100 GPU)
* **Optimizer:** Adam ($\beta_1 = 0.5$, Learning Rate = 0.0002)
* **Batch Size:** 1 (Pix2Pix mimarisi için standart olan instance normalization etkisi).
* **Süre:** Modellerin yakınsaması ve loss değerlerinin stabilize olması yaklaşık 4-5 saat sürmüştür.

---

## 5. Kullanılan Teknolojiler ve Araçlar

Projenin geliştirilmesinde aşağıdaki kütüphane ve araçlar kullanılmıştır:

* **Python 3.9+:** Ana programlama dili.
* **TensorFlow & Keras:** Derin öğrenme modellerinin (U-Net, PatchGAN) oluşturulması, eğitilmesi ve tensör işlemleri.
* **Streamlit:** Eğitilen modelin son kullanıcıya sunulması için interaktif web arayüzü geliştirilmesi.
* **OpenCV (cv2):** Görüntü okuma, gri tonlamaya çevirme ve ön işleme (Canny Edge, Thresholding) işlemleri.
* **NumPy:** Matris operasyonları ve veri manipülasyonu.
* **Matplotlib:** Eğitim sırasındaki Loss grafiklerinin görselleştirilmesi.
* **Gdown:** Büyük boyutlu model ağırlıklarının Google Drive üzerinden çalışma zamanında (runtime) indirilmesi.
* **Pillow (PIL):** Görüntü formatı dönüşümleri.

---

## 6. Proje Dokümantasyonu ve Dosya Yapısı

Proje dosyaları, yeniden üretilebilirlik (reproducibility) ilkesine uygun olarak organize edilmiştir.

```text
Landscape-Sketch-to-Paint/
├── app.py                     # Streamlit web arayüzü ana dosyası
├── requirements.txt           # Gerekli Python kütüphaneleri
├── style_utils.py             # Arayüz için CSS ve tasarım kodları
├── src/                       # Kaynak Kodlar
│   ├── model.py               # U-Net ve GAN mimari tanımları (Generator/Discriminator)
│   └── __init__.py
├── notebooks/                 # Model Eğitim Süreçleri (Kanıt Dosyaları)
│   ├── Training_UNet.ipynb        # 1. Aşama: U-Net Denemeleri ve Sonuçları
│   └── Training_Pix2Pix_GAN.ipynb # 2. Aşama: Final GAN Modeli Eğitimi
├── examples/                  # Test için örnek taslak görselleri
└── models/                    # (Otomatik iner) Eğitilmiş ağırlık dosyaları

````


## 7. Kurulum ve Çalıştırma
Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.Depoyu klonlayın:

````bash
git clone [https://github.com/Fatmanurkntr/Landscape-Sketch-to-Paint.git](https://github.com/Fatmanurkntr/Landscape-Sketch-to-Paint.git)
cd Landscape-Sketch-to-Paint

````

2.Gerekli kütüphaneleri yükleyin:

````bash
pip install -r requirements.txt
````
3.Uygulamayı başlatın:

````bash
streamlit run app.py
````

## 8. Referanslar

1.Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. Proceedings of the IEEE conference on computer vision and pattern recognition.

2.Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. International Conference on Medical Image Computing and Computer-Assisted Intervention.

3.Goodfellow, I., et al. (2014). Generative adversarial nets. Advances in neural information processing systems.



