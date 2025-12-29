# Landscape Sketch to Paint: Derin Öğrenme ile Görüntü Sentezi

Bu proje, Derin Öğrenme (Deep Learning) teknikleri kullanılarak el çizimi taslakların (sketch) fotorealistik manzara fotoğraflarına dönüştürülmesini sağlayan bir Görüntüden Görüntüye Çeviri (Image-to-Image Translation) uygulamasıdır. Proje, koşullu üretimsel çekişmeli ağlar (cGAN) tabanlı Pix2Pix mimarisi üzerine kurulmuştur.

Canlı Uygulama: https://landscape-sketch-to-paint-8jikjxxn4lxsqcxfcebwpr.streamlit.app/

## 1. Proje Hakkında ve Problem Tanımı

Bilgisayarlı görü alanında, taslak çizimlerden gerçekçi görüntüler üretmek zorlu bir problemdir. Geleneksel yöntemler manuel işlem gerektirirken, bu projede derin sinir ağları kullanılarak süreç otomatize edilmiştir.

Projenin temel amacı; kullanıcının girdiği siyah-beyaz kenar haritalarını (taslakları) anlamlandırarak, bu sınırlara uygun dağ, gökyüzü, nehir ve bitki örtüsü dokularını gerçekçi renklerle üretmektir.

## 2. Veri Seti ve Ön İşleme

Projede Kaggle üzerinde bulunan "Landscape Pictures" veri seti kullanılmıştır. Veri seti, gerçek manzara fotoğraflarını ve bu fotoğraflardan türetilmiş taslak görüntülerini içerir.

Veri Seti Kaynağı: https://www.kaggle.com/datasets/arnaud58/landscape-pictures

### Veri İşleme ve Teknik Kısıtlamalar
Eğitim süreci Kaggle kernel ortamında (Tesla P100 GPU) gerçekleştirilmiştir. Orijinal veri setinde yaklaşık 3.500+ görüntü çifti bulunmaktadır. Ancak GAN mimarisinin ve TensorFlow veri hattının (tf.data pipeline) yüksek bellek (RAM) kullanımı nedeniyle "ResourceExhaustedError" hataları ile karşılaşılmıştır.

Bu sorunu aşmak ve eğitimi stabilize etmek için aşağıdaki optimizasyonlar uygulanmıştır:

1.  Rastgele Alt Örnekleme (Random Subsampling): Veri seti içerisinden rastgele seçim yapılarak eğitim seti 2.500 görüntü ile sınırlandırılmıştır.
2.  Yeniden Boyutlandırma: Tüm giriş ve çıkış görüntüleri 256x256 piksel boyutuna sabitlenmiştir.
3.  Normalizasyon: Pikseller [0, 255] aralığından, Tanh aktivasyon fonksiyonuna uygun olan [-1, 1] aralığına normalize edilmiştir.
4.  Veri Çoğaltma (Augmentation): Overfitting'i önlemek için eğitim sırasında görüntülere rastgele yatay çevirme (Random Flip) ve kırpma (Random Crop) işlemleri uygulanmıştır.

## 3. Kullanılan Yöntem ve Mimari

Projede literatürde "Image-to-Image Translation with Conditional Adversarial Networks" (Isola et al., 2017) olarak bilinen Pix2Pix mimarisi kullanılmıştır.

### 3.1. Generator (Üreteç) Modeli
Üreteç olarak U-Net mimarisi tercih edilmiştir. Standart Encoder-Decoder yapılarının aksine, U-Net yapısındaki "Atlamalı Bağlantılar" (Skip Connections) sayesinde, giriş görüntüsündeki (taslak) yapısal detaylar darboğaz (bottleneck) katmanında kaybolmadan doğrudan çıkış katmanına aktarılır.

* Encoder: Conv2D, BatchNormalization ve LeakyReLU katmanlarından oluşan, görüntüyü 256x256 boyutundan 1x1 boyutuna kadar sıkıştıran yapı.
* Decoder: Conv2DTranspose (Upsampling), BatchNormalization ve Dropout katmanlarından oluşan yapı.
* Aktivasyon: Son katmanda Tanh fonksiyonu kullanılarak RGB renk uzayı üretilir.

### 3.2. Discriminator (Ayırt Edici) Modeli
Ayırt edici olarak PatchGAN mimarisi kullanılmıştır. Standart bir sınıflandırıcı tüm görüntü için tek bir "Gerçek/Sahte" değeri üretirken, PatchGAN görüntüyü 30x30 veya 70x70'lik yamalara (patches) böler ve her yama için ayrı karar verir.

Bu yöntem, modelin sadece genel şekli değil, yüksek frekanslı detayları (doku, keskinlik) da öğrenmesini zorunlu kılar.

### 3.3. Kayıp Fonksiyonları (Loss Functions)
Eğitim sırasında iki farklı kayıp fonksiyonunun ağırlıklı toplamı minimize edilmiştir:

1.  Adversarial Loss (GAN Loss): Üretecin, ayırt ediciyi kandırma başarısı. Gerçekçi doku üretimini sağlar.
2.  L1 Loss (Mean Absolute Error): Üretilen görüntünün, hedef görüntüye piksel bazında yakınlığı. Renk doğruluğunu sağlar.

Toplam Kayıp = GAN Loss + (Lambda * L1 Loss)
(Lambda değeri eğitimde 100 olarak belirlenmiştir.)

## 4. Kullanılan Teknolojiler ve Kütüphaneler

Proje Python dili ile geliştirilmiş olup, aşağıdaki temel kütüphaneler kullanılmıştır:

* TensorFlow / Keras: Model mimarisi, eğitim döngüsü ve tensör işlemleri.
* Streamlit: Web tabanlı kullanıcı arayüzü ve etkileşim.
* OpenCV & NumPy: Görüntü işleme ve matris operasyonları.
* Matplotlib: Eğitim sonuçlarının ve kayıp grafiklerinin görselleştirilmesi.
* Gdown: Büyük model dosyalarının Google Drive üzerinden çalışma zamanında indirilmesi.

## 5. Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.  Depoyu klonlayın:
    git clone https://github.com/Fatmanurkntr/Landscape-Sketch-to-Paint.git

2.  Proje dizinine gidin:
    cd Landscape-Sketch-to-Paint

3.  Gerekli bağımlılıkları yükleyin:
    pip install -r requirements.txt

4.  Uygulamayı başlatın:
    streamlit run app.py

Not: Uygulama ilk kez çalıştırıldığında, eğitilmiş model ağırlıklarını (yaklaşık 200MB) otomatik olarak indireceği için açılış süresi internet hızınıza bağlı olarak değişebilir.

## 6. Dosya Yapısı

```text
Landscape-Sketch-to-Paint/
├── app.py                     # Streamlit ana uygulama ve arayüz kodu
├── requirements.txt           # Gerekli Python kütüphaneleri listesi
├── style_utils.py             # CSS stilleri ve HTML düzenlemeleri
├── .gitignore                 # GitHub'a yüklenmeyecek dosyalar (büyük modeller vb.)
├── README.md                  # Proje dokümantasyonu
├── src/                       # Kaynak kodlar
│   ├── __init__.py            # Paket tanımlayıcısı
│   └── model.py               # U-Net ve Pix2Pix GAN mimari tanımları
├── notebooks/                 # Model eğitim süreçleri (Kaggle)
│   ├── Training_UNet.ipynb        # U-Net eğitim adımları ve grafikleri
│   └── Training_Pix2Pix_GAN.ipynb # GAN eğitim adımları ve grafikleri
├── examples/                  # Test amaçlı örnek taslak görselleri
│   ├── ornek1.jpg
│   └── ...
└── models/                    # (Otomatik oluşur) Google Drive'dan inen modeller buraya kaydedilir
