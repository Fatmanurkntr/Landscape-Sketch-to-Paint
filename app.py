import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import gdown
import io
from streamlit_drawable_canvas import st_canvas
from src.model import build_generator
from style_utils import apply_custom_styles, get_header

# --- MODEL AYARLARI ---
GAN_ID = '1yuWQ6nIji55GNv8wiv_9CMRSjVtHbsgZ' 
UNET_ID = '1aCxDdo3jimqdCjg-95ojKiE-YZNzE7hs'
GAN_PATH = 'models/best_generator.weights.h5'
UNET_PATH = 'models/best_landscape_unet.keras'

st.set_page_config(page_title="Landscape Sketch to Paint", layout="wide")

# --- HAFIZA ---
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'vision' not in st.session_state:
    st.session_state.vision = None
if 'unet' not in st.session_state:
    st.session_state.unet = None
if 'gan' not in st.session_state:
    st.session_state.gan = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# --- GÜNCELLEME ---
def update_from_upload():
    if st.session_state.uploader_key:
        img = Image.open(st.session_state.uploader_key).convert('RGB')
        st.session_state.current_image = img
        st.session_state.processed = False 

def update_from_example():
    ex_name = st.session_state.example_key
    if ex_name:
        ex_path = os.path.join('examples', ex_name)
        if os.path.exists(ex_path):
            img = Image.open(ex_path).convert('RGB')
            st.session_state.current_image = img
            st.session_state.processed = False

@st.cache_resource
def load_models():
    if not os.path.exists('models'): os.makedirs('models')
    if not os.path.exists(GAN_PATH):
        gdown.download(f'https://drive.google.com/uc?id={GAN_ID}', GAN_PATH, quiet=False)
    gan_model = build_generator()
    gan_model.load_weights(GAN_PATH)
    if not os.path.exists(UNET_PATH):
        gdown.download(f'https://drive.google.com/uc?id={UNET_ID}', UNET_PATH, quiet=False)
    unet_model = tf.keras.models.load_model(UNET_PATH)
    return gan_model, unet_model

gan, unet = load_models()

# CSS Uygula
apply_custom_styles()

# Başlık
st.markdown(get_header(), unsafe_allow_html=True)

# --- ÜST KISIM (TEK PARÇA) ---
c1, c2 = st.columns([1, 2.2], gap="large")

# SOL PANEL
with c1:
    st.markdown('<div class="config-header">KONFİGÜRASYON</div>', unsafe_allow_html=True)
    
    st.markdown('<label class="input-label">Çalışma Alanı</label>', unsafe_allow_html=True)
    work_mode = st.selectbox("wm", ["Taslak Çizimi (Sketch)", "Resim Yükle"], label_visibility="collapsed")
    
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    
    st.markdown('<label class="input-label">Manzara Tipi</label>', unsafe_allow_html=True)
    if work_mode == "Resim Yükle":
        opts = ["Beyaz Kağıt Çizimi", "Gerçek Fotoğraf", "Siyah Zemin"]
    else:
        opts = ["Beyaz Kağıt Çizimi"]
    manzara_tipi = st.selectbox("lt", opts, label_visibility="collapsed")
    
    stroke_width = 3
    if work_mode == "Taslak Çizimi (Sketch)":
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.markdown('<label class="input-label">Fırça Boyutu</label>', unsafe_allow_html=True)
        stroke_width = st.slider("sw", 1, 20, 3, label_visibility="collapsed")

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    process_btn = st.button("BOYAMAYA BAŞLA")

    st.markdown('<p style="font-size:0.75rem; color:#64748b; margin-top:15px;">Modeller (U-Net & GAN) manzara verileriyle eğitilmiştir.</p>', unsafe_allow_html=True)

# SAĞ PANEL
source_img = None
with c2:
    if work_mode == "Taslak Çizimi (Sketch)":
        # background_color="#FFFFFF" ve CSS desteği ile siyahlık giderildi
        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=400, 
            width=700, 
            drawing_mode="freedraw",
            key="canvas",
            display_toolbar=True
        )
        if canvas_result.image_data is not None:
            source_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('RGB')
    else:
        # Sekmeli Yapı
        tab1, tab2 = st.tabs(["📁 Dosya Yükle", "🖼️ Örnek Kullan"])
        
        with tab1:
            st.file_uploader("Bilgisayardan Seç", type=["jpg", "png", "jpeg"], key="uploader_key", on_change=update_from_upload)
        
        with tab2:
            example_files = []
            if os.path.exists('examples'):
                example_files = [f for f in os.listdir('examples') if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            if example_files:
                st.selectbox("Listeden Seçin", example_files, index=None, placeholder="Bir örnek seç...", key="example_key", on_change=update_from_example)
            else:
                st.info("Örnek klasörü boş.")

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        if st.session_state.current_image:
            st.image(st.session_state.current_image, width=400, caption="Seçilen Görsel")
            source_img = st.session_state.current_image
        else:
            st.markdown('<div class="placeholder" style="height:300px; color:#cbd5e1; border-color:#cbd5e1;">Görsel Bekleniyor...</div>', unsafe_allow_html=True)

# --- İŞLEME ---
if process_btn:
    if work_mode == "Taslak Çizimi (Sketch)":
        target_img = source_img
    else:
        target_img = st.session_state.current_image

    if target_img is not None:
        with st.spinner('Manzara oluşturuluyor...'):
            img_np = np.array(target_img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            if manzara_tipi == "Beyaz Kağıt Çizimi":
                 processed = 255 - gray
                 _, processed = cv2.threshold(processed, 100, 255, cv2.THRESH_BINARY)
            elif manzara_tipi == "Gerçek Fotoğraf":
                 processed = cv2.Canny(gray, 100, 200)
            else:
                 processed = gray
                 
            input_tensor = cv2.resize(processed, (256, 256))
            input_tensor = (input_tensor / 255.0).astype(np.float32)
            input_tensor = np.expand_dims(input_tensor, axis=(0, -1))
            
            unet_pred = unet.predict(input_tensor, verbose=0)
            gan_pred = gan(input_tensor, training=True)
            
            orig_w, orig_h = target_img.size
            st.session_state.vision = processed
            st.session_state.unet = cv2.resize(unet_pred[0], (orig_w, orig_h))
            st.session_state.gan = cv2.resize(gan_pred[0].numpy(), (orig_w, orig_h))
            st.session_state.processed = True

# --- ALT KARTLAR ---
st.markdown('<div style="height:40px;"></div>', unsafe_allow_html=True)
r1, r2, r3 = st.columns(3, gap="medium")

def show_card(col, title, badge, badge_style, img, fname):
    with col:
        with st.container(border=True): 
            st.markdown(f'<span class="badge" style="{badge_style}">{badge}</span>', unsafe_allow_html=True)
            
            if st.session_state.processed and img is not None:
                st.image(img, use_container_width=True)
                if fname:
                    buf = io.BytesIO()
                    Image.fromarray((img * 255).astype(np.uint8)).save(buf, format="PNG")
                    st.download_button("Resmi İndir", buf.getvalue(), fname, "image/png", key=f"dl_{fname}")
            else:
                st.markdown(f'<div class="placeholder">{title} Bekleniyor</div>', unsafe_allow_html=True)
                
            st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)

show_card(r1, "AI Gözü", "VISION", "background:#f1f5f9; color:#475569;", st.session_state.vision, None)
show_card(r2, "U-Net Sonucu", "U-NET", "background:#dcfce7; color:#15803d;", st.session_state.unet, "unet_sonuc.png")
show_card(r3, "GAN Boyama", "PIX2PIX GAN", "background:#ffedd5; color:#c2410c;", st.session_state.gan, "gan_sonuc.png")