import streamlit as st

def apply_custom_styles():
    style = """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;1,600&display=swap" rel="stylesheet">
    <style>
        /* --- 1. EN ARKA PLAN (DAĞ MANZARASI) --- */
        .stApp {
            background: url('https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&q=80&w=2070') no-repeat center center fixed !important;
            background-size: cover !important;
        }
        
        /* Gereksiz boşlukları ve barları temizle */
        header, footer, [data-testid="stSidebar"] { display: none !important; }
        
        .block-container {
            max-width: 1300px !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }

        /* --- 2. ANA BÜYÜK BEYAZ KUTU (KESİN ÇÖZÜM) --- */
        /* Streamlit'in ana içerik tutucusunu hedef alıyoruz */
        [data-testid="block-container"] {
            background-color: #ffffff !important; /* TAM BEYAZ */
            background-color: rgba(255, 255, 255, 0.95) !important; /* Hafif saydamlık (isteğe bağlı) */
            border-radius: 40px !important;
            padding: 4rem 5rem !important;
            box-shadow: 0 40px 100px rgba(0,0,0,0.5) !important; /* Güçlü gölge */
            border: 1px solid #ffffff !important;
            margin-top: 3rem !important;
            margin-bottom: 3rem !important;
        }

        /* --- 3. CANVAS VE SİYAH ALAN DÜZELTMESİ --- */
        iframe[title="streamlit_drawable_canvas.st_canvas"] {
            background-color: #ffffff !important; /* Siyahlığı beyaza çevir */
            border: 2px solid #e2e8f0 !important;
            border-radius: 15px !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            width: 100% !important; /* Tam genişlik */
        }

        /* --- 4. YAZI TİPLERİ VE RENKLER --- */
        h1 {
            font-family: 'Playfair Display', serif !important;
            color: #2c3e50 !important;
            text-align: center;
            font-size: 3.5rem !important;
            font-weight: 700 !important;
            font-style: italic !important;
            margin: 0 !important;
            padding-bottom: 10px !important;
            text-shadow: none !important;
        }
        
        .subtitle {
            font-family: 'Inter', sans-serif;
            text-align: center;
            font-size: 0.9rem;
            letter-spacing: 4px;
            color: #546e7a;
            text-transform: uppercase;
            margin-bottom: 4rem;
            font-weight: 600;
        }

        .config-header {
            color: #15803d;
            font-family: 'Inter', sans-serif;
            font-weight: 800;
            font-size: 0.9rem;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 1rem;
            border-bottom: 2px solid #bbf7d0;
            display: inline-block;
            padding-bottom: 5px;
        }

        .input-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            font-weight: 700;
            color: #334155;
            margin-bottom: 5px;
            display: block;
        }

        /* --- 5. BİLEŞENLER --- */
        div[data-baseweb="select"] > div {
            background-color: #f8fafc !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px;
            color: #0f172a !important;
        }

        /* Yeşil Buton */
        div.stButton > button {
            background: linear-gradient(135deg, #22c55e, #16a34a) !important;
            color: white !important;
            border: none !important;
            height: 55px;
            border-radius: 12px;
            font-weight: 700 !important;
            letter-spacing: 0.5px;
            box-shadow: 0 10px 20px rgba(22, 197, 94, 0.2) !important;
            width: 100%;
        }
        div.stButton > button:hover {
            transform: scale(1.02);
        }

        /* Gri İndir Butonu */
        div[data-testid="stDownloadButton"] button {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            color: #475569 !important;
            font-size: 0.8rem !important;
            width: 100%;
            font-weight: 600 !important;
        }
        div[data-testid="stDownloadButton"] button:hover {
            background-color: #f1f5f9 !important;
        }

        /* --- 6. ALT KARTLAR (HOVER EFEKTLİ) --- */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.03);
            transition: transform 0.3s ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }

        /* Resim Kutusu */
        div[data-testid="stImage"] {
            background-color: #f0fdf4;
            border: 2px dashed #86efac;
            border-radius: 15px;
            padding: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        div[data-testid="stImage"] img {
            max-height: 200px;
            object-fit: contain;
        }

        /* Placeholder */
        .placeholder {
            height: 200px;
            background-color: #f0fdf4;
            border: 2px dashed #86efac;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #16a34a;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .badge {
            background: #f1f5f9;
            color: #475569;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 700;
            display: table;
            margin: 0 auto 10px auto;
        }
        .card-title {
            text-align: center; font-weight: 700; color: #334155; margin-top: 10px;
        }
        .card-desc {
            text-align: center; font-size: 0.75rem; color: #94a3b8;
        }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def get_header():
    return """
    <h1>Landscape Sketch to Paint</h1>
    <div class="subtitle">Yapay Zeka ile Doğayı Yeniden Boya</div>
    """