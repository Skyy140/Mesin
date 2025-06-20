import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image

# --- Konfigurasi Halaman dan Judul ---
st.set_page_config(page_title="Prediksi Ekspresi Wajah", layout="wide")
st.title("Aplikasi Prediksi Ekspresi Wajah")
st.write("Aplikasi ini menggunakan model KNN dengan PCA untuk memprediksi ekspresi wajah.")

# --- Memuat Model dan Transformer (dengan Cache) ---
@st.cache_resource
def load_resources():
    """Memuat model KNN, transformer PCA, dan cascade classifier."""
    try:
        with open('pca_transformer.pkl', 'rb') as f_pca:
            pca_transformer = pickle.load(f_pca)
        with open('knn_tuned_model_0.80.pkl', 'rb') as f_model:
            model = pickle.load(f_model)
        # face_cascade is not strictly needed if only uploading full images without face detection
        # However, keeping it loaded for consistency if other parts of your model rely on its presence
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return pca_transformer, model, face_cascade
    except FileNotFoundError as e:
        st.error(f"Error: File model tidak ditemukan. Pastikan file '{e.filename}' ada di direktori yang sama.")
        return None, None, None

pca, knn_model, face_cascade = load_resources()

# --- Konstanta dan Pemetaan ---
IMG_WIDTH, IMG_HEIGHT = 56, 56
REVERSE_EXPRESSION_MAP = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry'}

# --- Fungsi Inti untuk Prediksi (diperbarui agar lebih fleksibel) ---
def predict_expression(image, pca_transformer, model):
    """Fungsi untuk melakukan pra-pemrosesan dan prediksi pada satu gambar."""
    if image is None:
        return "Gambar tidak valid"

    # Periksa apakah gambar sudah grayscale atau perlu dikonversi
    if len(image.shape) > 2 and image.shape[2] == 3:
        # Konversi dari BGR (format OpenCV) ke grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Gambar sudah grayscale
        gray_image = image
    
    # 2. Ubah ukuran
    resized_image = cv2.resize(gray_image, (IMG_WIDTH, IMG_HEIGHT))
    
    # 3. Normalisasi
    normalized_image = resized_image.astype('float32') / 255.0
    
    # 4. Ratakan (Flatten) dan ubah bentuk
    flattened_image = normalized_image.reshape(1, -1)
    
    # 5. Transformasi PCA
    pca_data = pca_transformer.transform(flattened_image)
    
    # 6. Prediksi
    prediction_index = model.predict(pca_data)[0]
    
    # 7. Kembalikan label ekspresi
    return REVERSE_EXPRESSION_MAP.get(prediction_index, "Tidak Diketahui")

# --- Pilihan Mode di Sidebar (Dihapus karena hanya satu mode) ---
# st.sidebar.title("Pilih Mode Prediksi")
# app_mode = st.sidebar.selectbox("Pilih antara Live Camera atau Upload Gambar:",
#                                 ["Live Camera", "Upload Gambar"])

# --- Logika Aplikasi Berdasarkan Pilihan Mode (Sekarang hanya untuk Upload Gambar) ---
if knn_model and pca and face_cascade: # face_cascade is still loaded, but not directly used in this mode
    st.header("Prediksi dari Gambar")
    st.info("Aplikasi ini akan memprediksi ekspresi wajah dari gambar yang diunggah. Gambar akan diubah menjadi grayscale sebelum diproses.")
    
    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 1. Baca gambar dan langsung konversi ke grayscale ('L' mode)
        image_gray_pil = Image.open(uploaded_file).convert('L')
        
        # Tampilkan gambar versi grayscale
        st.image(image_gray_pil, caption='Gambar Diunggah (Grayscale)', use_container_width=True)
        
        # 2. Konversi gambar grayscale PIL ke array numpy untuk diproses
        cv_image_gray = np.array(image_gray_pil)
        
        # 3. Langsung lakukan prediksi pada seluruh gambar grayscale
        prediction = predict_expression(cv_image_gray, pca, knn_model)
        
        # 4. Tampilkan hasil prediksi
        st.success(f"Hasil Prediksi pada Gambar: {prediction}")
else:
    st.warning("Model tidak dapat dimuat. Aplikasi tidak dapat berjalan.")