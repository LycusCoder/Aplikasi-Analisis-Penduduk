import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Fungsi log waktu
def log_time():
    return datetime.now().strftime("%H:%M:%S")

# Konfigurasi halaman
st.set_page_config(page_title="Aplikasi Analisis Penduduk", layout="wide")
st.title("📊 Aplikasi Analisis & Prediksi Data Penduduk")

# Sidebar navigasi
menu = st.sidebar.selectbox(
    "📚 Menu Navigasi", 
    ["Beranda", "Statistik & Visualisasi", "Prediksi Individual", "Prediksi Massal"]
)

# Path terstruktur
DATA_DIR = "data"
MODEL_DIR = "model"
OUTPUT_DIR = "output"

# Load model dan encoder
try:
    model_status = joblib.load(os.path.join(MODEL_DIR, 'model_status_nikah.pkl'))
    model_pendapatan = joblib.load(os.path.join(MODEL_DIR, 'model_pendapatan.pkl'))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
except Exception as e:
    st.error(f"❌ Gagal memuat model atau encoder: {e}")
    st.stop()

# Mapping label dari encoder
pendidikan_map = label_encoders.get('Pendidikan', {})
status_map = label_encoders.get('Status Nikah', {})

# Load data CSV utama
try:
    df = pd.read_csv(os.path.join(DATA_DIR, 'cleaned_data.csv'))
except Exception as e:
    st.error(f"❌ Gagal memuat data: {e}")
    st.stop()

if menu == "Beranda":
    st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🏠 Selamat Datang di Aplikasi Analisis Penduduk</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Aplikasi prediksi status pernikahan dan pendapatan berdasarkan umur dan tingkat pendidikan</p>", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/8044/8044624.png", width=200)

    st.markdown("### 🎯 Tujuan Aplikasi")
    st.markdown("""
    - Menganalisis pola data penduduk (umur, pendidikan, pendapatan, status nikah)  
    - Mempermudah prediksi status pernikahan dan pendapatan calon penduduk  
    - Memberikan insight menggunakan data visualisasi dan statistik deskriptif  
    """)

    st.markdown("### 📊 Dataset Awal")
    st.markdown(f"""
    - Jumlah baris awal: **20**
    - Setelah augmentasi: **{len(df)}**
    - Kolom utama: `Umur`, `Pendidikan`, `Pendapatan`, `Status Nikah`
    """)

    st.markdown("### 🔍 Fitur Utama Aplikasi")
    st.markdown("""
    1. **Statistik & Visualisasi**: Menampilkan distribusi dan hubungan antar variabel  
    2. **Prediksi Individual**: Prediksi status nikah dan pendapatan satu orang  
    3. **Prediksi Massal**: Unggah file CSV → Prediksi banyak data sekaligus  
    """)

    st.markdown("### 🤖 Teknologi yang Digunakan")
    st.markdown("""
    - **Python** dengan library: `Streamlit`, `Pandas`, `Scikit-learn`, `Matplotlib`, `Seaborn`
    - Model ML: `Random Forest Classifier` dan `Random Forest Regressor`
    """)

    st.markdown("---")
    st.markdown("<p style='text-align: center; font-size: 14px; color: gray;'>Dibuat sebagai proyek akhir analisis data penduduk</p>", unsafe_allow_html=True)
    
elif menu == "Statistik & Visualisasi":
    st.header("📊 Statistik Deskriptif dan Visualisasi")

    # 1. Statistik Numerik (Umur & Pendapatan)
    st.subheader("📈 Statistik Numerik")
    numeric_cols = df.select_dtypes(include=['number'])
    st.write(numeric_cols.describe())

    # Tambahkan penjelasan singkat
    st.markdown("""
    #### Penjelasan Statistik Numerik:
    - **Count**: Jumlah data non-kosong
    - **Mean**: Rata-rata
    - **Std**: Standar deviasi
    - **Min/Max**: Nilai minimum/maksimum
    - **25%, 50%, 75%**: Kuartil pertama, kedua, dan ketiga
    """)

    # 2. Statistik Kategorik (Pendidikan & Status Nikah)
    st.subheader("📊 Statistik Kategorik")
    categorical_cols = df.select_dtypes(include=['object'])
    for col in categorical_cols.columns:
        st.markdown(f"#### {col}")
        st.write(categorical_cols[col].value_counts())
    
    # Tambahkan penjelasan singkat
    st.markdown("""
    #### Penjelasan Statistik Kategorik:
    - Menunjukkan distribusi frekuensi setiap kategori.
    - Misalnya, berapa banyak orang dengan pendidikan SMA, D3, S1, atau S2.
    """)

    # 3. Gambar EDA
    st.subheader("📊 Visualisasi Data")
    if os.path.exists('eda_visualization.png'):
        st.image('eda_visualization.png', use_container_width=True)
    else:
        st.warning("File visualisasi belum tersedia. Silakan jalankan eda.py terlebih dahulu.")

elif menu == "Prediksi Individual":
    st.markdown("<h2 style='text-align: center; color: #1E90FF;'>🔮 Prediksi Status Nikah & Pendapatan</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Masukkan data penduduk untuk memprediksi apakah sudah kawin dan perkiraan pendapatannya.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧑‍🦱 Data Penduduk")
        umur = st.slider("Umur", 17, 80, 30, help="Pilih rentang umur antara 17 hingga 80 tahun")

    with col2:
        # Gunakan urutan yang diinginkan secara eksplisit
        pendidikan_options = ['SMA', 'D3', 'S1', 'S2']
        pendidikan = st.selectbox("Pendidikan", pendidikan_options, help="Pilih tingkat pendidikan terakhir")

    if st.button("🔍 Jalankan Prediksi", use_container_width=True):
        with st.spinner('Sedang memproses... Mohon tunggu sebentar'):
            pendidikan_encoded = pendidikan_map[pendidikan]
            input_data = [[umur, pendidikan_encoded]]

            status_pred = model_status.predict(input_data)[0]
            income_pred = model_pendapatan.predict(input_data)[0]

            status_label = 'Kawin' if status_pred == 1 else 'Belum'
            
            st.success(f"📌 **Status Nikah:** {'Kawin' if status_pred == 1 else 'Belum'}")
            st.success(f"💰 **Perkiraan Pendapatan:** Rp {income_pred:,.0f}")

        st.markdown("---")
        st.markdown("#### 📌 Ringkasan Input:")
        st.write({
            "Umur": umur,
            "Pendidikan": pendidikan
        })

        st.markdown("#### 📊 Hasil Prediksi:")
        st.json({
            "Status Nikah": status_label,
            "Pendapatan (Rp)": f"{income_pred:,.0f}"
        })

elif menu == "Prediksi Massal":
    st.markdown("<h2 style='text-align: center; color: #1E90FF;'>📦 Prediksi Massal dari CSV</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Unggah file CSV dengan kolom <code>Umur</code> dan <code>Pendidikan</code>, lalu dapatkan prediksi untuk setiap baris.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Unggah File CSV", type=["csv"], help="File harus memiliki kolom 'Umur' dan 'Pendidikan'")
    
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            df_upload_original = df_upload.copy()

            # Validasi kolom penting
            required_cols = ['Umur', 'Pendidikan']
            missing_cols = [col for col in required_cols if col not in df_upload.columns]

            if missing_cols:
                st.error(f"❌ Kolom wajib berikut tidak ditemukan dalam file: {', '.join(missing_cols)}")
            else:
                with st.spinner('🔄 Memproses data... Mohon tunggu'):
                    # Encoding pendidikan
                    df_upload['Pendidikan'] = df_upload['Pendidikan'].map(pendidikan_map)

                    # Prediksi
                    X_input = df_upload[['Umur', 'Pendidikan']]
                    df_upload['Status Nikah Prediksi'] = model_status.predict(X_input)
                    df_upload['Pendapatan Prediksi'] = model_pendapatan.predict(X_input)

                    # Decode prediksi status nikah
                    status_decoder = {v: k for k, v in status_map.items()}
                    df_upload['Status Nikah Prediksi'] = df_upload['Status Nikah Prediksi'].map(status_decoder)

                    # Tambahkan pendapatan ke format mata uang
                    df_upload['Pendapatan Prediksi'] = df_upload['Pendapatan Prediksi'].apply(lambda x: f"Rp {x:,.0f}")

                    # Tampilkan hasil
                    st.success("✅ Berhasil memprediksi semua data!")
                    st.subheader("📊 Hasil Prediksi")
                    st.dataframe(df_upload)

                    # Ringkasan statistik prediksi
                    st.subheader("📈 Ringkasan Prediksi")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Total Data", len(df_upload))
                    with col2:
                        kawin_count = df_upload['Status Nikah Prediksi'].value_counts().get('Kawin', 0)
                        st.metric("Jumlah Prediksi Kawin", kawin_count)

                    # Download file
                    csv = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ 📥 Unduh Hasil Prediksi",
                        data=csv,
                        file_name='hasil_prediksi_massal.csv',
                        mime='text/csv'
                    )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")