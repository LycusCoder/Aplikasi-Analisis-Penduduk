import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# === Setup direktori ===
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Fungsi untuk waktu log
def log_time():
    return datetime.now().strftime("%H:%M:%S")

print(f"[{log_time()}] 🧼 Mulai proses cleaning data...")

# Path file input/output
INPUT_PATH = os.path.join(DATA_DIR, "generated_data.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")

# Cek apakah file generated_data.csv tersedia
if not os.path.exists(INPUT_PATH):
    print(f"[{log_time()}] ❌ Error: File '{INPUT_PATH}' tidak ditemukan!")
    print(f"[{log_time()}] 🛠  Silakan jalankan dulu: python generate_data.py")
    exit(1)

# Muat data
df = pd.read_csv(INPUT_PATH)
print(f"[{log_time()}] 📥 Berhasil muat {len(df)} baris dari '{INPUT_PATH}'")

# Konversi kolom numerik ke tipe benar
df['Umur'] = pd.to_numeric(df['Umur'], errors='coerce')
df['Pendapatan'] = pd.to_numeric(df['Pendapatan'], errors='coerce')

# Encode Pendidikan
pendidikan_map = {'SMA': 0, 'D3': 1, 'S1': 2, 'S2': 3}
df['Pendidikan_Code'] = df['Pendidikan'].map(pendidikan_map)

# Imputasi numerik: Umur & Pendapatan
print(f"[{log_time()}] 📏 Imputasi nilai kosong (median)")
num_imputer = SimpleImputer(strategy='median')
df[['Umur', 'Pendapatan']] = num_imputer.fit_transform(df[['Umur', 'Pendapatan']])

# Bulatkan ke integer
df['Umur'] = df['Umur'].round(0).astype(int)
df['Pendapatan'] = df['Pendapatan'].round(0).astype(int)

# Imputasi Status Nikah dengan model jika ada NaN
print(f"[{log_time()}] 🤖 Prediksi Status Nikah dengan model jika perlu")
mask_missing = df['Status Nikah'].isna()

if mask_missing.any():
    df_train = df[~mask_missing]
    X_train = df_train[['Umur', 'Pendidikan_Code']]
    y_train = df_train['Status Nikah']
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    X_missing = df.loc[mask_missing, ['Umur', 'Pendidikan_Code']]
    df.loc[mask_missing, 'Status Nikah'] = clf.predict(X_missing)
    print(f"[{log_time()}] ✅ Terisi {mask_missing.sum()} nilai Status Nikah yang kosong")
else:
    print(f"[{log_time()}] 💡 Semua Status Nikah sudah lengkap, tidak ada yang diimputasi")

# Hapus kolom bantu
df.drop(columns=['Pendidikan_Code'], inplace=True)

# Simpan hasil
df.to_csv(OUTPUT_PATH, index=False)
print(f"[{log_time()}] 💾 Data bersih tersimpan di '{OUTPUT_PATH}'")
print(f"[{log_time()}] ✅ Proses cleaning selesai!")
