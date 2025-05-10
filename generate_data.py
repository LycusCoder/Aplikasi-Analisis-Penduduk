import os
import pandas as pd
import numpy as np
from datetime import datetime

# Pathing yang rapi
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Fungsi untuk tampilkan waktu saat ini dengan format rapi
def log_time():
    return datetime.now().strftime("%H:%M:%S")

print(f"[{log_time()}] 🔍 Memulai augmentasi data...")

# Muat dataset asli
df = pd.read_csv(os.path.join(DATA_DIR, 'dataset_penduduk.csv'))
print(f"[{log_time()}] 📥 Berhasil muat {len(df)} data dari 'dataset_penduduk.csv'")

# Aturan augmentasi: (mean, std) pendapatan per tingkat pendidikan (Tahun 2025)
pendidikan_income = {
    'SMA': (6_000_000, 800_000),  # Mean ± Std Deviasi
    'D3': (7_500_000, 1_000_000),
    'S1': (9_000_000, 1_200_000),
    'S2': (12_000_000, 1_500_000)
}

# Distribusi Pendidikan & Status Nikah
edu_weights = [0.5, 0.2, 0.2, 0.1]  # SMA, D3, S1, S2
marital_weights = [0.6, 0.4]       # Kawin, Belum

# Hitung jumlah data sintetis yang dibutuhkan
target_rows = 200 # Contoh dari saya 200 
synthetic_count = target_rows - len(df)

print(f"[{log_time()}] 🧮 Akan ditambahkan {synthetic_count} data sintetis")

# Generate data sintetis
np.random.seed(42)
synthetic_data = []
for i in range(synthetic_count):
    umur = int(np.random.randint(17, 80))
    pendidikan = np.random.choice(['SMA', 'D3', 'S1', 'S2'], p=edu_weights)
    mean, std = pendidikan_income[pendidikan]
    pendapatan = int(np.random.normal(mean, std))
    status_nikah = np.random.choice(['Kawin', 'Belum'], p=marital_weights)
    
    synthetic_data.append({
        'Umur': umur,
        'Pendidikan': pendidikan,
        'Pendapatan': pendapatan,
        'Status Nikah': status_nikah
    })

# Gabungkan data
df_synthetic = pd.DataFrame(synthetic_data)
df_final = pd.concat([df, df_synthetic], ignore_index=True)

# Hapus kolom ID jika sudah ada
if 'ID' in df_final.columns:
    df_final = df_final.drop(columns=['ID'])

# Tambahkan kembali ID sebagai kolom pertama
df_final.insert(0, 'ID', range(1, len(df_final) + 1))

# Hitung total baris
total_rows = len(df_final)

# Buat NaN secara acak pada kolom tertentu sesuai frekuensi dataset asli
nan_umur_pct = 2 / 20      # ~10%
nan_income_pct = 6 / 20    # ~30%
nan_marital_pct = 4 / 20   # ~20%

# Tambahkan NaN ke data sintetis
np.random.seed(42)
df_final['Umur'] = df_final['Umur'].mask(np.random.random(total_rows) < nan_umur_pct, np.nan)
df_final['Pendapatan'] = df_final['Pendapatan'].mask(np.random.random(total_rows) < nan_income_pct, np.nan)
df_final['Status Nikah'] = df_final['Status Nikah'].mask(np.random.random(total_rows) < nan_marital_pct, np.nan)

# Ganti np.nan dengan string 'NaN' agar terlihat di CSV
df_final = df_final.replace(np.nan, 'NaN')

# Simpan ke CSV
df_final.to_csv(os.path.join(DATA_DIR, 'generated_data.csv'), index=False)
print(f"[{log_time()}] 💾 Data berhasil disimpan ke 'generated_data.csv'")
print(f"[{log_time()}] ✅ Total baris: {len(df_final)}")
print(f"[{log_time()}] 🎯 Augmentasi selesai!")