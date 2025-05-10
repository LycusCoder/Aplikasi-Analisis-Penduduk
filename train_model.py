import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import joblib
from datetime import datetime

# === Setup direktori ===
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Fungsi untuk waktu log
def log_time():
    return datetime.now().strftime("%H:%M:%S")

print(f"[{log_time()}] 🤖 Mulai pelatihan model ML...")

# Load data bersih
cleaned_path = os.path.join(DATA_DIR, 'cleaned_data.csv')
df = pd.read_csv(cleaned_path)
print(f"[{log_time()}] 📥 Berhasil muat {len(df)} baris dari '{cleaned_path}'")

# Encode kategorik
le_pendidikan = LabelEncoder()
le_status = LabelEncoder()

df['Pendidikan'] = le_pendidikan.fit_transform(df['Pendidikan'])
df['Status Nikah'] = le_status.fit_transform(df['Status Nikah'])

# Simpan mapping label
label_mapping = {
    'Pendidikan': dict(zip(le_pendidikan.classes_, le_pendidikan.transform(le_pendidikan.classes_))),
    'Status Nikah': dict(zip(le_status.classes_, le_status.transform(le_status.classes_)))
}

# Split data
X = df[['Umur', 'Pendidikan']]
y_status = df['Status Nikah']
y_pendapatan = df['Pendapatan']

X_train, X_test, y1_train, y1_test = train_test_split(X, y_status, test_size=0.2, random_state=42)
X_train, X_test, y2_train, y2_test = train_test_split(X, y_pendapatan, test_size=0.2, random_state=42)

print(f"[{log_time()}] 🧮 Melatih model Status Nikah (Klasifikasi)...")

# Model Klasifikasi - Status Nikah
model_status = RandomForestClassifier(n_estimators=100, random_state=42)
model_status.fit(X_train, y1_train)
y1_pred = model_status.predict(X_test)
acc = accuracy_score(y1_test, y1_pred)
print(f"[{log_time()}] ✅ Akurasi Status Nikah: {acc:.2%}")

print(f"[{log_time()}] 🧠 Melatih model Pendapatan (Regresi)...")

# Model Regresi - Pendapatan
model_pendapatan = RandomForestRegressor(n_estimators=100, random_state=42)
model_pendapatan.fit(X_train, y2_train)
y2_pred = model_pendapatan.predict(X_test)
r2 = r2_score(y2_test, y2_pred)
mae = mean_absolute_error(y2_test, y2_pred)
print(f"[{log_time()}] ✅ R2 Score Pendapatan: {r2:.2f}")
print(f"[{log_time()}] 💰 Rata-rata kesalahan pendapatan: Rp {mae:,.0f}")

# Simpan model & encoder
joblib.dump(model_status, os.path.join(MODEL_DIR, 'model_status_nikah.pkl'))
joblib.dump(model_pendapatan, os.path.join(MODEL_DIR, 'model_pendapatan.pkl'))
joblib.dump(label_mapping, os.path.join(MODEL_DIR, 'label_encoders.pkl'))

print(f"[{log_time()}] 💾 Semua model dan encoder telah disimpan di folder '{MODEL_DIR}/'")
print(f"[{log_time()}] 🎯 Pelatihan model selesai!")
