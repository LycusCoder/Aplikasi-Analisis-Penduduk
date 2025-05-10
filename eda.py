import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pathing yang rapi
DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Muat data
df = pd.read_csv(os.path.join(DATA_DIR, 'cleaned_data.csv'))
print("📊 Statistik Deskriptif:\n")
print(df.describe(include='all'))

# Setup style visual
sns.set_style("whitegrid")
plt.style.use("ggplot")

# Buat figure besar untuk semua plot
fig = plt.figure(figsize=(20, 12))
fig.suptitle('📊 Exploratory Data Analysis (EDA)', fontsize=18, y=1.02)

# 1. Distribusi Pendidikan
plt.subplot(2, 3, 1)
sns.countplot(data=df, x='Pendidikan', palette='viridis')
plt.title('Distribusi Tingkat Pendidikan')
plt.xlabel('Pendidikan')
plt.ylabel('Jumlah Penduduk')

# 2. Rata-rata Pendapatan per Pendidikan
plt.subplot(2, 3, 2)
sns.barplot(data=df, x='Pendidikan', y='Pendapatan', estimator='mean', ci=None, palette='coolwarm')
plt.title('Rata-rata Pendapatan per Pendidikan')
plt.xlabel('Pendidikan')
plt.ylabel('Pendapatan (Rp)')

# 3. Boxplot Pendapatan per Pendidikan
plt.subplot(2, 3, 3)
sns.boxplot(data=df, x='Pendidikan', y='Pendapatan', palette='muted')
plt.title('Distribusi Pendapatan per Pendidikan')
plt.xlabel('Pendidikan')
plt.ylabel('Pendapatan (Rp)')

# 4. Distribusi Status Nikah
plt.subplot(2, 3, 4)
sns.countplot(data=df, x='Status Nikah', palette='Set2')
plt.title('Distribusi Status Nikah')
plt.xlabel('Status Nikah')
plt.ylabel('Jumlah Penduduk')

# 5. Hubungan Umur dan Pendapatan
plt.subplot(2, 3, 5)
sns.scatterplot(data=df, x='Umur', y='Pendapatan', hue='Pendidikan', alpha=0.7, palette='deep')
plt.title('Hubungan Umur dan Pendapatan')
plt.xlabel('Umur')
plt.ylabel('Pendapatan (Rp)')
plt.legend(title='Pendidikan')

# 6. Distribusi Umur
plt.subplot(2, 3, 6)
sns.histplot(df['Umur'], bins=15, kde=True, color='skyblue')
plt.title('Distribusi Umur')
plt.xlabel('Umur')
plt.ylabel('Frekuensi')

# Atur layout & simpan
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_visualization.png'), dpi=300, bbox_inches='tight')
plt.show()
