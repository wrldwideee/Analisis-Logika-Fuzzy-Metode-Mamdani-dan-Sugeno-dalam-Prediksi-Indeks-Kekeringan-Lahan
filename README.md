# 🏜️ Analisis Logika Fuzzy: Metode Mamdani dan Sugeno dalam Prediksi Indeks Kekeringan Lahan

> Sistem prediksi indeks kekeringan berbasis **Logika Fuzzy** (Mamdani & Sugeno) yang diintegrasikan dengan **Deep Learning (MLP Neural Network)** menggunakan data iklim harian BMKG.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Deskripsi Proyek

Proyek ini mengimplementasikan sistem cerdas untuk memprediksi **Indeks Kekeringan Lahan** menggunakan dua pendekatan utama:

1. **Logika Fuzzy Murni** — Sistem inferensi fuzzy dengan metode Mamdani (defuzzifikasi centroid) dan Sugeno (weighted average), menggunakan **fungsi keanggotaan trapesium** pada 5 variabel input dan **15 rule** inferensi.

2. **Hybrid Deep Learning + Fuzzy** — Model **Multi-Layer Perceptron (MLP)** memprediksi parameter cuaca esok hari berdasarkan data 3 hari terakhir, kemudian hasilnya diteruskan ke sistem fuzzy untuk menghasilkan Indeks Kekeringan.

Aplikasi dikembangkan menggunakan **Streamlit** sebagai antarmuka web interaktif, lengkap dengan visualisasi fungsi keanggotaan, heatmap fuzzifikasi, area defuzzifikasi, dan perbandingan output Mamdani vs Sugeno.

---

## 🗂️ Struktur Repository

```
├── app.py               # Aplikasi Streamlit utama (UI + alur prediksi)
├── fuzzy_logic.py       # Implementasi sistem fuzzy (fuzzifikasi, rule base, defuzzifikasi)
├── train_dl.py          # Script pelatihan model Deep Learning (MLP)
├── Code Analisa.ipynb   # Notebook analisis eksploratif dan eksperimen
├── dl_model.pkl         # Model MLP yang telah dilatih (hasil train_dl.py)
├── scaler.pkl           # Scaler normalisasi data (MinMaxScaler)
├── metrics.pkl          # Metrik evaluasi model (R², MAE, MSE, RMSE, MAPE)
└── __pycache__/         # Cache Python
```

---

## ⚙️ Cara Kerja Sistem

### Variabel Input
| Variabel | Kode BMKG | Deskripsi |
|---|---|---|
| Suhu Rata-rata | `Tavg` | Suhu udara harian rata-rata (°C) |
| Kelembapan | `RH_avg` | Kelembapan relatif rata-rata (%) |
| Curah Hujan | `RR` | Total curah hujan harian (mm) |
| Lama Penyinaran | `ss` | Lama penyinaran matahari (jam) |
| Kecepatan Angin | `ff_avg` | Kecepatan angin rata-rata (knot) |

### Pipeline Fuzzy
```
Input Crisp (Iklim) → Normalisasi → Fuzzifikasi (Trapesium)
       → Evaluasi 15 Rule → Agregasi → Defuzzifikasi
           ├─ Mamdani (Centroid) → Indeks Kekeringan (0–100)
           └─ Sugeno  (WA Const) → Indeks Kekeringan (0–100)
```

### Fungsi Keanggotaan Input (Trapesium)
| Term | Parameter `[a, b, c, d]` |
|---|---|
| Rendah | `[-0.5, -0.1, 0.2, 0.5]` |
| Sedang | `[0.2, 0.35, 0.65, 0.8]` |
| Tinggi | `[0.5, 0.8, 1.1, 1.5]` |

### Fungsi Keanggotaan Output (Trapesium)
| Term | Parameter `[a, b, c, d]` |
|---|---|
| Rendah (Basah) | `[-50, -10, 15, 40]` |
| Sedang (Normal) | `[15, 35, 65, 85]` |
| Tinggi (Kering) | `[60, 80, 110, 150]` |

### Rule Base (15 Aturan)
- **5 Rule Kering** — kombinasi suhu tinggi, hujan rendah, cahaya tinggi, angin tinggi, kelembapan rendah
- **5 Rule Normal** — kombinasi parameter pada level sedang
- **5 Rule Basah** — kombinasi hujan tinggi, suhu rendah, kelembapan tinggi, cahaya rendah

### Pipeline Hybrid Deep Learning + Fuzzy
```
Data 3 Hari Terakhir → Normalisasi (MinMaxScaler)
    → MLP Neural Network → Prediksi Fitur Cuaca Esok Hari
        → Denormalisasi → Sistem Fuzzy
            └─ Indeks Kekeringan Prediksi Esok Hari
```

---

## 🖥️ Tampilan Aplikasi

Aplikasi memiliki **3 halaman utama** yang dapat diakses melalui sidebar:

**1. Beranda & Dataset**
Menampilkan cuplikan data iklim historis dan tren suhu serta kelembapan.

**2. Sistem Fuzzy (Manual)**
Simulasi inferensi fuzzy secara interaktif menggunakan slider. Menampilkan:
- Visualisasi fungsi keanggotaan trapesium (5 variabel input)
- Heatmap derajat keanggotaan hasil fuzzifikasi
- Area defuzzifikasi Mamdani (centroid)
- Scatter plot perbandingan output Mamdani vs Sugeno untuk seluruh dataset

**3. Integrasi Deep Learning (Forecasting)**
Prediksi indeks kekeringan esok hari secara hybrid. Menampilkan:
- Metrik performa model MLP (R², MAE, MSE, RMSE, MAPE)
- Hasil prediksi parameter cuaca esok hari
- Grafik komparasi aktual vs prediksi
- Visualisasi keputusan fuzzy dari output Deep Learning

---

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/wrldwideee/Analisis-Logika-Fuzzy-Metode-Mamdani-dan-Sugeno-dalam-Prediksi-Indeks-Kekeringan-Lahan.git
cd Analisis-Logika-Fuzzy-Metode-Mamdani-dan-Sugeno-dalam-Prediksi-Indeks-Kekeringan-Lahan
```

### 2. Install Dependensi
```bash
pip install -r requirements.txt
```

Atau install secara manual:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
```

### 3. Siapkan Dataset
Letakkan file `climate_data.csv` (data iklim BMKG) di direktori yang sama. File harus memiliki kolom: `Tavg`, `RH_avg`, `RR`, `ss`, `ff_avg`.

### 4. Latih Model Deep Learning (opsional, jika belum ada file `.pkl`)
```bash
python train_dl.py
```
Script ini akan menghasilkan `dl_model.pkl`, `scaler.pkl`, dan `metrics.pkl`.

### 5. Jalankan Aplikasi
```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

---

## 📦 Dependensi

| Library | Kegunaan |
|---|---|
| `streamlit` | Framework antarmuka web interaktif |
| `pandas` | Manipulasi dan pembacaan data |
| `numpy` | Komputasi numerik dan array |
| `matplotlib` | Visualisasi grafik dan plot |
| `seaborn` | Heatmap dan visualisasi statistik |
| `scikit-learn` | MLP Neural Network dan preprocessing |
| `joblib` | Serialisasi model (save/load `.pkl`) |

---

## 📊 Metrik Evaluasi

Model Deep Learning dievaluasi menggunakan metrik regresi berikut:

| Metrik | Deskripsi |
|---|---|
| **R²** | Koefisien determinasi — seberapa baik model menjelaskan variansi data |
| **MAE** | Mean Absolute Error — rata-rata kesalahan absolut |
| **MSE** | Mean Squared Error — rata-rata kuadrat kesalahan |
| **RMSE** | Root Mean Squared Error — akar dari MSE |
| **MAPE** | Mean Absolute Percentage Error — kesalahan dalam persentase |

---

## 🛠️ Pengembangan Lebih Lanjut

Beberapa ide pengembangan yang dapat dilakukan:

- Menambahkan data stasiun BMKG lain untuk cakupan wilayah yang lebih luas
- Menggunakan arsitektur Deep Learning yang lebih kompleks (LSTM/GRU) untuk time series
- Mengoptimalkan rule base fuzzy menggunakan algoritma genetika atau PSO
- Menambahkan fitur peta distribusi kekeringan per wilayah
- Integrasi dengan API data cuaca real-time

---

## 👥 Kontributor

Lihat daftar kontributor di halaman [Contributors](https://github.com/wrldwideee/Analisis-Logika-Fuzzy-Metode-Mamdani-dan-Sugeno-dalam-Prediksi-Indeks-Kekeringan-Lahan/graphs/contributors).

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

*Dikembangkan sebagai implementasi sistem cerdas hibrida untuk pemantauan dan prediksi kekeringan lahan berbasis data iklim.*

---

## 📥 Dataset

Dataset iklim BMKG yang digunakan dalam proyek ini berukuran >20MB sehingga tidak di-host langsung di repository. Download melalui Google Drive:

[⬇️ Download climate_data.csv](https://drive.google.com/uc?export=download&id=1ROSwm5c7ERjDfOqK7mu6hoSsu08r0EqV)

> **Catatan:** Setelah download, letakkan file `climate_data.csv` di direktori yang sama dengan `app.py` sebelum menjalankan aplikasi.

---

## 🚗🍞 mobil roti

![mobil roti](./mobil_roti.png)
