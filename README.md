<div align="center">

<img src="assets/banner.svg" alt="OncoScan" width="100%"/>

<br/>

[![Live Demo](https://img.shields.io/badge/Live_Demo-oncoscan--mirza.streamlit.app-0066cc?style=for-the-badge&logo=streamlit&logoColor=white)](https://oncoscan-mirza.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-93.86%25-34c759?style=flat-square)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.988-34c759?style=flat-square)

</div>

---

## Tentang Proyek

**OncoScan** adalah aplikasi web interaktif untuk memprediksi apakah sebuah tumor payudara bersifat **ganas (malignant)** atau **jinak (benign)** berdasarkan 10 fitur hasil pengukuran inti sel. Model dilatih pada dataset *Breast Cancer Wisconsin (Diagnostic)* dan dibungkus dalam antarmuka bergaya Apple yang bersih, responsif, dan beranimasi.

Proyek ini dibuat sebagai tugas praktikum Machine Learning, Teknik Informatika, Universitas Pamulang.

> **Catatan:** aplikasi ini untuk tujuan **edukasi**, bukan alat diagnosis medis.

## Fitur

| | Fitur | Keterangan |
|---|---|---|
| 🏠 | **Beranda** | Ringkasan proyek, jumlah data, dan distribusi kelas |
| 📊 | **Dataset** | Pratinjau data, statistik deskriptif, dan visualisasi interaktif |
| 🔬 | **Prediksi** | Input 10 parameter lewat slider atau **upload hasil lab** (CSV/JSON), parameter terisi otomatis |
| 📈 | **Evaluasi Model** | Akurasi, precision, recall, F1, confusion matrix, dan kurva ROC |

### Upload Hasil Lab

Di menu **Prediksi**, unggah file hasil lab dan semua slider langsung menyesuaikan nilainya. Nama kolom dikenali otomatis (misalnya `radius_mean` atau `mean radius`). Tersedia dua file contoh:

- [`contoh_lab_ganas.csv`](contoh_lab_ganas.csv) → diprediksi **Ganas**
- [`contoh_lab_jinak.json`](contoh_lab_jinak.json) → diprediksi **Jinak**

## Performa Model

Model: `RobustScaler` + `RandomForestClassifier` (100 pohon, `max_depth=8`, `class_weight="balanced"`), data latih dan uji 80 : 20 (stratified, `random_state=42`).

| Metrik | Nilai |
|---|---|
| Accuracy | **93,86%** |
| Precision (Ganas) | 88,89% |
| Recall (Ganas) | **95,24%** |
| F1-score | 91,95% |
| ROC-AUC | **0,9884** |

Confusion matrix (data uji, 114 sampel):

|  | Prediksi Ganas | Prediksi Jinak |
|---|:---:|:---:|
| **Aktual Ganas** | 40 | 2 |
| **Aktual Jinak** | 5 | 67 |

Recall yang tinggi untuk kelas ganas penting di kasus ini, karena kasus ganas yang terlewat lebih berisiko daripada alarm palsu.

## Dataset

- **Sumber:** Breast Cancer Wisconsin (Diagnostic)
- **Jumlah data:** 569 sampel (212 ganas, 357 jinak)
- **Fitur dipakai:** 10 fitur *mean*, yaitu `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry`, dan `fractal dimension`

## Menjalankan di Komputer Sendiri

```bash
# 1. Clone repo
git clone https://github.com/Mirzakerenn/oncoscan-streamlit.git
cd oncoscan-streamlit

# 2. Buat dan aktifkan virtual environment
python -m venv env
env\Scripts\activate        # Windows
# source env/bin/activate   # macOS / Linux

# 3. Install dependensi
pip install -r requirements.txt

# 4. Jalankan aplikasi
streamlit run app.py
```

Aplikasi terbuka di `http://localhost:8501`.

## Struktur Folder

```
oncoscan-streamlit/
├── app.py                  # Aplikasi utama Streamlit
├── breast_cancer.csv       # Dataset
├── requirements.txt        # Dependensi Python
├── contoh_lab_ganas.csv    # Contoh file hasil lab (ganas)
├── contoh_lab_jinak.json   # Contoh file hasil lab (jinak)
├── assets/
│   └── banner.svg          # Banner README
└── .streamlit/
    └── config.toml         # Tema aplikasi
```

## Tim

**Kelompok 4** · Teknik Informatika · Universitas Pamulang

| Nama | NIM |
|---|---|
| Mirza Suria Sahir | 241011400498 |
| Daffa Ihsan | _isi NIM_ |
| Ganjar Nopandi | _isi NIM_ |

---

<div align="center">

Dibuat dengan Python dan Streamlit

</div>
