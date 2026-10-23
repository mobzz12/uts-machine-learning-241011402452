# UTS Machine Learning — Linear Regression & Logistic Regression

**Nama  :** Ilham Fajar Setiawan
**NIM   :** 241011402452
**Kelas :** Reg B
**Mata Kuliah :** Machine Learning

---

## 📌 Ringkasan Hasil

`RANDOM_STATE` = **52** (2 digit terakhir NIM 241011402452)

### Bagian A — Linear Regression (Ads_Data.csv, 40 data)

| Hal | Hasil |
|---|---|
| Model (normal equation) | $\hat{y} = 29,4375 + 9,0091X$ |
| Interpretasi $m$ | tiap tambahan 1 juta Rp biaya iklan → penjualan naik ±9,01 juta Rp |
| Evaluasi **training** | MAE 4,3895 · MSE 28,9508 · RMSE 5,3806 · **R² 0,9489** |
| Evaluasi **testing** | MAE 4,9646 · MSE 67,2248 · RMSE 8,1991 · **R² 0,9224** |
| Gradient Descent lr=0.001 | terlalu lambat (loss akhir 50,09, belum konvergen) |
| Gradient Descent lr=0.02 | **pas** — hasil identik normal equation (b=29,4375; m=9,0091) |
| Gradient Descent lr=0.1 | **divergen** — loss membesar → overflow → NaN (±iterasi ke-403) |
| Prediksi 4,5 / 7 / 12 juta | 69,98 / 92,50 / 137,55 juta (12 juta = ekstrapolasi, tidak terlalu bisa dipercaya) |

### Bagian B — Logistic Regression (Email_Data.csv, filter spam, 60 email)

Persamaan model: $P(\text{Spam}) = \sigma(-11,05 + 0,947 \cdot \text{JumlahLink} + 0,928 \cdot \text{KataPromo})$

| Metrik (data testing, 15 email) | Nilai |
|---|---|
| Accuracy | **86,67%** |
| Confusion matrix | TN=6, FP=1, FN=1, TP=7 |
| Precision / Recall / F1 | **0,875 / 0,875 / 0,875** |
| Loss akhir | 0,214 (turun stabil dari 0,693 → konvergen) |

Eksperimen threshold: 0.3 → P=0,700/R=0,875 · 0.5 → P=0,875/R=0,875 · 0.7 → P=1,000/R=0,625.
**Threshold terpilih: 0.5** — keseimbangan terbaik; FP dianggap lebih merugikan pengguna daripada FN.

Prediksi email baru (threshold 0.5): E1(1,2)→Bukan Spam (0,0003) · E2(5,6)→Bukan Spam (0,3213) · E3(9,10)→Spam (0,9988) · E4(2,11)→Spam (0,7411).

### Bagian C & Bonus

5 soal pemahaman konsep dijawab di notebook. Bonus: perbandingan dengan scikit-learn —

- Linear Regression: **identik** dengan implementasi sendiri (b=29,4375; m=9,0091).
- Logistic Regression: sklearn default (L2, C=1) → b=−10,26, m=[0,87; 0,86]; sklearn tanpa regularisasi (C=∞) → b=−12,20, m=[1,05; 1,02]; implementasi sendiri → b=−11,05, m=[0,95; 0,93]. Penyebab perbedaan: regularisasi L2 bawaan sklearn + metode optimasi berbeda (LBFGS konvergen penuh vs GD berhenti di 10.000 iterasi). Akurasi testing keduanya tetap sama: 86,67%.

---

## 📂 Isi Repositori

```
uts-machine-learning-241011402452/
├── README.md            ← file ini
├── tugas_UTS.ipynb      ← notebook utama, sudah Run All (output & grafik lengkap)
├── Ads_Data.csv         ← dataset biaya iklan vs penjualan (40 baris)
└── Email_Data.csv       ← dataset email spam (60 baris)
```

## 🛠️ Teknologi

- Implementasi model & metrik **sendiri dengan NumPy** (mengikuti materi `regression_1.ipynb` & `classification_1.ipynb`): `train_test_split`, `LinearRegression` (normal equation), `LinearRegressionGD`, `LogisticRegression` (gradient descent), `accuracy_score`, `confusion_matrix`, MAE/MSE/RMSE/R², precision/recall/F1.
- Library: `numpy`, `pandas`, `matplotlib` (sesuai ketentuan; `scikit-learn` hanya dipakai di bagian Bonus).
