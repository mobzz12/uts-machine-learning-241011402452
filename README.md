# UTS Machine Learning — Linear Regression & Logistic Regression

Nama: Ilham Fajar Setiawan
NIM: 241011402452
Kelas: Reg B

Tugas UTS mata kuliah Machine Learning. Semua model dan metrik evaluasi diimplementasikan sendiri dengan NumPy, disalin dari materi `simple_regression/regression_1.ipynb` dan `simple_classification/classification_1.ipynb` sesuai ketentuan soal. scikit-learn hanya dipakai di bagian bonus untuk pembanding.

RANDOM_STATE = 52 (2 digit terakhir NIM).

## Ringkasan hasil

### Bagian A — Linear Regression (Ads_Data.csv, 40 data)

Persamaan model: y = 29,44 + 9,01X. Tiap tambahan 1 juta Rp biaya iklan menaikkan penjualan sekitar 9 juta Rp per bulan.

Evaluasi (training / testing): R² 0,9489 / 0,9224, RMSE 5,38 / 8,20, MAE 4,39 / 4,96, MSE 28,95 / 67,22. Selisih train dan test kecil jadi tidak overfit.

Eksperimen gradient descent: lr=0.001 konvergen tapi lambat banget (loss akhir 50,09, belum sampai optimum), lr=0.02 pas (hasilnya sama persis dengan normal equation), lr=0.1 divergen — loss naik terus sampai overflow dan jadi NaN di sekitar iterasi ke-403.

Prediksi untuk biaya iklan 4,5 / 7 / 12 juta: 69,98 / 92,50 / 137,55 juta. Prediksi 12 juta ekstrapolasi (data cuma sampai 9,9 juta), jadi menurut saya tidak terlalu bisa dipercaya.

### Bagian B — Logistic Regression (Email_Data.csv, filter spam, 60 email)

P(Spam) = σ(-11,05 + 0,947·JumlahLink + 0,928·KataPromo)

Evaluasi data testing (15 email): accuracy 86,67% (13/15 benar), confusion matrix TN=6 FP=1 FN=1 TP=7, precision 0,875, recall 0,875, F1 0,875.

Eksperimen threshold: 0.3 → precision 0,700 / recall 0,875. 0.5 → 0,875 / 0,875. 0.7 → precision 1,000 tapi recall anjlok ke 0,625. Saya pilih threshold 0.5 karena keseimbangannya paling baik, sambil tetap jaga precision karena FP (email penting salah masuk spam) lebih merugikan daripada FN.

Prediksi email baru: E1(1,2) → Bukan Spam (P=0,0003), E2(5,6) → Bukan Spam (P=0,3213), E3(9,10) → Spam (P=0,9988), E4(2,11) → Spam (P=0,7411).

### Bonus — perbandingan scikit-learn

Linear Regression: identik dengan implementasi sendiri (b=29,4375, m=9,0091).

Logistic Regression: sklearn default (L2, C=1) → b=-10,26; sklearn tanpa regularisasi (C=∞) → b=-12,20; implementasi sendiri → b=-11,05. Bedanya karena sklearn pakai regularisasi L2 bawaan dan solver LBFGS yang konvergen penuh, sedangkan GD saya berhenti di 10.000 iterasi. Akurasi testing keduanya sama: 86,67%.

## Isi repositori

- `tugas_UTS.ipynb` — notebook utama, sudah dijalankan lengkap
- `Ads_Data.csv` — dataset biaya iklan vs penjualan
- `Email_Data.csv` — dataset email spam/bukan spam
