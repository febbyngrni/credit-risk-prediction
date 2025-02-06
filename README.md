# Credit Risk Prediction

**Article Report** : [Medium](https://medium.com/@febbyngrni/predicting-credit-risk-with-machine-learning-models-010f8d66beb6)<br>

## Problem Statement
Bank Republik adalah sebuah bank yang menyediakan layanan perbankan untuk individu dan bisnis, dengan tujuan memberikan akses kredit yang mudah dan aman bagi nasabahnya. Saat ini, Bank Republik menghadapi peningkatan Non-Performing Loan (NPL) yang berisiko mengganggu stabilitas keuangan dan operasional. Sistem penilaian kredit saat ini belum efektif dalam mendeteksi calon peminjam berisiko tinggi, sehingga diperlukan solusi berbasis prediksi untuk mengidentifikasi potensi kredit bermasalah sebelum pinjaman disetujui.

Dalam upaya menekan NPL, sistem ini akan mengutamakan recall yang tinggi untuk meminimalkan kemungkinan lolosnya peminjam berisiko. Meskipun berpotensi menghasilkan false positive, setiap prediksi gagal bayar akan divalidasi kembali secara manual sebelum keputusan akhir diambil.

Untuk mengatasi tantangan ini, analisis ini bertujuan untuk:
1. Memprediksi potensi kredit bermasalah sebelum pinjaman disetujui.
2. Mengindentifikasi faktor-faktor utama yang mempengaruhi risiko kredit.
3. Meningkatkan efektivitas sistem penilaian kredit, sehingga dapat mengurangi jumlah NPL di masa mendapatang.

## Modeling Approach
Dalam pemodelan prediksi risiko kredit ini, digunakan pendekatan machine learning untuk mengklasifikasikan peminjam berdasarkan kemungkinan kredit bermasalah. Model yang digunakan dalam analisis ini meliputi:
- **Baseline Model:** Menggunakan kelas mayoritas sebagai tolok ukur awal untuk mengukur performa dasar.
- **Vanilla Model:** Menggunakan model sederhana seperti **Logistic Regression** untuk memahami pola awal data.
- **Multiple Model:** Menguji berbagai model klasifikasi seperti **Random Forest, Decision Tree, KNN, dan SVM** dengan optimasi parameter guna meningkatkan akurasi prediksi.

Model dievaluasi menggunakan **F2 Score**, yang lebih menekankan pada recall dibanding precision. Hal ini bertujuan untuk memastikan bahwa model mampu mendeteksi sebanyak mungkin kasus kredit bermasalah agar bank dapat mengambil tindakan pencegahan lebih dini.

## Preprocessing
Preprocessing dilakukan untuk mempersiapkan data sebelum pelatihan model dengan tahapan berikut:

- **Handling Missing Values**: Mengisi loan_int_rate (9.59% missing) dan person_emp_length (2.80% missing) dengan median.
- **Handling Outliers**: Memfilter data tidak make sense, seperti person_age > 140 tahun dan person_emp_length > 120 tahun.
- **One-Hot Encoding**: Digunakan untuk fitur kategori nominal.
- **Label Encoding**: Diterapkan pada loan grade karena memiliki tingkatan dari A (terbaik) hingga G (terburuk).

## Result

Modeling dilakukan dengan beberapa algoritma, seperti Logistic Regression, Random Forest, Decision Tree, K-Nearest Neighbors (KNN), dan Support Vector Machine (SVM).

| Model                         | Accuracy   |
|-------------------------------|------------|
| Logistic Regression           | 0.823      |
| Random Forest                 | 0.935      |
| Decision Tree                 | 0.896      |
| K-Nearest Neighbors           | 0.843      |
| Support Vectoe Mchine (SVM)   | 0.807      |

Random Forest memiliki akurasi tertinggi sebesar 0.935, menunjukkan bahwa model ini mampu menangkap pola dari data dengan baik dan memberikan prediksi yang paling akurat dibandingkan model lainnya. Kemudian dilakukan hyperparamter tuning pada model ini, dan mendapatkan akurasi yang kurang lebih sama. Hal ini menunjukkan bahwa parameter default sudah cukup optimal untuk dataset ini.

### Threshold Tuning
Pada model Random Forest yang telah dipilih sebagai best model sebelumnya, dilakukan threshold tuning untuk memaksimalkan recall. Threshold tuning bertujuan untuk menyesuaikan batas keputusan model dalam mengklasifikasikan suatu observasi sebagai default atau non-default.

![output1](https://github.com/user-attachments/assets/bb01cda9-181c-4b4c-8955-925dee07a28e)

Dari percobaan berbagai nilai threshold, ditemukan bahwa threshold optimal untuk memaksimalkan F2-score adalah 0.2. Dengan threshold optimal 0.2, model mencapai F2-score maksimum sebesar 0.789. Nilai ini menunjukkan bahwa model memiliki keseimbangan yang lebih baik dalam menangkap nasabah yang berisiko default, dengan lebih menekankan pada recall dibanding precision.

``` 
  Classification Report After Threshold Tuning
  
                precision    recall  f1-score   support
  
             0       0.95      0.90      0.92      4052
             1       0.70      0.81      0.75      1134
  
      accuracy                           0.88      5186
     macro avg       0.82      0.86      0.84      5186
  weighted avg       0.89      0.88      0.89      5186
```

- Recall (Kelas 1) meningkat ke 0.81, memungkinkan model menangkap 81% pelanggan berisiko tinggi, sesuai dengan fokus utama dalam credit risk.
- Precision turun ke 0.70, menyebabkan lebih banyak false positives, yang bisa membuat beberapa pelanggan aman dianggap berisiko.
- Akurasi turun dari 0.94 ke 0.88, tetapi ini sejalan dengan tujuan meningkatkan recall dalam mendeteksi kredit macet.

## SHAP Analysis
![o1](https://github.com/user-attachments/assets/85288c42-bb04-4a0e-976e-3b117dfa70c8)

SHAP summary plot ini menunjukkan bagaimana setiap fitur berkontribusi terhadap keputusan model dalam menentukan risiko kredit. Beberapa insight dari plot ini:
- **Loan Percent Income** — Fitur paling berpengaruh dalam prediksi default. Semakin tinggi proporsi pinjaman terhadap pendapatan (warna merah), semakin besar dampaknya terhadap kemungkinan default (SHAP value positif).
- **Loan Grade** — Grade pinjaman juga memiliki pengaruh besar, dengan nilai yang lebih rendah (cenderung biru) berkontribusi pada risiko default yang lebih kecil.
- **Person Income** — Penghasilan individu memiliki korelasi negatif dengan kemungkinan default; semakin tinggi income (merah), semakin kecil kemungkinan gagal bayar.
- **Person Home Ownership (RENT)** — Pelanggan yang menyewa rumah cenderung memiliki risiko lebih tinggi dibandingkan mereka yang memiliki properti sendiri.
- **Loan Interest Rate** — Suku bunga yang lebih tinggi (merah) meningkatkan kemungkinan default, yang menunjukkan bahwa beban bunga berperan dalam ketidakmampuan membayar.

## Implementation
Untuk menjalankan model ini, gunakan FastAPI dengan command berikut:
``` bash
  fastapi dev api.py
```

Contoh data untuk clustering:
```json
{
  "person_age": 23,
  "person_income": 10000,
  "loan_amnt": 5000,
  "cb_person_cred_hist_length": 3,
  "person_emp_length": 2,
  "loan_int_rate": 7.49,
  "loan_percent_income": 0.10,
  "person_home_ownership": "MORTGAGE",
  "loan_intent": "EDUCATION",
  "loan_grade": "B",
  "cb_person_default_on_file": "N"
}
```

Output:
```json
{
  "res": "Found API",
  "credit_risk_predict": 0,
  "status_code": 200,
  "error_msg": ""
}
```

### Stramlit Interface
Untuk menjalankan model ini pada streamlit gunakan command ini:
``` bash
  streamlit run app.py
```

![Screenshot (101)](https://github.com/user-attachments/assets/b21982d4-7ec0-4d38-97d1-683b2143f9c9)

## Conclusion
1. **Threshold tuning meningkatkan recall** → Recall kelas default naik ke 0.81, meningkatkan deteksi pelanggan berisiko tinggi. Akurasi turun dari 0.94 ke 0.88, tetapi ini adalah trade-off yang sesuai dengan tujuan utama.
2. **Model stabil pada data uji** → Evaluasi menunjukkan recall 0.80 dan precision 0.68, selaras dengan performa pada data training.
3. **Faktor utama risiko kredit** →  
   - **Loan Percent Income** – Semakin besar persentase pinjaman terhadap pendapatan, semakin tinggi risiko default.  
   - **Loan Grade** – Grade pinjaman yang lebih rendah menunjukkan risiko lebih rendah.  
   - **Person Income** – Pendapatan tinggi menurunkan kemungkinan gagal bayar.  
   - **Home Ownership (RENT)** – Penyewa rumah lebih berisiko dibanding pemilik rumah.
   - **Loan Interest Rate** – Suku bunga tinggi meningkatkan risiko default.
