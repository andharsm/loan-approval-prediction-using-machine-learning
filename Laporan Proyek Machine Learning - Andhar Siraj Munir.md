
# Laporan Proyek Machine Learning - Andhar Siraj Munir

## 1. Domain Proyek

### Latar Belakang

Prediksi loan muncul karena krisis ekonomi global yang dipicu pandemi, inflasi, dan suku bunga tinggi [[1]](https://proceedings.unisba.ac.id/index.php/BCSES/article/view/43). Banyak individu melakukan peminjaman untuk bertahan dan meningkatkan perekonomian [[2].](https://idr.uin-antasari.ac.id/19639/) Sebagai lembaga non-perbankan, menghadapi masalah kredit macet yang dapat mempengaruhi deposit dan anggaran peminjaman [[3].](http://repository.ikippgribojonegoro.ac.id/1737/)

Prediksi Status Pinjaman menjadi penting karena membantu instansi keuangan, seperti bank dan swasta, dalam pengambilan keputusan pinjaman. Dengan menggunakan model pembelajaran mesin, dapat memproses informasi dari pemohon seperti pendapatan, pendidikan, dan riwayat kredit untuk memprediksi kemungkinan persetujuan atau penolakan pinjaman. Hal ini memungkinkan efisiensi dan objektivitas dalam proses penilaian risiko kredit, meminimalkan potensi default, dan mendukung kebijakan peminjaman yang lebih cerdas. Selain itu, prediksi status pinjaman juga memberikan manfaat bagi calon peminjam, membantu mereka memahami peluang mereka untuk mendapatkan pinjaman sebelum mengajukan aplikasi, dan memberikan transparansi dalam keputusan kredit.

Sebelumnya sudah terdapat beberapa penelitian terkait penggunaan machine learning dalam proses persetujuan pinjaman atau loan. Pada penelitian [[4]](https://teknosi.fti.unand.ac.id/index.php/teknosi/article/view/1555) menggunakan algoritma Random Forest dengan akurasi prediksi 77%. Berdasarkan penelitian tersebut, penelitian ini akan berfokus mengoptimasi algoritma Random Forest terhadap data [Loan Eligible Dataset.](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset)

## 2. Business Understanding

### 2.1. Problem Statements
1. Proses penilaian kredit dalam pemberian pinjaman seringkali kompleks dan melibatkan banyak faktor.
2. Sistem prediksi secara manual yang kurang efektif dapat menyebabkan ketidakpastian dalam menentukan persetujuan atau penolakan pinjaman.

### 2.2. Goals
1. Membuat model machine learning yang dapat memprediksi loan status
2. Mengoptimasi dan membandingkan optimasi yang terbaik untuk model Random Forest dengan dataset tersebut

### 2.3. Solution Statements
Dalam proses pengembangan model machine learning akan dilakukan beberapa percobaan:

1. Melatih model Random Forest
2. Melakukan hyperparameter tuning
3. Melakukan feature selection
4. Melakukan balancing data dengan SMOTE
5. Melakukan evaluasi model dengan metrik accuracy, precision, recall dan f1-score


## 3. Data Understanding

### 3.1. Sumber Data

Dataset yang digunakan pada proyek ini adalah dataset Loan Eligible Dataset yang didapatkan dari Kaggle. Berikut detailnya:

Jenis | Keterangan
--- | ---
Sumber | [Kaggle Dataset : Loan Eligible Datasett](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset)
Lisensi | [Database: Open Database, Contents: Database Contents](https://opendatacommons.org/licenses/dbcl/1-0/)
Kategori | Business, Education, Finance, Banking
Jenis dan Ukuran Berkas | CSV (59.97 kB)

### 3.2. Infomasi Dataset

Nama Kolom | Keterangan
--- | ---
Loan_ID |	Sebuah ID pinjaman unik.
Gender |	Baik laki-laki maupun perempuan.
Married |	Status perkawinan (ya) atau tidak menikah (tidak).
Dependents |	Jumlah orang yang bergantung pada klien.
Education |	Pendidikan pemohon (lulus atau tidak lulus).
Self_Employed |	Bekerja mandiri (Ya/Tidak).
ApplicantIncome |	Pendapatan pemohon.
CoapplicantIncome |	Pendapatan co-pemohon.
LoanAmount |	Jumlah pinjaman dalam ribuan.
Loan_Amount_Term |	Jangka waktu pinjaman dalam bulan.
Credit_History |	Riwayat kredit memenuhi panduan.
Property_Area |	Pemohon tinggal di perkotaan, semi perkotaan, atau pedesaan.
Loan_Status |	Pinjaman disetujui (Ya/Tidak).


### 3.3. Data Visualization
* Distribusi Data Loan Status
![Distribusi data loan status](https://i.postimg.cc/JzHjSPNg/distribusi-data-loan-status.png)

* Distribusi Data Kategorikal Terhadap Loan Status
![Distribusi data ketegorikal](https://i.postimg.cc/hj5pZ3xh/distribusi-data-kategorikal-terhadap-loan-status.png)

* Distribusi Data Numerikal
![Distribusi data numerikal](https://i.postimg.cc/MXktqM9S/ditribusi-data-numerikal.png)
## 4. Data Preparation

Dalam proyek ini, dilakukan beberapa tahap data preparation guna mendapatkan hasil yang maksimal sesuai tujuan.

### 4.1. Data Cleaning
Data cleaning merupakan proses pembersihan data dengan cara melakukan penghapusan data yang tidak terpakai, melakukan imputasi atau mengisi nilai pengganti pada data yang bernilai null. Berikut proses pada data cleaning:

Kolom | Jumlah Null | Penanganan
--- | --- | ---
Loan_ID |            0 | Hapus kolom
Gender |            13 | Isi dengan modus
Married |            3 | Isi dengan modus
Dependents |        15 | Isi dengan modus
Education |          0 | Isi dengan modus
Self_Employed |     32 | Isi dengan modus
ApplicantIncome |    0 | Isi dengan modus
CoapplicantIncome |  0 | Isi dengan modus
LoanAmount |        22 | Isi dengan median
Loan_Amount_Term |  14 | Isi dengan median
Credit_History |    50 | Isi dengan modus
Property_Area |      0 | -
Loan_Status |        0 | -

### 4.2. Encode Label
Tahap encode label dibantu library LabelEncoder dari sklearn untuk mengubah data kategorikal menjadi value numerik [[5]](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).

### 4.3. Feature Selection
Seleksi fitur dilakukan untuk mendapatkan fitur-fitur yang berpengaruh pada sebuah dataset terhadap kolom target. Pada proyek ini seleksi fitur dilakukan dengan bantuan model Random Forest [[6]](https://arxiv.org/abs/1106.5112). Didapatkan 7 fitur yang akan dilatih dan dites serta dibandingkan hasil evaluasinya dengan dataset original.

![Seleksi fitur](https://i.postimg.cc/rsp2xbTm/featur-importance-loan-prediction.png)

### 4.4. Over Sampling dengan Smote
Dikarenakan data target tidak seimbang, perlu dilakukan penyeimbangan data dan melakukan komparasi hasil evaluasi dengan data original. Proses oversampling dilakukan dengan bantuan SMOTE [[7]](https://www.jair.org/index.php/jair/article/view/10302).

![Over sampling SMOTE](https://i.postimg.cc/3wJwnLYs/over-sampling-smote.png)

### 4.5. Split Dataset
Tujuan dari pembagian dataset ini adalah untuk membagi data menjadi dua yang digunakan untuk melatih dan mengevaluasi kinerja model dengan bantuan library train_test_split [[5]](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Pada proyek ini, 80 persen dataset digunakan untuk melatih model, dan 20 persen sisanya digunakan untuk mengevaluasi model.

Data | Presentase
--- | ---
Train | 80%
Test | 20% 


## 5. Modeling
Setelah dilakukan data preparation didapatkan 4 dataset yang akan dilatih dan diujikan. Dataset tersebut adalah dataset original full fitur, dataset original 7 fitur, dataset SMOTE full fitur, dataset SMOTE 7 fitur. Pada proses pelatihan model akan dilakukan optimasi Hyperparameter tuning dengan model Random Forest. Hyperparameter tuning dilakukan dengan bantuan library GridSearchCV [[5]](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

### 5.1.  Hyperparameter tuning

Hyperparameter tuning dilakukan guna mendapatkan arsitektur model machine learning yang optimal [[8]](https://www.sciencedirect.com/science/article/abs/pii/S0925231220311693). Pada model Random Forest dilakukan tuning terhadap beberapa parameter dan dilakukan cross validation sebagai berikut: 

Parameter | Value
--- | ---
n_estimators | [50, 100, 200]
max_depth | [None, 10, 20, 30]
min_samples_split | [2, 5, 10]
min_samples_leaf | [1, 2, 4]
criterion | ['gini', 'entropy']
cross validation | 5 kali
scoring | accuracy

### 5.2. Result Model

Didapatkan 4 model hasil dari Hyperparameter tuning, berikut arsitekturnya:

Model | Parameter
--- | ---
Random Forest x Dataset Original | {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 50}
Random Forest x Dataset SMOTE | {'criterion': 'entropy', 'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
Random Forest x Dataset 7 Fitur | {'criterion': 'entropy', 'max_depth': 30, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50}
Random Forest x Dataset SMOTE 7 fitur | {'criterion': 'entropy', 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

## 6. Evaluasi

Evaluasi dilakukan dengan memprediksi data test dengan masing-masing model yang sudah dilatih. Metrik evaluasi yang dibandingkan adalah accuracy, precision, recall dan f1-score.

True Positive (TP): Jumlah observasi positif yang benar-benar diklasifikasikan dengan benar oleh model.

True Negative (TN): Jumlah observasi negatif yang benar-benar diklasifikasikan dengan benar oleh model.

False Positive (FP): Jumlah observasi negatif yang keliru diklasifikasikan sebagai positif oleh model (Type I error).

False Negative (FN): Jumlah observasi positif yang keliru diklasifikasikan sebagai negatif oleh model (Type II error).


* Accuracy
Accuracy mengukur sejauh mana model klasifikasi mampu mengklasifikasikan dengan benar keseluruhan observasi. Rumusnya adalah:

    accuracy=  (TP+TN)/(TP+TN+FP+FN)

* Precision
Precision (positive predictive value) mengukur sejauh mana observasi yang diklasifikasikan sebagai positif oleh model adalah benar positif. Rumusnya adalah:

    precision=  TP/(TP+FP)


* Recall
Recall mengukur sejauh mana model dapat mengidentifikasi secara benar semua observasi positif. Rumusnya adalah:

    recall=  TP/(TP+FN)

* F1-Score
F1-Score adalah ukuran gabungan dari precision dan recall. Ini memberikan keseimbangan antara kedua metrik tersebut. Rumusnya adalah:

    f1-score=2 ×  (precisio×recall)/(precision+recall)


![Komparasi evaluasi model](https://i.postimg.cc/fLy9LrhM/evaluasi-model-loan-approval-status.png)

Dari optimasi model Random Forest didapatkan model dan dataset terbaik adalah selected RF Smote yang mendapatkan akurasi 100% dari data test.
## Referensi
[1] N. Y. Puri and I. Amaliah, "Pengaruh Inflasi, Suku Bunga, PDB, Nilai Tukar dan Krisis Ekonomi terhadap Neraca Perdagangan Indonesia Periode 1995-2017," Bandung Conference Series: Economics Studies, vol. 1, no. 1, 2021.

[2] K. H. Alpasa, “Peran Pembiayaan Di Bank Syariah Indonesia Dalam Membantu Pembiayaan Usaha Mikro Kecil Dan Menengah (Umkm) Kota Banjarmasin,” UIN Antasari Banjarmasin, 2022. [Online]. Available: https://idr.uin-antasari.ac.id/19639/

[3]	D. A. Lestari, “Analisis Kinerja Keuangan Unit Pengelola Keuangan (Upk) Dalam Badan Keswadayaan Masyarakat (Bkm) Baroka,” IKIP PGRI Bojonegoro, 2021. [Online]. Available: http://repository.ikippgribojonegoro.ac.id/1737/

[4] B. Prasojo and E. Haryatmi, “Analisa Prediksi Kelayakan Pemberian Kredit Pinjaman dengan Metode Random Forest,” J. Nasionak Teknol. dan Sist. Inf., vol. 7, no. 2, 2021, doi: https://doi.org/10.25077/TEKNOSI.v7i2.2021.79-89.

[5] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011.

[6] M. B. Kursa and W. R. Rudnicki, “The All Relevant Feature Selection using Random Forest,” 2011. https://doi.org/10.48550/arXiv.1106.5112

[7] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic Minority Over-sampling Technique,” J. Artif. Intell. Res., vol. 16, pp. 321–357, 2002.

[8] L. Yang and A. Shami, “On hyperparameter optimization of machine learning algorithms: Theory and practice,” Neurocomputing, vol. 415, pp. 295–316, 2020, doi: https://doi.org/10.1016/j.neucom.2020.07.061.


