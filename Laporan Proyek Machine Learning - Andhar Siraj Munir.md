
# Laporan Proyek Machine Learning - Andhar Siraj Munir

## 1. Domain Proyek

### Latar Belakang

Prediksi loan muncul karena krisis ekonomi global yang dipicu pandemi, inflasi, dan suku bunga tinggi [[1]](https://proceedings.unisba.ac.id/index.php/BCSES/article/view/43). Banyak individu melakukan peminjaman untuk bertahan dan meningkatkan perekonomian [[2].](https://idr.uin-antasari.ac.id/19639/) Sebagai lembaga non-perbankan, menghadapi masalah kredit macet yang dapat mempengaruhi deposit dan anggaran peminjaman [[3].](http://repository.ikippgribojonegoro.ac.id/1737/)

Prediksi Status Pinjaman menjadi penting karena membantu instansi keuangan, seperti bank dan swasta, dalam pengambilan keputusan pinjaman. Dengan menggunakan model pembelajaran mesin, dapat memproses informasi dari pemohon seperti pendapatan, pendidikan, dan riwayat kredit untuk memprediksi kemungkinan persetujuan atau penolakan pinjaman. Hal ini memungkinkan efisiensi dan objektivitas dalam proses penilaian risiko kredit, meminimalkan potensi default, dan mendukung kebijakan peminjaman yang lebih cerdas. Selain itu, prediksi status pinjaman juga memberikan manfaat bagi calon peminjam, membantu mereka memahami peluang mereka untuk mendapatkan pinjaman sebelum mengajukan aplikasi, dan memberikan transparansi dalam keputusan kredit.

Sebelumnya sudah terdapat beberapa penelitian terkait penggunaan machine learning dalam proses persetujuan pinjaman atau loan. Pada penelitian [[4]](https://teknosi.fti.unand.ac.id/index.php/teknosi/article/view/1555) menggunakan algoritma *Random Forest* dengan akurasi prediksi 77%. Berdasarkan penelitian tersebut, penelitian ini akan berfokus mengoptimasi algoritma *Random Forest* terhadap data [*Loan Eligible Dataset*.](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset)

Dengan menggunakan model pembelajaran mesin, lembaga keuangan dapat secara cepat mengidentifikasi potensi risiko kredit macet melalui analisis data pemohon seperti pendapatan, pendidikan, dan riwayat kredit. Hal ini memungkinkan lembaga keuangan untuk mengambil keputusan pinjaman secara efisien dan objektif, mengurangi potensi bias manusia. Pada proyek ini akan berfokus dengan model pembelajaran mesin *Random Forest* dan melakukan optimasi untuk mendapatkan arsitektur terbaik.

## 2. Business Understanding

### 2.1. Problem Statements
1. Proses penilaian kredit dalam pemberian pinjaman seringkali kompleks dan melibatkan banyak faktor, menyebabkan ketidakpastian dalam menentukan persetujuan atau penolakan pinjaman. Karena kegiatan inti bank, pemberian pinjaman menjadi salah satu yang paling mencolok, dengan pendapatan bunga yang dihasilkan dari pinjaman membentuk bagian yang signifikan dari aset bank [[5]](https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.12707). Dalam sejarah dunia perbankan, proses persetujuan pinjaman selalu melibatkan tantangan untuk memilih peminjam yang dapat diandalkan dari sekian banyak calon peminjam [[6]](https://profesionalmudacendekia.com/index.php/jbmr/article/view/140).

2. Sistem prediksi secara manual yang kurang efektif dapat meningkatkan risiko ketidakakuratan dalam menilai kemungkinan kredit macet. Kegagalan dalam memprediksi kegagalan pinjaman telah membawa dampak yang luas, termasuk krisis perbankan (Musdholifah et al., 2020). Mengidentifikasi peminjam yang gagal membayar secara manual terbukti menjadi tugas yang rumit, mengingat kompleksitas lanskap perbankan saat ini dan permintaan pinjaman yang terus meningkat [[7]](https://www.sciencedirect.com/science/article/pii/S2666307423000293). 

### 2.2. Goals
1. Membuat model machine learning yang dapat memprediksi loan status.

Algoritma pembelajaran mesin (ML), yang memberdayakan sistem untuk menguraikan pola secara mandiri dan membuat prediksi berdasarkan data, telah muncul sebagai solusi yang menjanjikan untuk menilai kemungkinan gagal bayar pinjaman [[8]](https://www.sciencedirect.com/science/article/pii/S266630742300013X).

2. Mengoptimasi dan membandingkan optimasi yang terbaik untuk model *Random Forest* dengan dataset tersebut.

Optimasi akan dilakukan dengan melakukan *hyperparameter* *tuning* . *Hyperparameter* *tuning* adalah bagian penting dari proses pengembangan model machine learning Ini memastikan bahwa model machine learning memiliki *hyperparameter* yang tepat sehingga dapat memberikan hasil yang baik [[9]](https://ivosights.com/read/artikel/machine-learning-mengenal-apa-itu-hyperparameter-tuning-dalam).

### 2.3. Solution Statements
Dalam proses pengembangan model machine learning akan dilakukan beberapa percobaan:

1. Melatih model *Random Forest*
2. Melakukan *hyperparameter* *tuning*
3. Melakukan feature selection
4. Melakukan balancing data dengan SMOTE
5. Melakukan evaluasi model dengan metrik *accuracy, precision, recall* dan *f1-score*


## 3. Data Understanding

### 3.1. Sumber Data

Dataset yang digunakan pada proyek ini adalah dataset *Loan Eligible Dataset* yang didapatkan dari Kaggle. Detail sumber dataset ditunjukan pada Tabel 3.1.


Tabel 3.1. Deskripsi *Loan Eligible Dataset*
Jenis | Keterangan
--- | ---
Sumber | [Kaggle Dataset : *Loan Eligible Dataset*](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset)
Lisensi | [*Database: Open Database, Contents: Database Contents*](https://opendatacommons.org/licenses/dbcl/1-0/)
Kategori | *Business, Education, Finance, Banking*
Jenis dan Ukuran Berkas | CSV (59.97 kB)

### 3.2. Infomasi Dataset

Informasi atribut dataset ditunjukan pada Tabel 3.2 berikut.

Tabel 3.2. *Value Key* Dataset
Nama Kolom | Keterangan
--- | ---
*Loan_ID* |	Sebuah ID pinjaman unik.
*Gender* |	Baik laki-laki maupun perempuan.
*Married* |	Status perkawinan (ya) atau tidak menikah (tidak).
*Dependents* |	Jumlah orang yang bergantung pada klien.
*Education* |	Pendidikan pemohon (lulus atau tidak lulus).
*Self_Employed* |	Bekerja mandiri (Ya/Tidak).
*ApplicantIncome* |	Pendapatan pemohon.
*CoapplicantIncome* |	Pendapatan co-pemohon.
*LoanAmount* |	Jumlah pinjaman dalam ribuan.
*Loan_Amount_Term* |	Jangka waktu pinjaman dalam bulan.
*Credit_History* |	Riwayat kredit memenuhi panduan.
*Property_Area* |	Pemohon tinggal di perkotaan, semi perkotaan, atau pedesaan.
*Loan_Status* |	Pinjaman disetujui (Ya/Tidak).


### 3.3. Data Visualization
* Distribusi Data Loan Status
![distribusi data loan status](https://github.com/andharsm/loan-approval-prediction-using-machine-learning/assets/114974933/1785f6f8-fe54-425d-9ee5-4fa156be8dea)
Gambar 3.1. Distribusi Data Loan Status

Pada Gambar 3.1, status loan ditolak lebih rendah dari loan diterima. Berdasarkan data tersebut, dataset ini dapat dikatakan dataset yang tidak seimbang atau imbalance. 

* Distribusi Data Kategorikal Terhadap Loan Status
![distribusi data kategorikal](https://github.com/andharsm/loan-approval-prediction-using-machine-learning/assets/114974933/924f5515-795f-425f-a2de-65451661a9f5)
Gambar 3.2. Distribusi Data Kategorikal Terhadap Loan Status

* Distribusi Data Numerikal
![distribusi data numerikal](https://github.com/andharsm/loan-approval-prediction-using-machine-learning/assets/114974933/bfc29fca-9747-4593-b2ee-069eb8ff3818)
Gambar 3.3. Distribusi Data Numerikal

* Matrik Korelasi Dataset
![matriks korelasi](https://github.com/andharsm/loan-approval-prediction-using-machine-learning/assets/114974933/08c6558e-437f-4561-9f8d-25a6fd842016)
Gambar 3.4 Matrik Korelasi

Gambar 3.4 diatas menampilkan korelasi antar data numerikal, data yang paling berpengaruh satu sama lain hterdapat antara *LoanAmount* dan *ApplicantIncome* dengan nilai korelasi 0.57.

## 4. Data Preparation

Dalam proyek ini, dilakukan beberapa tahap data preparation guna mendapatkan hasil yang maksimal sesuai tujuan.

### 4.1. Data Cleaning
Data cleaning merupakan proses pembersihan data dengan cara melakukan penghapusan data yang tidak terpakai, melakukan imputasi atau mengisi nilai pengganti pada data yang bernilai null. Proses data cleaning ditunjukan pada Tabel 4.1 berikut.

Tabel 4.1. Proses Data Cleaning

Kolom | Jumlah Null | Penanganan
--- | --- | ---
*Loan_ID* |            0 | Hapus kolom
*Gender* |            13 | Isi dengan modus
*Married* |            3 | Isi dengan modus
*Dependents* |        15 | Isi dengan modus
*Education* |          0 | Isi dengan modus
*Self_Employed* |     32 | Isi dengan modus
*ApplicantIncome* |    0 | Isi dengan modus
*CoapplicantIncome* |  0 | Isi dengan modus
*LoanAmount* |        22 | Isi dengan median
*Loan_Amount_Term* |  14 | Isi dengan median
*Credit_History* |    50 | Isi dengan modus
*Property_Area* |      0 | -
*Loan_Status* |        0 | -

### 4.2. Encode Label
Tahap *encode* label dibantu library *LabelEncoder* dari *sklearn* untuk mengubah data kategorikal menjadi value numerik [[10]](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).

### 4.3. Feature Selection
Seleksi fitur dilakukan untuk mendapatkan fitur-fitur yang berpengaruh pada sebuah dataset terhadap kolom target. Pada proyek ini seleksi fitur dilakukan dengan bantuan model *Random Forest* [[11]](https://arxiv.org/abs/1106.5112). 

![fieature importance](https://github.com/andharsm/loan-approval-prediction-using-machine-learning/assets/114974933/1f1126cb-a453-40e2-9b29-d9081ae83512)

Gambar 4.1. Feature Selection

Ditunjukan pada Gambar 4.1 ditemukan urutan fitur paling berpengaruh. Berdasarkan data yang ada, ditentukan threshold 0.4 sehingga terpilih 7 fitur berpengaruh seperti yang ditunjukan pada gambar. 

### 4.4. Over Sampling dengan Smote
Dikarenakan data target tidak seimbang, perlu dilakukan penyeimbangan data dan melakukan komparasi hasil evaluasi dengan data original. Proses oversampling dilakukan dengan bantuan SMOTE [[12]](https://www.jair.org/index.php/jair/article/view/10302).

![hasil smote](https://github.com/andharsm/loan-approval-prediction-using-machine-learning/assets/114974933/b105f1d5-0a77-4972-8c28-41436c31b5cc)

Gambar 4.2. Distribusi Data Setelah Balancing dengan SMOTE

Hasil dari proses *balancing* data ditunjukanpada Gambar 4.2, data loan status dengan value N sudah seimbang dengan value Y. Dataset SMOTE ini nantinya akan dibandingkan dengan data original dan data seleksi fitur pada proses evaluasi untuk mendapatkan arsitektur model dan data yang terbaik.

### 4.5. Split Dataset
Tujuan dari pembagian dataset ini adalah untuk membagi data menjadi dua yang digunakan untuk melatih dan mengevaluasi kinerja model dengan bantuan library train_test_split [[10]](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Pada proyek ini, 80 persen dataset digunakan untuk melatih model, dan 20 persen sisanya digunakan untuk mengevaluasi model.


## 5. Modeling
Setelah dilakukan data preparation didapatkan 4 dataset yang akan dilatih dan diujikan. Dataset tersebut adalah dataset original full fitur, dataset original 7 fitur, dataset SMOTE full fitur, dataset SMOTE 7 fitur. Pada proses pelatihan model akan dilakukan optimasi *Hyperparameter* *tuning* dengan model *Random Forest*. *Hyperparameter* *tuning* dilakukan dengan bantuan library GridSearchCV [[10]](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

### 5.1.  *Hyperparameter* *tuning*
*Hyperparameter* *tuning* dilakukan guna mendapatkan arsitektur model machine learning yang optimal [[13]](https://www.sciencedirect.com/science/article/abs/pii/S0925231220311693). Pada model *Random Forest* dilakukan *tuning* terhadap beberapa parameter dan dilakukan cross validation sebagai berikut: 

Tabel 4.2. Daftar *Hyperparameter*

Parameter | Value
--- | ---
n_estimators | [50, 100, 200]
max_depth | [None, 10, 20, 30]
min_samples_split | [2, 5, 10]
min_samples_leaf | [1, 2, 4]
criterion | ['gini', 'entropy']
cross validation | 5 kali
scoring | accuracy

Berdasarkan Tabel 4.2, proyek ini akan dilakukan eksperimen terhadap beberapa parameter yang disediakan oleh *Random Forest*.

### 5.2. Result Model

Didapatkan 4 model hasil dari *Hyperparameter* *tuning*, berikut arsitekturnya:

Tabel 4.3. Hasil *Hyperparameter* *Tuning*

Model | Parameter
--- | ---
*Random Forest* x Dataset Original | {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 50}
*Random Forest* x Dataset SMOTE | {'criterion': 'entropy', 'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
*Random Forest* x Dataset 7 Fitur | {'criterion': 'entropy', 'max_depth': 30, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50}
*Random Forest* x Dataset SMOTE 7 fitur | {'criterion': 'entropy', 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

Tabel 4.3 menampilkan kumpulan parameter terbaik hasil *tuning* pada setiap dataset.

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

Gambar 5.1. Komparasi Evaluasi Model
![komparasi metrik evaluasi](https://github.com/andharsm/loan-approval-prediction-using-machine-learning/assets/114974933/210d5fc4-8a60-49ef-9541-a23950e77db7)

Berdasarkan hasil evaluasi yang ditunjukan pada Gambar 5.1. Kombinasi feature selection dan penggunaan SMOTE menghasilkan performa sempurna dengan nilai akurasi, precision, recall, dan F1-Score mencapai 100%. Namun, perlu diingat bahwa hasil sempurna seperti ini juga dapat menunjukkan kemungkinan adanya overfitting, terutama ketika diterapkan pada data yang baru.

Namun secara keseluruhan proses seleksi fitur dan SMOTE Penerapan pada model *Random Forest* dapat meningkatkan akurasi dan F1-Score. Dengan hasil tersebut optimasi model *Random Forest* berhasil diterapkan pada dataset *Loan Eligible Dataset*.


## Referensi
[1] N. Y. Puri and I. Amaliah, "Pengaruh Inflasi, Suku Bunga, PDB, Nilai Tukar dan Krisis Ekonomi terhadap Neraca Perdagangan Indonesia Periode 1995-2017," Bandung Conference Series: Economics Studies, vol. 1, no. 1, 2021.

[2] K. H. Alpasa, “Peran Pembiayaan Di Bank Syariah Indonesia Dalam Membantu Pembiayaan Usaha Mikro Kecil Dan Menengah (Umkm) Kota Banjarmasin,” UIN Antasari Banjarmasin, 2022. [Online]. Available: https://idr.uin-antasari.ac.id/19639/

[3]	D. A. Lestari, “Analisis Kinerja Keuangan Unit Pengelola Keuangan (Upk) Dalam Badan Keswadayaan Masyarakat (Bkm) Baroka,” IKIP PGRI Bojonegoro, 2021. [Online]. Available: http://repository.ikippgribojonegoro.ac.id/1737/

[4] B. Prasojo and E. Haryatmi, “Analisa Prediksi Kelayakan Pemberian Kredit Pinjaman dengan Metode *Random Forest*,” J. Nasionak Teknol. dan Sist. Inf., vol. 7, no. 2, 2021, doi: https://doi.org/10.25077/TEKNOSI.v7i2.2021.79-89.

[5] Dansana, Debabrata, et al. "Analyzing the impact of loan features on bank loan prediction using R andom F orest algorithm." Engineering Reports 6.2 (2024): e12707.

[6] Khairi, A., Bahri, B., & Artha, B. (2021). A literature review of non-performing loan. Journal of Business and Management Review, 2(5), 366-373.

[7] Uddin, N., Ahamed, M. K. U., Uddin, M. A., Islam, M. M., Talukder, M. A., & Aryal, S. (2023). An ensemble machine learning based bank loan approval predictions system with a smart application. International Journal of Cognitive Computing in Engineering, 4, 327-339.

[8] Mustaffa, Z., & Sulaiman, M. H. (2023). Stock price predictive analysis: An application of hybrid Barnacles Mating Optimizer with Artificial Neural Network. International Journal of Cognitive Computing in Engineering, 4, 109-117.

[9] Administrator, “Mengenal Apa Itu Hyperparameter Tuning dalam Machine Learning,” 2023. https://ivosights.com/read/artikel/machine-learning-mengenal-apa-itu-hyperparameter-tuning-dalam

[10] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011.

[11] M. B. Kursa and W. R. Rudnicki, “The All Relevant Feature Selection using *Random Forest*,” 2011. https://doi.org/10.48550/arXiv.1106.5112

[12] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic Minority Over-sampling Technique,” J. Artif. Intell. Res., vol. 16, pp. 321–357, 2002.

[13] L. Yang and A. Shami, “On *hyperparameter* optimization of machine learning algorithms: Theory and practice,” Neurocomputing, vol. 415, pp. 295–316, 2020, doi: https://doi.org/10.1016/j.neucom.2020.07.061.


