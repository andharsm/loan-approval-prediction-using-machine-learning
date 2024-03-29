# -*- coding: utf-8 -*-
"""Loan Status Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13TooUJ6kE3G70uX5jsdcNgkKgi_Y9SEB

# Predictive Analytics dengan Dataset Loan Status Prediction - Andhar Siraj Munir

# Domain Proyek

## Latar Belakang

Prediksi Status Pinjaman menjadi penting karena membantu instansi keuangan, seperti bank, dalam pengambilan keputusan pinjaman. Dengan menggunakan model pembelajaran mesin, dapat memproses informasi dari pemohon seperti pendapatan, pendidikan, dan riwayat kredit untuk memprediksi kemungkinan persetujuan atau penolakan pinjaman. Hal ini memungkinkan efisiensi dan objektivitas dalam proses penilaian risiko kredit, meminimalkan potensi default, dan mendukung kebijakan peminjaman yang lebih cerdas. Selain itu, prediksi status pinjaman juga memberikan manfaat bagi calon peminjam, membantu mereka memahami peluang mereka untuk mendapatkan pinjaman sebelum mengajukan aplikasi, dan memberikan transparansi dalam keputusan kredit.

## Referensi

Artikel [An Approach For Prediction Of Loan Approval
Using Machine Learning Algorithm](https://ijcrt.org/papers/IJCRT2106313.pdf)

# Business Understanding

## Problem Statements

1. Proses penilaian kredit dalam pemberian pinjaman seringkali kompleks dan melibatkan banyak faktor.
2. Sistem prediksi secara manual yang kurang efektif dapat menyebabkan ketidakpastian dalam menentukan persetujuan atau penolakan pinjaman.

## Goals

Tujuan utama proyek ini adalah membangun Model Pembelajaran Mesin yang dapat memprediksi status persetujuan atau penolakan pinjaman berdasarkan faktor-faktor seperti pendapatan, pendidikan, dan riwayat kredit. Dengan adanya model ini, diharapkan dapat meningkatkan efisiensi proses pengambilan keputusan kredit dan mengurangi risiko default. Selain itu, tujuan lainnya adalah memberikan transparansi kepada calon peminjam mengenai peluang mereka untuk mendapatkan pinjaman, sehingga memperkuat kepercayaan dan meminimalkan ketidakpastian dalam proses aplikasi pinjaman.

## Solution statements

Dalam proses pengembangan model machine learning akan dilakukan beberapa percobaan:

1. Melatih dataset dengan 2 model yaitu logistic regression dan random forest
2. Menggunakan hyperparameter tuning
3. Menggunakan feature selection

# Data Understanding

Pada proyek ini menggunakan dataset publik. Dataset tersebut adalah [Loan Eligible Dataset](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset) yang bersumber dari Kaggle

## Tentang Dataset

* Loan_ID: Sebuah ID pinjaman unik.
* Gender: Baik laki-laki maupun perempuan.
* Married: Status perkawinan (ya) atau tidak menikah (tidak).
* Dependents: Jumlah orang yang bergantung pada klien.
* Education: Pendidikan pemohon (lulus atau tidak lulus).
* Self_Employed: Bekerja mandiri (Ya/Tidak).
* ApplicantIncome: Pendapatan pemohon.
* CoapplicantIncome: Pendapatan co-pemohon.
* LoanAmount: Jumlah pinjaman dalam ribuan.
* Loan_Amount_Term: Jangka waktu pinjaman dalam bulan.
* Credit_History: Riwayat kredit memenuhi panduan.
* Property_Area: Pemohon tinggal di perkotaan, semi perkotaan, atau pedesaan.
* Loan_Status: Pinjaman disetujui (Ya/Tidak).

## Import Library
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

"""## Load Data

Import data dari Kaggle
"""

# Define your Kaggle API credentials
kaggle_credentials = {
    "username": "andharsm",
    "key": "aa01adb9e93691614d246ac084458990"
}

# Save the credentials to a file named kaggle.json
with open('/content/kaggle.json', 'w') as file:
    json.dump(kaggle_credentials, file)

!mkdir -p ~/.kaggle
!cp '/content/kaggle.json' ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d vikasukani/loan-eligible-dataset

# unzip
!mkdir loan-dataset
!unzip -qq loan-eligible-dataset.zip -d loan-dataset
!ls loan-dataset

"""Pada dataset terdapat 2 data yang sudah dipisah yaitu data train dan data test

## Data Exploratotion

Menampilkan dataset dalam dataframe dan melihat 5 dataset pertama
"""

df= pd.read_csv('/content/loan-dataset/loan-train.csv')

df.head()

"""Menampilkan dimensi dataset"""

df.shape

"""Terdapat 614 data latih dan 367 data uji

Menampilkan informasi dataset
"""

df.info()

"""Terdapat 13 kolom dataset, 8 data objek, 1 data integer dan 4 data float

Menampilkan ringkasan statistik dari kolom data numerik
"""

df.describe()

"""Menampilkan ringkasan statistik dari kolom data kategorikal"""

df.describe(include='O')

"""Menampilkan nilai unik setiap kolom"""

df.nunique()

"""Menampilkan missing value pada dataset"""

df.isnull().sum()

"""Terdapat beberapa data null pada beberapa kolom

Menampilkan data duplikasi
"""

df.duplicated().sum()

"""Tidak ada data duplikasi

## Visualisasi Data

### Distribusi Loan Status
"""

# Menghitung distribusi Loan Status
loan_status_distribution = df['Loan_Status'].value_counts()

# Membuat diagram batang
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(loan_status_distribution.index, loan_status_distribution.values, color=['lightblue', 'lightcoral'])
plt.title('Distribution of Loan Status (Bar Chart)')
plt.xlabel('Loan Status')
plt.ylabel('Count')

# Membuat diagram pie
plt.subplot(1, 2, 2)
plt.pie(loan_status_distribution, labels=loan_status_distribution.index, autopct='%1.1f%%', colors=['lightblue', 'lightcoral'], startangle=90)
plt.title('Distribution of Loan Status (Pie Chart)')

plt.tight_layout()
plt.show()

"""Pada dataset yang digunakan status loan ditolak lebih rendah dari loan diterima yaitu 31.3%. Berdasarkan data tersebut, dataset ini dapat dikatakan dataset yang tidak seimbang atau imbalance.

### Distribusi Gender
"""

# Membuat diagram batang untuk visualisasi Gender terhadap Loan Status
gender_loan_status_count = pd.crosstab(df['Gender'], df['Loan_Status'])
gender_loan_status_count

# Normalisasi data ke dalam presentase
gender_loan_status_percentage = gender_loan_status_count.div(gender_loan_status_count.sum(axis=1), axis=0) * 100
gender_loan_status_percentage

# Visualisasi Distribusi Gender
gender_loan_status_count.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'])
plt.title('Loan Status Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Loan Status', bbox_to_anchor=(1, 1))
plt.show()

# Plotting
fig, ax = plt.subplots()

# Bar plot dengan nilai presentase dan warna kustom
bars = gender_loan_status_percentage.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'], ax=ax, width=0.7)

# Menambahkan label
ax.set_ylabel('Percentage')
ax.set_title('Loan Status by Gender')

# Menambahkan legenda
ax.legend(title='Loan Status', bbox_to_anchor=(1.05, 1), loc='upper left')

# Menambahkan anotasi pada setiap stack
for bar in bars.patches:
    width, height = bar.get_width(), bar.get_height()
    x, y = bar.get_xy()
    ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center', fontsize=8, color='white')

"""Dari distribusi gender, didapatkan insight bahwa pemohon lebih banyak berjenis kelamin laki-laki. Presentase distirbusi setiap gender terhadap setatus pengajuan loan memiliki presentase yang hampir sama.

### Distribusi Married
"""

# Membuat diagram batang untuk visualisasi Married terhadap Loan Status
married_loan_status_count = pd.crosstab(df['Married'], df['Loan_Status'])
married_loan_status_count

# Normalisasi data ke dalam presentase
married_loan_status_percentage = married_loan_status_count.div(married_loan_status_count.sum(axis=1), axis=0) * 100
married_loan_status_percentage

# Visualisasi Distribusi Married
married_loan_status_count.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'])
plt.title('Loan Status Distribution by Married')
plt.xlabel('Married')
plt.ylabel('Count')
plt.legend(title='Loan Status', bbox_to_anchor=(1, 1))
plt.show()

# Plotting
fig, ax = plt.subplots()

# Bar plot dengan nilai presentase dan warna kustom
bars = married_loan_status_percentage.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'], ax=ax, width=0.7)

# Menambahkan label
ax.set_ylabel('Percentage')
ax.set_title('Loan Status by married')

# Menambahkan legenda
ax.legend(title='Loan Status', bbox_to_anchor=(1.05, 1), loc='upper left')

# Menambahkan anotasi pada setiap stack
for bar in bars.patches:
    width, height = bar.get_width(), bar.get_height()
    x, y = bar.get_xy()
    ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center', fontsize=8, color='white')

"""Pemohon loan lebih banyak yang berstatus sudah menikah daripada yang belum menikah. Presentasi pada masing-masing status menikah didominasi dengan status Yes

### Distribusi Dependents
"""

# Membuat diagram batang untuk visualisasi Dependents terhadap Loan Status
dependents_loan_status_count = pd.crosstab(df['Dependents'], df['Loan_Status'])
dependents_loan_status_count

# Normalisasi data ke dalam presentase
dependents_loan_status_percentage = dependents_loan_status_count.div(dependents_loan_status_count.sum(axis=1), axis=0) * 100
dependents_loan_status_percentage

# Visualisasi Distribusi Dependents
dependents_loan_status_count.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'])
plt.title('Loan Status Distribution by Dependents')
plt.xlabel('Dependents')
plt.ylabel('Count')
plt.legend(title='Loan Status', bbox_to_anchor=(1, 1))
plt.show()

# Plotting
fig, ax = plt.subplots()

# Bar plot dengan nilai presentase dan warna kustom
bars = dependents_loan_status_percentage.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'], ax=ax, width=0.7)

# Menambahkan label
ax.set_ylabel('Percentage')
ax.set_title('Loan Status by dependents')

# Menambahkan legenda
ax.legend(title='Loan Status', bbox_to_anchor=(1.05, 1), loc='upper left')

# Menambahkan anotasi pada setiap stack
for bar in bars.patches:
    width, height = bar.get_width(), bar.get_height()
    x, y = bar.get_xy()
    ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center', fontsize=8, color='white')

"""Berdasarkan tanggungan, pemohon tanpa tanggungan (dependents) lebih banyak dari pada pemohon dengan tanggungan. Pada jumlah tanggungan semua kategori memiliki loan status Yes lebih dari 60%

### Distribusi Education
"""

# Membuat diagram batang untuk visualisasi education terhadap Loan Status
education_loan_status_count = pd.crosstab(df['Education'], df['Loan_Status'])
education_loan_status_count

# Normalisasi data ke dalam presentase
education_loan_status_percentage = education_loan_status_count.div(education_loan_status_count.sum(axis=1), axis=0) * 100
education_loan_status_percentage

# Visualisasi Distribusi Education
education_loan_status_count.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'])
plt.title('Loan Status Distribution by Education')
plt.xlabel('Education')
plt.ylabel('Count')
plt.legend(title='Loan Status', bbox_to_anchor=(1, 1))
plt.show()

# Plotting
fig, ax = plt.subplots()

# Bar plot dengan nilai presentase dan warna kustom
bars = education_loan_status_percentage.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'], ax=ax, width=0.7)

# Menambahkan label
ax.set_ylabel('Percentage')
ax.set_title('Loan Status by education')

# Menambahkan legenda
ax.legend(title='Loan Status', bbox_to_anchor=(1.05, 1), loc='upper left')

# Menambahkan anotasi pada setiap stack
for bar in bars.patches:
    width, height = bar.get_width(), bar.get_height()
    x, y = bar.get_xy()
    ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center', fontsize=8, color='white')

"""Berdasarkan pendidikan, pemohon lulus pendidikan lebih banyak dari pada pemohon yang tidak lulus. Pada masing-masing kategori loan status yes lebih dari 60%

### Distribusi Self Employed
"""

# Membuat diagram batang untuk visualisasi self_employed terhadap Loan Status
self_employed_loan_status_count = pd.crosstab(df['Self_Employed'], df['Loan_Status'])
self_employed_loan_status_count

# Normalisasi data ke dalam presentase
self_employed_loan_status_percentage = self_employed_loan_status_count.div(self_employed_loan_status_count.sum(axis=1), axis=0) * 100
self_employed_loan_status_percentage

# Visualisasi Distribusi Self Employed
self_employed_loan_status_count.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'])
plt.title('Loan Status Distribution by Self Employed')
plt.xlabel('Self Employed')
plt.ylabel('Count')
plt.legend(title='Loan Status', bbox_to_anchor=(1, 1))
plt.show()

# Plotting
fig, ax = plt.subplots()

# Bar plot dengan nilai presentase dan warna kustom
bars = self_employed_loan_status_percentage.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'], ax=ax, width=0.7)

# Menambahkan label
ax.set_ylabel('Percentage')
ax.set_title('Loan Status by Self Employed')

# Menambahkan legenda
ax.legend(title='Loan Status', bbox_to_anchor=(1.05, 1), loc='upper left')

# Menambahkan anotasi pada setiap stack
for bar in bars.patches:
    width, height = bar.get_width(), bar.get_height()
    x, y = bar.get_xy()
    ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center', fontsize=8, color='white')

"""Berdasarkan pekerjaan mandiri (wirausaha), pemohon tidak bekerja secara mandiri lebih banyak dari pada pemohon yang bekerja secara mandiri. Pada setiap kategori presentase status loan yes adalah 68%

### Distribusi Self Credit History
"""

# Membuat diagram batang untuk visualisasi credit_history terhadap Loan Status
credit_history_loan_status_count = pd.crosstab(df['Credit_History'], df['Loan_Status'])
credit_history_loan_status_count

# Normalisasi data ke dalam presentase
credit_history_loan_status_percentage = credit_history_loan_status_count.div(credit_history_loan_status_count.sum(axis=1), axis=0) * 100
credit_history_loan_status_percentage

# Visualisasi Distribusi Credit History
credit_history_loan_status_count.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'])
plt.title('Loan Status Distribution by Credit History')
plt.xlabel('Credit History')
plt.ylabel('Count')
plt.legend(title='Loan Status', bbox_to_anchor=(1, 1))
plt.show()

# Plotting
fig, ax = plt.subplots()

# Bar plot dengan nilai presentase dan warna kustom
bars = credit_history_loan_status_percentage.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'], ax=ax, width=0.7)

# Menambahkan label
ax.set_ylabel('Percentage')
ax.set_title('Loan Status by Credit History')

# Menambahkan legenda
ax.legend(title='Loan Status', bbox_to_anchor=(1.05, 1), loc='upper left')

# Menambahkan anotasi pada setiap stack
for bar in bars.patches:
    width, height = bar.get_width(), bar.get_height()
    x, y = bar.get_xy()
    ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center', fontsize=8, color='white')

"""Berdasarkan credit history, pemohon yang pernah melakukan credit lebih banyak dari pada pemohon yang belum pernah, dan hanya sedikit loan yang diterima pada pemohon dengan belum ada riwayat credit. Pada riwayat 98% dari pemohon yang belum memiliki riwayat kredit ditolak, sedangkan jika sudah memiliki riwayat pemohom ditolak hanya 20% saja.

### Distribusi Self Property_Area
"""

# Membuat diagram batang untuk visualisasi property_area terhadap Loan Status
property_area_loan_status_count = pd.crosstab(df['Property_Area'], df['Loan_Status'])
property_area_loan_status_count

# Normalisasi data ke dalam presentase
property_area_loan_status_percentage = property_area_loan_status_count.div(property_area_loan_status_count.sum(axis=1), axis=0) * 100
property_area_loan_status_percentage

# Visualisasi Distribusi Property Area
property_area_loan_status_count.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'])
plt.title('Loan Status Distribution by Property Area')
plt.xlabel('Property Area')
plt.ylabel('Count')
plt.legend(title='Loan Status', bbox_to_anchor=(1, 1))
plt.show()

# Plotting
fig, ax = plt.subplots()

# Bar plot dengan nilai presentase dan warna kustom
bars = property_area_loan_status_percentage.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'], ax=ax, width=0.7)

# Menambahkan label
ax.set_ylabel('Percentage')
ax.set_title('Loan Status by Property Area')

# Menambahkan legenda
ax.legend(title='Loan Status', bbox_to_anchor=(1.05, 1), loc='upper left')

# Menambahkan anotasi pada setiap stack
for bar in bars.patches:
    width, height = bar.get_width(), bar.get_height()
    x, y = bar.get_xy()
    ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center', fontsize=8, color='white')

"""Berdasarkan area properti, properti pada area semi urban lebih banyak dari pada properti pada area lain. Setiap kategori properti area memiliki 60% lebih data yang loan status = yes"""

df

"""### Distribusi Data Numerik dengan Histogram"""

# Membuat histogram untuk semua kolom numerik
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Membuat grid plot untuk histogram
plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[column].dropna(), bins=20, kde=False, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Import library
import matplotlib.pyplot as plt
import seaborn as sns

# Membuat kategorikal data untuk stacking bar
categorical_data = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']]

# Membuat stacked bar plot
plt.figure(figsize=(15, 10))
for i, column in enumerate(categorical_data.columns, 1):
    plt.subplot(3, 3, i)
    sns.countplot(data=categorical_data, x=column, hue='Loan_Status', palette={'Y': 'lightblue', 'N': 'lightcoral'})
    plt.title(f'{column} vs Loan_Status')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.legend(title='Loan_Status')

plt.tight_layout()
plt.show()

"""# Data Preprocessing

## Data Cleaning

Melakukan data Cleaning terhadap df_train dan df_test
"""

df_prep = df.copy()
df_prep.head(5)

"""### Menghapus kolom yang tidak digunakan

Tahap ini akan dilakukan proses pengecekan dan pembersihan data yang kosong atau tidak terpakai

Menghapus kolom data yang tidak digunakan yaitu Loan ID
"""

df_prep = df_prep.drop(['Loan_ID'], axis=1)
df_prep.head(5)

# cek jumlah data kosong
print('Data null pada data train: ')
df_prep.isnull().sum()

# percobaan
#Mengisi nilai null pada kolom "Gender" dengan modus
df_prep['Gender'] = df_prep['Gender'].fillna(df['Gender'].mode()[0])

# Mengisi nilai null pada kolom "Married" dengan modus
df_prep['Married'] = df_prep['Married'].fillna(df['Married'].mode()[0])

# Mengisi nilai null pada kolom "Dependents" dengan modus
df_prep['Dependents'] = df_prep['Dependents'].fillna(df['Dependents'].mode()[0])

# Mengisi nilai null pada kolom "Self_Employed" dengan modus
df_prep['Self_Employed'] = df_prep['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

# Mengisi nilai null pada kolom "Credit_History" dengan modus
df_prep['Credit_History'] = df_prep['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Mengisi nilai null pada kolom "LoanAmount" dengan median
df_prep['LoanAmount'] = df_prep['LoanAmount'].fillna(df['LoanAmount'].median())

# Mengisi nilai null pada kolom "Loan_Amount_Term" dengan median
df_prep['Loan_Amount_Term'] = df_prep['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())

# cek jumlah data kosong
print('Data null pada data train: ')
df_prep.isnull().sum()

"""## Encode Label"""

df_prep.info()

df_prep.nunique()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_category = df_prep.select_dtypes(include=['object'])
label_mappings = {}  # Membuat kamus kosong untuk menyimpan label mappings

for kolom in df_category.columns:
    le.fit(df_category[kolom])
    df_prep[kolom] = le.transform(df_category[kolom])

    if kolom in df_category:
        print(f"kolom = {kolom}")
        label_mapping = {label: code for label, code in zip(le.classes_, le.transform(le.classes_))}
        label_mappings[kolom] = label_mapping  # Menambahkan label mapping ke kamus
        for label, code in label_mapping.items():
            print(f"label asli: {label} label encode: {code}")
        print()


print(df_prep.head(5))
print('===================')

"""## Feature Selection

Feature selection digunakan untuk meringkas fitur-fitur terpenting untuk mempermudah model dalam mempelajari data.

Feature selection dilakukan dengan algoritma random forest.

Hasil feature selection ini nantinya akan dikomparasi dengan data normal tanpa seleksi fitur.
"""

# drop target
X = df_prep.drop(['Loan_Status'], axis=1)
y = df_prep['Loan_Status']

print(X)
print(y)

# # drop target
# X_test = df_prep_test.drop(['Loan_Status'], axis=1)
# y_test = df_prep_test['Loan_Status']

# print(X_test)
# print(y_test)

from sklearn.ensemble import RandomForestClassifier

# Inisialisasi model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Melatih model pada data
rf_model.fit(X, y)

feature_importances = rf_model.feature_importances_
feature_importances

import matplotlib.pyplot as plt

# Mendapatkan nama fitur
feature_names = X.columns  # Gantilah ini dengan nama kolom sesuai dengan dataset Anda

# Membuat DataFrame untuk memudahkan visualisasi
import pandas as pd
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Mengurutkan DataFrame berdasarkan nilai kepentingan fitur
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

# Membuat horizontal bar chart dengan warna khusus untuk bar ke-8 dan seterusnya
plt.figure(figsize=(10, 6))

# Memberikan alpha global untuk seluruh diagram batang
alpha_value = 0.5

# Mengonfigurasi warna dan alpha untuk setiap bar
colors = ['lightcoral' if imp < 0.04 else 'lightcoral' for imp in feature_importance_df['Importance']]
alpha_values = [alpha_value if imp < 0.04 else 1.0 for imp in feature_importance_df['Importance']]

bars = plt.barh(range(len(feature_importance_df)), feature_importance_df['Importance'], align='center',
               color=colors, alpha=alpha_value)

# Memberikan alpha sesuai dengan kondisi
for i, alpha in enumerate(alpha_values):
    bars[i].set_alpha(alpha)

plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
plt.ylabel('Feature')
plt.xlabel('Importance')
plt.title('Feature Importance from Random Forest')
plt.show()

"""Dari hasil seleksi fitur terpilih 7 fitur terpenting berdasarkan random forest

Mengambil 7 fitur terbaik dengan library SelectFromModel berdasarkan ambang batas yang diketahui diatas yaitu 0.04
"""

from sklearn.feature_selection import SelectFromModel

# Menggunakan ambang batas untuk seleksi fitur
sfm = SelectFromModel(rf_model, threshold=0.04)  # Sesuaikan ambang batas sesuai kebutuhan
sfm.fit(X, y)

# Membuat masker untuk fitur yang dipilih
selected_features_mask = sfm.get_support()

# Mengambil nama fitur yang dipilih dari DataFrame
selected_features = X.columns[selected_features_mask]

# Menampilkan nama fitur yang dipilih
print("Selected Features:")
print(selected_features)

# Membuat DataFrame baru hanya dengan fitur yang dipilih
X_selected= pd.DataFrame(X, columns=selected_features)
# X_selected_test= pd.DataFrame(X_test, columns=selected_features)

# Menampilkan DataFrame baru
print("Selected Features in X_train:")
print(X_selected.shape)

# print("Selected Features in X_test:")
# print(X_selected_test.shape)

"""## Over Sampling SMOTE

Dikarenakan data tidak seimbang maka diperlukan tuning dengan menyeimbangkan data untuk mengetahui dataset dan model terbaiknya
"""

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)


X_smote, y_smote = smote.fit_resample(X, y)
X_selected_smote, y_selected_smote = smote.fit_resample(X_selected, y)

y_smote.value_counts()

# Mengganti nilai 0 menjadi 'tidak' dan nilai 1 menjadi 'iya'
# y_smote_labels = y_smote.replace({0: 'Tidak', 1: 'Iya'})

colors = ['lightblue', 'lightcoral']

# Membuat gambar dengan dua subplot (bar plot dan pie chart)
fig, axs = plt.subplots(1, figsize=(6, 6))

# Subplot 1 - Bar Plot
df['Loan_Status'].value_counts().plot(kind='bar', color=colors)
axs.set_title('Distribusi Data Original')
axs.set_xlabel('Death Event')
axs.set_ylabel('Jumlah')

plt.show()

# Mengganti nilai 0 menjadi 'tidak' dan nilai 1 menjadi 'iya'
y_smote_labels = y_smote.replace({0: 'Y', 1: 'N'})


# Membuat gambar dengan dua subplot (bar plot dan pie chart)
fig, axs = plt.subplots(1, figsize=(6, 6))

# Subplot 1 - Bar Plot
y_smote_labels.value_counts().plot(kind='bar', color=colors)
axs.set_title('Distribusi Data SMOTE')
axs.set_xlabel('Death Event')
axs.set_ylabel('Jumlah')

plt.show()

"""## Split Dataset

Split dataset jadi train dan test dengan skala 80:20. Split dilakukan diawal supaya tahap preprosesing yang dilakukan pure sesuai dari data train dan penyesuaian dilakukan terpisah ke data test.
"""

# Atribut dan target
X = df_prep.drop(['Loan_Status'], axis=1)
y = df_prep['Loan_Status']

print(X)
print(y)

X_selected = X_selected
y_selected = y = df_prep['Loan_Status']

print(X_selected)
print(y_selected)

# test_size = 0.20 artinya data testing 20% dan data training 80%
# random_state = 42 digunakan untuk menspesifikasikan random seed pada saat pembagian data training dan data testing

# split full dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)

# split featur selection dataset
X_selected_train, X_selected_test, y_selected_train, y_selected_test = train_test_split(X_selected, y_selected, test_size = 0.2, random_state = 42, stratify=y)

# Mengetahui dimensi data train dan data test
print('Dimensi feature data train :', X_train.shape)
print('Dimensi target data train :', y_train.shape)
print('Dimensi feature data test :', X_test.shape)
print('Dimensi target data test :', y_test.shape)
print()

print('Dimensi feature data train :', X_selected_train.shape)
print('Dimensi target data train :', y_selected_train.shape)
print('Dimensi feature data test :', X_selected_test.shape)
print('Dimensi target data test :', y_selected_test.shape)

"""#Modeling

## Random Forest

### Pelatihan model
"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def cv_randomforest(X_data, y_data, data_name):
    # Create a Random Forest Classifier
    rf_classifier = RandomForestClassifier()

    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']  # Tambahkan parameter criterion
    }

    # Create the GridSearchCV object
    grid = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Perform the cross-validation
    grid.fit(X_data, y_data)

    # Print the best parameters and corresponding accuracy
    print(f'Data {data_name}')
    print("Best Parameters: ", grid.best_params_)
    print("Best Accuracy: ", grid.best_score_)
    print()

    # Extract the best parameters for each dataset
    best_params = grid.best_params_

    # Create Random Forest Classifier with the best parameters for each dataset
    best_rf_classifier = RandomForestClassifier(**best_params)

    # Fit the models on the training data again (optional, as grid search already fits the best model)
    best_rf_classifier.fit(X_data, y_data)

    return best_rf_classifier

# Deklarasi best model hasil tuning
best_rf_ori = cv_randomforest(X, y, 'Ori')
best_rf_smote = cv_randomforest(X_smote, y_smote, 'Smote')
best_selected_rf_ori = cv_randomforest(X_selected, y, 'Ori_selected')
best_selected_rf_smote = cv_randomforest(X_selected_smote, y_selected_smote, 'Smote_selected')

"""### Evaluasi Model"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

def evaluasi_model(X_data, y_data, model, data_name):
    y_pred = model.predict(X_data)
    y_prob = model.predict_proba(X_data)[:, 1]  # Untuk ROC AUC, kita membutuhkan probabilitas positif

    acc = accuracy_score(y_data, y_pred)
    precision = precision_score(y_data, y_pred)
    recall = recall_score(y_data, y_pred)
    f1 = f1_score(y_data, y_pred)
    roc_auc = roc_auc_score(y_data, y_prob)  # ROC AUC Score

    print(f'{data_name}:')
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("ROC AUC:", roc_auc)
    print(f'Classification Report:\n{classification_report(y_data, y_pred)}')

    print()

    return acc, precision, recall, f1, roc_auc

acc_rf_ori, precision_rf_ori, recall_rf_ori, f1_rf_ori, roc_auc_rf_ori = evaluasi_model(X_test, y_test, best_rf_ori, 'Ori')
acc_rf_smote, precision_rf_smote, recall_rf_smote, f1_rf_smote, roc_auc_rf_smote = evaluasi_model(X_test, y_test, best_rf_smote, 'smote')
acc_selected_rf_ori, precision_selected_rf_ori, recall_selected_rf_ori, f1_selected_rf_ori, roc_auc_selected_rf_ori = evaluasi_model(X_selected_test, y_test, best_selected_rf_ori, 'Ori_selected')
acc_selected_rf_smote, precision_selected_rf_smote, recall_selected_rf_smote, f1_selected_rf_smote, roc_auc_selected_rf_smote = evaluasi_model(X_selected_test, y_test, best_selected_rf_smote, 'smote_selected')

"""# 7. Evaluasi"""

evaluasi = {'model_x_dataset': ['RF Ori', 'RF Smote', 'Selected RF Ori', 'Selected RF Smote'],
            'accuracy': [acc_rf_ori, acc_rf_smote, acc_selected_rf_ori, acc_selected_rf_smote],
            'precision': [precision_rf_ori, precision_rf_smote, precision_selected_rf_ori, precision_selected_rf_smote],
            'recall': [recall_rf_ori, recall_rf_smote, recall_selected_rf_ori, recall_selected_rf_smote],
            'f1-score': [f1_rf_ori, f1_rf_smote, f1_selected_rf_ori, f1_selected_rf_smote]}

import numpy as np

# Membulatkan semua angka dalam matriks evaluasi menjadi 2 digit dibelakang koma
for key in evaluasi:
    if key != 'model_x_dataset':
      evaluasi[key] = np.around(evaluasi[key], decimals=3)

# Menampilkan matriks evaluasi yang telah dibulatkan
print(evaluasi)

df_evaluasi = pd.DataFrame(evaluasi)
df_evaluasi

# Warna yang berbeda tapi tetap satu tone dan lembut
colors = ['lightcoral', 'lightblue', 'lightgreen', 'yellow']

# Batas atas sumbu y
y_upper_limit = 1.2

# Bar chart
barWidth = 0.15
space = 0.02  # Jarak antara grup bar

r1 = np.arange(len(df_evaluasi['model_x_dataset']))
r2 = [x + barWidth + space for x in r1]
r3 = [x + barWidth + space for x in r2]
r4 = [x + barWidth + space for x in r3]

plt.bar(r1, df_evaluasi['accuracy'], color=colors[0], width=barWidth, edgecolor='none', linewidth=0, label='Accuracy')
plt.bar(r2, df_evaluasi['precision'], color=colors[1], width=barWidth, edgecolor='none', linewidth=0, label='Precision')
plt.bar(r3, df_evaluasi['recall'], color=colors[2], width=barWidth, edgecolor='none', linewidth=0, label='Recall')
plt.bar(r4, df_evaluasi['f1-score'], color=colors[3], width=barWidth, edgecolor='none', linewidth=0, label='F1-Score')

# Menambahkan nilai di atas setiap bar dengan rotasi 90 derajat ke kiri
for i, val in enumerate(df_evaluasi['accuracy']):
    plt.text(r1[i], min(val + 0.01, y_upper_limit), round(val, 3), rotation=90, ha='center', va='bottom')

for i, val in enumerate(df_evaluasi['precision']):
    plt.text(r2[i], min(val + 0.01, y_upper_limit), round(val, 3), rotation=90, ha='center', va='bottom')

for i, val in enumerate(df_evaluasi['recall']):
    plt.text(r3[i], min(val + 0.01, y_upper_limit), round(val, 3), rotation=90, ha='center', va='bottom')

for i, val in enumerate(df_evaluasi['f1-score']):
    plt.text(r4[i], min(val + 0.01, y_upper_limit), round(val, 3), rotation=90, ha='center', va='bottom')

# Label dan legend horizontal di atas bar
plt.xlabel('')
plt.xticks([r + (barWidth*4/2) - space for r in range(len(df_evaluasi['model_x_dataset']))], df_evaluasi['model_x_dataset'])
plt.yticks([])  # Sembunyikan ticks pada sumbu y
plt.ylim(0, y_upper_limit)  # Set batas atas sumbu y
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)  # Atur legenda di atas bar
plt.title('Performance Metrics for Different Models and Datasets')

# Tampilkan plot
plt.show()

"""Dari optimasi model Random Forest didapatkan model dan data set terbaik adalah selected RF Smote yang mendapatkan akurasi 100% dari data tes."""

