# Laporan Proyek Pertama Modul Machine Learning Terapan

## Domain Proyek
Domain proyek yang dipilih untuk proyek pertama machine learning terapan adalah mengenai ekononi dengan membahas tentang harga emas yang dilakukan adalah memprediksi harga emas.

- Latar Belakang

Dalam catatan sejarah, emas telah digunakan sebagai mata uang di berbagai belahan dunia. Saat ini, logam mulia seperti emas dipegang oleh bank sentral di semua negara untuk menjamin pembayaran kembali utang luar negeri, dan juga untuk mengendalikan inflasi yaang meencerminkan kekuatan keuanga negara. Baru-baru ini, negara berkembang seperti Cina, Rusia, dan India menjadi pembeli emas yang besar, sedangkan Amerika Serikat, Afrika Selatan, dan Australia termasuk diantara penjual emas yang besar.

Memprediksi kenaikan dan penurunan harga emas harian dapat membantu investor memutuskan kapan harus membeli atau menjual komoditas terseut. tetapi harga emas bergantung pada banyak faktor seperti harga logam mulia lainnya seperti harga minyak mentah, kinerja bursa saham, harga obligasi, nilai tukar mata uang, dan sebagainya. 

Tantangan pada proyek ini adalah untuk secara akurat memprediksi haarga penutupan ETF emas yang disesuaikan di masa depan selama periode waktu tertentu di masa mendatang. Masalahnya adalah masalah regresi, karena nilai output yang merupakan harga penutupan yang disesuaikan dalam proyek ini adalah nilai kontinu.

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68459186/138650396-7b5242eb-4287-4b70-af33-ab6465db02e1.png">
</p>

- Uraian singkat informasi mengenai domain, jelaskan mengapa dan bagaimana masalah harus diselesaikan, menyertakan hasil riset terkait

## Business Understanding
- menjelaskan proses klarifikasi masalah dan manjagujakan minimal satu solusi untuk menyelesaikan permasalahan

### Problem (pernyataan masalah)

Dari latar belakang masalah di atas, berikut merupakan rumusan masalah yang didapatkan :
- Dari banyaknya fitur yang ada fitur manakah yang memiliki hubungan atau pengaruh terhadap data harga penjualan emas?
- Bagaimana cara pemrosesan data yang dapat dilakukan pada data harga penjualan emas?
- Bagaimana cara membuat model machine learning yang cocok untuk data harga penjualan emas?

### Goals (tujuan)
Berikut ini adalah tujuan yang akan dicapai :
- Memilih fitur-fitur yang memiliki hubungan atau pengaruh terhadap data harga penjualan emas
- Melakukan pemrosesan terhadap data harga penjualan eams
- Membuat model machine learning terbaik untuk memprediksi harga penjualan emas 

### Solution Statements (pernyataan solusi-dua atau lebih)
Berikut ini adalah solusi yang mungkin dapat dilakukan :
- Melihat persebaran data pada data penjualan emas dan memilih variabel utama yang berhubungan langsung dengan harga penjualan emas.
- Pemrosesan terhadap data penjualan emas yang dapat dilakukan antara lain, melihat apakah ada data yang hilang/kosong, memvisualisasikan data, melakukan beberapa perhitungan (MACD, RSI, SMA, dan Bollinger Bands), normalisasi, encoding fitur dan membagi data menjadi data latih dan data uji. 
- Membuat beberapa algoritma model seperti Decision Tree, Support Vector Regressor, Random Forest dan sebagainya, serta menerapkan hyperparamater tuning pada beberap model.

## Data Understanding
- memberikan informasi mengenai data yang digunakan dan sumber data, menguraikan seluruh variabel atau fitur pada data, visualisasi data

## Data Preparation
- menerapkan minimal satu (baiknya 2 atau lebih) teknik preparation, menjelaskan mengapa diperlukan tahapan tersebut, 

## Modeling
- membuat model dan menjelaskan proses pemodelan

## Evaluation
- menyebutkan metrik evaluasi yang digunakan, menjelaskan hasil royek berdasarkan metrik, menyajikan model terbaik sebagai solusi

## Referensi
