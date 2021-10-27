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

## Business Understanding

### Problem (pernyataan masalah)

Dari latar belakang masalah di atas, berikut merupakan rumusan masalah yang didapatkan :
- Dari banyaknya fitur yang ada fitur manakah yang memiliki hubungan atau pengaruh terhadap data harga penjualan emas?
- Bagaimana cara pemrosesan data yang dapat dilakukan pada data harga penjualan emas?
- Bagaimana cara membuat model machine learning yang cocok untuk data harga penjualan emas?

### Goals
Berikut ini adalah tujuan yang akan dicapai :
- Memilih fitur-fitur yang memiliki hubungan atau pengaruh terhadap data harga penjualan emas
- Melakukan pemrosesan terhadap data harga penjualan eams
- Membuat model machine learning terbaik untuk memprediksi harga penjualan emas 

### Solution Statements
Berikut ini adalah solusi yang mungkin dapat dilakukan :
- Melihat persebaran data pada data penjualan emas dan memilih variabel utama yang berhubungan langsung dengan harga penjualan emas.
- Pemrosesan terhadap data penjualan emas yang dapat dilakukan antara lain, melihat apakah ada data yang hilang/kosong, memvisualisasikan data, melakukan beberapa perhitungan (MACD, RSI, SMA, dan Bollinger Bands), normalisasi, encoding fitur dan membagi data menjadi data latih dan data uji. 
- Membuat beberapa algoritma model seperti Decision Tree, Support Vector Regressor, Random Forest dan sebagainya, serta menerapkan hyperparamater tuning pada beberap model.

## Data Understanding

![image](https://user-images.githubusercontent.com/68459186/138727998-25fa3504-e84f-473c-af31-68eeb6de0a34.png)

Informasi dataset :

| Hal                     | Keterangan                                                                              |
| ----------------------- | --------------------------------------------------------------------------------------- |
| Sumber                  | [Kaggle Dataset : Gold Price Prediction Dataset](https://www.kaggle.com/sid321axn/gold-price-prediction-dataset) |
| Lisensi                 | CC0: Public Domain                                                                      |
| Kategori                | Finace, Tabulat data, Beginner, Economics, Regression                                   |
| Rating Penggunaan       | 9.4                                                                                     |
| Jenis dan Ukuran Berkas | CSV (1.04 MB)                                                                           |

Atribut pada dataset :
- Fitur
  - Gold ETF :- Date, Open, High, Low, Close and Volume.
  - S&P 500 Index :- 'SP_open', 'SP_high', 'SP_low', 'SP_close', 'SP_Ajclose', 'SP_volume'
  - Dow Jones Index :- 'DJ_open','DJ_high', 'DJ_low', 'DJ_close', 'DJ_Ajclose', 'DJ_volume'
  - Eldorado Gold Corporation (EGO) :- 'EG_open', 'EG_high', 'EG_low', 'EG_close', 'EG_Ajclose', 'EG_volume'
  - EURO - USD Exchange Rate :- 'EU_Price','EU_open', 'EU_high', 'EU_low', 'EU_Trend'
  - Brent Crude Oil Futures :- 'OF_Price', 'OF_Open', 'OF_High', 'OF_Low', 'OF_Volume', 'OF_Trend'
  - Crude Oil WTI USD :- 'OS_Price', 'OS_Open', 'OS_High', 'OS_Low', 'OS_Trend'
  - Silver Futures :- 'SF_Price', 'SF_Open', 'SF_High', 'SF_Low', 'SF_Volume', 'SF_Trend'
  - US Bond Rate (10 years) :- 'USB_Price', 'USB_Open', 'USB_High','USB_Low', 'USB_Trend'
  - Platinum Price :- 'PLT_Price', 'PLT_Open', 'PLT_High', 'PLT_Low','PLT_Trend'
  - Palladium Price :- 'PLD_Price', 'PLD_Open', 'PLD_High', 'PLD_Low','PLD_Trend'
  - Rhodium Prices :- 'RHO_PRICE'
  - US Dollar Index : 'USDI_Price', 'USDI_Open', 'USDI_High','USDI_Low', 'USDI_Volume', 'USDI_Trend'
  - Gold Miners ETF :- 'GDX_Open', 'GDX_High', 'GDX_Low', 'GDX_Close', 'GDX_Adj Close', 'GDX_Volume'
  - Oil ETF USO :- 'USO_Open','USO_High', 'USO_Low', 'USO_Close', 'USO_Adj Close', 'USO_Volume'
- Variabel target
  - Gold ETF :- Adjusted Close  

Atribut pada dataset yang dipilih untuk digunakan untuk mengembangkan model :
- Fitur
  - Open, High, Low, Close
- Variabel target
  - Adj Close

Kemudian dilakukan perhitungan pada data variabel target antara lain :
- MACD (Moving Average Convergence Divergence), sebuah indikator dalam analisis teknikal yang menggambarkan hubungan antara dua moving average dalam sebuah tren harga aset. Adapum, moving average merupakan rerata harga, baik pembukaan atau penutupan perdagangan setiap harinya yang digambarkan dalam sebuah garis tren. Kegunaan untuk memahami kapan harga aset tersebut akan bersifat bullish atau bearish. Pada dasarnya, MACD menghitung Exponential Moving Average (EMA) selama 12 hari dan 26 hari terakhir. EMA adalah jenis moving average yang menitikberatkan pada bobot dan signifikansi dari data yang paling baru. Rumus MACD sebagai berikut.

  ![MACD](https://user-images.githubusercontent.com/68459186/138883468-f5dd57fb-f173-4bb3-89d5-55c1af4242d7.png)

  Dengan demikian, MACD akan bernilai positif jika EMa 12 hari lebih besar dari EMA 26 hari dan berlaku sebaliknya. 

- RSI (Relative Strength Index), indikator yang digunakan dalam mengukur besarnya volatilitas harga sebuah aset. Indikator ini dilakukan untuk mengevaluasi apakah aset tersebut terbilang dalam posisi jenuh beli (overbounght) atau jenuh jual (oversold). RSI ditampilkan sebagai osilator (grafik garis yang bergerak antara dua titik ekstrem) dengan nilai berada di antara 0 hingga 100. Rumus RSI sebagai berikut.

  ![RSI](https://user-images.githubusercontent.com/68459186/138889935-e40b7183-a5bc-411a-a4f9-7b9a0e8a9ab4.png)

  Rata-rata keuntungan atau kerugian yang digunakan dalam perhitungan adalah presentase keuntungan atau kerugian rata-rata selama periode kilas balik (dua titik yang dipilih untuk dibandingkan, bisa selama 7 hari, bisa selama 14 hari, dst).

- SMA (Simple Moving Average), bentuk simpel dari Moving Average. Moving Average untuk memberi petunjuk mengenai arah tren harga sebuah aset di masa depan. Pada Simpel Moving Average indikator dihitung dengan menggunakan rerata aritmatika dari salah satu set nilai tertentu, biasanya harga penutupan dengan jumlah periode dalam kisaran itu. Dengan kata lain, serangkaian data aset digabungkan dulu bersama-sama untuk kemudian dibagi menjadi harga aset di set tertentu tersebut. Rumus SMA sebagai berikut.
  ![SMA](https://user-images.githubusercontent.com/68459186/138896477-358fc5e2-a4af-49e7-a493-95a3da06d5a8.png)

- Bolliger Band, alat analisis teknis yang dikembangkan oleh John Bollinger untuk menghasilkan sinyak oversold atau overbought. Ada tiga baris yang membentuk Bollinger Bands, SMA (middle band), upper band, dan lower band. Upper dan lower band biasanya 2 standar deviasi +/- dari rata-rata bergerak sederhana selama 20 hari, tetapi dapat dimodifikasi. Bollinger band dimanfaatkan untuk menganalisis pergerakan harga sebuah aset atau komoditas tertentu. Rumus bollinger band sebagai berikut.
  ![image](https://user-images.githubusercontent.com/68459186/138906534-cc3772a6-99e3-4ada-87a1-17bb252b8a49.png)

Sehingga fitur yang digunakan bertambah dengan adanya hasil perhitungan yang dilakukan, maka variabel yang digunakan antara lain :
- open, high, low, close, adj close, adj close_returns, rsi_adj close, upper_band_adj close, lower_band_adj close, dif_adj close, dan macd_adj close.

Berikut ini merupakan visualisasi dari data fitur yang digunakan :
- open,

  ![image](https://user-images.githubusercontent.com/68459186/138909541-8a92fc56-f127-4c09-a743-98472eae798f.png) 

- high,

  ![image](https://user-images.githubusercontent.com/68459186/138909692-d4b5b164-bb33-4bfe-8913-f91098087d49.png)

- low,
  
  ![image](https://user-images.githubusercontent.com/68459186/138909743-38b3213d-64de-430d-bf03-422ad2203469.png)
 
- close, 
  
  ![image](https://user-images.githubusercontent.com/68459186/138909825-10187dcf-41ae-441a-9788-aa8e770ab756.png)

- adj close,
  
  ![image](https://user-images.githubusercontent.com/68459186/138909863-d5c03339-2f50-455a-a004-2ca83e36bd55.png)
 
- adj close_returns,
  
  ![image](https://user-images.githubusercontent.com/68459186/138909893-7667571c-4cfd-44e4-b498-ace379546a0b.png)
 
- rsi_adj close, 
  
  ![image](https://user-images.githubusercontent.com/68459186/138909956-0e378d72-54f7-41eb-a9e4-2f4190869321.png)

- upper_band_adj close, 
  
  ![image](https://user-images.githubusercontent.com/68459186/138910004-9c9b1c2a-c3f4-4a97-9feb-97960756a659.png)

- lower_band_adj close,
  
  ![image](https://user-images.githubusercontent.com/68459186/138910050-a61d4011-81a6-4876-89bf-89de292268ed.png)
 
- dif_adj close, dan 
 
  ![image](https://user-images.githubusercontent.com/68459186/138910082-0e059120-7d5f-4a15-9a00-3efaa1b7edf8.png)

- macd_adj close.

  ![image](https://user-images.githubusercontent.com/68459186/138908914-a5b878b8-d44d-4fe5-9016-8d634d351597.png)

## Data Preparation
- menerapkan minimal satu (baiknya 2 atau lebih) teknik preparation, menjelaskan mengapa diperlukan tahapan tersebut, 
Teknik preparation yang digunakan pada proyek ini antara lain :
- Menghilangkan data yang bernilai 0 atau kosong
  
  ![image](https://user-images.githubusercontent.com/68459186/139034942-5b1ea3b5-439d-4e8a-a4a3-889ef4df4488.png)
  
  Bisa dilihat pada gambar diatas menunjukan jumlah nilai yang kosong atau NaN yang terdapat pada data dikarenakan jumlahnya tidak terlalu banyak sehingga diputuskan untuk menghapusnya. Selain itu, karena jumlah yang tidak terlalu banyak sehingga tidak terlalu mempengaruhi fitur atau hilangnya informasi yang dibutuhkan. 
  
- Normalisasi
  Normalisasi dilakukan dengan tujuan untuk mengubah nilai kolom numerik dalam data ke skala yang sama, tanpa menggangu perbedaan dalam rentang nilai. Normalisasi dilakukan pada fitur-fitur yang akan digunakan. Proses normalisasi dilakukan menggunakan fungsi MinMaxScaler dari sklearn.

- Train-Test-Split 
  Dilakukan pembagian dataset menjadi 3 bagian, yaitu data latih, data validasi, dan data uji. Pertama, dilakukan pengambilan data untuk validasi sebanyak 89 data terakhir pada setiap variabel. Kemudian sisanya dibagi menjadi data latih dan data uji dengan perbandingan 80:20. Data latih digunakan untuk proses pelatihan model dengan data sebanyak 80% dari dataset yang sudah dikurangi, sedangkan data uji digunakan untuk menguji model yang sudah dilatih, serta data validasi digunakan untuk mengecek akurasi dari model yang sudah dilatih sebelum digunakan pada data uji. Pembagian dataset dilakukan menggunakan fungsi train_test_split dari sklearn. 

## Modeling

Pada pemodelan menggunakan beberapa algoritma sebagai berikut :
- Decision Tree Regression, salah satu metode klasifikasi populer, karena mudah diinterpretasi oleh manusia. Decision tree adalah model prediksi menggunakan struktur pohon atau struktur berhirarki. Konsepnya adalah mengubah data menjadi aturan-aturan keputusan. Manfaat untuk mem-break down proses pengambilan keputusan yang kompleks menjadi lebih simpel, sehingga pengambilan keputusan akan lebih menginterpretasikan solusi dari permasalahan. Selain itu, decision tree juga berguna untuk mengekplorasi data, menemukan hubungan tersembunyi antara sejumlah calon variabel input dengan sebuah variabel target. Kelebihan, mampu mengeliminasi perhitungan atau data-data yang sekiranya tidak diperlukan.  
- Support Vector Regressor, Support Vector Machine juga dapat digunakan sebagai metode regresi, mempertahankan semua fitur utama yang menjadi ciri algoritma (margin maksimal). Support Vector Regression (SVR) menggunakan prinsip yang sama dengan SVM untuk klasifikasi, dengan hanya beberapa perbedaan kecil. Pertama-tama, karena keluaran adalah bilangan real, menjadi sangat sulit untuk memprediksi informasi yang ada, yang memiliki kemungkinan tak terbatas. Dalam kasus regresi, margin toleransi (epsilon) diatur dalam perkiraan ke SVM yang akan sudah diminta dari masalah. Tapi selain fakta ini, ada juga alasan yang lebih rumit, algoritmanya lebih rumit sehingga harus dipertimbangkan. Namun, ide utamanya selalu sama: untuk meminimalkan kesalahan, individualisasi hyperplane yang memaksimalkan margin, dengan mengingat bahwa bagian dari kesalahan ditoleransi.
- Random Forest, salah satu algoritma supervised learning yang digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest termasuk ke dalam kelompok model ensemble (model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama). Sehingga, tingkat keberhasilan akan lebih tinggi dibandingkan model yang bekerja sendirian. Random forest pada dasarnya adalah versi bagging dari algoritma decision tree. Algoritma ini disebut sebagai random forest karena disusun dari banyak algoritma pohon (decision tree) yang pembagian data dan fiturnya dipilih secara acak. 
  ![image](https://user-images.githubusercontent.com/68459186/139058578-936b1b4d-2c83-406d-8cd1-397b871fd6cc.png)
 
- LassoCV dan RidgeCV, Pada regresi ridge ada 3 konsep, yaitu regularization, L1 Loss function atau L1 Regularization, dan L2 Loss function atau L2 Regularization. Regularization digunakan untuk menyelesaikan masalah performa model yang tidak sinkron. Maksudnya, suatu model memiliki performa yang baik untuk data latih tetapi memiliki performa yang buruk untuk data uji. regularization menyelesaikan masalah ini dengan menambahkan penalti ke fungsi tujuan dan mengontrol kompleksitas model dengan penalti tersebut. Regularization biasanya digunakan untuk situasi variabel berjumlah besar, rasio jumlah observasi dan jumlah variabel kecil, dan adanya multikolinieritas. Maksud dari istilah L1 Loss function atau L1 Regularization adalah meminimalkan fungsi tujuan dengan menambahkan penalti ke jumlah nilai absolut koefisien atau biasa dikenal dengan metode deviasi absolut terkecil sedangkan maksud dari L2 Loss function atau L2 Regularization adalah meminimalkan fungsi tujuan dengan menambahkan penalti ke jumlah kuadrat koefisien.
  Kata Lasso pada regresi lasso merupakan singkatan dari Least Absolute Shrinkage and Selection Operator. Metode ini menggunakan teknik L1 Regularization dalam fungsi tujuan. Keuntungan regresi lasso dibandingkan regresi ridge adalah regresi lasso dapat memilih variabel bawaan serta penyusutan parameter. Persamaan regresi ridge dan laso adalah sama-sama digunakan untuk menangani multikolinieritas. Regresi ridge secara komputasi lebih efisien jika dibandingkan regresi lasso. 
- Gradient Boosting Regressor, Termasuk dalam algoritma ensemble yang menggunakan pningkatan akurasi prediktor. Gradient boost membangun tree dengan 8 sampai 32 daun, menggunakan boosting untuk proses pengoptimalan dengan menggunakan loss function untuk meminimalisir kesalahan, dan cara kerja algoritma gradient boost adalah membangun satu tree untuk menyesuaikan data, lalu tree berikutnya dibangun untuk mengurangi residual (error).
- Stochastic Gradient Descent, Pendekatan sederhana namun sangat efisien untuk menyesuaikan pengklasifikasi linier dan regressor di bawah kerugian cembung seperti (linier) Support Vector Machines dan Logistic Regression. kelebihan, efisien dan kemudahan implementasi. kekurangan, SGD membutuhkan sejumlah hyperparameter seperti parameter regularisasi dan jumlah iterasi dan sensitif terhadap penskalaan fitur.  

## Evaluation

Matrik evalusi yang digunakan :
- RMSE, Metrik kesalahan akar rata-rata kuadrat adalah ukuran yang sering digunakan untuk perbedaan antara nilai yang diprediksi oleh model atau penaksir dan nilai yang diamati. Metrik ini berkisaran dari nol hingga tak terhingga; nilai yang lebih rendah menunjukkan kualitas model yang lebih tinggi.
- R2 Score, Kuadrat dari koefisien korelasi Pearson antara label dan nilai prediksi. Metrik ini berkisaran antara nol dan satu; nilai yang lebih tinggi menunjukkan kualitas model yang lebih tinggi. 

Berikut ini merupakan hasil dari proses pemodelan yang telah dilakukan menggunakan beberapa algoritma, model dilatih menggunakan data latih dan dicoba memprediksi pada data validasi yang dapat dilihat pada gambar visualisasi berikut ini.
  
  ![image](https://user-images.githubusercontent.com/68459186/139044524-76a204b6-f169-4a03-a9d3-f67edfe6a153.png)

Berikut ini grafik yang digunakan untuk membandingkan nilai RMSE dari setiap model yang telah dibuat.

![image](https://user-images.githubusercontent.com/68459186/139089280-5cf54886-3f46-4b24-87f4-eb132104fea9.png)

Dari data diatas sehingga dibuat sebuah tabel untuk mengurutkan algoritma dengan performa tinggi ke rendah, sebagai berikut ini. 

![image](https://user-images.githubusercontent.com/68459186/139089080-cd43ad37-46cc-49d7-8f36-8a5efbb84fc7.png)

Dari hasil yang didapatkan dari pembuatan beberapa model diatas maka didapatkan nilai RMSE dan R2 Score seperti diatas. Sehingga dapat digunakan model dengan nilai RMSE terendah dan R2 Score yang tinggi yang memiliki performa yang baik untuk digunakan sebagai model prediksi harga emas. Model yang mungkin cocok untuk digunakan adalah LassoCV, SVR yang diberikan Hyperparameter Tunning, dan RidgeCV. 


## Referensi
- Fernando, Jason. (2021). _Moving Average Convergence Divergence (MACD)_. Diakses pada 26 Oktober 2021, dari https://www.investopedia.com/terms/m/macd.asp
-  Fernando, Jason. (2021). _Relative Strength Index (RSI)_. Diakses pada 26 Oktober 2021, dari https://www.investopedia.com/terms/r/rsi.asp
-  Hayes, Adam. (2021). _Simple Moving Average (SMA)_. Diakses pada 26 Oktober 2021, dari https://www.investopedia.com/terms/s/sma.asp
- Hayes, Adam. (2021). _Bollinger Band Definition_. Diakses pada 26 Oktober 2021, dari https://www.investopedia.com/terms/b/bollingerbands.asp
- Anonim. (2021). _Jenis-Jenis Metode Regresi dalam Algoritma Supervised Learning_. Diakses pada 27 Oktober 2021, dari https://www.dqlab.id/jenis-metode-regresi-algoritma-supervised-learning
- IYKRA. (2018). _Mengenal Decision Tree dan Manfaatnya_. Diakses pada 27 Oktober 2021, dari https://medium.com/iykra/mengenal-decision-tree-dan-manfaatnya-b98cf3cf6a8d
- Anonim. (). _Support Vector Machine-Regression (SVR)_. Diakses pada 27 Oktober 2021, dari https://www.saedsayad.com/support_vector_machine_reg.htm
- Anonim. (). _1.5. Stochastic Gradient Descent_. Diakses pada 27 Oktober 2021, dari https://scikit-learn.org/stable/modules/sgd.html
- Anonim. (2021). _Algoritma Machine Learning yang Harus Kamu Pelajari di Tahun 2021_. Diakses pada 27 Oktober 2021, dari https://www.dqlab.id/algoritma-machine-learning-yang-perlu-dipelajari
- Anonim. (2021). _AutoML Tables Documentation : Guide_. Diakses pada 27 Oktober 2021, dari https://cloud.google.com/automl-tables/docs/evaluate?hl=en
- Majumder, Sohom. (2021). _Gold price prediction-Time Series split & LGBM_. Diakses pada 18 Oktober 2021, dari https://www.kaggle.com/sohommajumder21/gold-price-prediction-time-series-split-lgbm
- Siddhartha, Manu. (2021). _Gold Price Prediction Using Machine Learning_. Diakses pada 17 Oktober 2021, dari https://www.kaggle.com/sid321axn/gold-price-prediction-using-machine-learning
- Dranitsyna, Ekaterina. (2021). _gold_price_model_. Diakses pada 18 Oktober 2021, dari https://www.kaggle.com/ekaterinadranitsyna/gold-price-model
-  Venable, Hayden. (2021). _GOld Price Predicion with PCA and Regression_. Diakses pada 17 Oktober 2021, dari https://www.kaggle.com/haydenvenable/gold-price-prediction-with-pca-and-regression
