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

  ![image](https://user-images.githubusercontent.com/68459186/138907632-773f173c-88be-4403-a471-71cf90223ca2.png)

- high,

  ![image](https://user-images.githubusercontent.com/68459186/138907675-d7e6b0de-6f7f-45e1-97f7-96b9ac3b45ac.png)

- low,
  
  ![image](https://user-images.githubusercontent.com/68459186/138907719-34d2056c-d90f-42af-a2c1-ce7cc44eaad7.png)
 
- close, 
  
  ![image](https://user-images.githubusercontent.com/68459186/138907809-a54c85ea-2e1f-4883-a758-aa59506a881a.png)

- adj close,
  
  ![image](https://user-images.githubusercontent.com/68459186/138907842-af659089-fe8b-4d1c-a891-d8fd757bc658.png)
 
- adj close_returns,
  
  ![image](https://user-images.githubusercontent.com/68459186/138907883-d0914879-3bce-49a1-941a-145987d09aaf.png)
 
- rsi_adj close, 
  
  ![image](https://user-images.githubusercontent.com/68459186/138907915-8377cbd6-47ea-431e-86f9-112bc3ebe65a.png)

- upper_band_adj close, 
  
  ![image](https://user-images.githubusercontent.com/68459186/138907942-f314551b-9f15-4fb1-b024-51b9cd6fcc36.png)

- lower_band_adj close,
  
  ![image](https://user-images.githubusercontent.com/68459186/138907967-e0e196cc-d27e-4c48-8641-6ba609a8fdef.png)
 
- dif_adj close, dan 
 
  ![image](https://user-images.githubusercontent.com/68459186/138907987-d283a18c-c504-47dc-9e28-19ed31dfec1a.png)

- macd_adj close.

![image](https://user-images.githubusercontent.com/68459186/138908914-a5b878b8-d44d-4fe5-9016-8d634d351597.png)


## Data Preparation
- menerapkan minimal satu (baiknya 2 atau lebih) teknik preparation, menjelaskan mengapa diperlukan tahapan tersebut, 

## Modeling
- membuat model dan menjelaskan proses pemodelan

## Evaluation
- menyebutkan metrik evaluasi yang digunakan, menjelaskan hasil royek berdasarkan metrik, menyajikan model terbaik sebagai solusi

## Referensi
- Fernando, Jason. (2021). _Moving Average Convergence Divergence (MACD)_. Diakses pada 26 Oktober 2021, dari https://www.investopedia.com/terms/m/macd.asp
-  Fernando, Jason. (2021). _Relative Strength Index (RSI)_. Diakses pada 26 Oktober 2021, dari https://www.investopedia.com/terms/r/rsi.asp
-  Hayes, Adam. (2021). _Simple Moving Average (SMA)_. Diakses pada 26 Oktober 2021, dari https://www.investopedia.com/terms/s/sma.asp
- 
