# Hello World

DTS PRO-A Machine Learning
Membuat model untuk handwriting recognition

1. Create repo: https://github.com/kurniawan2805/handwriting-recognition

Proposal 16 Jul

### Explorasi dataset:
<ul>
  <li> cek panjang karakter, max di 34 (df_train) </li>
  <li> untuk label = 'UNREADABLE' dihapus saja, kriteria kurang jelas </li>
  <li> mayoritas label dengan huruf kapital, tapi ada label dengan huruf kecil, dicek gambar visual huruf besar, hanya ada 1 karakter 'i' yang ditulis kecil dengan titik di atas => di cast ke kapital semua label  </li>
</ul>


  
  

### Metode Solve:



CNN + RNN + CTC

CNN = untuk deteksi fitur dari gambar, line segmentation, 

Input gambar grayscale,
Untuk ekstraksi fitur, namun output berupa multiple character
kelas CNN berupa char yg digunakan

RNN untuk menentukan sequence char sehingga menjadi sebuah kata/kalimnat

CTC mencari jalur terbaik, karena dalam extraksi karakter dan sequence dimungkinkan ada repetisi char yang tidak diinginkan
