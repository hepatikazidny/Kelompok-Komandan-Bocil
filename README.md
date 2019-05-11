# Kelompok-Komandan-Bocil

# Deskripsi Kelompok
Nama Kelompok : Komandan Bocil
Title : Cat Classification

Dataset diambil pada situs : Visual Geometry Group, kemudian dilakukan pemilihan data untuk dijadikan dataset yang akan digunakan dalam proses classification nanti

# Progress 1
melakukan preprocessing dan feature extraction pada data gambar kucing 

# Progress 2
melakukan segmentasi menggunakan 2D Discrete Wavelet Transform (DWT) dan melakukan eksperimen sementara, yaitu melakukan klasifikasi sementara dengan hasil global feature extraction yang telah dilakukan pada progress 1. Dengan hasil global feature extraction tersebut, dilakukan testing kepada beberapa model machine learning yang ada, antara lain adalah Logistic Regression, LDA, KNN, Decision Tree, Gaussian Naive Bayes, Random Forest, dan SVM. sehingga progress 2 ini kami mengupload 2 file yaitu Progress2_Kelompok_Komandan_Bocil yang isinya adalah segmentasi dengan 2D Discrete Wavelet Transform (DWT) dan file Test_Global_Feature_Extraction yang isinya adalah melakukan klasifikasi sementara dengan hasil global feature extraction yang telah dilakukan pada progress 1 dan testing menggunakan beberapa model machine learning yang telah disebutkan sebelumnya.

#Final Report
Berdasarkan hasil eksperimen progress 1, telah dilakukan proses preprocessing dan ekstraksi fitur citra dengan menggunakan global feature extraction. Penggunaan metode global feature extraction dikarenakan beberapa jenis ras kucing memiliki tekstur, warna, dan bentuk yang serupa. Dalam progress 1, ukuran dataset citra kucing yang awalnya berbeda-beda telah difiksasi menjadi ukuran yang sama. Kemudian setelah dilakukan Global Feature Extraction, terlihat gambar menunjukkan adanya pemisahan objek dengan background sehingga akan memudahkan proses segmentasi yang akan kami lakukan.

Setelah berhasil melakukan ekstraksi fitur pada seluruh dataset, bagian selanjutnya yang akan dilakukan adalah segmentasi citra dengan menggunakan Watershed Algorithm Segmentation dan mencoba melakukan klasifikasi citra. Metode segmentasi yang digunakan berubah dari proposal yang telah diajukan sebelumnya, yaitu Discrete Wavelet Transform (DWT). Hal ini dikarenakan ada kendala pada penyesuaian dimensi matrix DWT untuk dilanjutkan ke proses klasifikasi. Untuk mengantisipasi hal tersebut kami mengubah metode segmentasi ke Watershed Algorithm Segmentation. Metode ini mengisi setiap isolated valley (local minima) dengan beberapa warna air (water) berbeda. Dengan adanya perbedaan water pada valleys yang berbeda, dengan perbedaan warna akan memulai mensegmentasi. Pada eksperimen kali ini, kami telah berhasil melakukan segmentasi menggunakan Watershed Algorithm Segmentation pada dataset hasil ekstraksi fitur dengan hasil berikut ini. Awalnya Watershed Algorithm Segmentation akan mensegmentasi secara keseluruhan pada citra untuk memisahkan objek dengan background.

Selanjutnya dilakukan testing kepada beberapa model machine learning yang ada, antara lain adalah Logistic Regression, LDA, KNN, Decision Tree, Gaussian Naive Bayes, Random Forest, dan SVM.


# Referensi
[1] The Random Forest Algorithm : Niklas Donges, 2018. https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd , diakses 7  Maret 2019.

[2] Dataset (Training dan Test), 2012. http://www.robots.ox.ac.uk/~vgg/data/pets/ , diakses 5 Maret 2019.

[3] Image Classification using Python and Scikit-Learn, 2018. https://gogul09.github.io/software/image-classification-python, diakses 18 Maret 2019.
[4] Dataset (Training dan Test), 2012. http://www.robots.ox.ac.uk/~vgg/data/pets/ , diakses 5 Maret 2019.

