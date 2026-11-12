# Aplikasi identifikasi penulis tulisan tangan

Aplikasi web Flask yang menebak penulis dari satu baris tulisan tangan, memakai
ResNet18 hasil UAS (50 penulis, akurasi uji 0,9808).

Kamera dipegang **browser** lewat `getUserMedia`, bukan server. Server hanya
menerima satu frame JPEG per permintaan. Preview video jalan sepenuhnya di sisi
klien, jadi tidak ada beban inferensi per frame.

## Pasang

```
pip install -r requirements.txt
```

Lalu salin bobot model dari Drive (`MyDrive/UAS_ML/hasil/`) ke folder `model/`:

```
model/resnet18_fold1.pt
model/resnet18_fold2.pt      (opsional)
model/resnet18_fold3.pt      (opsional)
model/resnet18_fold4.pt      (opsional)
model/resnet18_fold5.pt      (opsional)
```

Satu berkas sudah cukup. Kalau kelima ada, aplikasi memakai ensemble: softmax
dirata-ratakan dan jumlah model yang sepakat ditampilkan sebagai sinyal
keyakinan. Tanpa bobot, halaman tetap terbuka tetapi menampilkan peringatan.

Bobot harus dari notebook yang **baru** (50 kelas, input 128 x 1024).
`model_ResNet18.pt` dari arsip lama tidak kompatibel: 49 kelas, input 384 x 64.

Tanpa akses ke Drive, bobot bisa dilatih sendiri di CPU dalam sekitar 15 menit —
lihat bagian "Melatih sendiri tanpa Colab" di bawah.

## Jalankan

```
python app.py
```

Buka `http://localhost:5000`.

### Kamera dari ponsel

`getUserMedia` hanya jalan di secure context. `http://192.168.x.x:5000` dari
ponsel **tidak** akan membuka kamera. Pilih salah satu:

```
flask --app app run --host 0.0.0.0 --cert=adhoc
```

atau tunnel (`cloudflared tunnel --url http://localhost:5000`). Tab Unggah tetap
berfungsi di HTTP biasa.

### Produksi

```
gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 app:app
```

Satu worker saja — tiap worker menduplikasi model di memori. Thread aman karena
operasi PyTorch melepas GIL. `torch.set_num_threads(2)` sudah diset di
`model_service.py`.

## Isi folder

| Berkas | Keterangan |
|---|---|
| `app.py` | Route Flask saja |
| `model_service.py` | Muat model, preprocessing, prediksi, Grad-CAM |
| `quality.py` | Gerbang mutu: ukuran, kontras, ketajaman, rasio tinta |
| `segment.py` | Potong baris dengan projection profile |
| `latih_lokal.py` | Melatih ulang satu fold di CPU, tanpa Colab |
| `kalibrasi_ambang.py` | Menghitung ambang "tidak dikenal" dari test set |
| `meta.json` | 50 nama kelas urut `label_id`, ukuran input, normalisasi, ambang |
| `contoh/` | 3 crop per penulis untuk panel pembanding (820 KB) |
| `templates/`, `static/` | Halaman, CSS, JS kamera |
| `Klasifikasi_Tulisan_Tangan.ipynb` | Notebook UAS yang menghasilkan bobot model (berisi output) |

Notebook disertakan sebagai asal-usul model: di situlah split data, ekstraksi
GLCM, dan pelatihan ResNet18 serta SqueezeNet dilakukan. Datasetnya sendiri
(81 MB) tidak masuk repo.

## Endpoint

| Route | Method | Keterangan |
|---|---|---|
| `/` | GET | Halaman utama |
| `/api/predict` | POST | Gambar -> top-5, keyakinan, flag `tidak_dikenal` |
| `/api/verify` | POST | Gambar + `penulis` -> skor 1:1 |
| `/api/gradcam` | POST | Gambar (+ `label_id`) -> PNG overlay |
| `/api/writers` | GET | Daftar 50 penulis |
| `/api/health` | GET | Model termuat, device, ambang |
| `/contoh/<penulis>/<berkas>` | GET | Contoh tulisan penulis |

Gambar dikirim sebagai multipart `image`, atau JSON `{"image": "data:image/jpeg;base64,..."}`.
Batas unggahan 10 MB.

Grad-CAM dipisah dari `/api/predict` karena butuh backward pass. Kalau digabung,
setiap prediksi jadi 2-3 kali lebih lambat padahal heatmap jarang dibuka.

## Pipeline di server

1. Decode + `exif_transpose` (foto ponsel sering tersimpan berotasi)
2. Pangkas ke area tulisan, **otomatis, hanya untuk gambar sebesar foto kamera**
3. Gerbang mutu — tolak dengan pesan spesifik, bukan "gagal"
4. Potong baris, **hanya jika diminta** (checkbox "foto satu halaman penuh")
5. Resize tinggi 128 px (rasio dijaga), pad putih sampai 1024 px
6. 3 channel, normalisasi ImageNet
7. Softmax, dirata-ratakan antar fold
8. Di bawah `ambang_keyakinan` -> "tidak dikenal"

Langkah 5 dan 6 **harus** sama dengan sel 5.2 notebook. Kalau berbeda, akurasi
turun tanpa satu pun pesan error.

Langkah 2 ada karena foto kamera gagal total tanpa itu. Preprocessing memakai
skala `min(128/tinggi, 1024/lebar)`, jadi foto potret 3000x4000 menyusut jadi
96x128 px di kanvas 1024x128 dan tulisannya tinggal beberapa piksel. Diuji pada
foto simulasi: tanpa pemangkasan 0 dari 25 benar, dengan pemangkasan 25 dari 25.
Pemangkasan dibatasi gambar setinggi minimal 600 px dengan cakupan tinta di
bawah 50%, supaya crop dataset tidak ikut tersentuh — memangkas crop dataset
justru menurunkan akurasi dari 0,9850 ke 0,9398. Pada 532 gambar uji, aturan ini
aktif di nol gambar dan akurasinya tetap 0,9850.

Langkah 4 mati secara default karena hasil pengukuran. Model dilatih pada gambar
dataset **utuh** yang dimampatkan ke tinggi 128, bukan pada hasil potong satu
baris, dan banyak gambar dataset memang berisi beberapa baris. Diuji pada 531
gambar uji, memotong aktif di 265 di antaranya dan menurunkan akurasi dari
**0,9849 ke 0,9454** — 23 gambar yang benar berubah jadi salah. Memotong baru
masuk akal untuk foto satu halaman penuh, yang bentuknya memang jauh dari data
latih.

Batas gerbang mutu juga diukur dari dataset, bukan dikarang: rasio tinta di
2.658 gambar berkisar 0,0017 sampai 0,613, jadi batasnya dipasang 0,0015 dan
0,75 supaya gambar yang sah tidak ikut tertolak.

## Melatih sendiri tanpa Colab

Kalau bobot Colab tidak tersedia, satu fold bisa dilatih ulang di CPU:

```
python latih_lokal.py --dataset "../data_uas/dataset_clean" --fold 1
```

Split, preprocessing, dan hyperparameter-nya sama persis dengan notebook, jadi
pembagian datanya identik (seed 42 → 532 uji, 426/425 per fold). Bobotnya tidak
akan sama dengan hasil Colab karena RNG dan hardware berbeda.

Hasil terukur di mesin 32 core: **15 menit**, berhenti di epoch 15 (terbaik
epoch 10), dan pada test set Precision 0,9863 / Recall 0,9848 / Akurasi 0,9850 /
F1 0,9848 — sebanding dengan 0,9812 milik fold 1 di Colab.

Skrip juga menulis `hasil_lokal/split_data.csv` supaya kalibrasi ambang bisa
langsung dijalankan.

## Kalibrasi ambang

```
python kalibrasi_ambang.py --dataset "../data_uas/dataset_clean" --split "hasil_lokal/split_data.csv" --tulis
```

Skrip mengukur sebaran keyakinan pada 532 gambar uji, mengambil persentil bawah
dari prediksi yang benar, lalu melaporkan berapa prediksi salah yang tersaring.

Hasil pada model hasil latih lokal: prediksi benar rata-rata 0,9632 (terendah
0,2278), prediksi salah rata-rata 0,4502 (tertinggi 0,9360). Persentil 1 memberi
ambang **0,4023**: 10 dari 532 gambar dijawab "tidak dikenal", dan 4 dari 8
prediksi salah tersaring.

Sebarannya tumpang tindih — ada prediksi salah dengan keyakinan 93,6% — jadi
tidak ada ambang yang bisa menyaring semua kesalahan. Ambang ini juga khusus
untuk bobot yang sedang dipakai; kalau bobotnya diganti, kalibrasi harus
diulang.

## Batas yang perlu disebut di laporan

- **Closed set.** Model hanya tahu 50 penulis ini. Tulisan orang lain akan
  dipaksa masuk ke salah satunya — itulah sebabnya ambang penolakan ada, dan
  sebabnya ambang itu harus dikalibrasi, bukan dikira-kira.
- Model dilatih dari crop satu baris pada satu sesi penulisan. Foto dengan
  kertas, pena, atau pencahayaan berbeda belum teruji.
- Gerbang mutu menangkap buram, gelap, dan kosong, tetapi **meloloskan derau**
  yang tajam dan berkontras. Penyaring akhir untuk kasus seperti itu adalah
  ambang keyakinan.
- Aplikasi tidak membaca isi tulisan. Itu tugas OCR, model lain.
